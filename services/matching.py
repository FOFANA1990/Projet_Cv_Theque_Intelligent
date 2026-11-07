"""
services/matching.py
---------------------
Scoring des candidatures en base par rapport à une recherche recruteur
(formulaire "Confier une mission" de WizeSource).

Le scoring est délégué au Space Hugging Face (endpoint /scorer_lot, même
modèle Qwen2.5-7B que l'extraction) via gradio_client. Les profils sont
envoyés par petits lots : un seul accès GPU ZeroGPU par lot.

Sécurité des threads : la conversion Candidate (ORM) -> dict se fait en amont,
de façon synchrone ; les appels réseau ne manipulent ensuite que des dicts.
"""

import asyncio
import json
import logging
import re
import uuid

from config import settings
from database import SessionLocal
from models import Candidate, Mission, MissionMatch
from schemas import CandidateMatch, MissionSearchRequest
from services import extractor as extractor_mod

logger = logging.getLogger(__name__)

SCORING_API_NAME = "/scorer_lot"
BATCH_SIZE = 5  # <= MAX_PROFILES_PER_BATCH (8) côté Space


class MatchingError(Exception):
    """Levée quand le scoring échoue de façon inattendue."""


# ---------------------------------------------------------------------------
# Conversion ORM -> profil (dict Python simple)
# ---------------------------------------------------------------------------

def _candidate_to_profile(candidate: Candidate) -> dict:
    """
    Reconstruit un profil structuré (même forme que l'ancien JSON du LLM)
    à partir des tables normalisées liées au candidat. Suppose que les
    relations (competences, experiences, formations, langues,
    centres_interet) sont déjà chargées (eager loading côté requête).
    """
    return {
        "id": candidate.id,
        "nom": candidate.nom,
        "prenom": candidate.prenom,
        "email": candidate.email,
        "telephone": candidate.telephone,
        "poste_vise": candidate.poste_vise,
        "competences": [c.competence for c in candidate.competences],
        "experiences": [
            {
                "poste": e.poste,
                "entreprise": e.entreprise,
                "periode": e.periode,
                "description": e.description,
            }
            for e in candidate.experiences
        ],
        "formations": [
            {
                "diplome": f.diplome,
                "etablissement": f.etablissement,
                "annee": f.annee,
            }
            for f in candidate.formations
        ],
        "langues": [l.langue for l in candidate.langues],
        "centres_interet": [c.centre_interet for c in candidate.centres_interet],
    }


_WORD_PATTERN = re.compile(r"[a-zàâäéèêëïîôöùûüç0-9+#\.]{3,}", re.IGNORECASE)


def _extract_keywords(text: str) -> set[str]:
    """Extrait un ensemble de mots significatifs (3 caractères ou plus) d'un texte."""
    return {w.lower() for w in _WORD_PATTERN.findall(text or "")}


def _profile_searchable_text(profile: dict) -> str:
    """Concatène les champs du profil pertinents pour le préfiltrage rapide."""
    parts = [
        profile.get("poste_vise") or "",
        " ".join(profile.get("competences", [])),
    ]
    for exp in profile.get("experiences", []):
        parts.append(exp.get("poste", ""))
        parts.append(exp.get("description", ""))
    return " ".join(parts)


def prefilter_profiles(
    profiles: list[dict], criteria: MissionSearchRequest, top_n: int
) -> list[dict]:
    """
    Filtre rapide (sans appel LLM) qui ne garde que les `top_n` profils les
    plus susceptibles de correspondre à la recherche, par recoupement de
    mots-clés entre le profil candidat et la recherche recruteur.

    But : plafonner le nombre d'appels au LLM (donc le temps total de la
    recherche) indépendamment de la taille de la base de données. C'est une
    heuristique volontairement simple (pas de recherche sémantique) : un
    candidat pertinent mais formulé avec des synonymes différents peut être
    écarté à ce stade. Si la base grandit significativement, remplacer par
    une recherche plein-texte PostgreSQL ou des embeddings serait plus robuste.
    """
    if len(profiles) <= top_n:
        return profiles

    recherche_keywords = _extract_keywords(criteria.expertise) | _extract_keywords(criteria.message)

    scored: list[tuple[int, dict]] = []
    for profile in profiles:
        profile_keywords = _extract_keywords(_profile_searchable_text(profile))
        overlap = len(recherche_keywords & profile_keywords)
        scored.append((overlap, profile))

    # Tri par recoupement décroissant ; en cas d'égalité, on garde l'ordre
    # d'origine (stable) plutôt qu'un ordre arbitraire.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [profile for _, profile in scored[:top_n]]



def _profile_for_llm(profile: dict) -> dict:
    """Profil envoyé au Space : sans identifiants ni coordonnées (inutiles au scoring)."""
    return {k: v for k, v in profile.items() if k not in ("id", "email", "telephone")}


def _score_batch_sync(batch: list[dict], criteria: MissionSearchRequest) -> list[tuple[int, str]]:
    """Appelle /scorer_lot pour un lot ; retourne (score, justification) alignés sur `batch`."""
    payload = json.dumps(
        {
            "criteria": {
                "expertise": criteria.expertise,
                "engagement": criteria.engagement,
                "demarrage": criteria.demarrage,
                "message": criteria.message,
            },
            "profiles": [_profile_for_llm(p) for p in batch],
        },
        ensure_ascii=False,
    )

    try:
        client = extractor_mod._get_client()
        raw = client.predict(payload, settings.HF_SPACE_API_KEY, api_name=SCORING_API_NAME)
    except Exception as exc:
        extractor_mod._client = None  # connexion neuve au prochain essai
        raise MatchingError(f"Échec de l'appel au Space pour le scoring : {exc}") from exc

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MatchingError(f"Réponse du Space non-JSON : {str(raw)[:200]}") from exc

    if isinstance(data, dict) and "error" in data:
        raise MatchingError(f"Le Space a renvoyé une erreur : {data['error']}")
    if not isinstance(data, list):
        raise MatchingError("Réponse du Space inattendue (liste attendue).")

    by_index = {}
    for item in data:
        try:
            by_index[int(item["index"])] = (
                max(0, min(100, int(item.get("score", 0)))),
                str(item.get("justification", "")),
            )
        except (KeyError, TypeError, ValueError):
            continue

    return [
        by_index.get(i, (0, "Score indisponible (réponse du modèle inexploitable)."))
        for i in range(len(batch))
    ]


def _to_match(profile: dict, score: int, justification: str) -> CandidateMatch:
    return CandidateMatch(
        id=profile["id"],
        nom=profile["nom"],
        prenom=profile["prenom"],
        email=profile["email"],
        telephone=profile["telephone"],
        poste_vise=profile["poste_vise"],
        competences=profile["competences"],
        score=score,
        justification=justification,
    )


async def match_candidates(
    candidates: list[Candidate], criteria: MissionSearchRequest
) -> list[CandidateMatch]:
    """
    Score les candidats les plus pertinents et retourne les résultats triés
    par score décroissant. Les lots sont traités séquentiellement (ZeroGPU
    n'apporte rien à la concurrence). Lève MatchingError si TOUS les lots échouent.
    """
    profiles = [_candidate_to_profile(c) for c in candidates]
    a_scorer = prefilter_profiles(profiles, criteria, settings.MAX_CANDIDATES_TO_SCORE)
    if not a_scorer:
        return []

    results: list[CandidateMatch] = []
    failures = 0
    batches = [a_scorer[i : i + BATCH_SIZE] for i in range(0, len(a_scorer), BATCH_SIZE)]

    for batch in batches:
        try:
            scores = await asyncio.to_thread(_score_batch_sync, batch, criteria)
        except MatchingError as exc:
            failures += 1
            logger.error("Lot de %d profil(s) non scoré : %s", len(batch), exc)
            continue
        results.extend(_to_match(p, s, j) for p, (s, j) in zip(batch, scores))

    if failures == len(batches):
        raise MatchingError("Aucun lot n'a pu être scoré (Space indisponible ?).")

    results.sort(key=lambda m: m.score, reverse=True)
    return results

async def process_mission_matching(mission_id: uuid.UUID) -> None:
    """
    Tâche de fond complète pour une mission recruteur : charge la mission et
    les candidats en base (nouvelle session dédiée, indépendante de celle de
    la requête HTTP déjà terminée), lance le scoring, sauvegarde les
    résultats (table `mission_matches`) et met à jour le statut de la
    mission ("terminee" ou "echec").

    Les résultats ne sont JAMAIS renvoyés au recruteur — uniquement
    consultables par les administrateurs via les endpoints /api/v1/admin/...
    """
    from sqlalchemy.orm import selectinload  # import local pour éviter tout cycle

    db = SessionLocal()
    try:
        mission = db.query(Mission).filter(Mission.id == mission_id).first()
        if mission is None:
            logger.error("Mission %s introuvable pour le matching en tâche de fond.", mission_id)
            return

        candidates = (
            db.query(Candidate)
            .options(
                selectinload(Candidate.competences),
                selectinload(Candidate.experiences),
                selectinload(Candidate.formations),
                selectinload(Candidate.langues),
                selectinload(Candidate.centres_interet),
            )
            .all()
        )

        criteria = MissionSearchRequest(
            societe=mission.societe,
            nom=mission.nom_recruteur or "",
            email=mission.email_recruteur,
            engagement=mission.engagement or "",
            expertise=mission.expertise,
            demarrage=mission.demarrage or "",
            message=mission.description_texte or mission.expertise,
        )

        mission.statut = "en_cours"
        db.commit()

        resultats = await match_candidates(candidates, criteria)

        # Relance éventuelle : on remplace les anciens résultats de cette mission.
        db.query(MissionMatch).filter(MissionMatch.mission_id == mission.id).delete()

        for r in resultats:
            db.add(
                MissionMatch(
                    mission_id=mission.id,
                    candidate_id=r.id,
                    score=r.score,
                    justification=r.justification,
                )
            )

        mission.statut = "terminee"
        db.commit()
        logger.info(
            "Matching terminé pour la mission %s : %d résultat(s).", mission_id, len(resultats)
        )
    except Exception:
        db.rollback()
        logger.exception("Échec du matching en tâche de fond pour la mission %s", mission_id)
        try:
            mission = db.query(Mission).filter(Mission.id == mission_id).first()
            if mission is not None:
                mission.statut = "echec"
                db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "Échec supplémentaire lors de la mise à jour du statut d'échec pour %s", mission_id
            )
    finally:
        db.close()