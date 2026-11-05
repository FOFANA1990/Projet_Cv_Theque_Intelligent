"""
services/extractor.py
----------------------
Pipeline d'extraction structurée des informations d'un CV, en s'appuyant
sur Qwen2.5-7B-Instruct hébergé sur un Space Hugging Face (SDK Gradio),
appelé via gradio_client.

Architecture :
    Render (ce backend) --gradio_client--> HF Space (modèle Qwen2.5)

Remplace l'ancienne version qui appelait un serveur Ollama. Le schéma de
sortie, les règles d'extraction et la fonction publique `extract_cv_info`
restent identiques — seul le mécanisme d'appel change, aucun autre fichier
du projet n'a besoin d'être modifié.

Configuration requise (config.py / variables d'environnement Render) :
    HF_SPACE_URL      ex. "FOFANA1990/cv-extractor-qwen"
    HF_SPACE_API_KEY  même valeur que le secret EXTRACTION_API_KEY du Space
    HF_TOKEN          token Hugging Face en LECTURE (obligatoire si le Space
                      est privé, ce qui est le cas ici)
"""

import json
import logging
import time

from gradio_client import Client

from config import settings
from schemas import CandidateCVData

logger = logging.getLogger(__name__)

# Gradio 6.x nomme l'endpoint d'après la fonction Python : `def extraire(...)`
# dans hf_space/app.py  ->  "/extraire"  (et non "/predict" comme avant).
API_NAME = "/extraire"

# Un Space gratuit s'endort après 48 h d'inactivité : au réveil, il doit
# recharger le modèle, ce qui prend un peu de temps. On réessaie donc la
# connexion plusieurs fois avant d'abandonner.
CONNECT_ATTEMPTS = 6
CONNECT_DELAY_SECONDS = 30


class ExtractionError(Exception):
    """Levée quand le pipeline LLM échoue à produire une extraction exploitable."""


_client: Client | None = None


def _get_client() -> Client:
    """
    Construit (une seule fois) le client vers le Space, avec nouvelles
    tentatives si le Space est en cours de réveil.
    """
    global _client
    if _client is not None:
        return _client

    if not settings.HF_SPACE_URL:
        raise ExtractionError(
            "HF_SPACE_URL n'est pas configuré (ex. 'FOFANA1990/cv-extractor-qwen') : "
            "ajoutez cette variable d'environnement sur Render."
        )

    last_exc: Exception | None = None
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            _client = Client(
                settings.HF_SPACE_URL,
                token=settings.HF_TOKEN or None,
                verbose=False,
            )
            return _client
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Connexion au Space '%s' impossible (tentative %d/%d) : %s",
                settings.HF_SPACE_URL, attempt, CONNECT_ATTEMPTS, exc,
            )
            if attempt < CONNECT_ATTEMPTS:
                time.sleep(CONNECT_DELAY_SECONDS)

    raise ExtractionError(
        f"Impossible de se connecter au Space Hugging Face '{settings.HF_SPACE_URL}' "
        f"après {CONNECT_ATTEMPTS} tentatives. Vérifiez : le Space est-il démarré (pas en "
        f"erreur) ? HF_SPACE_URL est-il correct ? Le Space est privé : HF_TOKEN est-il "
        f"défini avec un token valide ? Dernière erreur : {last_exc}"
    ) from last_exc


def _call_space(raw_text: str) -> str:
    """Appelle le Space et retourne le contenu brut de sa réponse (JSON en chaîne)."""
    global _client

    truncated_text = raw_text[: settings.MAX_INPUT_CHARS]
    client = _get_client()

    try:
        result = client.predict(
            truncated_text,
            settings.HF_SPACE_API_KEY,
            api_name=API_NAME,
        )
    except Exception as exc:
        # On oublie le client en cache : si le Space a redémarré entre-temps,
        # le prochain appel repartira sur une connexion neuve.
        _client = None
        raise ExtractionError(
            f"Échec de l'appel au Space Hugging Face '{settings.HF_SPACE_URL}' : {exc}"
        ) from exc

    if not result or not str(result).strip():
        raise ExtractionError("Le Space a renvoyé une réponse vide.")

    return result


def extract_cv_info(raw_text: str) -> CandidateCVData:
    """
    Exécute le pipeline d'extraction sur le texte brut d'un CV et retourne
    les données structurées et validées via Pydantic.

    Args:
        raw_text: texte brut extrait du CV (voir services/parser.py).

    Returns:
        Une instance validée de `CandidateCVData`.

    Raises:
        ExtractionError: si l'appel au Space ou le parsing/la validation échoue.
    """
    if not raw_text.strip():
        raise ExtractionError("Texte source vide : rien à extraire.")

    content = _call_space(raw_text)

    try:
        parsed_json = json.loads(content)
    except json.JSONDecodeError as exc:
        preview = str(content)[:500].replace("\n", "\\n")
        logger.error("Réponse du Space non-JSON (extrait) : %s", preview)
        raise ExtractionError(f"La réponse du Space n'est pas un JSON valide : {exc}") from exc

    if isinstance(parsed_json, dict) and "error" in parsed_json:
        raise ExtractionError(f"Le Space a renvoyé une erreur : {parsed_json['error']}")

    try:
        return CandidateCVData.model_validate(parsed_json)
    except Exception as exc:
        raise ExtractionError(
            f"La sortie du Space ne respecte pas le schéma attendu : {exc}"
        ) from exc