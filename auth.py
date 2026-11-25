"""
auth.py
-------
Comptes candidats / recruteurs : création automatique au dépôt (CV ou mission),
connexion par e-mail + mot de passe, et espace personnel.

Principe :
- Un compte = (email, rôle) avec le rôle "candidat" ou "recruteur".
- Au dépôt d'un CV / d'une mission, le visiteur saisit un mot de passe :
    * e-mail inconnu  -> le compte est créé automatiquement ;
    * e-mail connu    -> le mot de passe doit correspondre, sinon refus (401).
- Les CV / missions restent rattachés au compte par l'e-mail (aucune colonne
  ajoutée aux tables existantes : seule la table `comptes` est nouvelle).
- Le recruteur ne voit JAMAIS les résultats du matching, seulement le statut.
"""

import logging
import secrets
import smtplib
import time
import uuid
from email.message import EmailMessage
from html import escape
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, Form, Header, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import Boolean, DateTime, String, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, Session, mapped_column

from config import settings
from database import Base, get_db
from models import Candidate, Mission

logger = logging.getLogger(__name__)

ROLES = ("candidat", "recruteur")
MIN_PASSWORD_LENGTH = 8
TOKEN_VALIDITY_HOURS = 12
VERIF_VALIDITY_HOURS = 48
RESEND_MIN_SECONDS = 60
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60

# Si JWT_SECRET n'est pas défini sur Render, on en génère un au démarrage :
# fonctionnel, mais les sessions sont perdues à chaque redémarrage.
_JWT_SECRET = settings.JWT_SECRET or secrets.token_hex(32)
if not settings.JWT_SECRET:
    logger.warning("JWT_SECRET non défini : secret aléatoire utilisé (sessions perdues au redémarrage).")


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

class Compte(Base):
    __tablename__ = "comptes"
    __table_args__ = (UniqueConstraint("email", "role", name="uq_compte_email_role"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    mot_de_passe_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    email_verifie: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    date_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


def ensure_schema(engine) -> None:
    """Ajoute la colonne `email_verifie` si la table `comptes` existait déjà (PostgreSQL)."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE comptes ADD COLUMN IF NOT EXISTS email_verifie BOOLEAN NOT NULL DEFAULT FALSE"))


# ---------------------------------------------------------------------------
# Mots de passe, jetons, limitation des tentatives
# ---------------------------------------------------------------------------

def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode("utf-8"))
    except ValueError:
        return False


def create_token(compte: Compte) -> str:
    payload = {
        "sub": str(compte.id),
        "email": compte.email,
        "role": compte.role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_VALIDITY_HOURS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm="HS256")


_failed: dict[str, list[float]] = {}


def _check_not_locked(key: str) -> None:
    now = time.time()
    recent = [t for t in _failed.get(key, []) if now - t < LOCKOUT_SECONDS]
    _failed[key] = recent
    if len(recent) >= MAX_FAILED_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Trop de tentatives. Réessayez dans quelques minutes.",
        )


def _register_failure(key: str) -> None:
    _failed.setdefault(key, []).append(time.time())


def _clear_failures(key: str) -> None:
    _failed.pop(key, None)


_last_sent: dict[str, float] = {}


def create_verification_token(compte: Compte) -> str:
    payload = {
        "sub": str(compte.id),
        "purpose": "verif",
        "exp": datetime.now(timezone.utc) + timedelta(hours=VERIF_VALIDITY_HOURS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm="HS256")


def _send_mail(to: str, subject: str, text_body: str, html_body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{settings.MAIL_FROM_NAME} <{settings.MAIL_FROM}>"
    msg["To"] = to
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    with smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT, timeout=20) as smtp:
        if settings.MAIL_STARTTLS:
            smtp.starttls()
        if settings.MAIL_USE_CREDENTIALS and settings.MAIL_USERNAME:
            smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
        smtp.send_message(msg)


def send_verification_email(compte: Compte, force: bool = False) -> None:
    """Envoie le lien de confirmation (au plus un envoi par minute et par compte)."""
    now = time.time()
    if not force and now - _last_sent.get(str(compte.id), 0) < RESEND_MIN_SECONDS:
        return
    link = f"{settings.PUBLIC_BASE_URL.rstrip('/')}/api/v1/auth/verifier?token={create_verification_token(compte)}"
    texte = (
        "Bonjour,\n\nPour activer votre compte, confirmez votre adresse e-mail en ouvrant ce lien "
        f"(valable {VERIF_VALIDITY_HOURS} h) :\n{link}\n\n"
        "Si vous n'êtes pas à l'origine de cette demande, ignorez ce message."
    )
    html = (
        "<p>Bonjour,</p><p>Pour activer votre compte, confirmez votre adresse e-mail :</p>"
        f'<p><a href="{escape(link)}" style="background:#002A3F;color:#fff;padding:12px 20px;text-decoration:none">'
        "Confirmer mon e-mail</a></p>"
        f"<p style=\"color:#666;font-size:13px\">Lien valable {VERIF_VALIDITY_HOURS} h. Si vous n'êtes pas à l'origine de cette demande, ignorez ce message.</p>"
    )
    try:
        _send_mail(compte.email, "Confirmez votre adresse e-mail", texte, html)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Envoi de l'e-mail de confirmation impossible pour %s", compte.email)
        raise HTTPException(
            status_code=502, detail="Impossible d'envoyer l'e-mail de confirmation pour le moment. Réessayez plus tard."
        ) from exc
    _last_sent[str(compte.id)] = now


def authenticate_or_create(db: Session, email: str, password: str, role: str) -> Compte:
    """
    Cœur du dépôt sécurisé : crée le compte si l'e-mail est nouveau, sinon
    vérifie le mot de passe. Lève 401 / 422 / 429 en cas de problème.
    """
    email = normalize_email(email)
    if role not in ROLES:
        raise HTTPException(status_code=422, detail="Rôle de compte invalide.")
    if not email or "@" not in email:
        raise HTTPException(status_code=422, detail="Adresse e-mail invalide.")

    key = f"{role}:{email}"
    _check_not_locked(key)

    compte = db.query(Compte).filter(Compte.email == email, Compte.role == role).first()

    if compte is None:
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"Le mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères.",
            )
        compte = Compte(email=email, role=role, mot_de_passe_hash=hash_password(password))
        db.add(compte)
        db.commit()
        db.refresh(compte)
        logger.info("Compte %s créé (e-mail à confirmer) pour %s", role, email)
        send_verification_email(compte, force=True)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Compte créé. Un e-mail de confirmation vient d'être envoyé à {email} : "
                   "cliquez sur le lien qu'il contient, puis connectez-vous.",
        )

    if not verify_password(password or "", compte.mot_de_passe_hash):
        _register_failure(key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Un compte existe déjà avec cet e-mail : mot de passe incorrect.",
        )
    _clear_failures(key)

    if not compte.email_verifie:
        send_verification_email(compte)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Adresse e-mail pas encore confirmée. Ouvrez le lien envoyé à {email} "
                   "(un nouveau lien vient de vous être envoyé si besoin), puis connectez-vous.",
        )
    return compte


def get_current_compte(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
) -> Compte:
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Connexion requise.")
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=["HS256"])
        compte = db.get(Compte, uuid.UUID(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Session expirée, reconnectez-vous.")
    if compte is None:
        raise HTTPException(status_code=401, detail="Compte introuvable.")
    return compte


# ---------------------------------------------------------------------------
# Routes : connexion + espace personnel
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1", tags=["Compte"])


class TokenResponse(BaseModel):
    token: str
    email: str
    role: str


@router.post("/auth/login", response_model=TokenResponse, summary="Connexion (e-mail + mot de passe)")
def login(
    email: str = Form(...),
    mot_de_passe: str = Form(...),
    role: str = Form(...),
    db: Session = Depends(get_db),
) -> TokenResponse:
    email_n = normalize_email(email)
    key = f"{role}:{email_n}"
    _check_not_locked(key)

    compte = db.query(Compte).filter(Compte.email == email_n, Compte.role == role).first()
    if compte is None or not verify_password(mot_de_passe, compte.mot_de_passe_hash):
        _register_failure(key)
        raise HTTPException(status_code=401, detail="E-mail ou mot de passe incorrect.")
    _clear_failures(key)
    if not compte.email_verifie:
        send_verification_email(compte)
        raise HTTPException(
            status_code=403,
            detail=f"Adresse e-mail pas encore confirmée : ouvrez le lien envoyé à {compte.email}.",
        )
    return TokenResponse(token=create_token(compte), email=compte.email, role=compte.role)


_PAGE = (
    '<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    "<title>Confirmation e-mail</title></head>"
    '<body style="font-family:system-ui,sans-serif;background:#EAF3F6;margin:0;display:grid;place-items:center;min-height:100vh">'
    '<div style="background:#fff;border:1px solid #cfdde3;padding:32px;max-width:440px;margin:16px">'
    "<h1 style=\"margin:0 0 12px;font-size:1.3rem;color:#002A3F\">{titre}</h1><p>{corps}</p>{suite}</div></body></html>"
)


@router.get("/auth/verifier", response_class=HTMLResponse, include_in_schema=False)
def verifier_email(token: str = "", db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=["HS256"])
        if payload.get("purpose") != "verif":
            raise ValueError
        compte = db.get(Compte, uuid.UUID(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        return HTMLResponse(
            _PAGE.format(titre="Lien invalide ou expiré", corps="Renvoyez votre formulaire pour recevoir un nouveau lien.", suite=""),
            status_code=400,
        )
    if compte is None:
        return HTMLResponse(_PAGE.format(titre="Compte introuvable", corps="Ce compte n'existe plus.", suite=""), status_code=404)
    compte.email_verifie = True
    db.commit()
    retour = "/missions" if compte.role == "recruteur" else "/cv"
    suite = (
        f'<p><a href="{retour}" style="color:#002A3F;font-weight:600">Retourner au formulaire</a> · '
        '<a href="/espace" style="color:#002A3F;font-weight:600">Mon espace</a></p>'
    )
    return HTMLResponse(
        _PAGE.format(titre="E-mail confirmé ✔", corps="Votre compte est activé. Vous pouvez maintenant envoyer votre dépôt.", suite=suite)
    )


def issue_token(compte: Compte) -> str:
    return create_token(compte)


class MonCV(BaseModel):
    id: uuid.UUID
    nom_fichier: str
    statut: str
    poste_vise: str | None
    date_reception: datetime


class MaMission(BaseModel):
    id: uuid.UUID
    societe: str
    expertise: str
    engagement: str | None
    statut: str
    date_reception: datetime


@router.get("/me/cv", response_model=list[MonCV], summary="Mes CV déposés (candidat)")
def mes_cv(compte: Compte = Depends(get_current_compte), db: Session = Depends(get_db)):
    if compte.role != "candidat":
        raise HTTPException(status_code=403, detail="Espace réservé aux candidats.")
    return (
        db.query(Candidate)
        .filter(func.lower(Candidate.email) == compte.email)
        .order_by(Candidate.date_reception.desc())
        .all()
    )


@router.delete("/me/cv/{cv_id}", summary="Supprimer l'un de mes CV et toutes ses données")
def supprimer_cv(cv_id: uuid.UUID, compte: Compte = Depends(get_current_compte), db: Session = Depends(get_db)):
    if compte.role != "candidat":
        raise HTTPException(status_code=403, detail="Espace réservé aux candidats.")
    cand = db.query(Candidate).filter(Candidate.id == cv_id, func.lower(Candidate.email) == compte.email).first()
    if cand is None:
        raise HTTPException(status_code=404, detail="CV introuvable.")
    db.delete(cand)
    db.commit()
    return {"supprime": True}


@router.get("/me/missions", response_model=list[MaMission], summary="Mes missions déposées (recruteur)")
def mes_missions(compte: Compte = Depends(get_current_compte), db: Session = Depends(get_db)):
    if compte.role != "recruteur":
        raise HTTPException(status_code=403, detail="Espace réservé aux recruteurs.")
    return (
        db.query(Mission)
        .filter(func.lower(Mission.email_recruteur) == compte.email)
        .order_by(Mission.date_reception.desc())
        .all()
    )


@router.delete("/me/missions/{mission_id}", summary="Supprimer l'une de mes missions")
def supprimer_mission(mission_id: uuid.UUID, compte: Compte = Depends(get_current_compte), db: Session = Depends(get_db)):
    if compte.role != "recruteur":
        raise HTTPException(status_code=403, detail="Espace réservé aux recruteurs.")
    m = db.query(Mission).filter(Mission.id == mission_id, func.lower(Mission.email_recruteur) == compte.email).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Mission introuvable.")
    db.delete(m)
    db.commit()
    return {"supprime": True}


@router.delete("/me", summary="Supprimer mon compte")
def supprimer_compte(compte: Compte = Depends(get_current_compte), db: Session = Depends(get_db)):
    db.delete(compte)
    db.commit()
    return {"supprime": True}


def authorize_deposit(db: Session, email: str, mot_de_passe: str, role: str, authorization: str = "") -> Compte:
    """
    Autorise un dépôt (CV ou mission). Cas normal : le visiteur est connecté
    (en-tête Authorization: Bearer ...) et l'e-mail du formulaire est celui de
    son compte. Repli : e-mail + mot de passe dans le formulaire.
    """
    if authorization.strip():
        compte = get_current_compte(authorization, db)
        if compte.role != role:
            raise HTTPException(status_code=403, detail=f"Ce dépôt est réservé aux comptes « {role} ».")
        if compte.email != normalize_email(email):
            raise HTTPException(status_code=403, detail="L'e-mail du formulaire doit être celui de votre compte connecté.")
        return compte
    return authenticate_or_create(db, email, mot_de_passe, role)


@router.post("/auth/inscription", summary="Créer un compte (un e-mail de confirmation est envoyé)")
def inscription(
    email: str = Form(...),
    mot_de_passe: str = Form(...),
    role: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        authenticate_or_create(db, email, mot_de_passe, role)
    except HTTPException as exc:
        if exc.status_code == 403:  # compte créé / e-mail à confirmer : message d'information
            return {"ok": True, "message": exc.detail}
        raise
    return {"ok": True, "message": "Ce compte existe déjà et est confirmé : connectez-vous."}