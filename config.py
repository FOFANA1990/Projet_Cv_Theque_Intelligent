"""
config.py
---------
Configuration centralisée du projet via Pydantic Settings.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "CV-Thèque Intelligente API"
    APP_ENV: str = Field(default="development")
    DEBUG: bool = Field(default=True)

    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    DATABASE_URL: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/cv_theque"
    )

    # --- Extraction du CV via le Space Hugging Face (Qwen2.5-7B-Instruct) ---
    # HF_SPACE_URL     : "utilisateur/nom-du-space" (ex. FOFANA1990/cv-extractor-qwen)
    # HF_SPACE_API_KEY : même valeur que le secret EXTRACTION_API_KEY du Space
    # HF_TOKEN         : token Hugging Face en lecture (utile si le Space est privé)
    HF_SPACE_URL: str = Field(default="")
    HF_SPACE_API_KEY: str = Field(default="")
    HF_TOKEN: str = Field(default="")

    # --- Ollama (encore utilisé par le matching / scoring) ---
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="qwen2.5:7b-instruct")
    OLLAMA_AUTH_USER: str = Field(default="")
    OLLAMA_AUTH_PASSWORD: str = Field(default="")
    OLLAMA_NUM_CTX: int = Field(default=8192)
    OLLAMA_SCORING_MODEL: str = Field(default="qwen2.5:3b-instruct")
    MATCHING_CONCURRENCY: int = Field(default=3)
    MAX_CANDIDATES_TO_SCORE: int = Field(default=15)
    MAX_INPUT_CHARS: int = Field(default=16000)

    TESSERACT_CMD: str | None = Field(default=None)

    ADMIN_API_KEY: str = Field(default="changez-moi-en-dev")

    # Secret de signature des sessions (comptes candidats/recruteurs).
    # À définir sur Render (ex. 64 caractères aléatoires).
    JWT_SECRET: str = Field(default="")

    # URL publique du site (liens de confirmation e-mail).
    PUBLIC_BASE_URL: str = Field(default="https://projet-cv-theque-intelligent.onrender.com")

    UPLOAD_TMP_DIR: str = Field(default="./tmp_uploads")
    MAX_UPLOAD_SIZE_MB: int = Field(default=10)

    MAIL_USERNAME: str = Field(default="")
    MAIL_PASSWORD: str = Field(default="")
    MAIL_FROM: str = Field(default="no-reply@etech-origin.com")
    MAIL_FROM_NAME: str = Field(default="ETECH ORIGIN — Recrutement")
    MAIL_PORT: int = Field(default=587)
    MAIL_SERVER: str = Field(default="smtp.gmail.com")
    MAIL_STARTTLS: bool = Field(default=True)
    MAIL_SSL_TLS: bool = Field(default=False)
    MAIL_USE_CREDENTIALS: bool = Field(default=True)
    MAIL_VALIDATE_CERTS: bool = Field(default=True)

    # --- API France Travail (recherche d'offres d'emploi externes) ---
    # Inscription gratuite sur https://francetravail.io -> créer une
    # application -> souscrire à l'API "Offres d'emploi v2" -> récupérer
    # Client ID / Client Secret. Sans ces identifiants, la recherche
    # d'offres externes est désactivée proprement (message clair côté admin).
    FRANCE_TRAVAIL_CLIENT_ID: str = Field(default="")
    FRANCE_TRAVAIL_CLIENT_SECRET: str = Field(default="")
    FRANCE_TRAVAIL_SCOPE: str = Field(default="api_offresdemploiv2 o2dsoffre")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()