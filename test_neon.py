"""Teste la connexion à Neon (PostgreSQL).

Usage :
    pip install sqlalchemy psycopg2-binary
    python test_neon.py

Collez la chaîne de connexion quand elle est demandée (elle ne s'affiche pas).
Format attendu :
    postgresql+psycopg2://USER:MOTDEPASSE@HOTE/neondb?sslmode=require
"""

import getpass

from sqlalchemy import create_engine, text


def main() -> None:
    url = getpass.getpass("Collez la chaîne Neon corrigée (rien ne s'affiche) : ").strip()

    if not url.startswith("postgresql+psycopg2://"):
        print("ERREUR : la chaîne doit commencer par 'postgresql+psycopg2://'")
        return
    if "channel_binding" in url:
        print("ERREUR : supprimez '&channel_binding=require' à la fin de la chaîne.")
        return

    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            version = conn.execute(text("select version()")).scalar()
        print("Connexion OK ->", version)
    except Exception as exc:
        # On n'affiche que le type et le début du message, jamais la chaîne.
        print("ECHEC de connexion :", type(exc).__name__)
        print(str(exc).splitlines()[0][:300])


if __name__ == "__main__":
    main()
