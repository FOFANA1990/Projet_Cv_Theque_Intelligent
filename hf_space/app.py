"""
app.py — Space Hugging Face (SDK Gradio) qui héberge Qwen2.5-7B-Instruct
(format GGUF quantifié Q4_K_M, via llama-cpp-python) pour l'extraction
structurée de CV.

Appelé depuis le backend Render via `gradio_client.Client(...).predict(...)`
— voir services/extractor.py côté backend.

Le modèle est téléchargé une seule fois au démarrage du Space (mis en
cache ensuite par Hugging Face), puis chargé en mémoire pour toute la
durée de vie du Space.
"""

import json
import os

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Compatibilité ZeroGPU ---
# Hugging Face exige qu'au moins une fonction soit décorée @spaces.GPU au
# démarrage d'un Space Gradio (sans quoi : "No @spaces.GPU function detected
# during startup"), y compris quand le matériel réel est CPU Basic. Cette
# application n'a jamais besoin de GPU (llama-cpp-python tourne en CPU pur) :
# la fonction ci-dessous ne fait rien, elle sert uniquement à satisfaire
# cette vérification de démarrage.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_noop() -> None:
        return None
except Exception:
    pass  # exécution locale ou environnement sans le paquet "spaces" : sans impact

MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"  # quantification Q4_K_M : qualité/mémoire équilibrées (~4.4 Go)
# Remarque : le dépôt officiel Qwen/Qwen2.5-7B-Instruct-GGUF scinde ce
# fichier en deux parties (...-00001-of-00002.gguf / -00002...) ; ce
# mirroir bartowski (très utilisé, fiable) le fournit en un seul fichier.

# Clé partagée optionnelle : définissez le secret EXTRACTION_API_KEY dans
# Settings > Repository secrets du Space pour exiger une clé à chaque
# appel. Laissez non défini pour un Space sans restriction d'accès.
API_KEY = os.environ.get("EXTRACTION_API_KEY", "")

print(f"Téléchargement du modèle {MODEL_REPO}/{MODEL_FILE} (une seule fois, mis en cache)...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

print("Chargement du modèle en mémoire (peut prendre une minute)...")
llm = Llama(
    model_path=model_path,
    n_ctx=8192,
    # os.cpu_count() renvoie souvent le nombre de coeurs de la machine HOTE
    # partagee (environnement conteneurise type Hugging Face Spaces), pas le
    # quota reellement alloue a ce Space -> demander trop de threads cree de
    # la contention et RALENTIT l'inference au lieu de l'accelerer (constate :
    # 1345s avec n_threads=2, puis 2567s -- deux fois plus lent -- avec
    # n_threads=os.cpu_count()). On plafonne volontairement a une valeur
    # modeste et fixe plutot que de faire confiance a la detection automatique.
    n_threads=4,
    verbose=False,
)
print("Modèle chargé — Space prêt.")


SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'extraction d'informations structurées à partir de CV (curriculum vitae).

Analyse le texte du CV fourni par l'utilisateur et extrait UNIQUEMENT les informations qui y sont explicitement présentes. N'invente rien et ne duplique jamais une information.

Réponds STRICTEMENT avec un objet JSON respectant exactement ce schéma (aucun champ supplémentaire, aucun texte en dehors du JSON) :

{
  "nom": "string",
  "prenom": "string",
  "email": "string",
  "telephone": "string",
  "poste_vise": "string",
  "competences": ["string", "..."],
  "experiences": [
    {"poste": "string", "entreprise": "string", "periode": "string", "description": "string"}
  ],
  "formations": [
    {"diplome": "string", "etablissement": "string", "annee": "string"}
  ],
  "langues": ["string", "..."],
  "centres_interet": ["string", "..."]
}

Règles importantes :
- Chaque expérience professionnelle distincte du CV doit apparaître UNE SEULE FOIS dans "experiences".
- Chaque formation distincte doit apparaître UNE SEULE FOIS dans "formations".
- "poste_vise" correspond à l'intitulé du poste recherché par le candidat, pas à un poste occupé dans le passé.
- Si une information n'est pas présente, laisse le champ vide ("" ou []) plutôt que d'inventer une valeur.
"""


def extraire(raw_text: str, api_key: str) -> str:
    """
    Reçoit le texte brut d'un CV + une clé d'accès, retourne le JSON
    structuré (en chaîne) extrait par le modèle.

    C'est cette fonction que le backend Render appelle via gradio_client.
    """
    if API_KEY and api_key != API_KEY:
        return json.dumps({"error": "unauthorized"})

    if not raw_text or not raw_text.strip():
        return json.dumps({"error": "texte vide"})

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text[:16000]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1200,  # largement suffisant pour le schema JSON attendu ;
                              # borne le pire des cas en temps de generation
        )
        return response["choices"][0]["message"]["content"]
    except Exception as exc:
        return json.dumps({"error": f"échec du modèle : {exc}"})


demo = gr.Interface(
    fn=extraire,
    inputs=[
        gr.Textbox(label="Texte brut du CV", lines=20),
        gr.Textbox(label="Clé API", type="password"),
    ],
    outputs=gr.Textbox(label="JSON extrait"),
    title="Extraction CV — Qwen2.5-7B-Instruct",
    description="API d'extraction structurée de CV (usage interne — backend CV-Thèque Intelligente).",
)

demo.launch()