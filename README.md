# 🤝 RH·IA — Chatbot RH intelligent + Matching CV/Poste

> Outil RH augmenté par IA : posez des questions sur des CVs en langage naturel et obtenez un score de matching automatique entre un candidat et une fiche de poste.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-orange?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat)

---

## 🎯 Problème résolu

Les recruteurs passent en moyenne **23 secondes** à lire un CV (source : Ladders Study).
Ce projet automatise deux tâches chronophages :
1. **Interroger des CVs** en langage naturel sans lire chaque page
2. **Scorer automatiquement** la compatibilité candidat/poste

---

## 🏗️ Architecture RAG

```
┌──────────────────────────────────────────────────────────┐
│                     INGESTION (offline)                   │
│                                                           │
│  PDF (CV / Fiche de poste)                               │
│       ↓                                                   │
│  PyPDFLoader → texte brut                                │
│       ↓                                                   │
│  RecursiveCharacterTextSplitter                          │
│  (chunk_size=800, overlap=100)                           │
│       ↓                                                   │
│  OpenAI text-embedding-3-small → vecteurs 1536 dim       │
│       ↓                                                   │
│  ChromaDB (in-memory vector store)                       │
└──────────────────────┬───────────────────────────────────┘
                        │
┌──────────────────────▼───────────────────────────────────┐
│                     RETRIEVAL (online)                    │
│                                                           │
│  Question utilisateur → embedding → similarité cosinus   │
│       ↓                                                   │
│  Top-4 chunks les plus pertinents                        │
│       ↓                                                   │
│  Injection dans prompt template (contexte + question)    │
│       ↓                                                   │
│  GPT-4o-mini → réponse ancrée sur les documents          │
└──────────────────────────────────────────────────────────┘
```

### Pourquoi ces choix ?

| Composant | Choix | Raison |
|-----------|-------|--------|
| Splitter | RecursiveCharacterTextSplitter | Coupe sur paragraphes → préserve le contexte sémantique |
| Embeddings | text-embedding-3-small | Rapport qualité/coût optimal, 1536 dim |
| Vector store | ChromaDB | Zéro config, in-memory pour prototype |
| LLM | GPT-4o-mini | Temp=0.2 pour réponses factuelles |
| k retrieval | 4 | Compromis contexte suffisant / token budget |
| Chain type | "stuff" | Concatène les chunks — OK si k≤6 |

---

## ✨ Fonctionnalités

### Tab 1 — Chatbot RH
- Upload multi-CVs (PDF)
- Indexation automatique dans ChromaDB
- Questions en langage naturel
- Réponses avec **citation des sources** (nom du fichier)
- Questions suggérées pré-définies
- Historique de conversation

### Tab 2 — Matching CV / Poste
- Upload CV + fiche de poste (PDF)
- **Extraction automatique** des exigences via LLM
- **Vérification compétence par compétence** dans le CV
- Score pondéré : technique (65%) + soft skills (35%)
- Détail avec evidence extraite du CV pour chaque compétence

---

## 🚀 Installation

```bash
git clone https://github.com/Haristocratee/rh-ia-chatbot.git
cd rh-ia-chatbot

python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## 📐 Prompt Engineering — Matching

Le scoring hybride évite les hallucinations :

```
1. LLM extrait les exigences de la fiche de poste → JSON structuré
2. Pour chaque exigence :
   a. Retrieval : top-3 chunks du CV les plus similaires
   b. LLM vérifie : présent ou non + cite l'evidence
3. Score = moyenne pondérée (tech × 0.65 + soft × 0.35)
```

**Pourquoi ne pas faire un simple cosine score CV ↔ Poste ?**
→ Manque de granularité : on ne sait pas QUELLE compétence manque.
→ L'approche hybride donne un résultat exploitable par un RH non-technique.

---

## 🔮 Améliorations prévues

- [ ] Support multi-candidats avec classement automatique
- [ ] Export rapport PDF du matching
- [ ] Mémoire conversationnelle (LangChain ConversationBufferMemory)
- [ ] Mode batch : scorer 50 CVs vs 1 fiche de poste automatiquement
- [ ] Alternative gratuite : sentence-transformers (zéro coût embedding)

---

## 👤 Auteur

**Harry Patrice TEGUE KEMGNE**  
Alternant Data Scientist / AI Engineer  
📧 harrytegue@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/harry-tegue-0aa9a4288/) | [GitHub](https://github.com/Haristocratee)
