"""
app.py — RH·IA : Chatbot RH + Matching CV/Poste
═══════════════════════════════════════════════════
Stack : Streamlit · LangChain · ChromaDB · OpenAI · PyPDF2

Architecture RAG complète :
  1. Ingestion : PDF → texte brut → chunks (RecursiveCharacterTextSplitter)
  2. Indexation : chunks → embeddings (OpenAI text-embedding-3-small) → ChromaDB
  3. Retrieval : question → embedding → similarité cosinus → top-k chunks
  4. Génération : chunks récupérés + question → LLM → réponse ancrée

NOTE ENTRETIEN — Pourquoi ChromaDB ?
  → Vector store in-memory, zéro config, parfait pour prototype
  → En production : Pinecone (managed) ou pgvector (PostgreSQL)
  → FAISS est plus rapide mais moins pratique pour les métadonnées
"""

import streamlit as st
import os
import tempfile
from pathlib import Path

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RH·IA — Chatbot & Matching",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;700;800&family=Instrument+Sans:wght@300;400;500&display=swap');

    :root {
        --bg: #f8f7f4;
        --surface: #ffffff;
        --surface2: #f0ede8;
        --border: #e2ddd8;
        --accent: #1a1a2e;
        --accent2: #e63946;
        --text: #1a1a1e;
        --muted: #6b6b7a;
        --success: #2d6a4f;
        --success-bg: #d8f3dc;
        --warning: #b5451b;
        --warning-bg: #fde8d8;
    }

    html, body, [class*="css"] {
        font-family: 'Instrument Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }
    h1,h2,h3,h4 { font-family: 'Cabinet Grotesk', sans-serif; }
    .main { background-color: var(--bg); }
    .block-container { padding: 2rem 2.5rem; max-width: 1200px; }

    [data-testid="stSidebar"] {
        background-color: var(--accent);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stTextInput input {
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.2) !important;
        color: white !important;
    }

    /* Header */
    .main-header {
        background: var(--accent);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: var(--accent2);
        border-radius: 50%;
        opacity: 0.15;
    }
    .main-title {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    .main-subtitle { color: rgba(255,255,255,0.65); font-size: 0.92rem; margin-top: 0.4rem; }

    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: var(--accent);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Score */
    .score-circle {
        width: 120px; height: 120px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        border: 4px solid;
    }
    .score-value {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label { font-size: 0.7rem; color: var(--muted); }

    /* Messages chatbot */
    .msg-user {
        background: var(--accent);
        color: white;
        padding: 0.9rem 1.2rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.5rem 0 0.5rem 3rem;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .msg-bot {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        padding: 0.9rem 1.2rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 3rem 0.5rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .msg-sources {
        font-size: 0.75rem;
        color: var(--muted);
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid var(--border);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .badge-green { background: var(--success-bg); color: var(--success); }
    .badge-red { background: var(--warning-bg); color: var(--warning); }
    .badge-gray { background: var(--surface2); color: var(--muted); }

    /* Boutons */
    .stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 700;
        padding: 0.55rem 1.2rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.8; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* Progress bar custom */
    .progress-bar-container {
        background: var(--surface2);
        border-radius: 99px;
        height: 10px;
        width: 100%;
        margin: 0.3rem 0;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #1a1a2e, #e63946);
        transition: width 0.5s ease;
    }

    .stTextInput input, .stTextArea textarea {
        border-radius: 8px !important;
        border-color: var(--border) !important;
    }

    hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE — Persistance entre reruns Streamlit
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_rh" not in st.session_state:
    st.session_state.vectorstore_rh = None
if "vectorstore_match" not in st.session_state:
    st.session_state.vectorstore_match = None
if "docs_charges" not in st.session_state:
    st.session_state.docs_charges = []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input(
        "Clé API OpenAI",
        type="password",
        placeholder="sk-...",
        help="Nécessaire pour embeddings + LLM"
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.markdown("✅ Clé configurée")
    else:
        st.markdown("⚠️ Clé requise pour fonctionner")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; line-height:1.8; opacity:0.7'>
    <b>Stack technique</b><br>
    LangChain · ChromaDB<br>
    OpenAI Embeddings<br>
    GPT-4o-mini · PyPDF2<br><br>
    <b>Pattern</b><br>
    RAG (Retrieval-Augmented Generation)<br><br>
    <b>Auteur</b><br>
    Harry Patrice TEGUE KEMGNE<br>
    Alternant Data Scientist / AI Engineer
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">RH·IA</p>
    <p class="main-subtitle">Chatbot RH intelligent + Matching CV/Poste — Propulsé par RAG & GPT-4o-mini</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FONCTIONS CORE
# ─────────────────────────────────────────────

def load_and_chunk_pdfs(uploaded_files: list, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
    """
    Charge des PDFs uploadés et les découpe en chunks.

    Pourquoi RecursiveCharacterTextSplitter ?
    → Essaie d'abord de couper sur les paragraphes (\n\n),
      puis les phrases (\n), puis les mots ( ).
    → Préserve mieux le contexte sémantique qu'un simple split fixe.

    chunk_size=800 : compromis entre contexte suffisant et token budget LLM
    chunk_overlap=100 : évite de couper une info à cheval sur deux chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    all_docs = []
    for uploaded_file in uploaded_files:
        # Streamlit donne un BytesIO — PyPDFLoader a besoin d'un vrai fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Ajout métadonnée : nom du fichier source (utile pour les citations)
        for page in pages:
            page.metadata["source_file"] = uploaded_file.name

        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)
        Path(tmp_path).unlink()  # Nettoyage fichier temp

    return all_docs


def build_vectorstore(docs: list, collection_name: str = "rh_docs") -> FAISS:
    """
    Construit le vector store FAISS depuis une liste de documents.

    Embedding model : text-embedding-3-small
    → Dimension : 1536 vectorielles par chunk
    → Coût : ~$0.00002 / 1000 tokens

    Pourquoi FAISS plutôt que ChromaDB ici ?
    → FAISS est plus stable sur Python 3.14 (Streamlit Cloud)
    → Zéro dépendance système, pure Python + C bindings
    → En production on garderait ChromaDB/pgvector pour la persistance
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore


def build_rag_chain(vectorstore: FAISS, system_context: str = "RH"):
    """
    Construit la chaîne RAG avec LCEL (LangChain Expression Language).

    LCEL remplace les anciennes Chain classes (RetrievalQA, etc.)
    depuis LangChain 0.2. Syntaxe pipe : retriever | prompt | llm | parser

    k=4 : on récupère les 4 chunks les plus similaires à la question
    Temperature=0.2 : réponses factuelles, peu de créativité
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt_template = f"""Tu es un assistant RH expert. Réponds uniquement à partir du contexte fourni.
Contexte : {system_context}

Règles strictes :
- Réponds uniquement avec les informations présentes dans le CONTEXTE
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis, concis et professionnel
- Ne jamais inventer de compétences ou d'expériences non mentionnées

CONTEXTE :
{{context}}

QUESTION : {{question}}

RÉPONSE :"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # LCEL chain : syntaxe moderne LangChain 0.2+
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def compute_matching_score(cv_store: Chroma, poste_store: Chroma, llm) -> dict:
    """
    Calcule le score de matching entre un CV et une fiche de poste.

    Méthode :
    1. On extrait les compétences requises de la fiche de poste (via LLM)
    2. Pour chaque compétence, on query le CV store
    3. Le LLM évalue si la compétence est présente dans le CV (0/1)
    4. Score global = moyenne pondérée

    NOTE ENTRETIEN :
    Une approche plus robuste utiliserait des embeddings directs
    (similarité cosinus entre chunk fiche de poste et chunk CV).
    Ici on hybride : LLM pour l'extraction + retrieval pour la vérification.
    C'est plus lisible et défendable qu'un simple cosine score brut.
    """

    # Step 1 : Extraire les compétences de la fiche de poste
    poste_retriever = poste_store.as_retriever(search_kwargs={"k": 6})
    poste_docs = poste_retriever.get_relevant_documents("compétences requises profil recherché")
    poste_text = "\n".join([d.page_content for d in poste_docs])

    extraction_prompt = f"""
Analyse cette fiche de poste et extrais EXACTEMENT les compétences/exigences requises.
Retourne UNIQUEMENT un JSON valide avec ce format :
{{
  "competences_techniques": ["comp1", "comp2", ...],
  "competences_soft": ["soft1", "soft2", ...],
  "experience_requise": "X ans dans ...",
  "formation_requise": "Bac+X en ..."
}}

FICHE DE POSTE :
{poste_text}

JSON :"""

    import json
    response = llm.invoke(extraction_prompt)
    try:
        # Nettoyage du JSON (le LLM peut ajouter des backticks)
        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        exigences = json.loads(raw)
    except:
        exigences = {
            "competences_techniques": [],
            "competences_soft": [],
            "experience_requise": "Non spécifié",
            "formation_requise": "Non spécifié"
        }

    # Step 2 : Vérifier chaque compétence dans le CV
    cv_retriever = cv_store.as_retriever(search_kwargs={"k": 3})

    resultats_tech = []
    for comp in exigences.get("competences_techniques", []):
        docs = cv_retriever.get_relevant_documents(comp)
        cv_excerpt = "\n".join([d.page_content for d in docs])

        check_prompt = f"""
Le candidat a-t-il la compétence suivante dans son CV ?
Compétence : {comp}
Extrait CV : {cv_excerpt}

Réponds UNIQUEMENT avec ce JSON :
{{"present": true/false, "evidence": "citation courte ou 'non mentionné'"}}
"""
        check_resp = llm.invoke(check_prompt)
        try:
            raw = check_resp.content.strip().replace("```json","").replace("```","").strip()
            result = json.loads(raw)
        except:
            result = {"present": False, "evidence": "Erreur parsing"}

        resultats_tech.append({
            "competence": comp,
            "present": result.get("present", False),
            "evidence": result.get("evidence", "")
        })

    resultats_soft = []
    for soft in exigences.get("competences_soft", []):
        docs = cv_retriever.get_relevant_documents(soft)
        cv_excerpt = "\n".join([d.page_content for d in docs])

        check_prompt = f"""
Le candidat démontre-t-il cette soft skill dans son CV ?
Soft skill : {soft}
Extrait CV : {cv_excerpt}
Réponds UNIQUEMENT : {{"present": true/false, "evidence": "..."}}
"""
        check_resp = llm.invoke(check_prompt)
        try:
            raw = check_resp.content.strip().replace("```json","").replace("```","").strip()
            result = json.loads(raw)
        except:
            result = {"present": False, "evidence": "Erreur parsing"}

        resultats_soft.append({
            "competence": soft,
            "present": result.get("present", False),
            "evidence": result.get("evidence", "")
        })

    # Step 3 : Calcul du score
    total_items = len(resultats_tech) + len(resultats_soft)
    items_present = sum(1 for r in resultats_tech + resultats_soft if r["present"])
    score = round((items_present / total_items * 100) if total_items > 0 else 0)

    # Pondération : tech compte double
    score_tech = round(sum(1 for r in resultats_tech if r["present"]) / max(len(resultats_tech), 1) * 100)
    score_soft = round(sum(1 for r in resultats_soft if r["present"]) / max(len(resultats_soft), 1) * 100)
    score_pondere = round(score_tech * 0.65 + score_soft * 0.35)

    return {
        "score_global": score_pondere,
        "score_technique": score_tech,
        "score_soft": score_soft,
        "exigences": exigences,
        "resultats_tech": resultats_tech,
        "resultats_soft": resultats_soft,
        "nb_competences_evaluees": total_items
    }


# ─────────────────────────────────────────────
# TABS PRINCIPAUX
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chatbot RH", "🎯 Matching CV / Poste"])


# ═══════════════════════════════════════════
# TAB 1 — CHATBOT RH
# ═══════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1.8], gap="large")

    with col_left:
        st.markdown('<div class="card"><div class="card-title">📄 Charger les CVs</div>', unsafe_allow_html=True)

        uploaded_cvs = st.file_uploader(
            "Glissez vos CVs ici",
            type=["pdf"],
            accept_multiple_files=True,
            key="upload_chatbot",
            label_visibility="collapsed"
        )

        if uploaded_cvs:
            st.markdown(f"**{len(uploaded_cvs)} fichier(s) chargé(s) :**")
            for f in uploaded_cvs:
                st.markdown(f"• {f.name}")

        if uploaded_cvs and api_key:
            if st.button("⚡ Indexer les CVs", use_container_width=True):
                with st.spinner("Lecture et indexation en cours..."):
                    docs = load_and_chunk_pdfs(uploaded_cvs)
                    st.session_state.vectorstore_rh = build_vectorstore(docs, "rh_chatbot")
                    st.session_state.docs_charges = [f.name for f in uploaded_cvs]
                    st.session_state.chat_history = []
                st.success(f"✅ {len(docs)} chunks indexés dans ChromaDB")
        elif uploaded_cvs and not api_key:
            st.warning("Configurez votre clé API OpenAI dans la sidebar.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Stats indexation
        if st.session_state.vectorstore_rh:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📊 Index actif</div>
                <div style="font-size:0.85rem; color:#6b6b7a; line-height:1.8">
                    Fichiers : {', '.join(st.session_state.docs_charges)}<br>
                    Modèle embedding : text-embedding-3-small<br>
                    Vector store : ChromaDB (in-memory)<br>
                    Retrieval : top-4 similarité cosinus
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Questions suggérées
        st.markdown('<div class="card"><div class="card-title">💡 Questions suggérées</div>', unsafe_allow_html=True)
        questions_suggeres = [
            "Ce candidat maîtrise-t-il Python ?",
            "Quelle est son expérience en data science ?",
            "A-t-il travaillé en environnement agile ?",
            "Quelles sont ses certifications ?",
            "Parle-t-il anglais ?"
        ]
        for q in questions_suggeres:
            if st.button(q, key=f"suggest_{q}", use_container_width=True):
                st.session_state["question_input"] = q
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card-title">💬 Conversation</div>', unsafe_allow_html=True)

        # Zone de chat
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center; padding:3rem 1rem; color:#6b6b7a;">
                    <div style="font-size:2.5rem; margin-bottom:1rem">🤝</div>
                    <div style="font-size:0.95rem">Chargez des CVs et posez vos questions RH.<br>
                    Le chatbot répond en citant les sources documentaires.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="msg-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        sources_html = ""
                        if msg.get("sources"):
                            sources_html = f'<div class="msg-sources">📎 Sources : {msg["sources"]}</div>'
                        st.markdown(
                            f'<div class="msg-bot">🤖 {msg["content"]}{sources_html}</div>',
                            unsafe_allow_html=True
                        )

        # Input question
        st.markdown("---")
        question = st.text_input(
            "Votre question",
            placeholder="Ex: Ce candidat a-t-il de l'expérience en Machine Learning ?",
            key="question_input",
            label_visibility="collapsed"
        )

        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send = st.button("Envoyer →", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Reset", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send and question:
            if not st.session_state.vectorstore_rh:
                st.error("Chargez et indexez des CVs d'abord.")
            elif not api_key:
                st.error("Clé API manquante.")
            else:
                # Ajout question à l'historique
                st.session_state.chat_history.append({"role": "user", "content": question})

                with st.spinner("Recherche dans les CVs..."):
                    chain, retriever = build_rag_chain(
                        st.session_state.vectorstore_rh,
                        system_context="Tu analyses des CVs de candidats pour aider le recruteur."
                    )
                    answer = chain.invoke(question)
                    source_docs = retriever.invoke(question)

                # Extraction sources
                sources = list(set([
                    doc.metadata.get("source_file", "Document")
                    for doc in source_docs
                ]))
                sources_str = " · ".join(sources)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_str
                })
                st.rerun()


# ═══════════════════════════════════════════
# TAB 2 — MATCHING CV / POSTE
# ═══════════════════════════════════════════
with tab2:
    col_cv, col_poste = st.columns(2, gap="large")

    with col_cv:
        st.markdown('<div class="card"><div class="card-title">👤 CV du candidat</div>', unsafe_allow_html=True)
        cv_file = st.file_uploader(
            "CV (PDF)", type=["pdf"], key="cv_match", label_visibility="collapsed"
        )
        if cv_file:
            st.markdown(f"✅ **{cv_file.name}**")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_poste:
        st.markdown('<div class="card"><div class="card-title">📋 Fiche de poste</div>', unsafe_allow_html=True)
        poste_file = st.file_uploader(
            "Fiche de poste (PDF)", type=["pdf"], key="poste_match", label_visibility="collapsed"
        )
        if poste_file:
            st.markdown(f"✅ **{poste_file.name}**")
        st.markdown('</div>', unsafe_allow_html=True)

    # Bouton analyse
    if cv_file and poste_file and api_key:
        if st.button("🎯 Lancer l'analyse de matching", use_container_width=True):
            with st.spinner("Indexation des documents..."):
                # Indexation CV
                cv_docs = load_and_chunk_pdfs([cv_file])
                cv_store = build_vectorstore(cv_docs, "cv_matching")

                # Indexation fiche de poste
                poste_docs = load_and_chunk_pdfs([poste_file])
                poste_store = build_vectorstore(poste_docs, "poste_matching")

            with st.spinner("Analyse IA en cours (30-60 secondes)..."):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
                matching = compute_matching_score(cv_store, poste_store, llm)

            # ─── AFFICHAGE RÉSULTATS ───

            st.markdown("---")
            st.markdown("## 📊 Résultats du matching")

            # Score global
            score = matching["score_global"]
            if score >= 70:
                color = "#2d6a4f"
                mention = "Profil compatible ✓"
                bg = "#d8f3dc"
            elif score >= 50:
                color = "#b5451b"
                mention = "Partiellement compatible"
                bg = "#fde8d8"
            else:
                color = "#c1121f"
                mention = "Profil insuffisant"
                bg = "#fde8e8"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="text-align:center; padding:1.5rem; background:{bg}; border-radius:12px;">
                    <div style="font-family:'Cabinet Grotesk',sans-serif; font-size:3rem; font-weight:800; color:{color}">{score}%</div>
                    <div style="font-size:0.85rem; color:{color}; font-weight:600">{mention}</div>
                    <div style="font-size:0.75rem; color:#6b6b7a; margin-top:0.3rem">Score global pondéré</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="text-align:center; padding:1.5rem; background:#f0ede8; border-radius:12px;">
                    <div style="font-family:'Cabinet Grotesk',sans-serif; font-size:3rem; font-weight:800; color:#1a1a2e">{matching['score_technique']}%</div>
                    <div style="font-size:0.85rem; color:#1a1a2e; font-weight:600">Compétences techniques</div>
                    <div style="font-size:0.75rem; color:#6b6b7a; margin-top:0.3rem">Pondération 65%</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="text-align:center; padding:1.5rem; background:#f0ede8; border-radius:12px;">
                    <div style="font-family:'Cabinet Grotesk',sans-serif; font-size:3rem; font-weight:800; color:#1a1a2e">{matching['score_soft']}%</div>
                    <div style="font-size:0.85rem; color:#1a1a2e; font-weight:600">Soft skills</div>
                    <div style="font-size:0.75rem; color:#6b6b7a; margin-top:0.3rem">Pondération 35%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Détail compétences techniques
            col_tech, col_soft = st.columns(2, gap="large")

            with col_tech:
                st.markdown('<div class="card"><div class="card-title">⚙️ Compétences techniques</div>', unsafe_allow_html=True)
                for item in matching["resultats_tech"]:
                    badge_class = "badge-green" if item["present"] else "badge-red"
                    icon = "✓" if item["present"] else "✗"
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem; padding-bottom:0.8rem; border-bottom:1px solid #e2ddd8">
                        <span class="badge {badge_class}">{icon} {item['competence']}</span>
                        <div style="font-size:0.78rem; color:#6b6b7a; margin-top:0.3rem; font-style:italic">
                            {item['evidence']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_soft:
                st.markdown('<div class="card"><div class="card-title">🤝 Soft skills</div>', unsafe_allow_html=True)
                for item in matching["resultats_soft"]:
                    badge_class = "badge-green" if item["present"] else "badge-red"
                    icon = "✓" if item["present"] else "✗"
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem; padding-bottom:0.8rem; border-bottom:1px solid #e2ddd8">
                        <span class="badge {badge_class}">{icon} {item['competence']}</span>
                        <div style="font-size:0.78rem; color:#6b6b7a; margin-top:0.3rem; font-style:italic">
                            {item['evidence']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Exigences extraites
            exig = matching["exigences"]
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📋 Exigences extraites de la fiche de poste</div>
                <div style="font-size:0.85rem; color:#6b6b7a; line-height:1.8">
                    <b>Formation :</b> {exig.get('formation_requise', 'Non spécifié')}<br>
                    <b>Expérience :</b> {exig.get('experience_requise', 'Non spécifié')}<br>
                    <b>Compétences évaluées :</b> {matching['nb_competences_evaluees']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif (cv_file or poste_file) and not api_key:
        st.warning("Configurez votre clé API OpenAI dans la sidebar pour lancer le matching.")
    elif not (cv_file and poste_file):
        st.markdown("""
        <div style="text-align:center; padding:4rem 1rem; color:#6b6b7a;">
            <div style="font-size:2.5rem; margin-bottom:1rem">🎯</div>
            <div style="font-size:0.95rem">Uploadez un CV et une fiche de poste (PDF)<br>
            pour obtenir une analyse de compatibilité détaillée.</div>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown(f"""
<div style="display:flex; justify-content:space-between; color:#6b6b7a; font-size:0.78rem;">
    <span>RH·IA · Harry Patrice TEGUE KEMGNE</span>
    <span>Stack : LangChain · ChromaDB · OpenAI · Streamlit</span>
    <span>Pattern : RAG (Retrieval-Augmented Generation)</span>
</div>
""", unsafe_allow_html=True)    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;700;800&family=Instrument+Sans:wght@300;400;500&display=swap');

    :root {
        --bg: #f8f7f4;
        --surface: #ffffff;
        --surface2: #f0ede8;
        --border: #e2ddd8;
        --accent: #1a1a2e;
        --accent2: #e63946;
        --text: #1a1a1e;
        --muted: #6b6b7a;
        --success: #2d6a4f;
        --success-bg: #d8f3dc;
        --warning: #b5451b;
        --warning-bg: #fde8d8;
    }

    html, body, [class*="css"] {
        font-family: 'Instrument Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }
    h1,h2,h3,h4 { font-family: 'Cabinet Grotesk', sans-serif; }
    .main { background-color: var(--bg); }
    .block-container { padding: 2rem 2.5rem; max-width: 1200px; }

    [data-testid="stSidebar"] {
        background-color: var(--accent);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stTextInput input {
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.2) !important;
        color: white !important;
    }

    /* Header */
    .main-header {
        background: var(--accent);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: var(--accent2);
        border-radius: 50%;
        opacity: 0.15;
    }
    .main-title {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    .main-subtitle { color: rgba(255,255,255,0.65); font-size: 0.92rem; margin-top: 0.4rem; }

    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: var(--accent);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Score */
    .score-circle {
        width: 120px; height: 120px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        border: 4px solid;
    }
    .score-value {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label { font-size: 0.7rem; color: var(--muted); }

    /* Messages chatbot */
    .msg-user {
        background: var(--accent);
        color: white;
        padding: 0.9rem 1.2rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.5rem 0 0.5rem 3rem;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .msg-bot {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        padding: 0.9rem 1.2rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 3rem 0.5rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .msg-sources {
        font-size: 0.75rem;
        color: var(--muted);
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid var(--border);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .badge-green { background: var(--success-bg); color: var(--success); }
    .badge-red { background: var(--warning-bg); color: var(--warning); }
    .badge-gray { background: var(--surface2); color: var(--muted); }

    /* Boutons */
    .stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 700;
        padding: 0.55rem 1.2rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.8; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* Progress bar custom */
    .progress-bar-container {
        background: var(--surface2);
        border-radius: 99px;
        height: 10px;
        width: 100%;
        margin: 0.3rem 0;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #1a1a2e, #e63946);
        transition: width 0.5s ease;
    }

    .stTextInput input, .stTextArea textarea {
        border-radius: 8px !important;
        border-color: var(--border) !important;
    }

    hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE — Persistance entre reruns Streamlit
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_rh" not in st.session_state:
    st.session_state.vectorstore_rh = None
if "vectorstore_match" not in st.session_state:
    st.session_state.vectorstore_match = None
if "docs_charges" not in st.session_state:
    st.session_state.docs_charges = []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input(
        "Clé API OpenAI",
        type="password",
        placeholder="sk-...",
        help="Nécessaire pour embeddings + LLM"
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.markdown("✅ Clé configurée")
    else:
        st.markdown("⚠️ Clé requise pour fonctionner")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; line-height:1.8; opacity:0.7'>
    <b>Stack technique</b><br>
    LangChain · ChromaDB<br>
    OpenAI Embeddings<br>
    GPT-4o-mini · PyPDF2<br><br>
    <b>Pattern</b><br>
    RAG (Retrieval-Augmented Generation)<br><br>
    <b>Auteur</b><br>
    Harry Patrice TEGUE KEMGNE<br>
    Alternant Data Scientist / AI Engineer
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">RH·IA</p>
    <p class="main-subtitle">Chatbot RH intelligent + Matching CV/Poste — Propulsé par RAG & GPT-4o-mini</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FONCTIONS CORE
# ─────────────────────────────────────────────

def load_and_chunk_pdfs(uploaded_files: list, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
    """
    Charge des PDFs uploadés et les découpe en chunks.

    Pourquoi RecursiveCharacterTextSplitter ?
    → Essaie d'abord de couper sur les paragraphes (\n\n),
      puis les phrases (\n), puis les mots ( ).
    → Préserve mieux le contexte sémantique qu'un simple split fixe.

    chunk_size=800 : compromis entre contexte suffisant et token budget LLM
    chunk_overlap=100 : évite de couper une info à cheval sur deux chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    all_docs = []
    for uploaded_file in uploaded_files:
        # Streamlit donne un BytesIO — PyPDFLoader a besoin d'un vrai fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Ajout métadonnée : nom du fichier source (utile pour les citations)
        for page in pages:
            page.metadata["source_file"] = uploaded_file.name

        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)
        Path(tmp_path).unlink()  # Nettoyage fichier temp

    return all_docs


def build_vectorstore(docs: list, collection_name: str = "rh_docs") -> Chroma:
    """
    Construit le vector store ChromaDB depuis une liste de documents.

    Embedding model : text-embedding-3-small
    → Dimension : 1536 → 1536 dimensions vectorielles par chunk
    → Coût : ~$0.00002 / 1000 tokens (très faible)
    → Alternative gratuite : sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)

    ChromaDB in-memory : les données sont perdues au redémarrage.
    En production : persist_directory="./chroma_db" pour la persistance.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name
    )
    return vectorstore


def build_rag_chain(vectorstore: Chroma, system_context: str = "RH") -> RetrievalQA:
    """
    Construit la chaîne RAG : Retriever + LLM + Prompt.

    k=4 : on récupère les 4 chunks les plus similaires à la question
    → Trop peu (k=1) : réponse trop partielle
    → Trop (k=10) : dépasse le context window + bruit

    Temperature=0.2 : réponses factuelles, peu de créativité
    → Pour du matching CV on veut des faits, pas de l'invention
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Prompt template strict — ancrage sur le contexte documentaire
    prompt_template = f"""
Tu es un assistant RH expert. Réponds uniquement à partir du contexte fourni.
Contexte : {system_context}

Règles strictes :
- Réponds uniquement avec les informations présentes dans le CONTEXTE
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis, concis et professionnel
- Ne jamais inventer de compétences ou d'expériences non mentionnées

CONTEXTE :
{{context}}

QUESTION : {{question}}

RÉPONSE :"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = on concatène tous les chunks dans le prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


def compute_matching_score(cv_store: Chroma, poste_store: Chroma, llm) -> dict:
    """
    Calcule le score de matching entre un CV et une fiche de poste.

    Méthode :
    1. On extrait les compétences requises de la fiche de poste (via LLM)
    2. Pour chaque compétence, on query le CV store
    3. Le LLM évalue si la compétence est présente dans le CV (0/1)
    4. Score global = moyenne pondérée

    NOTE ENTRETIEN :
    Une approche plus robuste utiliserait des embeddings directs
    (similarité cosinus entre chunk fiche de poste et chunk CV).
    Ici on hybride : LLM pour l'extraction + retrieval pour la vérification.
    C'est plus lisible et défendable qu'un simple cosine score brut.
    """

    # Step 1 : Extraire les compétences de la fiche de poste
    poste_retriever = poste_store.as_retriever(search_kwargs={"k": 6})
    poste_docs = poste_retriever.get_relevant_documents("compétences requises profil recherché")
    poste_text = "\n".join([d.page_content for d in poste_docs])

    extraction_prompt = f"""
Analyse cette fiche de poste et extrais EXACTEMENT les compétences/exigences requises.
Retourne UNIQUEMENT un JSON valide avec ce format :
{{
  "competences_techniques": ["comp1", "comp2", ...],
  "competences_soft": ["soft1", "soft2", ...],
  "experience_requise": "X ans dans ...",
  "formation_requise": "Bac+X en ..."
}}

FICHE DE POSTE :
{poste_text}

JSON :"""

    import json
    response = llm.invoke(extraction_prompt)
    try:
        # Nettoyage du JSON (le LLM peut ajouter des backticks)
        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        exigences = json.loads(raw)
    except:
        exigences = {
            "competences_techniques": [],
            "competences_soft": [],
            "experience_requise": "Non spécifié",
            "formation_requise": "Non spécifié"
        }

    # Step 2 : Vérifier chaque compétence dans le CV
    cv_retriever = cv_store.as_retriever(search_kwargs={"k": 3})

    resultats_tech = []
    for comp in exigences.get("competences_techniques", []):
        docs = cv_retriever.get_relevant_documents(comp)
        cv_excerpt = "\n".join([d.page_content for d in docs])

        check_prompt = f"""
Le candidat a-t-il la compétence suivante dans son CV ?
Compétence : {comp}
Extrait CV : {cv_excerpt}

Réponds UNIQUEMENT avec ce JSON :
{{"present": true/false, "evidence": "citation courte ou 'non mentionné'"}}
"""
        check_resp = llm.invoke(check_prompt)
        try:
            raw = check_resp.content.strip().replace("```json","").replace("```","").strip()
            result = json.loads(raw)
        except:
            result = {"present": False, "evidence": "Erreur parsing"}

        resultats_tech.append({
            "competence": comp,
            "present": result.get("present", False),
            "evidence": result.get("evidence", "")
        })

    resultats_soft = []
    for soft in exigences.get("competences_soft", []):
        docs = cv_retriever.get_relevant_documents(soft)
        cv_excerpt = "\n".join([d.page_content for d in docs])

        check_prompt = f"""
Le candidat démontre-t-il cette soft skill dans son CV ?
Soft skill : {soft}
Extrait CV : {cv_excerpt}
Réponds UNIQUEMENT : {{"present": true/false, "evidence": "..."}}
"""
        check_resp = llm.invoke(check_prompt)
        try:
            raw = check_resp.content.strip().replace("```json","").replace("```","").strip()
            result = json.loads(raw)
        except:
            result = {"present": False, "evidence": "Erreur parsing"}

        resultats_soft.append({
            "competence": soft,
            "present": result.get("present", False),
            "evidence": result.get("evidence", "")
        })

    # Step 3 : Calcul du score
    total_items = len(resultats_tech) + len(resultats_soft)
    items_present = sum(1 for r in resultats_tech + resultats_soft if r["present"])
    score = round((items_present / total_items * 100) if total_items > 0 else 0)

    # Pondération : tech compte double
    score_tech = round(sum(1 for r in resultats_tech if r["present"]) / max(len(resultats_tech), 1) * 100)
    score_soft = round(sum(1 for r in resultats_soft if r["present"]) / max(len(resultats_soft), 1) * 100)
    score_pondere = round(score_tech * 0.65 + score_soft * 0.35)

    return {
        "score_global": score_pondere,
        "score_technique": score_tech,
        "score_soft": score_soft,
        "exigences": exigences,
        "resultats_tech": resultats_tech,
        "resultats_soft": resultats_soft,
        "nb_competences_evaluees": total_items
    }


# ─────────────────────────────────────────────
# TABS PRINCIPAUX
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chatbot RH", "🎯 Matching CV / Poste"])


# ═══════════════════════════════════════════
# TAB 1 — CHATBOT RH
# ═══════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1.8], gap="large")

    with col_left:
        st.markdown('<div class="card"><div class="card-title">📄 Charger les CVs</div>', unsafe_allow_html=True)

        uploaded_cvs = st.file_uploader(
            "Glissez vos CVs ici",
            type=["pdf"],
            accept_multiple_files=True,
            key="upload_chatbot",
            label_visibility="collapsed"
        )

        if uploaded_cvs:
            st.markdown(f"**{len(uploaded_cvs)} fichier(s) chargé(s) :**")
            for f in uploaded_cvs:
                st.markdown(f"• {f.name}")

        if uploaded_cvs and api_key:
            if st.button("⚡ Indexer les CVs", use_container_width=True):
                with st.spinner("Lecture et indexation en cours..."):
                    docs = load_and_chunk_pdfs(uploaded_cvs)
                    st.session_state.vectorstore_rh = build_vectorstore(docs, "rh_chatbot")
                    st.session_state.docs_charges = [f.name for f in uploaded_cvs]
                    st.session_state.chat_history = []
                st.success(f"✅ {len(docs)} chunks indexés dans ChromaDB")
        elif uploaded_cvs and not api_key:
            st.warning("Configurez votre clé API OpenAI dans la sidebar.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Stats indexation
        if st.session_state.vectorstore_rh:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📊 Index actif</div>
                <div style="font-size:0.85rem; color:#6b6b7a; line-height:1.8">
                    Fichiers : {', '.join(st.session_state.docs_charges)}<br>
                    Modèle embedding : text-embedding-3-small<br>
                    Vector store : ChromaDB (in-memory)<br>
                    Retrieval : top-4 similarité cosinus
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Questions suggérées
        st.markdown('<div class="card"><div class="card-title">💡 Questions suggérées</div>', unsafe_allow_html=True)
        questions_suggeres = [
            "Ce candidat maîtrise-t-il Python ?",
            "Quelle est son expérience en data science ?",
            "A-t-il travaillé en environnement agile ?",
            "Quelles sont ses certifications ?",
            "Parle-t-il anglais ?"
        ]
        for q in questions_suggeres:
            if st.button(q, key=f"suggest_{q}", use_container_width=True):
                st.session_state["question_input"] = q
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card-title">💬 Conversation</div>', unsafe_allow_html=True)

        # Zone de chat
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center; padding:3rem 1rem; color:#6b6b7a;">
                    <div style="font-size:2.5rem; margin-bottom:1rem">🤝</div>
                    <div style="font-size:0.95rem">Chargez des CVs et posez vos questions RH.<br>
                    Le chatbot répond en citant les sources documentaires.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="msg-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        sources_html = ""
                        if msg.get("sources"):
                            sources_html = f'<div class="msg-sources">📎 Sources : {msg["sources"]}</div>'
                        st.markdown(
                            f'<div class="msg-bot">🤖 {msg["content"]}{sources_html}</div>',
                            unsafe_allow_html=True
                        )

        # Input question
        st.markdown("---")
        question = st.text_input(
            "Votre question",
            placeholder="Ex: Ce candidat a-t-il de l'expérience en Machine Learning ?",
            key="question_input",
            label_visibility="collapsed"
        )

        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send = st.button("Envoyer →", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Reset", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send and question:
            if not st.session_state.vectorstore_rh:
                st.error("Chargez et indexez des CVs d'abord.")
            elif not api_key:
                st.error("Clé API manquante.")
            else:
                # Ajout question à l'historique
                st.session_state.chat_history.append({"role": "user", "content": question})

                with st.spinner("Recherche dans les CVs..."):
                    chain = build_rag_chain(
                        st.session_state.vectorstore_rh,
                        system_context="Tu analyses des CVs de candidats pour aider le recruteur."
                    )
                    result = chain.invoke({"query": question})

                # Extraction sources
                sources = list(set([
                    doc.metadata.get("source_file", "Document")
                    for doc in result.get("source_documents", [])
                ]))
                sources_str = " · ".join(sources)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["result"],
                    "sources": sources_str
                })
                st.rerun()


# ═══════════════════════════════════════════
# TAB 2 — MATCHING CV / POSTE
# ═══════════════════════════════════════════
with tab2:
    col_cv, col_poste = st.columns(2, gap="large")

    with col_cv:
        st.markdown('<div class="card"><div class="card-title">👤 CV du candidat</div>', unsafe_allow_html=True)
        cv_file = st.file_uploader(
            "CV (PDF)", type=["pdf"], key="cv_match", label_visibility="collapsed"
        )
        if cv_file:
            st.markdown(f"✅ **{cv_file.name}**")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_poste:
        st.markdown('<div class="card"><div class="card-title">📋 Fiche de poste</div>', unsafe_allow_html=True)
        poste_file = st.file_uploader(
            "Fiche de poste (PDF)", type=["pdf"], key="poste_match", label_visibility="collapsed"
        )
        if poste_file:
            st.markdown(f"✅ **{poste_file.name}**")
        st.markdown('</div>', unsafe_allow_html=True)

    # Bouton analyse
    if cv_file and poste_file and api_key:
        if st.button("🎯 Lancer l'analyse de matching", use_container_width=True):
            with st.spinner("Indexation des documents..."):
                # Indexation CV
                cv_docs = load_and_chunk_pdfs([cv_file])
                cv_store = build_vectorstore(cv_docs, "cv_matching")

                # Indexation fiche de poste
                poste_docs = load_and_chunk_pdfs([poste_file])
                poste_store = build_vectorstore(poste_docs, "poste_matching")

            with st.spinner("Analyse IA en cours (30-60 secondes)..."):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
                matching = compute_matching_score(cv_store, poste_store, llm)

            # ─── AFFICHAGE RÉSULTATS ───

            st.markdown("---")
            st.markdown("## 📊 Résultats du matching")

            # Score global
            score = matching["score_global"]
            if score >= 70:
                color = "#2d6a4f"
                mention = "Profil compatible ✓"
                bg = "#d8f3dc"
            elif score >= 50:
                color = "#b5451b"
                mention = "Partiellement compatible"
                bg = "#fde8d8"
            else:
                color = "#c1121f"
                mention = "Profil insuffisant"
                bg = "#fde8e8"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="text-align:center; padding:1.5rem; background:{bg}; border-radius:12px;">
                    <div style="font-family:'Cabinet Grotesk',sans-serif; font-size:3rem; font-weight:800; color:{color}">{score}%</div>
                    <div style="font-size:0.85rem; color:{color}; font-weight:600">{mention}</div>
                    <div style="font-size:0.75rem; color:#6b6b7a; margin-top:0.3rem">Score global pondéré</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="text-align:center; padding:1.5rem; background:#f0ede8; border-radius:12px;">
                    <div style="font-family:'Cabinet Grotesk',sans-serif; font-size:3rem; font-weight:800; color:#1a1a2e">{matching['score_technique']}%</div>
                    <div style="font-size:0.85rem; color:#1a1a2e; font-weight:600">Compétences techniques</div>
                    <div style="font-size:0.75rem; color:#6b6b7a; margin-top:0.3rem">Pondération 65%</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="text-align:center; padding:1.5rem; background:#f0ede8; border-radius:12px;">
                    <div style="font-family:'Cabinet Grotesk',sans-serif; font-size:3rem; font-weight:800; color:#1a1a2e">{matching['score_soft']}%</div>
                    <div style="font-size:0.85rem; color:#1a1a2e; font-weight:600">Soft skills</div>
                    <div style="font-size:0.75rem; color:#6b6b7a; margin-top:0.3rem">Pondération 35%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Détail compétences techniques
            col_tech, col_soft = st.columns(2, gap="large")

            with col_tech:
                st.markdown('<div class="card"><div class="card-title">⚙️ Compétences techniques</div>', unsafe_allow_html=True)
                for item in matching["resultats_tech"]:
                    badge_class = "badge-green" if item["present"] else "badge-red"
                    icon = "✓" if item["present"] else "✗"
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem; padding-bottom:0.8rem; border-bottom:1px solid #e2ddd8">
                        <span class="badge {badge_class}">{icon} {item['competence']}</span>
                        <div style="font-size:0.78rem; color:#6b6b7a; margin-top:0.3rem; font-style:italic">
                            {item['evidence']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_soft:
                st.markdown('<div class="card"><div class="card-title">🤝 Soft skills</div>', unsafe_allow_html=True)
                for item in matching["resultats_soft"]:
                    badge_class = "badge-green" if item["present"] else "badge-red"
                    icon = "✓" if item["present"] else "✗"
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem; padding-bottom:0.8rem; border-bottom:1px solid #e2ddd8">
                        <span class="badge {badge_class}">{icon} {item['competence']}</span>
                        <div style="font-size:0.78rem; color:#6b6b7a; margin-top:0.3rem; font-style:italic">
                            {item['evidence']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Exigences extraites
            exig = matching["exigences"]
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📋 Exigences extraites de la fiche de poste</div>
                <div style="font-size:0.85rem; color:#6b6b7a; line-height:1.8">
                    <b>Formation :</b> {exig.get('formation_requise', 'Non spécifié')}<br>
                    <b>Expérience :</b> {exig.get('experience_requise', 'Non spécifié')}<br>
                    <b>Compétences évaluées :</b> {matching['nb_competences_evaluees']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif (cv_file or poste_file) and not api_key:
        st.warning("Configurez votre clé API OpenAI dans la sidebar pour lancer le matching.")
    elif not (cv_file and poste_file):
        st.markdown("""
        <div style="text-align:center; padding:4rem 1rem; color:#6b6b7a;">
            <div style="font-size:2.5rem; margin-bottom:1rem">🎯</div>
            <div style="font-size:0.95rem">Uploadez un CV et une fiche de poste (PDF)<br>
            pour obtenir une analyse de compatibilité détaillée.</div>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown(f"""
<div style="display:flex; justify-content:space-between; color:#6b6b7a; font-size:0.78rem;">
    <span>RH·IA · Harry Patrice TEGUE KEMGNE</span>
    <span>Stack : LangChain · ChromaDB · OpenAI · Streamlit</span>
    <span>Pattern : RAG (Retrieval-Augmented Generation)</span>
</div>
""", unsafe_allow_html=True)
