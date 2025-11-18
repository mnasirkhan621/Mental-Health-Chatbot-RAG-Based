import os
import pandas as pd
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- PATHS ----------
ROOT = os.path.dirname(os.path.dirname(__file__))
KB   = os.path.join(ROOT, "knowledge_base")
VS   = os.path.join(ROOT, "vector_store")

# ---------- FAST EMBEDDING ----------
print("Loading embedding model (first run: ~10-20s)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
print("Model ready!")

# ---------- SMART SPLIT: Only split long docs ----------
def smart_split(docs):
    short_docs = [d for d in docs if len(d.page_content) < 1000]
    long_docs  = [d for d in docs if len(d.page_content) >= 1000]

    print(f"Keeping {len(short_docs)} short docs as-is")
    if long_docs:
        print(f"Splitting {len(long_docs)} long docs (PDFs)...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_chunks = splitter.split_documents(long_docs)
        print(f"Split into {len(split_chunks)} chunks")
    else:
        split_chunks = []

    return short_docs + split_chunks

# ---------- LOAD CSVs (NO SPLIT) ----------
def load_csv_no_split(path, cols, fmt):
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
    docs = []
    for _, r in df.head(500).iterrows():  # LIMIT TO 500 ROWS
        try:
            txt = fmt.format(**r.to_dict())
        except:
            txt = " | ".join(f"{c}: {r.get(c,'')}" for c in cols)
        docs.append(Document(page_content=txt[:1500], metadata={"source": path}))  # Cap length
    return docs

# ---------- MAIN ----------
def ingest():

    docs = []

    # 1. PDFs → will be split later
    pdf_loader = DirectoryLoader(KB, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
    docs.extend(pdf_loader.load())
    print(f"Loaded {len(docs)} PDF pages")

    # 2. CSVs → keep as single docs, limit rows
    cfg = {
        "data.csv": ("Questions", "Answers", "Q: {Questions} A: {Answers}"),
        "dataset.csv": ("title", "content", "Title: {title} | {content}"),
        "Indicators_of_Anxiety_or_Depression.csv": ("Indicator", "Value", "Ind: {Indicator} Val: {Value}"),
        "MentalHealthSurvey.csv": ("depression", "anxiety", "Dep: {depression} Anx: {anxiety} Relief: {stress_relief_activities}"),
    }

    for name, (c1, c2, fmt) in cfg.items():
        path = os.path.join(KB, name)
        if not os.path.exists(path):
            path = path.replace(".csv", ".xlsx")
        if os.path.exists(path):
            docs.extend(load_csv_no_split(path, [c1, c2], fmt))
            print(f"Loaded {name} (500 rows)")

    # 3. Smart split
    final_docs = smart_split(docs)
    print(f"Final: {len(final_docs)} documents to embed")

    # 4. Embed with progress
    print("Embedding... (this will take ~30-60 seconds)")
    texts = [d.page_content for d in final_docs]
    embeddings_list = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Batching"):
        batch = texts[i:i+batch_size]
        embeddings_list.extend(embeddings.embed_documents(batch))

    # 5. Build FAISS with precomputed embeddings — FIXED
    print("Building FAISS index...")
    text_embeddings = list(zip(texts, embeddings_list))  # Zip for from_embeddings
    vector = FAISS.from_embeddings(text_embeddings, embedding=embeddings)
    os.makedirs(VS, exist_ok=True)
    vector.save_local(os.path.join(VS, "faiss_index"))
    print("SUCCESS! Vector store saved at vector_store/faiss_index")

if __name__ == "__main__":
    ingest()