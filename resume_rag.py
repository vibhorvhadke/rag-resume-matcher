import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

def load_resume(path):
    with open(path, "r") as f:
        return f.read()

def chunk_resume(text):
    section_pattern = r'\n(?=[A-Z][a-zA-Z ]*:)'
    raw_sections = re.split(section_pattern, text.strip())
    return [s.strip() for s in raw_sections if s.strip()]

def extract_metadata(text):
    name_match = re.search(r'Name:\s*(.+)', text)
    skills_match = re.search(r'Skills:\s*(.+)', text)
    name = name_match.group(1).strip() if name_match else "Unknown"
    skills = skills_match.group(1).strip() if skills_match else ""
    return {"name": name, "skills": skills}

def build_resume_database(resumes_folder="resumes", db_path="vector_db"):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection("resumes")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="resumes")

    for filename in os.listdir(resumes_folder):
        filepath = os.path.join(resumes_folder, filename)
        text = load_resume(filepath)
        chunks = chunk_resume(text)
        embeddings = embedding_model.encode(chunks)
        metadata = extract_metadata(text)

        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": filename, **metadata} for _ in chunks]
        )
        print(f"Added {filename}: {len(chunks)} chunks")

    return collection, embedding_model

if __name__ == "__main__":
    collection, model = build_resume_database()
    print("Total chunks:", collection.count())
