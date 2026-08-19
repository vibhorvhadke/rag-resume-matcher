import re
import json
import chromadb
from sentence_transformers import SentenceTransformer

def search_resumes(collection, embedding_model, job_description, top_k=10):
    jd_embedding = embedding_model.encode([job_description])
    results = collection.query(
        query_embeddings=jd_embedding.tolist(),
        n_results=top_k
    )
    return results

def extract_matched_skills(job_description, candidate_skills_str):
    if not candidate_skills_str:
        return []
    jd_lower = job_description.lower()
    candidate_skills = [s.strip() for s in candidate_skills_str.split(",")]
    return [skill for skill in candidate_skills if skill.lower() in jd_lower]

def get_candidate_skills(collection, source):
    result = collection.get(where={"source": source}, limit=1)
    if result["metadatas"]:
        return result["metadatas"][0].get("skills", "")
    return ""

def score_candidates(collection, embedding_model, job_description, top_k=10):
    results = search_resumes(collection, embedding_model, job_description, top_k=top_k)
    candidates = {}

    for i in range(len(results['documents'][0])):
        source = results['metadatas'][0][i]['source']
        chunk = results['documents'][0][i]
        distance = results['distances'][0][i]
        similarity = 1 / (1 + distance)

        if source not in candidates:
            candidates[source] = {
                "name": results['metadatas'][0][i].get('name', 'Unknown'),
                "matched_chunks": [],
                "similarities": []
            }
        candidates[source]["matched_chunks"].append(chunk)
        candidates[source]["similarities"].append(similarity)

    scored = []
    for source, data in candidates.items():
        avg_similarity = sum(data["similarities"]) / len(data["similarities"])
        match_score = round(avg_similarity * 100, 2)
        skills_str = get_candidate_skills(collection, source)
        matched_skills = extract_matched_skills(job_description, skills_str)

        scored.append({
            "candidate_name": data["name"],
            "resume_path": f"resumes/{source}",
            "match_score": match_score,
            "matched_skills": matched_skills,
            "relevant_excerpts": data["matched_chunks"],
            "reasoning": f"Matched on {len(data['matched_chunks'])} section(s): "
                         f"{', '.join(c.split(':')[0] for c in data['matched_chunks'])}"
        })

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored

def match_job_to_resumes(collection, embedding_model, job_description, top_k=10):
    candidates = score_candidates(collection, embedding_model, job_description, top_k=top_k)
    return {
        "job_description": job_description,
        "top_matches": candidates
    }

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="vector_db")
    collection = client.get_collection(name="resumes")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    sample_jd = "Looking for a Machine Learning Engineer with Python and TensorFlow experience"
    output = match_job_to_resumes(collection, embedding_model, sample_jd)

    with open("outputs/match_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
