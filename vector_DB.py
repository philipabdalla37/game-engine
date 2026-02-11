from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorDB:
    def __init__(self, db_name, prompts, model_name="all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.db_name = db_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        docs: list[Document] = []
        for i, ev in enumerate(prompts):
            text = VectorDB.event_to_text(ev)
            metadata = {"event_id": ev.get("event_id", ""), "index": i}
            docs.append(Document(page_content=text, metadata=metadata))

        db = FAISS.from_documents(docs, self.embeddings)
        db.save_local(self.db_name)

    @staticmethod
    def event_to_text(ev: dict) -> str:
        event_id = ev.get("event_id", "")
        narration = (ev.get("narration") or "").strip()

        sc = ev.get("skill_check") or {}
        skill = (sc.get("skill") or "").strip()
        dc = sc.get("dc", "")
        success = (sc.get("success") or "").strip()
        failure = (sc.get("failure") or "").strip()
        return (
            f"Event ID: {event_id}\n"
            f"Scene: {narration}\n"
            f"Required Check: {skill} (DC {dc})\n"
            f"Success Outcome: {success}\n"
            f"Failure Outcome: {failure}"
        )

    @staticmethod
    def load_db(db_name: str, model_name: str = "all-MiniLM-L6-v2"):
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return FAISS.load_local(
            db_name,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    @staticmethod
    def embed_query(db_name,query: str, k: int):
        db = VectorDB.load_db(db_name)
        return db.similarity_search(query, k=k)

    @staticmethod
    def embed_query_with_scores(db_name, query: str, k: int):
        db = VectorDB.load_db(db_name)
        return db.similarity_search_with_score(query, k=k)
