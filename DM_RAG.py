from langchain_ollama import ChatOllama
from vector_DB import VectorDB

class DM_RAG:

    def __init__(self, db_name, ollama_model="llama3.2:3b"):
        self.db_name = db_name
        self.llm = ChatOllama(model=ollama_model, temperature=0.7)
        print("Using: "+ollama_model)

    def format_retrieved(self, docs):
        return "\n\n".join(
            f"CANDIDATE EVENT:\n{d.page_content}"
            for d in docs
        )

    def first_turn(self, party_desc: str, k: int = 3) -> str:

        docs = VectorDB.embed_query(self.db_name, query= party_desc, k=k)
        retrieved = self.format_retrieved(docs)
        prompt = (
            "DM START ENGINE.\n"
            "Pick ONE retrieved candidate as base and adapt to PARTY. Do not merge plots.\n"
            "Output JSON only. 1 event. Exactly 1 skill check (Cha/Str/Int/Per). DC 10-18. 2-4 sentences. Non-lethal.\n"
            "Return JSON shape: "
            "{\"event_id\":1,\"narration\":\"...\",\"skill_check\":{\"skill\":\"Perception\",\"dc\":14,\"success\":\"...\",\"failure\":\"...\"}}\n\n"
            f"PARTY:\n{party_desc}\n\n"
            f"RETRIEVED:\n{retrieved}\n"
        )

        out = self.llm.invoke(prompt)
        return out.content

    def next_turn(self, previous_turn: str, player_input: str, k: int = 3) -> str:

        query = (previous_turn.strip() + "\n" + player_input.strip()).strip()
        docs = VectorDB.embed_query(self.db_name, query=query, k=k)
        retrieved = self.format_retrieved(docs)

        prompt = (
            "DM CONTINUATION ENGINE.\n"
            "Use PREV + PLAYER + ONE retrieved candidate as the base. Do not merge plots.\n"
            "Output JSON only. 1 event. Exactly 1 skill check (Cha/Str/Int/Per). DC 10-18. 2-4 sentences. Non-lethal.\n"
            "Return JSON shape: "
            "{\"event_id\":1,\"narration\":\"...\",\"skill_check\":{\"skill\":\"Perception\",\"dc\":14,\"success\":\"...\",\"failure\":\"...\"}}\n\n"
            f"PREV:\n{previous_turn}\n\n"
            f"PLAYER:\n{player_input}\n\n"
            f"RETRIEVED:\n{retrieved}\n"
        )

        out = self.llm.invoke(prompt)
        return out.content
