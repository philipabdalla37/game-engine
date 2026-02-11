import os

from openai import OpenAI
import json

class GPT_prompt_generator:

    output_prompts: list[str] = []

    def __init__(self, model = "gpt-5.2"):
        self.model = model

    def generate_prompts(self,num_prompts: int):
        prompt = f'''You are an expert Dungeon Master running a tabletop Dungeons & Dragons session.

        Generate {num_prompts} distinct game turns that will be used in a RAG setup with a smaller LLM. 
        
        RULES:
        - All generated events must loosely follow the same plot and build on each other, so that the smaller LLM can piece them together to create a story.
        - Each event MUST include exactly ONE required skill check.
        - The skill check must be ONE of the following:
          • Charisma
          • Strength
          • Intelligence
          • Perception
        - Rotate skills naturally; do not repeat the same skill every time.
        - Events should be suitable for a low-to-mid fantasy setting.
        - Write in a vivid but concise DM narration style.
        
        FORMAT (VERY IMPORTANT — FOLLOW EXACTLY):
        Return a JSON object with a top-level key called "events".
        The value of "events" must be an array of objects, where each object follows this schema:
        
        {{
          "event_id": <integer>,
          "narration": "<2–4 sentences describing the situation>",
          "skill_check": {{
            "skill": "<Charisma | Strength | Perception | Intelligence>",
            "dc": <integer between 10 and 18>,
            "success": "<what happens on a successful roll>",
            "failure": "<what happens on a failed roll>"
          }}
        }}
        
        ADDITIONAL GUIDELINES:
        - Success and failure outcomes must meaningfully change the situation.
        - Avoid lethal outcomes; consequences should advance the story.
        - Do not include dice mechanics beyond the skill and DC.
        - Do not include explanations, comments, or extra text outside the JSON.
        
        Begin.
        '''



        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={"format": {"type": "json_object"}}
        )

        raw = (response.output_text or "").strip()
        if not raw:
            raise RuntimeError("Empty output_text from model.")

        data = json.loads(raw)

        new_events = data.get("events", [])

        if not isinstance(new_events, list):
            raise TypeError(f"Expected 'events' to be a list, got {type(new_events)}")

        self.output_prompts.extend(new_events)



