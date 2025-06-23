# rag/generator.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


#"tiiuae/falcon-rw-1b"

class Generator:
    def __init__(self, model_name="openai-community/gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate_answer(self, context, question):
        input_text = context + "\n\nQuestion: " + question + "\nAnswer:"

        # Tokenize input manually
        tokens = self.tokenizer.encode(input_text, add_special_tokens=False)

        # Truncate tokens to GPT-2's max length (1024)
        max_len = 1024
        tokens = tokens[:max_len]
        input_ids = torch.tensor([tokens])

        # Explicitly set position_ids to avoid IndexError
        position_ids = torch.arange(0, input_ids.shape[-1]).unsqueeze(0)

        outputs = self.model.generate(
            input_ids=input_ids,
            position_ids=position_ids,  # 🛡️ This avoids index errors
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id  # Avoid pad token warnings
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
