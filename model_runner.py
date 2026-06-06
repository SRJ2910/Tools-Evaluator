import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


class FunctionGemmaRunner:

    def __init__(self, model_path):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype
        )

        self.model.to(self.device)
        self.model.eval()

    def generate(self, tools, messages):

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True
        )
        print("=== Prompt ===")
        print(prompt)
        print("=== End Prompt ===")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=1.0,
                top_p=0.9
            )

        generated = outputs[0][
            inputs.input_ids.shape[1]:
        ]

        generated_text = self.tokenizer.decode(
            generated,
            skip_special_tokens=False
        )
        print("=== Generated ===")
        print(generated_text)
        print("=== End Generated ===")
        return {
            "generated_text": generated_text,
            "prompt_tokens": inputs.input_ids.shape[1],
            "generated_tokens": len(generated)
        }