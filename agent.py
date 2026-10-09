import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_tool import rag_search

MODEL_PATH = r"D:\gex3-work\gex3-1\gex3-1\models\qwen3-0.6b"


def has_phone(text: str) -> bool:
    return bool(re.search(r"(\+?\d[\d\s\-()]{6,}\d)", text))


def main():
    question = "What exact phone number should I call for the University IT Service Desk?"

    context = rag_search(question)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a university IT support assistant. "
                "Answer only from the given policy context. "
                "If the context does not explicitly contain a phone number, "
                "you must say that you do not know it. "
                "Do not invent a phone number. Do not provide unrelated instructions."
            ),
        },
        {
            "role": "user",
            "content": f"Policy context:\n{context}\n\nQuestion: {question}",
        },
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    if not has_phone(context) and has_phone(answer):
        answer = (
            "I don't know the exact phone number for the University IT Service Desk. "
            "It is not provided in the available policy documents."
        )

    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()