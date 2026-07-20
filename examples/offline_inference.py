import os

from nanovllm_plus import LLM, SamplingParams


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    params = SamplingParams(temperature=0.6, max_tokens=128)
    outputs = llm.generate(["Introduce nano-vLLM-Plus."], params)
    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
