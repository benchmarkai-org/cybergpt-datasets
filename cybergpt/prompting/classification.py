import argparse
import pickle
import random
import tiktoken
import json
import dotenv
import os
from openai import OpenAI
from tqdm.auto import tqdm

MIN_TEST_SIZE = 5


if __name__ == "__main__":
    """
    Example usage:
    python -m cybergpt.prompting.classification \
        --dataset_path data/contrastive/classification_dataset.pkl \
        --system-prompt cybergpt/prompting/class_system_prompt.txt \
        --output-path data/contrastive/classification_results.pkl \
        --sample-size 200 \
        --max-system-tokens 80000
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_path",
        type=str,
        default="data/contrastive/classification_dataset.pkl",
    )
    args.add_argument(
        "--system-prompt",
        type=str,
        default="cybergpt/prompting/class_system_prompt.txt",
    )
    args.add_argument(
        "--output-path", type=str, default="data/contrastive/classification_results.pkl"
    )
    args.add_argument("--sample-size", type=int, default=None)
    args.add_argument("--max-system-tokens", type=int, default=80000)
    args = args.parse_args()

    RANDOM_SEED = 42
    SYSTEM_PROMPT = open(args.system_prompt, "r").read()
    max_system_tokens = int(args.max_system_tokens)

    dataset = pickle.load(open(args.dataset_path, "rb"))

    random.seed(RANDOM_SEED)
    dataset = [d for d in dataset if len(d["test_values"]) >= MIN_TEST_SIZE]
    if args.sample_size is not None:
        sample_size = int(args.sample_size)
        dataset = random.sample(dataset, sample_size)
        assert len(dataset) == sample_size

    dotenv.load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    class_results = []
    for d in tqdm(dataset, desc="Classifying dataset"):

        # Take items until we hit the token budget for the system prompt
        full_system_prompt = f"{SYSTEM_PROMPT}\n<HISTORY>\n"
        token_count = len(
            tiktoken.get_encoding("cl100k_base").encode(full_system_prompt)
        )

        random.shuffle(d["values"])
        for i, item in enumerate(d["values"]):
            string_value = f"{item}\n"
            new_tokens = len(tiktoken.get_encoding("cl100k_base").encode(string_value))
            if token_count + new_tokens > max_system_tokens:
                break
            full_system_prompt += string_value
            token_count += new_tokens
        full_system_prompt += "</HISTORY>"

        # Mix d["test_values"] with d["contrastive_values"] randomly, keeping track of which is which
        mixed_values = d["test_values"] + d["contrastive_values"]
        mixed_values_is_test = [True] * len(d["test_values"]) + [False] * len(
            d["contrastive_values"]
        )
        mixed_values_tuples = random.sample(
            list(zip(mixed_values, mixed_values_is_test)),
            len(mixed_values),
        )
        mixed_values = [x[0] for x in mixed_values_tuples]
        mixed_values_is_test = [x[1] for x in mixed_values_tuples]

        query_prompt = ""
        for i, item in enumerate(mixed_values):
            query_prompt += f"{i+1}. {item}\n"

        # Make a call to openai model with system prompt and user query prompt
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": query_prompt},
            ],
        )

        # Get response
        response_text = response.choices[0].message.content
        try:
            response_json = json.loads(
                response_text.replace("```json", "").replace("```", "")
            )
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {response_text}")
            response_json = None

        class_results.append(
            {
                "class": d["class"],
                "response_text": response_text,
                "is_test": mixed_values_is_test,
                "response_json": response_json,
            }
        )

    pickle.dump(class_results, open(args.output_path, "wb"))
