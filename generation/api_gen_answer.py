from openai import OpenAI
import os
import json
from tqdm import tqdm
from pathlib import Path

client = OpenAI(api_key="you api key")


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=True,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


dataset = 'dataset'
data_json = json.load(open(f'path to your base dataset', 'r'))

# FIXME Adjust the prompt as needed
QNA_PROMPT = """In a short sentence including the following content, {data}"""


logprobs_data = []
text_data = []
output = []
for idx, data in enumerate(tqdm(data_json)):
    API_RESPONSE = get_completion(
        [{"role": "user", "content": QNA_PROMPT.format(data=data['question'])}],
        model="gpt-3.5-turbo", # FIXME Update the model as needed
        logprobs=True,
        # top_logprobs=5,
    )
    logprobs = [token.logprob for token in API_RESPONSE.choices[0].logprobs.content]
    response_text = API_RESPONSE.choices[0].message.content
    print(response_text)

    output.append(
        {
            "question":data['question'],
            "answer":data['answer'],
            "generated_sequence":response_text,
            "transformed_sequence":data['turker_sequence'] if 'turker_sequence' in data else data['transformed'],
            "logprobs": logprobs,
        }
    )


json.dump(output, open(f'output path', 'w'), indent=4)
