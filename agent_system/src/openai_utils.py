import logging
import os
import tiktoken
import threading
import time
import openai
from concurrent.futures import ThreadPoolExecutor
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)


# Custom thread pool executor for gpt requests
class CustomThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers):
        super().__init__(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        return super().submit(fn, *args, **kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_gpt_response(messages, model="gpt-4-turbo-preview", temperature=0):
    logging.info("Getting gpt response ...")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=False
    )
    logging.info(f"GPT response: {response}")
    try:
        output = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error getting gpt response: {e}")
        output = ""
    return output


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_gpt_response_multithread(messages, model="gpt-4-turbo-preview", temperature=0.0):
    counter_api_request = 0
    counter_tokens = 0
    lock = threading.Lock()

    openai.api_key = os.environ.get('OPENAI_API_KEY')

    timeout_seconds = 60  # Set the timeout to 60 seconds (1 minute)

    start_time = time.time()
    done = False
    user_prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else messages[-2]["content"]

    token_count = num_tokens_from_string(user_prompt, "cl100k_base")

    with lock:
        while not done and (time.time() - start_time) < timeout_seconds:
            print(counter_api_request)
            if counter_api_request >= 100 or counter_tokens + token_count >= 20000:

                time_diff = 60 - (time.time() - start_time)
                if time_diff > 0:
                    print(f'........user_msg is {user_prompt}')
                    print('Waiting to avoid hitting rate and token limits per minute.')
                    time.sleep(time_diff)
                    counter_api_request = 1
                    counter_tokens = token_count
                    start_time = time.time()
                else:
                    counter_api_request += 1
                    counter_tokens += token_count

            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=120
                )
                # print(f'response is {response}')
            except Exception as e:
                logging.info(f"Exception encountered {e}, going to sleep")
                time.sleep(60)
                continue
            done = True

    if not done:
        logging.info("Timeout exceeded, stopping the loop.")
    gpt_msg = response.choices[0].message.content
    total_tokens_usage = response.usage.total_tokens
    return gpt_msg, total_tokens_usage

