import queue
import time
import logging
logging.basicConfig(level=logging.INFO)
import requests
import json

from config import KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class LLM():
    def __init__(self):
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": KEY['OPEN_ROUTER_AUTHORIZATION_TOKEN'],
            "Content-Type": "application/json",
        }
        self.llm_queue = queue.Queue()
        self.stop_signal = object()
        self.max_word_limit = 5

        self.interruption_word = "ধন্যবাদ"
        self.is_interruption_found = False

        logging.info("-------------------------- LLM Started ------------------------------")

    def get_response(self, tts):
        """
        Generates response from the LLM.
        """
        while True:
            try:
                text = self.llm_queue.get_nowait()
                self.is_interruption_found = False

                query_list = text.split()
                if self.interruption_word in query_list:
                    logging.info(f"---------------------------------------------------------------------------> LLM: Stop word found. Query={text}")
                    self.is_interruption_found = True
                    continue

                t1 = time.time()
                prev = t1
                first_space = False

                logging.info(f"\n\n****************************** Text input to LLM: text={text} time={time.time()} s")

                payload = {
                    "model": "openai/gpt-4.1-nano", 
                    "messages": [
                        {
                            'role': "system",
                            'content': 'You are a helpful chatbot who understands Bengali regional dialects and only speaks standard Bengali language. Please be concise and end every sentence with \"।\"'
                        },
                        {
                            "role": "user", 
                            'content': f"Please genrate response for only valid query. For invalid query print only a \"$\". Here is the query in Bengali regional dialect\n\"{text}\""
                        }
                    ],
                    "stream": True,
                }

                text = ""
                response = ""
                with requests.post(self.url, headers=self.headers, json=payload, stream=True) as r:
                    r.encoding = "utf-8"
                    for raw in r.iter_lines(decode_unicode=False):
                        if not raw:
                            continue

                        line = raw.decode("utf-8", errors="ignore").strip()
                        if line.startswith(":"):
                            continue
                        
                        if not line.startswith("data: "):
                            continue

                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")

                            if delta.strip() == "$":
                                logging.info("Invalid response from user: $")
                                # break
                                # continue


                            if delta:
                                text += delta
                                response += delta

                                if len(response) > 0 and"$" in response:
                                    logging.info(f"Invalid user query, response={response}")
                                    # break

                                if ("।" in text or "?" in text or "!" in text) and len(text) > 1:
                                    tts.text_queue.put(text)
                                    logging.info(f"LLM response={text}")
                                    logging.info(f"Time taken in LLM={(time.time() - prev):.2f} s")
                                    prev = time.time()
                                    text = ""
                        except json.JSONDecodeError:
                            logging.info("LLM response error")

                    if len(text) > 0 and len(response) > 0 and "$" not in response and text.strip() != "।":
                        logging.info(f"LLM response={text}")
                        logging.info(f"Time taken in LLM={(time.time() - prev):.2f} s")
                        prev = time.time()
                        tts.text_queue.put(text)

                    logging.info(f"Time taken in LLM for generating full response={time.time() - t1:.2f} s\n\n")
            except queue.Empty:
                time.sleep(0.001)
