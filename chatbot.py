import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import gradio as gr
import time
from embedding import doc_transformer, SUPPORTED_FILE_TYPES
import json

# Set Chatbot Configuration
with open('Settings\\user_selections.json', 'r') as file:
        data = json.load(file)
        
openai.api_key = data["API Key"]
print(data["API Key"])
EMBEDDING_MODEL = data["Embedding Model"]
print(data["Embedding Model"])
GPT_MODEL = data["Chat Model"]
print(data["Chat Model"])
MAX_TOKENS = data["Max Tokens"]
print(data["Max Tokens"])
TEMPERATURE = data["Temperature"]
print(data["Temperature"])
FREQ_PENALTY = data["Frequency Penalty"]
print(data["Frequency Penalty"])
PRESENCE_PENALTY = data["Presence Penalty"]
print(data["Presence Penalty"])

# # Set the OpenAI API key
# openai.api_key = OPENAI_API_KEY

# Necessary varirables
messages = [
    {"role": "system", "content": "You are a chatbot that answer questions provided a context file or without."},
]
df = pd.DataFrame()

# search function


def strings_ranked_by_relatedness(query: str, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
                                  top_n: int = 5) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def query_message(query: str, model: str, token_budget: int) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    if df.empty:
        return query
    strings, relatednesses = strings_ranked_by_relatedness(query)
    #introduction = 'Use the below documents as aid to answer the subsequent question. If the answer cannot be found, write "Not enough information provided."'
    introduction = 'Use the below documents as context to answer the subsequent question.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nRelevant article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(query: str, model: str = GPT_MODEL, token_budget: int = 4096 - 500, print_message: bool = False,) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query=query, model=model,
                            token_budget=token_budget)
    for x in messages:
        print(x)
    if print_message:
        print(message)
    if (len(messages) > 8):
        del messages[1:3]
    messages.append({"role": "user", "content": message})
    print("propmt token = " + str(num_tokens_from_messages(messages, model=model)))
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS if MAX_TOKENS > 0 else None,
        frequency_penalty=FREQ_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    response_message = response["choices"][0]["message"]["content"]
    print("total tokens = " + str(num_tokens(response_message + message, model=model)))
    messages.append({"role": "assistant", "content": response_message})
    return response_message


# Creating the Chatbot Interface

# Displays user input
def add_usertext(history, text):
    global user_text
    user_text = text
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

# Displays bot response


def add_bottext(history):
    response = ask(user_text)
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.02)
        yield history

# Uploads context file


def add_file(history, file):
    if file.name.endswith("json"):
        global df
        df = pd.read_json(file.name)
        response = "Context File uploaded."
    else:
        response = "Invalid file type. Please upload a JSON file."
    history = history + [[file.name.split('\\')[-1], ""]]
    for character in response:
        history[-1][1] += character
        time.sleep(0.02)
        yield history

def add_dir(history, dir):
    # print(', '.join([x.name.split('\\')[-1] for x in dir]))
    # return history
    global df
    df, file_path = doc_transformer(dir)
    files = ', '.join([x.name.split('\\')[-1] for x in dir if x.name.split(".")[-1].upper() in SUPPORTED_FILE_TYPES])
    history = history + [[files, ""]]
    response = "JSON data has been saved to: "+ file_path
    for character in response:
        history[-1][1] += character
        time.sleep(0.02)
        yield history

# Chatbot interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [(None, "Hi! How may I assist you?"),], elem_id="chatbot", height=600)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload context data",
                container=False,
            )
        with gr.Column(scale=0.075, min_width=0):
            json_btn = gr.UploadButton("📤", file_types=[".json"])
        with gr.Column(scale=0.075, min_width=0):
            dir_btn = gr.UploadButton("📁", file_count = "directory")
    txt_msg = txt.submit(add_usertext, [chatbot, txt], [chatbot, txt], queue=False).then(
        add_bottext, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = json_btn.upload(add_file, [chatbot, json_btn], [chatbot])
    dir_msg = dir_btn.upload(add_dir, [chatbot, dir_btn], [chatbot])

demo.queue()
demo.launch()
