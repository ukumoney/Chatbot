import gradio as gr
import json

def save_to_json(api_key, embedding_model, chat_model, max_tokens, temperature, freq_penalty, presence_penalty):
    # Create a dictionary with the user's selections
    user_selections = {
        "API Key": api_key,
        "Embedding Model": embedding_model,
        "Chat Model": chat_model,
        "Max Tokens": max_tokens,
        "Temperature": temperature,
        "Frequency Penalty": freq_penalty,
        "Presence Penalty": presence_penalty,
    }

    # Save the user's selections to a JSON file
    with open("Settings\\user_selections.json", "w") as f:
        json.dump(user_selections, f)

    return "Selections saved successfully!"

# Define the interface
api_key_input = gr.components.Textbox(label="API Key", lines=1, info="Your OpenAI API key")
embedding_model_input = gr.components.Dropdown(choices=['text-embedding-ada-002', 'text-similarity-*-001', 'text-search-*-*-001', 'code-search-*-*-001'], label="Embedding Model", info="The model used to embed the user's input")
chat_model_input = gr.components.Dropdown(choices=['gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613'], label="Chat Model", info="The model used to generate the bot's response")
max_tokens_input = gr.components.Number(label="Max Tokens", info="The maximum number of tokens to generate for each request")
temperature_input = gr.components.Slider(value=1.0, maximum=2.0, minimum=0, label="Temperature", info="Controls randomness. As the temperature approaches zero, the model will become deterministic and repetitive while higher temperature results in more random completions.")
freq_penalty_input = gr.components.Slider(value=0.0, maximum=2.0, minimum=-2.0, label="Frequency Penalty", info="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")
presence_penalty_input = gr.components.Slider(value=0.0, maximum=2.0, minimum=-2.0, label="Presence Penalty", info="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")

# Create the Gradio interface
gr.Interface(
    fn=save_to_json,
    inputs=[
        api_key_input,
        embedding_model_input,
        chat_model_input,
        max_tokens_input,
        temperature_input,
        freq_penalty_input,
        presence_penalty_input
    ],
    outputs=gr.components.Textbox(label="Status", lines=1, value="Waiting for user input..."),
    allow_flagging="never",
    title="Chatbot Configuration",
    description="Saves user selections to a JSON file.",
    examples=[
        ["your_secret_api_key", "text-embedding-ada-002", "gpt-3.5-turbo", 256, 0.7, 1.0, -0.5]
    ]
).launch()
