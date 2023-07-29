import os
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import PyPDF2
import json
import csv
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set Embedding Configuration
with open('Settings\\user_selections.json', 'r') as file:
        data = json.load(file)

openai.api_key = data["API Key"]
EMBEDDING_MODEL = data["Embedding Model"]
GPT_MODEL = data["Chat Model"]

# Supported file types
SUPPORTED_FILE_TYPES = ["PDF", "JSON", "CSV", "DOCX", "XLSX", "XLS", "TXT"]

# Process input data into chunks and embeddings
def doc_transformer(path) -> tuple[pd.DataFrame, str]:
    # Create a dataframe to store the text and embeddings
    df = pd.DataFrame(columns=["text", "embedding"])
    
    for file_obj in path:
        file = file_obj.name
        filetype = file.split(".")[-1].upper()
        if filetype == "PDF":
            with open(file, 'rb') as pdf_file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Extract the text from the PDF document
                document_text = ''.join([page.extract_text() for page in pdf_reader.pages])

        elif filetype == "JSON":
            with open(file, 'r') as json_file:
                # Load the JSON data
                json_data = json.load(json_file)

                # Extract the text from the JSON
                document_text = json.dumps(json_data)

        elif filetype == "CSV":
            with open(file, 'r') as csv_file:
                # Read the CSV file
                csv_reader = csv.reader(csv_file)

                # Extract the text from the CSV
                document_text = '\n'.join([' '.join(row) for row in csv_reader])

        elif filetype == "DOCX":
            doc = docx.Document(file)
            paragraphs = [p.text for p in doc.paragraphs]

            # Extract the text from the DOCX document
            document_text = '\n'.join(paragraphs)

        elif filetype == "XLSX" or filetype == "XLS":
            # Read the XLSX file
            xls_data = pd.read_excel(file)

            # Convert the data to text
            document_text = xls_data.to_string(index=False)

        elif filetype == "TXT":
            with open(file, 'r') as txt_file:
                # Read the TXT file
                document_text = txt_file.read()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        split_texts = text_splitter.split_text(document_text)

        # Add the text and embeddings to the dataframe
        for text in split_texts:
            text_embedding_response = openai.Embedding.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            text_embedding = text_embedding_response["data"][0]["embedding"]
            df = df.append({"text": text, "embedding": text_embedding}, ignore_index=True)

    # Convert DataFrame to JSON
    json_data = df.to_json(orient="records")

    # Define the file path where to save the JSON data
    iteration = len(os.listdir("embedding data"))
    file_path = f"embedding data\\record_embeddings_{iteration}.json"

    # Write the JSON data to the file
    with open(file_path, 'w') as json_file:
        json_file.write(json_data)

    return df, file_path