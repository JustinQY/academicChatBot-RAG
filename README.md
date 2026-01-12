# Academy QA System for McMaster Graduates V1.0

This is an online system built for academic question answering based on the materials in the library and users update.

➡️ Try my system here:
https://macacademychatbot.streamlit.app/

## Functions

- **Academic Materials Batch Uploading**
- **Uploaded Documents Management _(check and delete)_**
- **Question Answering in a few seconds**
- **Questions History Management _(check and clear)_**

## Technical Stack

- **LLM**: OpenAI GPT-3.5-turbo (question answering), OpenAI text-embedding-3-large (embedding)
- **Vector Database**: Chroma
- **Framework**: LangChain
- **Front End**: Streamlit
- **Material Loading**: PyPDFLoader

## Project Structure

```
academicChatBot-RAG/
├── app.py                      # Streamlit Web App
├── requirements.txt            # Project Dependencies
├── config.example.json         # Example of config.json
└── chroma_db
    ├── base                    # Default materials
    └── user                    # Uploaded materials
```

## Example Questions
- Can you list some of the hyperparameters in the FFN?
- What is backpropagation?
- Explain the concept of gradient descent.

## Procedure Under the Hood

1. **Documents Loading**：read and load all pdf files from chroma_db/base and chroma_db/user
2. **Split**：Split files into chunks with maximum length of 300 tokens and 50 tokens overlap.
3. **Vectorization**：Vectorize and Store the file chunks into the Chrome vectordatabase.
4. **Retrieval**：Retrieve top 10 most related file chunks according to the question.
5. **Generation**：Custom prompt with the retrieval chunks and user question, generate answer by LLM.

## Highlights

- **PDF Documents Process**：Upload, analyze and process academic pdf files.
- **Semantic Retrieval**：Retrieving reliable files in the vectordatabases. (Embedding process powered by OpenAI Model: text-embedding-3-large)
- **AI Question Answering**：Generate answers by OpenAI: GPT-3.5-turbo.
- **Reliable Information**：For now the answer given only based on all the materials in the database.
- **Web Frontend UI**：Friendly front end interaction UI powered by Streamlit.


## Installation

1. **Clone the Repo**
```bash
git clone https://github.com/JustinQY/academicChatBot-RAG.git
cd academicChatBot-RAG
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Config API Keys**
Create `config.json` and insert your API keys based on the `config.example.json`

```bash
cp config.example.json config.json
```

```json
{
  "OpenAIAPIKey": "your-openai-api-key",
  "LangChainAPIKey": "your-langchain-api-key"
}
```

## Run

### 1: Web（Recommand）

```bash
streamlit run app.py
```

### 2: Python Script

```bash
python academicChatBot.py
```

### Document Requirements

- Only accept .pdf type files.
- Suggested size for single file is 50MB.


## Notes

- ⚠️ It takes several seconds to process default documents for the first time of cold loading.
- ⚠️ Keep an eye out on your OpenAI API Usage. [check it here](https://platform.openai.com/settings/organization/usage)

## Owner
JustinQY

