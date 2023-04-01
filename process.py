from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt
from docx import Document
import chardet
import sys, os

#openai.api_key = ("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")

os.chdir("/Users/deepak.bhutada/Downloads/Custom-Chatbot")

def read_file_with_detected_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

def train():
    file_path = "/Users/deepak.bhutada/Downloads/Custom-Chatbot/training/facts/"
    trainingData = list(Path(file_path).glob("**/*.*"))

    if len(trainingData) < 1:
        print("The folder training/facts should be populated with at least one .txt, .doc, or .md file.", file=sys.stderr)
        return

    data = []
    for training in trainingData:
        if training.suffix == '.doc' or training.suffix == '.docx':
            doc = Document(training)
            text = "\n".join([para.text for para in doc.paragraphs])
            print(f"Add {training.name} to dataset")
            data.append(text)
        else:
            print(f"Add {training.name} to dataset")
            text = read_file_with_detected_encoding(training)
            data.append(text)

    textSplitter = CharacterTextSplitter(chunk_size=2000, separator="\n")

    docs = []
    for sets in data:
        docs.extend(textSplitter.split_text(sets))

    store = FAISS.from_texts(docs, OpenAIEmbeddings())
    faiss.write_index(store.index, "training.index")
    store.index = None

    with open("faiss.pkl", "wb") as f:
        pickle.dump(store, f)

def runPrompt():
  index = faiss.read_index("training.index")

  with open("faiss.pkl", "rb") as f:
    store = pickle.load(f)

  store.index = index

  with open("training/master.txt", "r") as f:
    promptTemplate = f.read()

  prompt = Prompt(template=promptTemplate, input_variables=["history", "context", "question"])

  llmChain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0.25))

  def onMessage(question, history):
    docs = store.similarity_search(question)
    contexts = []
    for i, doc in enumerate(docs):
      contexts.append(f"Context {i}:\n{doc.page_content}")
      answer = llmChain.predict(question=question, context="\n\n".join(contexts), history=history)
    return answer

  history = []
  while True:
    question = input("Ask a question > ")
    answer = onMessage(question, history)
    print(f"Bot: {answer}")
    history.append(f"Human: {question}")
    history.append(f"Bot: {answer}")