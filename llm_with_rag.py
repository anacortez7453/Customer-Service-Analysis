"""
rag_utils.py

Once you've completed and tested Parts 1-5 in the assignment notebook,
paste your finished code into the matching sections below. This file is
what your Streamlit app (app.py) will import from.
"""
 
import os
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Part 1: your knowledge base ---
documents = [
    # TODO: paste your 10 finished documents from Part 1 here
    "Vamos offers free returns within 30 days for domestic orders only.",
    "Vamos' standard ground shipping takes 3-5 business days within the US, with the exception of overseas territories.",
    "Vamos guarantees on-time delivery for Express shipments, or the shipping fee is refunded.",
    "Vamos offers same-day delivery in select metro areas for orders placed before 11 AM local time.",
    "Vamos' international shipments typically take 7-14 business days depending on customs processing.",
    "Packages over 70 lbs require freight shipping and cannot use Vamos' standard parcel rates.",
    "Lost packages must be reported to Vamos within 30 days of the expected delivery date to qualify for a refund.",
    "Hazardous materials, including lithium batteries over 100Wh, require special handling and cannot ship via Vamos' standard air freight.",
    "To speak to a Vamos support agent, call during business hours: Monday-Friday, 8 AM-5 PM.",
    "Vamos accepts online payments via PayPal, Apple Pay, credit, or debit card.",
]

assert len(documents) == 10, f"You need exactly 10 documents, you have {len(documents)}"
doc_ids = [f"doc_{i}" for i in range(len(documents))]


# --- Part 2: build the vector store ---
def build_vector_store(documents, doc_ids):
    # TODO: paste your finished function from Part 2 here
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMADB_API_KEY"),
        tenant=os.getenv("CHROMADB_TENANT"),
        database=os.getenv("CHROMADB_DB")
    )

    collection = client.get_or_create_collection(
        name="vamos_docs",
        embedding_function=None,
    )

    vectorizer = TfidfVectorizer()      # 
    doc_embeddings = vectorizer.fit_transform(documents).toarray().tolist() 
    collection.add(
        documents=documents,
        embeddings=doc_embeddings,
        ids=doc_ids,
)
    return collection, vectorizer

# --- Part 3: retrieval ---
def retrieve(collection, vectorizer, question, n_results=3):
    # TODO: paste your finished function from Part 3 here
    question_embedding = vectorizer.transform([question]).toarray().tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
    )
    return results["documents"][0]

# --- Part 4: prompt building + generation ---
SYSTEM_PROMPT = (
    "You are a customer service assistant for Vamos."
    "Answer customer questions using ONLY the provided context."
    "If the answer is not available in the context, say to call a real life agent, type speak to a real agent."
)

def build_prompt(question, retrieved_docs):
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt

def generate_answer(messages):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "(No DEEPSEEK_API_KEY found -- check your .env file)"

    client_llm = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    response = client_llm.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        max_tokens=300,
    )
    return response.choices[0].message.content

# --- Part 5: end-to-end chat ---
def chat(collection, vectorizer, question, history=None, n_results=3):
    # TODO: paste your finished function from Part 5 here
    retrieved_docs = retrieve(collection, vectorizer, question, n_results)
    prompt = build_prompt(question, retrieved_docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": prompt}
    )

    answer = generate_answer(messages)
    return answer