# -----------------------------
# IMPORTS
# -----------------------------
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# API KEYS
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AI_Travel_Planner"

# -----------------------------
# EMBEDDINGS
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# SAFE STATIC DATA (NO CRASH)
# -----------------------------
def create_travel_db():
    docs = [
        Document(page_content="Goa: Baga Beach, Calangute, nightlife, seafood."),
        Document(page_content="Hyderabad: Charminar, Golconda Fort, biryani."),
        Document(page_content="Kerala: Alleppey backwaters, Munnar hills."),
        Document(page_content="Jaipur: Amber Fort, Hawa Mahal, City Palace."),
        Document(page_content="Paris: Eiffel Tower, Louvre Museum."),
        Document(page_content="Dubai: Burj Khalifa, Desert Safari."),
    ]

    return Chroma.from_documents(docs, embedding)

vectorstore = create_travel_db()

# -----------------------------
# MEMORY
# -----------------------------
chat_memories = {}

def get_memory(chat_id):
    if chat_id not in chat_memories:
        chat_memories[chat_id] = InMemoryChatMessageHistory()
    return chat_memories[chat_id]

# -----------------------------
# LLM
# -----------------------------
def get_llm(temp=0.7):
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=temp
    )

# -----------------------------
# QUERY REWRITE
# -----------------------------
def rewrite_query(question):
    prompt = ChatPromptTemplate.from_template("""
Convert user query into a clear travel planning request.

Query: {question}
""")
    chain = prompt | get_llm(0) | StrOutputParser()
    return chain.invoke({"question": question})

# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve_docs(q):
    return vectorstore.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(q)

def build_context(docs):
    return "\n\n".join([d.page_content for d in docs])

# -----------------------------
# EXTRACT LOCATIONS
# -----------------------------
def extract_locations(text):
    prompt = ChatPromptTemplate.from_template("""
Extract place names. Return comma-separated only.

{text}
""")
    chain = prompt | get_llm(0) | StrOutputParser()
    res = chain.invoke({"text": text})
    return [p.strip() for p in res.split(",") if p.strip()]

# -----------------------------
# GENERATE RESPONSE
# -----------------------------
def generate_response(question, memory):
    refined = rewrite_query(question)
    docs = retrieve_docs(refined)
    context = build_context(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an AI Travel Planner.

Provide:
- Day-wise itinerary
- Must visit places
- Food suggestions
- Tips

Keep it simple and useful.
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "Query: {question}\n\nContext:\n{context}")
    ])

    chain = prompt | get_llm() | StrOutputParser()

    return chain.invoke({
        "question": refined,
        "context": context,
        "chat_history": memory.messages
    })

# -----------------------------
# MAIN
# -----------------------------
def get_response(question, chat_id):
    memory = get_memory(chat_id)

    answer = generate_response(question, memory)
    places = extract_locations(answer)

    memory.add_message(HumanMessage(content=question))
    memory.add_message(AIMessage(content=answer))

    return answer, places
