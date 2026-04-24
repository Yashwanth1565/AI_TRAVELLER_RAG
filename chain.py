# -----------------------------
# CLEAN ENV
# -----------------------------
import os
os.environ.pop("LANGCHAIN_TRACING", None)

# -----------------------------
# LOAD ENV (LOCAL ONLY)
# -----------------------------
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# SAFE IMPORT WIKIPEDIA
# -----------------------------
try:
    import wikipedia
    WIKI_AVAILABLE = True
except:
    WIKI_AVAILABLE = False

# -----------------------------
# IMPORTS
# -----------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# API KEYS (LOCAL + CLOUD SAFE)
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AI_Travel_Planner"

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_DIR = "./travel_db"

# -----------------------------
# EMBEDDINGS
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# DATA FETCH
# -----------------------------
def fetch_wikipedia(place):
    if not WIKI_AVAILABLE:
        return ""
    try:
        return wikipedia.page(place).content
    except:
        return ""

def fallback_docs():
    return [
        Document(page_content="Goa beaches Baga, Calangute, nightlife."),
        Document(page_content="Hyderabad Charminar, Golconda Fort."),
        Document(page_content="Kerala backwaters, Munnar hills."),
        Document(page_content="Paris Eiffel Tower, Louvre."),
    ]

# -----------------------------
# CREATE DB
# -----------------------------
def create_travel_db():
    places = ["Goa", "Hyderabad", "Kerala", "Paris"]

    raw_docs = []
    for p in places:
        data = fetch_wikipedia(p)
        if data:
            raw_docs.append(Document(page_content=data))

    if not raw_docs:
        raw_docs = fallback_docs()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)

    db = Chroma.from_documents(docs, embedding, persist_directory=CHROMA_DIR)
    db.persist()
    return db

# -----------------------------
# LOAD DB
# -----------------------------
if not os.path.exists(CHROMA_DIR):
    vectorstore = create_travel_db()
else:
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

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
Convert query into detailed travel query.

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
Extract place names. Return comma-separated.

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
        ("system", "You are a travel planner. Give itinerary, food, tips."),
        MessagesPlaceholder("chat_history"),
        ("human", "Query: {question}\nContext:\n{context}")
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