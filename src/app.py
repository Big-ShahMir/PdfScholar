import streamlit as st
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import openai
from PyPDF2 import PdfMerger
import shelve


# Performs similarity search from vector space.
def content_finder(dataset, query, k=80):
    similar = dataset.similarity_search(query, k=k)
    data = " ".join([d.page_content for d in similar])
    return data


# Creates the dataset OR "content" to be used by the AI.
def create_dataset(pdf: str, embeddings) -> FAISS:
    pdf_loader = PyPDFLoader(pdf)
    document = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
    docs = text_splitter.split_documents(document)

    dt_set = FAISS.from_documents(docs, embeddings)
    return dt_set


# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat_history()

st.title("PDF Scholar ")
st.caption("Upload a PDF file to begin your Query Quest.")

USER_LOGO = "🧑"
BOT_LOGO = "📚"

if "memory" not in st.session_state.keys():
    st.session_state["memory"] = ConversationBufferMemory()
    st.session_state["files"] = []

# Sidebar with a button to delete chat history, upload file(s), and input
# OpenAI API key.
with st.sidebar:
    Api_Key = st.text_input("Input OpenAI API Key", key="APIkey",
                            type="password")
    st.divider()
    files = st.file_uploader("Choose PDF file(s)", type="pdf",
                             accept_multiple_files=True)
    st.divider()
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        profile_pic = USER_LOGO
    else:
        profile_pic = BOT_LOGO
    with st.chat_message(message["role"], avatar=profile_pic):
        st.markdown(message["content"])

query = st.chat_input("How can I help?",
                      disabled=not files)

if query and files and not Api_Key:
    st.info("Please Enter Your OpenAI API Key")
    st.stop()

if not Api_Key:
    st.stop()

if files:
    merge = PdfMerger()
    if files != st.session_state["files"]:  # Need to confirm logic
        st.session_state["files"] = []  #
        for file in files:
            st.session_state["files"].append(file)
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())

            merge.append(file.name)

        merge.write("combined.pdf")
        merge.close()

embedding = OpenAIEmbeddings(openai_api_key=Api_Key)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6, openai_api_key=
Api_Key)

# Rewrite prompt for concision --> too many wasted tokens.
template = """ Act as a informative study assistant to undergraduate and
        graduate university students who answers questions about topics
        that appear in academic papers and other educational texts.
        
        DATASET = {content}
        QUESTION = {question}
        
        Search and analyze the following text to answer the QUESTION, 
        and only use information from the DATASET.
        
        In the event the user asks "are you sure" or asks you to reevaluate
        your response present the user with excerpts from the text to show 
        your reasoning.

        Be considerate in your responses and make sure to only answer
        the question QUESTION.

        If the question the user provides does not
        state a question but rather a statement or an opinion,
        provide a mindful response acknowledging the statement and ask them
        politely if they have a question.

        If you feel like you don't have enough information to answer any
        part or the whole the question, say "I don't know" and specify
        which part of the question you cannot answer. You should not
        recommend the user to continue to inquire about that topic.
        
        Do not leave any incomplete sentences or ideas within your
        responses.
        """

chat_history_template = """

    CHAT_HISTORY = {history}
    
    Given a chat history CHAT_HISTORY and the latest user question "{question}" 
    which might reference context in the chat history, formulate a standalone 
    question which can be understood without the chat history. Do NOT answer the
    question, just reformulate it if needed otherwise return the question as is.
"""

if query and files and Api_Key:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_LOGO):
        st.markdown(query)

    with st.chat_message("assistant", avatar=BOT_LOGO):
        message_placeholder = st.empty()

        dataset = create_dataset("combined.pdf", embedding)
        content = content_finder(dataset, query)

        prompt = PromptTemplate.from_template(template).partial(content=content)
        prompt_reformulate = PromptTemplate.from_template(
            chat_history_template).partial(
            history=st.session_state["memory"].load_memory_variables({}))

        reformulate_agent = LLMChain(
            llm=model,
            prompt=prompt_reformulate,
            verbose=True
        )
        final_query = reformulate_agent.invoke({"question": query})

        agent = LLMChain(
            llm=model,
            prompt=prompt,
            memory=st.session_state["memory"],
            verbose=True
        )
        pre_answer = agent.invoke({"question": final_query["text"]})

        message_placeholder.markdown(pre_answer["text"])
        st.session_state.messages.append(
            {"role": "assistant", "content": pre_answer["text"]})

save_chat_history(st.session_state["messages"])

