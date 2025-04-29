import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import os
os.environ["OPENAI_API_KEY"] =""
# Helper functions
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunks["text"] for chunks in transcript_list)
        return transcript
    except TranscriptsDisabled:
        st.error("Captions are disabled for this video.")
        return None

def process_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain

def extract_video_id(url):
    """ Simple function to extract video ID from full YouTube URL """
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        return None


st.title(" YouTube Video Q&A Bot")
st.write("Paste a YouTube video link and ask any question based on its transcript!")

video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Load Video Transcript"):
    video_id = extract_video_id(video_url)
    if video_id:
        transcript = get_transcript(video_id)
        if transcript:
            st.session_state['main_chain'] = process_transcript(transcript)
            st.success("Transcript loaded and vector store created!")
    else:
        st.error("Invalid YouTube URL.")


if 'main_chain' in st.session_state:
    user_question = st.text_input("Ask a question about the video:")
    if st.button("Get Answer"):
        if user_question:
            answer = st.session_state['main_chain'].invoke(user_question)
            st.subheader("Answer:")
            st.success(answer)
