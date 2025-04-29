# YouTube-Video-Q-A-Bot
An interactive chatbot that allows users to ask questions about the content of any YouTube video — powered by **transcript-based retrieval**, **OpenAI's GPT-4o-mini**, and **FAISS vector search**.

> 💡 This app extracts the transcript from a YouTube video, chunks it, embeds it, and uses semantic search + LLMs to answer user questions based only on that video’s transcript.

---

## 🚀 Features

- ✅ Extracts transcripts from English-language YouTube videos.
- ✅ Splits and embeds transcript content using `RecursiveCharacterTextSplitter`.
- ✅ Stores chunked embeddings in FAISS vector store.
- ✅ Answers questions using `ChatOpenAI` (`gpt-4o-mini`) with strict context-based responses.
- ✅ Fully interactive UI with [Streamlit](https://streamlit.io/).

---


---

## 🧠 Tech Stack

- [Python 3.10+](https://www.python.org/)
- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/youtube-qa-bot.git
cd youtube-qa-bot
