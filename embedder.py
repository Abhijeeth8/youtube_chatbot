from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
load_dotenv()

def load_transcript(video_url):
  if not video_url.startswith("https://www.youtube.com/watch?v="):
      print("Please enter a valid YouTube video URL.")
      exit()

  yt_loaded_doc = YoutubeLoader.from_youtube_url(youtube_url=video_url)
  yt_doc = yt_loaded_doc.load()
  yt_transcript = yt_doc[0].page_content
  return yt_transcript  




def embed_transcript(yt_transcript):

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

  split_text = text_splitter.create_documents([yt_transcript])
  # print(f"Number of chunks: {len(split_text)}")

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY"))
  vectorstore = Chroma.from_documents(split_text, embeddings, persist_directory="./transcript_chroma_db")