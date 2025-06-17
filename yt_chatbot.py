from dotenv import load_dotenv
load_dotenv()
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from embedder import load_transcript, embed_transcript

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def summarize_transcript(yt_transcript):
    print("Summarizing the transcript...")
    summary = llm.invoke("Summarize the following transcript briefly in 20 points and always satart by saying the video discusses about and never use the word transcript:\n\n" + yt_transcript)
    print("Summary of the transcript:")
    print(summary.content)
    print("Do you have any questions about the summary or the video? (yes/no)")
    input_response = input().strip().lower()
    if input_response == "yes":
        print("You can start a conversation with the chatbot using the transcript.")
        start_chatbot(yt_transcript)
    else:
        print("Thank you! If you have any more questions in the future, feel free to ask.")
        exit()
    

def start_chatbot(yt_transcript):
    print("Embedding the video for the chatbot...")
    embed_transcript(yt_transcript)
    print("Video embedded successfully. You can now start chatting with the bot.")

    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY"))

    vectorstore = Chroma(
        persist_directory="./transcript_chroma_db",
        embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided context. Verify if the provided context is relevant to the question asked and Answer the question based on the context if only the context is relevant to the question, and do not try to come up with own answers and if either the context is not sufficient to answer the question or the question is irrelevant to the context, say that appropriate issue in a formal tone."""),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("user", "Context: \n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    conversation = []
    question = ""
    while True:
      question = input("You (Type exit to exit): ")
      if question.lower() == "exit":
          break
      conversation.append(HumanMessage(content=question))
      retrieved_docs = retriever.invoke(question)
      retrieved_context = " ".join([retrieved_doc.page_content for retrieved_doc in retrieved_docs])
      prompt = prompt_template.invoke({"context": retrieved_context, "question": question, "chat_history": conversation})

      
      response = llm.invoke(prompt)
      conversation.append(AIMessage(content=response.content))
      print("Assistant:", response.content)


if __name__ == "__main__":
    link = input("Enter the YouTube video URL: ")
    yt_transcript = load_transcript(link)

    summarize_or_converse = input("Do you want to summarize the transcript or start a conversation? (Type 'summarize' or 'chat'): ").strip().lower()
    if summarize_or_converse not in ["summarize", "chat"]:
        print("Invalid input. Please type 'summarize' or 'chat'.")
        exit()

    if summarize_or_converse == "chat":
        print("You chose to start a conversation. The chatbot will use the transcript to answer your questions.")
        start_chatbot(yt_transcript)
    else:
        print("You chose to summarize the transcript. The summary will be generated in 20 points.")
        summarize_transcript(yt_transcript)
