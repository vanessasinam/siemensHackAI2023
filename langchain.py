import os
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import ContextualCompressionRetriever, SVMRetriever, TFIDFRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import gradio as gr
import time

# build environment
def load_pdf(pdf, chain_type='stuff'):
    # loader = PyPDFLoader(pdf)
    loader = PyPDFLoader(pdf)
    data = loader.load()
    
    #split
    global all_splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200, length_function = len)
    all_splits = text_splitter.split_documents(data)
    
    #store
    global vectorstore
    global embedding
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)
   
    # docs = vectorstore.similarity_search(question)

def init():
     # Build prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer. 
    Always say the source and the pages of the document where you got the answer from.
    {context}
    Question: {question}
    Helpful Answer:"""

    template2 = """Use the following pieces of context to answer the users question.
    If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer. 
    Keep the answer as concise as possible. Reply with three sentences maximum.
    ALWAYS return a "Sources" part in your answer.
    The "Sources" part should be a reference to the source of the document from which you got your answer.

    Example of your response should be:

    ```
    The answer is foo

    Sources:
    1. abc
    2. xyz
    ```
    Begin!
    ----------------
    {summaries}
    """

    messages = [
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Chat History:\n{chat_history}\n Current Human Message: {question}\n AI answer:")
        ]
    QA_CHAIN_PROMPT = ChatPromptTemplate.from_messages(messages)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

    retriever=vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
                                                    #  combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(template)})
    return qa_chain

def qa_langchain(question):
       
    return 

# What are Siemens' sustainability goals?
# Which of these goals has a higher impact on people's well-being?
# How exactly Siemens achieve this goal?
if __name__ == "__main__":
    
    load_pdf('https://assets.new.siemens.com/siemens/assets/api/uuid:c1088e4f-4d7f-4fa5-8e8e-33398ecf5361/sustainability-report-fy2022.pdf')
    print('Finished loading PDF!')

    messages = [
        # SystemMessage('You are a helpful assistant')
    ]

    chatbot = gr.themes.Base(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.gray).set(
                    body_background_fill="*neutral_500", body_background_fill_dark="*neutral_500",
                    block_background_fill="*color_green_900", block_background_fill_dark="*color_green_900",
                    body_text_color="*neutral_900", body_text_color_dark="*neutral_900",
                    color_accent_soft="*color_grey_500", color_accent_soft_dark="*color_grey_500",
                    background_fill_primary="*neutral_400", background_fill_primary_dark="*neutral_400",
                    background_fill_secondary="*neutral_200", background_fill_secondary_dark="*neutral_200",
                    input_background_fill="#92a19a", input_background_fill_dark="#92a19a",
                    block_label_background_fill="*color_black", block_label_background_fill_dark="*color_black",
                    #button_secondary_background_fill="*color_grey_800"
                )
    
    with gr.Blocks(theme=chatbot,
                    css='.gradio-container {background-image:url("https://assets.new.siemens.com/siemens/assets/api/uuid:8bd182a9-61a5-4f7e-8569-9d875ff5a5d1/width:1024/quality:HIGH/8bd182a9-61a5-4f7e-8569-9d875ff5a5d1-high.webp");}'
                ) as demo:

        chatbot = gr.Chatbot(label="\U0001F951 Chatbot", avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))))
        dt = gr.Textbox(label="Introduction")

        def intro_msg():
            intro_m="Hi, this is the Siemens' sustainability chatbot, I have already read the sustainability report of 2022. Feel free to ask me any questions about the report?"
            return intro_m
            
        demo.load(intro_msg, inputs=None, outputs=dt)
        msg = gr.Textbox(label="Questions", placeholder="Welcome to Siemens' sustainability chatbot! Enter text and press enter!",
                            autofocus=True)

        clear = gr.Button("Clear")    

    #llm_chain, llm = init_chain(model, tokenizer)

        def user(user_message, history):

            return "", history + [[user_message, None]]

        def bot(history):

            print("Question: ", history[-1][0])

            qa_chain = init()
            greetings = [ "hi!", "hi", " ", "hii", "hello!", "hello", "how can you help me?", "hallo", "what do you do?", "what can i ask you?", "what do you do?", "who are you?", "was ist das?" ]
            
            if history[-1][0].lower() in greetings:
                bot_message = "Hello, I am Siemens Sustainability Chatbot! I can give you information about the Siemens Sustainability Report 2022."
            else:
                bot_message = qa_chain({"question": history[-1][0]})['answer']

            print("Response: ", bot_message)

            history[-1][1] = ""

            for character in bot_message:

                history[-1][1] += character

                time.sleep(0.009)

                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)

        clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()

demo.launch(share=True)