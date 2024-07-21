import gradio as gr
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


def initialize_sales_bot(vector_store_dir: str = "real_estates_sale_homework"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(

    ), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0,

                     )

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                      search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = False

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        prompt_template = """你是一个优秀的显卡销售，你会介绍好自己的产品，会想办法把自己的产品推销出去，
           当话题超出范围时或当顾客问到你不知道或不确定的问题，你会委婉的说出自己不清楚，
           礼貌地回避这个问题并给到顾客解决方案和情绪价值。
           历史对话: {history}
           问题: {question}
           回答:
           """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "query"]
        )
        llm = ChatOpenAI(model_name="gpt-4", temperature=0,

                         )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(history=history, question=message)


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="显卡销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="localhost")


if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
