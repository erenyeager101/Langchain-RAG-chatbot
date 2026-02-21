from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from src.retrieve import HybridRetriever
from dotenv import load_dotenv

load_dotenv()

class ChatAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.retriever = HybridRetriever()

        # Define Tools
        self.tools = [
            Tool(
                name="KnowledgeBase",
                func=self.retriever_tool,
                description="Use this tool to search for information in the uploaded documents and knowledge graph. Always use this first for specific questions about the context."
            ),
            Tool(
                name="WebSearch",
                func=DuckDuckGoSearchRun().run,
                description="Use this tool to search the internet for current events or general knowledge not in the documents."
            ),
            Tool(
                name="Calculator",
                func=self.calculator_tool,
                description="Use this tool for mathematical calculations."
            )
        ]

        # Define Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with access to a knowledge base and the internet. "
                       "Use the KnowledgeBase tool for questions about uploaded documents. "
                       "Use WebSearch for external information. "
                       "Always cite your sources when using the KnowledgeBase."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create Agent
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True # To show sources/steps in UI
        )

    def retriever_tool(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found."

        result = ""
        for i, doc in enumerate(docs):
            result += f"Source {i+1} ({doc.metadata.get('source', 'Unknown')}): {doc.page_content}\n\n"
        return result

    def calculator_tool(self, query: str):
        chain = LLMMathChain.from_llm(llm=self.llm, verbose=True)
        return chain.run(query)

    def chat(self, user_input: str):
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response
        except Exception as e:
            return {"output": f"Error: {str(e)}", "intermediate_steps": []}

if __name__ == "__main__":
    pass
