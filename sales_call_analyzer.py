from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

class SalesCallAnalyzer:
    def __init__(self, api_key: str, model: str = "command-r-plus"):
        self.llm = Cohere(model=model, cohere_api_key=api_key)
        self.prompt_template = """
Given the following sales call transcript, predict the likelihood(accurate) of closing the sale (in percentage), the current stage of the conversation (e.g., initial call, negotiation, closed deal), and explain why you think so in 3 sentences at the max.

Respond in the following format:
Likelihood of closing the sale: [answer]
Stage of the call: [answer]
Reason: [answer](make a point based on customer)
Suggestions : in 3 crisp short points explain how can the agent improve their strategy or enhance the conversation to close the deal, if the deal is already closed highlight what the sales agent did that made the deal close.
Sales Call Transcript: {transcript}
"""
        self.prompt = PromptTemplate(input_variables=["transcript"], template=self.prompt_template)
        self.llm_chain = RunnableSequence(self.prompt | self.llm)

    def analyze_transcript(self, transcript: str) -> str:
        response = self.llm_chain.invoke({"transcript": transcript})
        return response