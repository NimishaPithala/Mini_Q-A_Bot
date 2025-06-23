# rag/rag_pipeline.py
from rag.retriever import Retriever
from rag.generator import Generator

class MiniQABot:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def answer(self, question):
        chunks = self.retriever.retrieve(question)
        context = "\n".join(chunks)
        return self.generator.generate_answer(context, question)
