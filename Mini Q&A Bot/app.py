# app.py
from rag.rag_pipeline import MiniQABot

def main():
    print("Mini Q&A Bot 🤖")
    bot = MiniQABot()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break

        print("\nGenerating answer...")
        answer = bot.answer(query)
        print("\n🤖 Answer:", answer)

if __name__ == "__main__":
    main()
