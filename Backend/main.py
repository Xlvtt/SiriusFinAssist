from dotenv import load_dotenv
load_dotenv()


from graph import assistant


if __name__ == "__main__":
    query = input("Ask your question:")
    for step in assistant.stream({"input": query}):
        for key, value in step.items():
            print(value)