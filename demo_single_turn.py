from starter.agent import Agent

agent = Agent("data/catalog.jsonl")
agent.reset("test_session", {})

response = agent.respond("test_session", "I'm looking for black running shoes, size 9", turn=1, top_k=5)

print("Message:", response["message"])
print("Recommendations:", response["recommendations"])
print("Ask attribute:", response["ask_attribute"])