import ollama
import pandas as pd

# Read prompts from CSV file
prompts_df = pd.read_csv("dataset/prompts.csv")
print(prompts_df.columns.tolist())
# Store responses
responses = []

# Generate answers
for index, row in prompts_df.iterrows():

    prompt = row["prompt"]

    print(f"\nQuestion {row['prompt_id']}: {prompt}")

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    print("Answer:", answer)

    responses.append({
        "prompt_id": row["prompt_id"],
        "domain": row["domain"],
        "prompt": prompt,
        "response": answer
    })

# Save responses to CSV
responses_df = pd.DataFrame(responses)

responses_df.to_csv(
    "dataset/llm_responses.csv",
    index=False
)

print("\nSuccess! All responses saved in dataset/llm_responses.csv")
# Compare AI responses
comparison = []

for item in responses:
    comparison.append({
        "prompt_id": item["prompt_id"],
        "domain": item["domain"],
        "response_length": len(item["response"].split()),
        "observation": "Response generated successfully"
    })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    "dataset/response_comparison.csv",
    index=False
)

print("Response comparison saved successfully!")
# Save conversation history
with open("dataset/conversation_history.txt", "w", encoding="utf-8") as file:
    for item in responses:
        file.write(f"Question: {item['prompt']}\n")
        file.write(f"Answer: {item['response']}\n")
        file.write("-" * 50 + "\n")

print("Conversation history saved successfully!")