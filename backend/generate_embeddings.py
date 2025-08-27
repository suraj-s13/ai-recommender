# generate_embeddings.py
import json
from sentence_transformers import SentenceTransformer

# Load the model. This will be downloaded the first time you run it.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your tool data
with open('tools.json', 'r') as f:
    tools = json.load(f)

# Create a list of descriptions to be encoded
# We combine key fields to create a rich text for embedding
texts_to_embed = [
    f"{tool['toolName']}: {tool['description']} Keywords: {', '.join(tool['keywords'])}" 
    for tool in tools
]

# Generate embeddings
print("Generating embeddings... This may take a moment.")
embeddings = model.encode(texts_to_embed, show_progress_bar=True)
print("Embeddings generated.")

# Add embeddings to your tool data
for i, tool in enumerate(tools):
    tool['embedding'] = embeddings[i].tolist() # Convert numpy array to list for JSON

# Save the processed data to a new file
with open('tools_with_embeddings.json', 'w') as f:
    json.dump(tools, f, indent=2)

print("Saved data with embeddings to tools_with_embeddings.json")