import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('tools.json', 'r') as f:
    tools = json.load(f)

