from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

with open('tools_withembeddings.json', 'r') as f:
    tools_data = json.load(f)

tools_embeddings = np.array