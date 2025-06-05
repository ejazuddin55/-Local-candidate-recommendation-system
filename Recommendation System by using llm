
#Install required packages

pip install -q torch transformers sentence-transformers faiss-cpu accelerate
pip install -q -U bitsandbytes

#using a  small model for local use

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use a small, non-gated model that fits in T4 memory without quantization
model_name = "microsoft/phi-1_5"  # 1.3B parameter model that works well

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

#Generating a CV for 20 Candidates

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time
import re

# Load model and tokenizer
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16
)

# Helper to clean and format the CV output
def clean_cv(text, cv_id):
    # Remove repeated prompts or incomplete generations
    text = re.sub(r"(Generate a fictional CV.*?)\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    # Ensure basic formatting
    lines = text.split("\n")
    formatted = f"CV ID: {cv_id}\n" + "\n".join([line.strip() for line in lines])
    return formatted

# Prompt for CV generation
cv_prompt = (
    "Generate a fictional CV for a professional in the tech industry. "
    "Choose fields like software engineer, data scientist, product manager, UX designer, or cybersecurity analyst. "
    "Include realistic details: name, experience, skills, education, and location."
)

# Generate 20 CVs
cvs = []
for i in range(20):
    print(f"🚀 Generating CV {i+1}/20...")
    inputs = tokenizer(cv_prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.9)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    cleaned = clean_cv(decoded, cv_id=f"CANDIDATE_{i+1:03}")
    cvs.append(cleaned)
    time.sleep(1)  # throttle generation slightly

# Save to file
with open("candidate_profiles.json", "w") as f:
    json.dump(cvs, f, indent=2)

print("✅ All 20 CVs generated and saved to 'candidate_profiles.json'")

#Converting CV into embeddings and storing it in a FAISS index

from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Install FAISS (CPU version only)
pip install faiss-cpu --quiet
import faiss
import json
import numpy as np

# Load the saved CVs
with open("candidate_profiles.json", "r") as f:
    cv_texts = json.load(f)

# Load a small, efficient sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Runs locally, very fast

# Encode the CVs
print("🔍 Encoding CVs...")
embeddings = model.encode(cv_texts, show_progress_bar=True)

# Store in FAISS index
embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))

# Save the index and raw CVs for retrieval
faiss.write_index(index, "cv_faiss.index")
with open("cv_texts.json", "w") as f:
    json.dump(cv_texts, f, indent=2)

print("✅ FAISS index and CV texts saved.")

#Implementation of GPT-2 with Configurable Skipped Transformer Layers

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import torch

class SkippedGPT2(GPT2LMHeadModel):
    def __init__(self, config, skip_last_m=2):
        super().__init__(config)
        self.skip_last_m = skip_last_m

    def forward(self, input_ids, **kwargs):
        hidden_states = self.transformer.wte(input_ids)
        for block in self.transformer.h[:len(self.transformer.h) - self.skip_last_m]:
            hidden_states = block(hidden_states)[0]
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = SkippedGPT2(config, skip_last_m=4)
model.eval()

# Input text
prompt = "My name is Ejaz and I love"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Predict
with torch.no_grad():
    logits = model(input_ids)

# Decode prediction
next_token_id = logits[0, -1].argmax().item()
next_word = tokenizer.decode([next_token_id], skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Next word prediction: {next_word}")

#Interactive Candidate Filtering Using Sentence Transformers and FAISS

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

# Load CVs
with open("candidate_profiles.json") as f:
    cvs = json.load(f)

# Load model and encode CVs
embedder = SentenceTransformer('all-MiniLM-L6-v2')
cv_embeddings = embedder.encode(cvs, convert_to_numpy=True)

# Full FAISS index
dimension = cv_embeddings.shape[1]

# Start with all indices
current_indices = list(range(len(cvs)))
threshold = 5

print("🧠 System: Welcome! What kind of candidate are you looking for?")

while len(current_indices) > threshold:
    user_input = input("👤 You: ")
    query_vec = embedder.encode([user_input])

    # Filter current pool
    pool_embeddings = np.array([cv_embeddings[i] for i in current_indices])
    temp_index = faiss.IndexFlatL2(dimension)
    temp_index.add(pool_embeddings)

    # Shrink candidate pool: keep only top 50% most similar
    top_k = max(1, len(current_indices) // 2)  # reduce by half each time
    D, I = temp_index.search(query_vec, top_k)

    # Map filtered indices back
    new_indices = [current_indices[i] for i in I[0]]
    current_indices = new_indices

    print(f"🧠 System: {len(current_indices)} candidates match your request.")
    if len(current_indices) > threshold:
        print("🧠 System: Please narrow down further (e.g., add skills, years, etc.)")

# ✅ Final results
print("\n🎉 Final Candidates:")
for i in current_indices:
    print("------")
    print(cvs[i])
