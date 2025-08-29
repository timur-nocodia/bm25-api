from fastembed import TextEmbedding
import pandas as pd

# List all supported dense models
models = TextEmbedding.list_supported_models()
df = pd.DataFrame(models)
print("Available Dense Embedding Models:")
print(df[['model', 'dim', 'size_in_GB', 'license']].to_string())

# Test a popular model
print("\nTesting BAAI/bge-small-en-v1.5:")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(["test document", "another test"]))
print(f"Output shape: {len(embeddings)} texts, {len(embeddings[0])} dimensions")
print(f"First 5 dims: {embeddings[0][:5]}")