from fastembed import TextEmbedding

# List all supported dense models
print("Available Dense Embedding Models:")
print("-" * 80)
models = TextEmbedding.list_supported_models()
for model in models[:10]:  # Show first 10
    print(f"Model: {model['model']}")
    print(f"  Dimensions: {model['dim']}")
    print(f"  Size: {model.get('size_in_GB', 'N/A')} GB")
    print(f"  License: {model.get('license', 'N/A')}")
    print(f"  Description: {model.get('description', 'N/A')[:100]}...")
    print()

# Popular models for local deployment
print("\nRecommended models for local deployment:")
print("1. BAAI/bge-small-en-v1.5 - 384 dims, 0.13 GB, fast & accurate")
print("2. sentence-transformers/all-MiniLM-L6-v2 - 384 dims, 0.09 GB, very fast")
print("3. BAAI/bge-base-en-v1.5 - 768 dims, 0.44 GB, higher accuracy")
print("4. intfloat/multilingual-e5-large - 1024 dims, 2.24 GB, multilingual")