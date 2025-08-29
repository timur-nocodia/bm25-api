from fastembed import TextEmbedding

# Check if BGE-M3 is in supported models
print("Checking for BGE-M3 in FastEmbed supported models...")
print("-" * 50)

models = TextEmbedding.list_supported_models()
bge_models = [m for m in models if 'bge' in m['model'].lower()]

print(f"Found {len(bge_models)} BGE models:")
for model in bge_models:
    print(f"  - {model['model']}")
    print(f"    Dimensions: {model['dim']}")
    print(f"    Size: {model.get('size_in_GB', 'N/A')} GB")
    print()

# Check specifically for BGE-M3
m3_models = [m for m in models if 'm3' in m['model'].lower()]
if m3_models:
    print("BGE-M3 models found:")
    for model in m3_models:
        print(f"  - {model['model']}")
else:
    print("\nBGE-M3 not directly supported in FastEmbed yet.")
    print("Alternative: Use sentence-transformers or FlagEmbedding library")