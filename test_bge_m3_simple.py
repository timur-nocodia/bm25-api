#!/usr/bin/env python3
"""
Simple test script to check BGE-M3 availability and basic functionality
"""
import sys

def test_imports():
    """Test if required libraries are available"""
    print("Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch available:", torch.__version__)
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        from FlagEmbedding import BGEM3FlagModel
        print("✅ FlagEmbedding available")
        return True
    except ImportError:
        print("❌ FlagEmbedding not available")
        print("Install with: pip install FlagEmbedding")
        return False

def test_fastembed_fallback():
    """Test FastEmbed as fallback"""
    print("\nTesting FastEmbed fallback...")
    try:
        from fastembed import TextEmbedding
        models = TextEmbedding.list_supported_models()
        
        # Look for multilingual models
        multilingual_models = []
        for model in models:
            name = model['model'].lower()
            if any(keyword in name for keyword in ['multilingual', 'e5', 'bge']):
                multilingual_models.append(model)
        
        print(f"✅ FastEmbed available with {len(models)} models")
        print(f"   Found {len(multilingual_models)} multilingual models:")
        
        for model in multilingual_models[:5]:  # Show first 5
            print(f"   - {model['model']} ({model['dim']} dims)")
        
        return True
        
    except Exception as e:
        print(f"❌ FastEmbed error: {e}")
        return False

def test_bge_m3_basic():
    """Test basic BGE-M3 functionality"""
    print("\nTesting BGE-M3...")
    
    try:
        from FlagEmbedding import BGEM3FlagModel
        print("Loading BGE-M3 model (this may take a few minutes)...")
        
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        print("✅ BGE-M3 model loaded successfully")
        
        # Test with English and Russian text
        test_texts = [
            "Hello world, this is a test",
            "Привет мир, это тест",
            "Test document for embedding"
        ]
        
        print("Generating embeddings...")
        embeddings = model.encode(
            test_texts, 
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        print("✅ Embeddings generated successfully!")
        print(f"   Dense shape: {len(embeddings['dense_vecs'])} x {len(embeddings['dense_vecs'][0])}")
        print(f"   Sparse vectors: {len(embeddings['lexical_weights'])}")
        
        # Show sample sparse weights for first text
        if embeddings['lexical_weights']:
            first_sparse = embeddings['lexical_weights'][0]
            top_tokens = sorted(first_sparse.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top sparse tokens: {top_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ BGE-M3 test failed: {e}")
        return False

if __name__ == "__main__":
    print("BGE-M3 Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n⚠️  BGE-M3 not available, testing fallbacks...")
        
    # Test FastEmbed fallback
    fastembed_ok = test_fastembed_fallback()
    
    # Try BGE-M3 basic test
    bge_m3_ok = test_bge_m3_basic()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"BGE-M3: {'✅ Working' if bge_m3_ok else '❌ Not available'}")
    print(f"FastEmbed: {'✅ Working' if fastembed_ok else '❌ Not available'}")
    
    if not bge_m3_ok and not fastembed_ok:
        print("\n❌ No embedding models available!")
        sys.exit(1)
    elif not bge_m3_ok:
        print("\n⚠️  BGE-M3 not available, will use FastEmbed fallback")
    else:
        print("\n✅ All systems ready!")