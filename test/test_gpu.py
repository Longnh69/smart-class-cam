print("="*60)
print("CUDA 13.0 GPU TEST")
print("="*60)

# Test 1: ONNX Runtime
print("\n[1] Testing ONNX Runtime...")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"Available: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("✓ CUDA provider found!")
    else:
        print("✗ CUDA provider NOT found")
        print("   → Install cuDNN 9.x")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: InsightFace with GPU
print("\n[2] Testing InsightFace GPU...")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    provider = app.det_model.session.get_providers()[0]
    print(f"✓ InsightFace using: {provider}")
    
    if 'CUDA' in provider:
        print("✓ GPU WORKING!")
    else:
        print("✗ Still using CPU")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Task Manager Check
print("\n[3] Manual Check:")
print("Open Task Manager → Performance → GPU")
print("Run your program and check if GPU usage increases")
print("\nIf GPU usage stays at 0% → cuDNN not installed correctly")