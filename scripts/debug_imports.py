import sys
print("Python executable:", sys.executable)
print("Path:", sys.path)

try:
    print("Importing torch...")
    import torch
    print("Torch version:", torch.__version__)
except Exception as e:
    print("Torch failed:", e)

try:
    print("Importing torchvision...")
    import torchvision
    print("Torchvision version:", torchvision.__version__)
except Exception as e:
    print("Torchvision failed:", e)

try:
    print("Importing facenet_pytorch...")
    from facenet_pytorch import InceptionResnetV1
    print("Facenet imported.")
    model = InceptionResnetV1(pretrained='vggface2').eval()
    print("Facenet model loaded.")
except Exception as e:
    print("Facenet failed:", e)

try:
    print("Importing local src...")
    import src
    from src.generator import StyleGAN2Wrapper
    print("Local src imported.")
except Exception as e:
    print("Local src failed:", e)
