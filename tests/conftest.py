import os
import sys
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

# Add project root to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def dummy_image():
    """Generates a simple 600x600 RGB PIL image with a red square representing a face."""
    img = Image.new("RGB", (600, 600), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a mock face box in the middle
    draw.rectangle([200, 150, 400, 400], fill=(200, 150, 100), outline=(0, 0, 0))
    # Draw eyes
    draw.rectangle([250, 230, 270, 250], fill=(255, 255, 255))
    draw.rectangle([330, 230, 350, 250], fill=(255, 255, 255))
    return img

@pytest.fixture
def real_or_dummy_image(dummy_image):
    """Loads test_image.jpg if available, else returns the dummy image."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img_path = os.path.join(project_root, "test_image.jpg")
    if os.path.exists(test_img_path):
        try:
            return Image.open(test_img_path)
        except Exception:
            return dummy_image
    return dummy_image
