"""Setup configuration for doc-reconstruction-engine."""
from setuptools import find_packages, setup

setup(
    name="doc-reconstruction-engine",
    version="0.1.0",
    description="AI-based Document Reconstruction Engine",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "python-docx>=1.1.0",
        "reportlab>=4.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "ocr": ["paddleocr>=2.7.0", "paddlepaddle>=2.5.0", "pytesseract>=0.3.10"],
        "dl": ["torch>=2.0.0", "torchvision>=0.15.0"],
        "layout": ["layoutparser>=0.3.4"],
        "ui": ["gradio>=4.0.0"],
        "all": [
            "paddleocr>=2.7.0",
            "paddlepaddle>=2.5.0",
            "pytesseract>=0.3.10",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "layoutparser>=0.3.4",
            "gradio>=4.0.0",
            "scipy>=1.11.0",
        ],
    },
)
