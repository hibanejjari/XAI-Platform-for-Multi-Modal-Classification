"""
Unified XAI Platform - Main Application
Entry point for the Gradio interface
"""

import warnings
warnings.filterwarnings("ignore")

from src.ui import create_interface


if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_interface()
    demo.launch(share=False)
