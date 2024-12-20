# 🐝 Beevibe

### **Feel the vibe of building smarter models.**

---

## 🐝 **About Beevibe**
Beevibe is a Python package designed to make it easier to train advanced language models, like CamemBERT, on text datasets with specific themes and perform accurate inference on new sentences. Inspired by the intelligence and collaboration of bees, Beevibe is built to empower developers and researchers with tools that are efficient, intuitive, and scalable.

Beevibe leverages modern AI features to simplify workflows and enhance user experience. It integrates:
- OpenAI and Ollama interfaces for intelligent interaction.
- AI-assisted research, documentation, and result analysis.
- Automatic dataset summarization.
- Example-guided usage to facilitate onboarding.
- Easy creation of classification heads.

---

## ✨ **Features**

- **Simplified Usage**: Designed for simplicity and efficiency, making it lightweight and easy to use.
- **AI-Powered Functionalities**:
  - OpenAI and Ollama integration for modern interaction.
  - Dataset analysis and synthesis.
  - Automated classification head creation and result interpretation.
- **Tutorial-Ready**: Includes synthetic datasets and CamemBERT tutorials.
- **Streamlined Development**:
  - Supports GitHub Codespaces and VSCode for development.
  - Colab integration for GPU testing.
- **Quality Assurance**:
  - Implements `ruff` for code linting and `pydantic` for parameter validation.
  - Comprehensive test suite for non-regression verification.

---

## 📦 **Installation**

Install Beevibe using pip:

```bash
pip install Beevibe
```

---

## 🚀 **Quickstart**

### **1. Training a Model**
Train CamemBERT on your custom thematic dataset:

```python
from Beevibe import ModelTrainer

# Initialize the trainer
trainer = ModelTrainer(model="camembert-base", dataset="path/to/dataset.csv")

# Train the model
trainer.train(epochs=3, batch_size=16)

# Save the trained model
trainer.save("trained_model")
```

### **2. Performing Inference**
Use the trained model to classify or extract themes from new sentences:

```python
from Beevibe import ModelInference

# Load the trained model
inference = ModelInference("trained_model")

# Infer themes for a new sentence
result = inference.predict("This is a new sentence to classify.")

print("Predicted Theme:", result)
```

---

## 📚 **Documentation**

Comprehensive documentation is available [here](https://github.com/fbullier/Beevibe/wiki) with detailed guides, API references, and examples.

---

## 🤝 **Contributing**

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Added feature"`).
4. Push the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## 📜 **License**

Beevibe is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### **Copyright Notice**

```
Copyright (c) 2024 Beevibe Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📖 **Citing Beevibe**

If you use Beevibe in your research, projects, or publications, please cite it as follows:

```
@misc{Beevibe2024,
  title={Beevibe: Feel the vibe of building smarter models},
  author={The Beevibe Team},
  year={2024},
  url={https://github.com/fbullier/Beevibe},
  note={Version 0.1}
}
```

By citing Beevibe, you help others discover and build upon this work!

---

## 🌟 **Acknowledgments**
- Created with the assistance of AI tools like ChatGPT.
- Inspired by the brilliance of **CamemBERT** and the power of Python.
- Special thanks to the vibrant community of developers and data scientists who make innovation possible.

---

![Beevibe Logo](logo.png)
