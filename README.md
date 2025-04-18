
# 🦉 Open-Vocabulary Object Detection with OWL-ViT

This project demonstrates **Open-Vocabulary Object Detection** using the [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) model from Hugging Face. It supports **zero-shot detection** based on custom text queries as well as **image-guided detection** using a reference image.

---

## 🚀 Features

- 🔎 **Zero-Shot Object Detection** – Detect objects by specifying natural language queries like `"zebra"` or `"elephant"` without any fine-tuning.
- 🖼️ **Image-Guided Detection** – Provide a query image and find similar objects in the target image.
- 📦 Powered by `transformers`, `PyTorch`, `OpenCV`, and `Pillow`
- 📊 Visualize predictions with bounding boxes and confidence scores using `matplotlib`

---

## 🧠 Model

- **Model**: [`google/owlvit-base-patch32`](https://huggingface.co/google/owlvit-base-patch32)
- **Architecture**: Vision Transformer with dual text-image encoder, trained with contrastive and detection objectives.
- **Framework**: PyTorch

---

## 🛠️ Installation

```bash
pip install -q git+https://github.com/huggingface/transformers.git
pip install torch torchvision
pip install Pillow matplotlib opencv-python scikit-image
```

---

## 📂 Project Structure

```
open_vocabulary_detection/
│
├── open_vocabulary.py         # Main implementation
├── sample_images/             # Folder to store local test images
└── README.md                  # Project documentation
```

---

## 📸 How It Works

### 1. Zero-Shot Detection

Provide a list of text queries and an input image. The model detects objects in the image that match the queries.

```python
text_queries = ["zebra", "elephant"]
image = Image.open("sample_images/zebra.jpg")
inputs = processor(text=text_queries, images=image, return_tensors="pt")
outputs = model(**inputs)
```

Visualize bounding boxes on the image based on prediction confidence.

---

### 2. Image-Guided Detection

Provide both a target image and a reference query image.

```python
inputs = processor(images=target_image, query_images=query_image, return_tensors="pt")
outputs = model.image_guided_detection(**inputs)
```

Use post-processing to extract and display matching regions.

---

## 📊 Example Outputs

| Input Image | Query ("Zebra") | Output |
|-------------|------------------|--------|
| ![zebra](sample_images/zebra.jpg) | 🟩 Detected: Zebra with 95% confidence | ![output](output_zebra.jpg) |

---

## ✅ TODOs

- [ ] Add Gradio or Streamlit-based UI
- [ ] Webcam input for real-time detection
- [ ] Deploy demo on Hugging Face Spaces
- [ ] Extend with video input support

---

## 💡 Use Cases

- Wildlife monitoring
- Surveillance systems
- Visual content search
- Educational tools in computer vision

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

- Hugging Face 🤗 Transformers
- Google Research for OWL-ViT
- PyTorch & PIL community
