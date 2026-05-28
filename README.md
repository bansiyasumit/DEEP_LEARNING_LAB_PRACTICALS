# Deep Learning & Computer Vision Repository

A collection of implementation blueprints spanning classical neural networks, modern computer vision pipelines, sequence modeling, and generative AI across **TensorFlow, Keras, PyTorch, and Darknet**.

---

## Architecture Roadmap

```
[1. Foundations] ──> [2. Deep Multi-Layer] ──> [3. Sequential & Temporal] ──> [4. Advanced Vision & GenAI]
   - Perceptrons        - Dense Keras MLPs       - Text RNN Classifiers       - Vision Transformers (ViT)
   - Optimizers         - Native TF Graph MLPs   - Hybrid CNN-RNN Video       - Latent Stable Diffusion

```

---

## Foundational & Multi-Layer Networks

### 1. Single-Layer Perceptron Gradient Optimizer

* **File:** `single_layer_perceptron_gradient_optimizer_pdfFile.pdf`
* **Objective:** Optimization of base parameters ($w$, $b$) via raw gradient computation to learn a fundamental linear transformation.
* **Stack:** TensorFlow v1 (`tf.Variable`, `tf.placeholder`, `tf.Session`).
* **Details:** Computes Root Mean Squared Error (RMSE) via `tf.reduce_sum(tf.square(...))`. Uses a `GradientDescentOptimizer` (learning rate: $0.01$) over 1000 epochs to reduce error from **20.16** to **$4.66 \times 10^{-11}$**.

### 2. Sonar Dataset Classification (Perceptron)

* **File:** `sonar_single_layer_perceptron_pdfFile.pdf`
* **Objective:** Binary classification of acoustic reflections to distinguish between "Rocks" (R) and "Mines" (M).
* **Stack:** TensorFlow v1, Pandas, Scikit-Learn.
* **Details:** Maps 60 analytical frequency bands using a single-layer feedforward graph paired with a 2-class `tf.nn.softmax` output layer. Achieves a test accuracy of **57.14%**.

### 3. Room Classification using MLP (Keras API)

* **File:** `room_classification_mlp_keras_pdfFile.pdf`
* **Objective:** Multi-class image recognition separating indoor scenes into Bedrooms, Dining Rooms, and Living Rooms.
* **Stack:** Keras, TensorFlow v2, OpenCV (`cv2`).
* **Details:** Resizes imagery to $300 \times 300 \times 3$ and normalizes pixels to $[0, 1]$. Features a `Flatten` layer ($270,000$ inputs), a `Dense` hidden layer (256 nodes, `tanh`), and a 3-node Softmax layer. Tracks $69,121,027$ params optimized via `AdamOptimizer`.

### 4. Room Classification using MLP (Native TensorFlow)

* **File:** `rooms_classification_using_mlp_tensorflow_pdfFile.pdf`
* **Objective:** Re-engineers the 3-class room recognition objective using low-level tensor graph operations.
* **Stack:** TensorFlow v1 Graph Architecture.
* **Details:** Downsamples inputs to a flat 30,000-dimensional matrix ($100 \times 100 \times 3$). Manually wires weight/bias dictionaries using a dual-hidden-layer network (256 neurons each) running manual `ReLU` mappings. Secure a final test precision of **35.0%**.

---

## Sequential & Temporal Architectures

### 5. IMDB Text Sentiment Classification

* **File:** `rnn_classification_pdfFile.pdf`
* **Objective:** Sequence modeling executing text classification over movie reviews for binary sentiment determination.
* **Stack:** PyTorch (`nn.Module`, `DataLoader`), Hugging Face Datasets.
* **Details:** Encodes text blocks into a uniform, 100-token index vector via zero-padding. Routes sequences through an `nn.Embedding` (50-dim), an `nn.RNN` (64 hidden states), a 50% `nn.Dropout` layer, and an `nn.Linear` projection matrix. Finalizes an absolute test accuracy score of **63.74%**.

### 6. Hybrid CNN-RNN Video Action Recognition

* **File:** `video_classification_pdfFile.pdf`
* **Objective:** Spatio-temporal analysis executing video activity recognition over a subsampled UCF101 dataset.
* **Stack:** TensorFlow v2, Keras, OpenCV.
* **Details:** Extracts 2048-dimensional spatial footprints per frame using a frozen **InceptionV3** network. Sequences frame blocks (capped at length 20) into stacked Gated Recurrent Units (`keras.layers.GRU`). Incorporates masking to shield padding from downstream gradient updates.

---

## Bounding Box Regression & Modern Object Detection

### 7. Bounding Box Detection (CUB-200)

* **File:** `Pytorch1_Bounding_Box_Detection_Solutions_pdfFile.pdf`
* **Objective:** Bounding box coordinate regression pipeline across the avian CUB_200_2011 dataset.
* **Stack:** PyTorch, Albumentations, Kagglehub.
* **Details:** Customizes a pre-trained **ResNet34** (21M parameters) to produce a 4-node output layer for spatial positioning ($x, y, w, h$). Employs a custom vectorized `BboxIOU` module to track Intersection over Union metrics alongside a mixed-precision `torch.cuda.amp` training loop.

### 8. Object Detection with Pre-trained Models

* **File:** `Object_detection_pretrained.ipynb - Colab.pdf`
* **Objective:** Out-of-the-box spatial inference comparisons using highly established vision models.
* **Stack:** PyTorch, Torchvision Models, Pillow.
* **Details:** Evaluates image inference structures across two backbones: **Faster R-CNN** with a ResNet50-FPN backbone for precision, and **SSD Lite** paired with a MobileNet V3 Large backbone for edge efficiency.

### 9. YOLOv4 Traffic Sign Detection

* **File:** `yolov4_traffic_sign_detection_pdfFile.pdf`
* **Objective:** Localization and multi-class classification of custom road signs using the C-based Darknet engine.
* **Stack:** Darknet (AlexeyAB fork) compiled with OpenCV, GPU, and cuDNN.
* **Details:** Configures multi-scale `[YOLO]` layers for 4 custom classes. Pre-YOLO convolutional filtering depths are tuned explicitly to 27 via the formula: $(\text{classes} + 5) \times 3$. Implements transfer learning starting from `yolov4.conv.137` anchors.

### 10. YOLOv6 Finetuning Implementation

* **File:** `yolov6_implementation_pdfFile.pdf`
* **Objective:** Sub-centisecond edge object detection finetuning on target custom imagery.
* **Stack:** PyTorch, YOLOv6 Native Engine, CUDA (NVIDIA GeForce RTX 3090).
* **Details:** Runs accelerated training routines via `tools/train.py` mapped to a custom configuration script (`yolov6s_finetune.py`) using a batch scale of 16 over 50 epochs. Logs COCO validation precision bounds ($AP \text{ @ } IoU=0.50:0.95$).

---

## Generative AI & Vision Transformers

### 11. MNIST Image Reconstruction (Autoencoder)

* **File:** `mnist_autoencoder_pdfFile.pdf`
* **Objective:** Unsupervised representation learning and spatial compression using a symmetric bottleneck topology.
* **Stack:** PyTorch Core, Torchvision Datasets.
* **Details:** Features an **Encoder** compressing input grids ($28 \times 28 \rightarrow 748$ pixels) into a low-dimensional latent space vector, and a **Decoder** mapping features back into structural image arrays via Mean Squared Error (`nn.MSELoss`) optimization.

### 12. Generative Art via Stable Diffusion

* **File:** `stable_diffusion_pdfFile.pdf`
* **Objective:** High-fidelity latent text-to-image synthesis using conditional diffusion modeling.
* **Stack:** Hugging Face `diffusers`, `transformers`, `accelerate` framework.
* **Details:** Pipelines textual prompt vectors through a frozen CLIP text encoder, maps spatial attributes into an optimized low-dimensional Variational Autoencoder (VAE) space, and applies an 860M parameter U-Net engine to iteratively eliminate Gaussian noise.

### 13. Emotion Recognition via Vision Transformer (ViT)

* **File:** `Emotion-Recognition-using-the-Vision-Transformer_pdfFile.pdf`
* **Objective:** Fine-tuning an attention-based Vision Transformer for facial expression classification.
* **Stack:** Hugging Face Hub (`ViTForImageClassification`), PyTorch.
* **Details:** Slices $224 \times 224$ facial images into non-overlapping local grids (patches). Passes patch linear projections combined with 1D spatial position embeddings into standard Transformer Encoder blocks to isolate long-range expression relationships via Multi-Head Self-Attention.

### 14. Multi-Modality Image-Image Registration

* **File:** `Image-Image multimodality registration_pdfFile.pdf`
* **Objective:** Cross-modality geometric alignment mapping image properties across separate sensor environments.
* **Details:** Core alignment relies on maximizing **Mutual Information (MI)** entropy indicators instead of absolute pixel differences (e.g., matching structural MRI sequences with clinical CT grids). Optimizes spatial arrays using rigid, affine, or non-rigid deformable scaling rules until spatial grids line up.

---

## Setup & Environment Initialization

### 1. Repository Setup

```bash
# Clone custom detection framework engines
git clone https://github.com/AlexeyAB/darknet
git clone https://github.com/meituan/YOLOv6

# Instantiate python virtual dependency isolation space
python -m venv venv
source venv/bin/activate # Win: venv\Scripts\activate

# Install foundational deep learning environment stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow albumentations diffusers transformers accelerate pandas opencv-python matplotlib

```

### 2. Native Darknet Compilation (YOLOv4)

```bash
cd darknet/
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
sed -i 's/LIBSO=0/LIBSO=1/' Makefile
make

```
