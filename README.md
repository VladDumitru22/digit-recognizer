# Digit Recognizer ✏️

## Overview 📝

This project is a **Python-based handwritten digit recognition application** using **PyTorch**. Users can draw digits on a canvas and see the model’s prediction. This project demonstrates **machine learning workflow, data preprocessing, neural network training, and deployment with a simple GUI**.  

---

## Features 🖥️

- **MNIST-based Neural Network**: Fully connected network trained on MNIST dataset.  
- **Data Preprocessing**: Load CSV data, normalize, and split into training/validation sets.  
- **Real-time Prediction**: Users can draw digits on a canvas and see predictions live.  
- **Visualization**: Probabilities for each digit displayed with bar charts.  
- **Streamlit Interface**: Modern, interactive UI for quick testing and demonstration.  

---

## Project Structure 📂
```
digit-recognizer/
├── data/
│   └── raw/               
│       ├── mnist_train.csv
│       └── mnist_test.csv
├── models/
│   └── mnist_model_best.pth
|
├── notebooks/
|       └── eda.ipynb
├── src/
│   ├── data_utils.py       
│   ├── model.py
|   ├── app.py            
│   └── train.py              
│   
├── environment.yml
├── .gitattributes
└── README.md
```
## Key Programming Concepts Demonstrated 🛠️

1. **Machine Learning Workflow**  
   - Data cleaning, normalization, train-validation split, and batching.  
   - Model training using **PyTorch** with **Adam optimizer** and **CrossEntropyLoss**.  

2. **Neural Network Design**  
   - Fully connected feedforward network with batch normalization and ReLU activations.  
   - Multi-class classification for digits 0-9.  

3. **Python Best Practices**  
   - Modular project structure with reusable functions.  

5. **Version Control & Environment Management**  
   - Git for source control.  
   - Conda for reproducible environment setup.  

## How to Run ▶️

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate digit-recognizer
```

### 2. Train the Model (Optional)

```bash
python src/train.py
```

### 3. Run GUI App

```bash
python src/app.py
```

## Future Improvements 🚀

- Add **real-time webcam digit recognition**.  
- Enable **saving user drawings and predictions** for further analysis.  
- Swutcg GUI framework for a better and modern GUI.
- Deploy as a **web app**.  

---

## Learning Outcomes 🎓

By completing this project, the following skills were demonstrated:

- Building and training a neural network in **PyTorch**.  
- Handling CSV data, preprocessing, and normalization.  
- Creating an interactive GUI using tkinter.  
- Understanding multi-class classification and softmax probabilities.  
- Structuring a Python project with best practices and version control. 


