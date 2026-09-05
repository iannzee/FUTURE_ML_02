# FUTURE_ML_02: Support Ticket Classification & Prioritization

This repository contains a Machine Learning decision-support system designed to automatically classify unstructured customer support tickets and assign urgency levels[cite: 1]. It was developed as part of the Future Interns Machine Learning track (Task 2)[cite: 1].

## 📌 Project Overview
Customer support teams handle hundreds of emails and complaints daily, often losing time to manual sorting[cite: 1]. This Natural Language Processing (NLP) pipeline reads raw ticket text, automatically categorizes the specific issue (e.g., Billing, Technical Issue, Account, General Query), and predicts a priority level (High, Medium, Low)[cite: 1]. This ensures critical issues are escalated immediately and reduces the overall ticket backlog[cite: 1].

## ✨ Key Features
*   **Text Preprocessing:** Cleans raw text by lowercasing, handling punctuation, and removing noise/stopwords using NLTK[cite: 1].
*   **Feature Extraction:** Converts cleaned text into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization[cite: 1].
*   **Multi-Class Classification:** Utilizes machine learning algorithms (like Random Forest and Logistic Regression) to predict both the ticket category and its required priority[cite: 1].
*   **Model Evaluation:** Outputs rigorous performance metrics including accuracy, precision, and recall, alongside visual Confusion Matrices for deep class-wise analysis[cite: 1].

## 🛠️ Technology Stack
*   **Core Language:** Python[cite: 1]
*   **NLP & ML Libraries:** NLTK, Scikit-learn[cite: 1]
*   **Data Handling & Visualization:** Pandas, NumPy, Matplotlib, Seaborn[cite: 1]

## 📁 Repository Structure
```text
FUTURE_ML_02/
├── data/                       # Contains raw or synthetic support ticket datasets
├── src/                        # Core source code modules
│   ├── __init__.py             # Package marker
│   ├── preprocessing.py        # Text cleaning and priority assignment logic
│   ├── train.py                # TF-IDF vectorization and model training
│   └── visualize.py            # Confusion matrix generation
├── outputs/                    # Saved ML models, vectorizers, and visual plots
├── .gitignore                  # Git tracking exclusion rules
├── main.py                     # Master execution pipeline
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```[cite: 1]

## 🚀 Getting Started

**1. Create and activate a virtual environment**
```bash
python -m venv venv
```[cite: 1]
*   *Windows:* `venv\Scripts\activate`[cite: 1]
*   *Mac/Linux:* `source venv/bin/activate`[cite: 1]

**2. Install dependencies**
Ensure NLTK and other required packages are installed[cite: 1]:
```bash
pip install -r requirements.txt
```[cite: 1]

**3. Run the pipeline**
Executing the main script will load the data, train the models, print the classification reports, and generate confusion matrices[cite: 1]:
```bash
python main.py
```[cite: 1]

## 💼 Operational Business Impact
*   **Elimination of Manual Triage:** Automatically routing tickets to specialized teams reduces response delays by up to 80%[cite: 1].
*   **SLA Breach Prevention:** Priority detection guarantees that critical system failures or churn risks jump ahead of routine queries[cite: 1].
*   **Operational Analytics:** Class-wise performance statistics allow management to spot emerging software bugs or recurring pipeline defects early[cite: 1].
