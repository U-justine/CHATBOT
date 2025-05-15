# 📚 ICT Support Chatbot – Marwadi University

An AI-powered virtual assistant built to answer frequently asked questions by students and staff of the ICT Department at Marwadi University. 
This chatbot helps users retrieve important academic, placement, and department-related information instantly through a simple, conversational interface.


## 🚀 Demo Video

▶️ **YouTube Link:** [Click here to watch the demo](https://youtu.be/FZib2bWeB4U)

---

## 🧠 Features

- 💬 Natural language query handling
- 🔎 Accurate answer matching using TF-IDF and cosine similarity
- 🗃️ Predefined Q&A dataset derived from ICT Department materials
- 🧾 Text preprocessing using NLP techniques
- 📁 Optional file upload/download support (for document sharing)
- 🌐 Web-based interface (Flask) & desktop compatibility
- ⚙️ Modular backend with scalable design
- 🔐 JWT-based authentication support (configurable)

---

## 🛠️ Technologies Used

- **Python 3.13**
- **Flask** (for web framework)
- **Scikit-learn** (for TF-IDF & similarity)
- **NLTK** (for NLP preprocessing)
- **Transformers & PyTorch** (optional ML extensions)
- **Tkinter** (for GUI, if used locally)
- **Openpyxl / Pandas** (for dataset handling)
- **Flask-JWT-Extended**, **Flask-Limiter** (for auth and rate limiting)

---

## 📁 Dataset

The dataset was constructed manually based on department presentations, including:
- Faculty information
- Course offerings
- Government job opportunities
- Placement statistics
- Vision and mission of the department

Stored as `ICT_QA.xlsx` in structured format:
- **Question**
- **Answer**
- *(Optional)* Category label

---

## 🖥️ Installation & Run Locally

