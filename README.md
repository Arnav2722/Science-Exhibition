# Medical_Diagnosis | A Machine Learning Based Web Application

![Pyhon 3.4](https://img.shields.io/badge/ide-Jupyter_notebook-blue.svg) ![Python](https://img.shields.io/badge/Language-Python-brightgreen.svg) ![Frontend](https://img.shields.io/badge/Frontend-Bootstrap-purple.svg)

## Table of Content

- [Problem statment / Why this topic?](#Problem-statment)
- [Flow Chart / Archeticture](#Flow-chart)
- [Directory Tree](#directory-tree)
- [Installation](#installation)
- [Quick start](#Quick-start)
- [Technical Aspect](#technical-aspect)
- [Team](#team)

## Problem Statment

The envisioned project holds significant value in the medical domain. It proposes the development of a machine learning-driven web application dedicated to medical diagnostics. The project entails the creation of a machine learning model, seamlessly integrated into the web application, enabling users to upload their medical data for analysis. The application will employ the developed machine learning model to detect health diseases. Subsequently, users can schedule appointments with doctors through the web application if they seek professional advice. Additionally, a chat (email) feature will facilitate communication between patients and doctors within the platform.

## Why this Project?

While humans are prone to errors, machines are not, and their predicted outcomes can be assessed for accuracy through machine learning. Bearing this in mind, extensive research was conducted in the realms of allopathic, homeopathic, and ayurvedic data. Due to the limited availability of research papers for patient datasets in homeopathy and ayurvedic medicine, the decision was made to utilize allopathic datasets accessible on platforms such as Kaggle and the UCI machine learning portals.

## Flow chart

Front-end UX/UI, Back-end Machine learning, Deep learning flow chart

![ml](https://user-images.githubusercontent.com/62024355/120781058-4fac3300-c546-11eb-83be-dfc8319fd2f3.png)

## Directory Tree

```
├── data
├── Pyhon notebooks code files
├── trained models.pkl file
├── static logos
├── Templates
│   ├── Home.html
│   ├── contact.html
│   ├── about us.html
│   ├── services.html
│   ├── css folder
│   ├── js folder
│   ├── images folder
│   └── fonts folder
│         ├── Diabetes
│         ├── Breast Cancer
│         ├── Heart Disease
│         ├── Kidney Disease
│         ├── Liver Disease
│         ├── Malaria
│         └── Pneumonia
├── app.py
├── readme.md
└── requirements.txt


```

## Installation

This project was made using [python 3.6.8](https://www.python.org/downloads/release/python-368/). Download and install the same version before running this program.

## Quick start

**Step-1:** Download the files in the repository.<br>
**Step-2:** Get into the downloaded folder, open command prompt in that directory and install all the dependencies using following command<br>

```python
pip install -r requirements.txt
```

**Step-3:** After successfull installation of all the dependencies, run the following command<br>

```python
python app.py
```

```python
or
flask run
```

**Step-4:** Go to the **New command prompt** of root folder, run the following commands in new cmd terminal<br>

```
cd templates
index.html
```

## Technical aspect

This webapp was developed using Flask Web Framework. The models used to predict the diseases were trained on large Datasets. All the links for datasets and the python notebooks used for model creation are mentioned below in this readme. The webapp can predict following Diseases:

- Diabetes
- Breast Cancer
- Heart Disease
- Kidney Disease
- Liver Disease
- Malaria
- Pneumonia

**Models with their Accuracy of Prediction**

| Disease        | Type of Model            | Accuracy |
| -------------- | ------------------------ | -------- |
| Diabetes       | Machine Learning Model   | 74.03%   |
| Breast Cancer  | Machine Learning Model   | 96.49%   |
| Heart Disease  | Machine Learning Model   | 100%     |
| Kidney Disease | Machine Learning Model   | 96.88%   |
| Liver Disease  | Machine Learning Model   | 77.97%   |
| Malaria        | Deep Learning Model(CNN) | 94.78%   |
| Pneumonia      | Deep Learning Model(CNN) | 95%      |

**NOTE**
<br>
==> Python version 3.6.8 was used for the whole project.<br>

**Dataset Links**
All the datasets were used from kaggle.

- [Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- [Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- [Heart Disease Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)
- [Kidney Disease Dataset](https://www.kaggle.com/mansoordaku/ckdisease)
- [Liver Disease Dataset](https://www.kaggle.com/uciml/indian-liver-patient-records)

## Team

[Arnav Sharma](https://github.com/Arnav2722) <br>
[Mehul Verma](https://www.mehulverma.netlify.app)
