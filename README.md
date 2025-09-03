# EEG + NHIS Explorer

## 📖 Overview
The **EEG + NHIS Explorer** is an interactive [Streamlit](https://streamlit.io/) web application that allows users to explore relationships between **laboratory EEG** (Electroencephalography) findings and **population-level sleep patterns** from the **National Health Interview Survey (NHIS)**.

The app is designed for **educational and descriptive** purposes. It does not diagnose sleep disorders or make causal claims—it helps users understand how objective brainwave measures and self-reported sleep patterns can complement each other.

---

## 🧠 What Are EEG and NHIS?

### EEG (Electroencephalography)
EEG records the brain’s electrical activity using sensors placed on the scalp.  
In this app, EEG measures come from an *open-source laboratory study* comparing **Normal Sleep (NS)** and **Sleep Deprivation (SD)** conditions in 71 participants. Measures include:
- Brainwave power in **theta**, **alpha**, and **beta** bands.
- Mood scores from the **Positive and Negative Affect Schedule (PANAS)**.
- Attention performance from the **Psychomotor Vigilance Test (PVT)**.
- Sleep questionnaires such as the **Pittsburgh Sleep Quality Index (PSQI)**.

### NHIS (National Health Interview Survey)
The NHIS is a U.S. nationwide household survey conducted by the CDC’s National Center for Health Statistics. It provides annual, de-identified microdata about health and lifestyle factors for a representative sample of the population.  
In this app, NHIS data includes:
- **Sleep hours**, restfulness, trouble sleeping, and sleep-aid use.
- Demographics such as **age**, **sex**, **education**, and **race/ethnicity**.

---

## ✨ Features
- **🧠 EEG Viewer** – Select a participant, condition, and task to view EEG signals and channel maps.
- **📈 EEG Dashboard** – Explore mood, attention, and brainwave power patterns with plain-language explanations.
- **⚡ Reaction Time Demo** – Try a short reaction-time test inspired by the PVT.
- **🗺️ NHIS Dashboard** – View national sleep patterns by demographics.
- **🔗 Lab ↔ Survey Comparison** – Compare EEG measures to population sleep reports side-by-side.

---

## 📂 Data Access Statement
**EEG Dataset**  
- Source: OpenNeuro Dataset [ds004902](https://openneuro.org/datasets/ds004902/versions/1.0.8)  
- License: [CC0 Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)  
- Description: Resting-state EEG for sleep deprivation study (71 participants; eyes open/closed; mood and vigilance measures).  

**NHIS Dataset**  
- Source: [National Health Interview Survey 2024 Public-Use Microdata](https://www.cdc.gov/nchs/nhis/index.html)  
- License: Public domain, U.S. Government work (17 U.S.C. §105)  
- Description: U.S. household survey including sleep measures and demographics.

**Reproducibility**  
- Download CSV from the links. Processing steps are noted in each module; code paths and assumptions are documented in app text. The app is descriptive, not diagnostic.

---
## 🌐 Live App

- Here’s our live, interactive Streamlit app:  
-- [**EEG + NHIS Explorer on Streamlit**](https://eeg-nhis-app.streamlit.app/)
---

## ⚙️ Requirements
To install dependencies, run:
```bash
pip install -r requirements.txt





