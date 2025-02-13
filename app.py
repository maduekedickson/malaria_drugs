import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_filename = "malaria_drug_resistance_model.pkl"
with open(model_filename, "rb") as model_file:
    model = pickle.load(model_file)

# Load the encoders (you must save them during training)
encoder_snp1 = LabelEncoder()
encoder_snp2 = LabelEncoder()
encoder_drug = LabelEncoder()

# Use the same categories as during training
snp1_categories = np.array(["A", "B", "C", "D", "E"])  # Replace with actual values
snp2_categories = np.array(["X", "Y", "Z", "W"])  # Replace with actual values
drug_type_categories = np.array(["Type1", "Type2", "Type3"])  # Replace with actual values

encoder_snp1.fit(snp1_categories)
encoder_snp2.fit(snp2_categories)
encoder_drug.fit(drug_type_categories)

# Load image
st.image("image.jpg")

# Streamlit App Title
st.title("Malaria Drug Resistance Prediction")

# Description
st.markdown("""
### **Description**
This application predicts whether a malaria strain is **drug-resistant** based on genetic and protein expression data.

### **How It Works**
- Select **SNP (Single Nucleotide Polymorphisms)** and **Drug Type** from the dropdown menus.
- Enter numerical values for **gene expression** and **protein levels**.
- Click **'Predict'** to determine whether the malaria strain is resistant to the selected drug.

### **Why This Matters**
- Helps in understanding **drug resistance patterns**.
- Supports **malaria research and drug development**.
- Provides insights for **effective treatment strategies**.

""")

# User Input (Dropdowns for categorical features)
st.sidebar.header("Enter Feature Values")

SNP_1 = st.sidebar.selectbox("SNP_1", snp1_categories)
SNP_2 = st.sidebar.selectbox("SNP_2", snp2_categories)
Drug_Type = st.sidebar.selectbox("Drug Type", drug_type_categories)

# Convert selected values to encoded form
SNP_1_encoded = encoder_snp1.transform([SNP_1])[0]
SNP_2_encoded = encoder_snp2.transform([SNP_2])[0]
Drug_Type_encoded = encoder_drug.transform([Drug_Type])[0]

# Numerical inputs for other features
Gene_Exp_1 = st.sidebar.number_input("Gene Expression 1", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
Gene_Exp_2 = st.sidebar.number_input("Gene Expression 2", min_value=0.0, max_value=10.0, value=4.2, step=0.1)
Protein_1 = st.sidebar.number_input("Protein 1", min_value=0.0, max_value=10.0, value=1.1, step=0.1)
Protein_2 = st.sidebar.number_input("Protein 2", min_value=0.0, max_value=10.0, value=0.9, step=0.1)

# Button to make predictions
if st.sidebar.button("Predict"):
    # Prepare input as numpy array
    input_data = np.array([[SNP_1_encoded, SNP_2_encoded, Drug_Type_encoded, Gene_Exp_1, Gene_Exp_2, Protein_1, Protein_2]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Show result
    if prediction == 1:
        st.success("Prediction: Drug Resistant")
    else:
        st.warning("Prediction: Not Drug Resistant")
