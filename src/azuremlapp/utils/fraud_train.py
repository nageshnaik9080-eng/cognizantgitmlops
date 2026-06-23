import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Load environment variables from the .env file
    # Since .env is in src/azuremlapp, and this file is in src/azuremlapp/utils,
    # we point to the parent directory's .env file.
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    # 2. Retrieve the file path from the environment variable
    relative_file_path = os.getenv("file_path")
    if not relative_file_path:
        raise ValueError("Error: 'file_path' variable not found in .env file.")

    # Convert the relative path to an absolute path relative to the workspace root
    # (assuming execution from the root folder /workspaces/cognizantgitmlops)
    workspace_root = os.getcwd()
    csv_path = os.path.join(workspace_root, relative_file_path)
    
    print(f"Loading dataset from: {csv_path}\n")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}. Please verify execution directory.")
        return

    # 3. Data Inspection
    print("--- Dataset Head ---")
    print(df.head(), "\n")
    
    print("--- Label Distribution ---")
    print(df['label'].value_counts(), "\n")

    # 4. Feature Engineering & Preprocessing
    # Drop identifier columns not useful for training
    X = df.drop(columns=['transaction_id', 'timestamp', 'label'])
    y = df['label']

    # One-Hot Encode categorical variables (like 'card_type')
    X = pd.get_dummies(X, columns=['card_type'], drop_first=True)

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Model Training (Random Forest Classifier)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    print("Model training complete.\n")

    # 7. Model Evaluation
    y_pred = model.predict(X_test)
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()