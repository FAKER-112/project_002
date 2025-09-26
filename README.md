# Customer Satisfaction Prediction - Custom Version__incomplete

This repository contains a custom implementation of an end-to-end machine learning pipeline for predicting customer satisfaction scores using the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The pipeline is built with [ZenML](https://zenml.io/) and includes data ingestion, preprocessing, model training, and evaluation steps.

## 📦 Project Structure

```
ml_eng_ops-project/
│
├── README.md
├── requirement.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── utills.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── data/
│   │   └── olist_customers_dataset.csv
│   └── pipeline/
│       └── training_pipeline.py
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd my-own-version
```

### 2. Install Requirements

```bash
pip install -r requirement.txt
```

### 3. Run the Training Pipeline

```bash
python src/pipeline/training_pipeline.py
```

## 🏗 Pipeline Overview

The pipeline consists of the following main steps:

- **Data Ingestion**: Loads the raw dataset from `src/data/olist_customers_dataset.csv`.
- **Data Transformation**: Cleans and preprocesses the data, handling missing values and feature selection.
- **Model Training**: Trains a machine learning model to predict the customer satisfaction score.
- **Evaluation**: Evaluates the model performance using appropriate metrics.

## 📝 Customization

- All core logic is implemented in the `src/` directory.
- You can modify or extend the pipeline steps in `src/components/` and `src/pipeline/`.

## 📚 References

- [ZenML Documentation](https://docs.zenml.io/)
- [Olist E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## 📝 License

This project is for educational purposes.

---

*For any questions or suggestions, feel free to open an issue or pull request!*