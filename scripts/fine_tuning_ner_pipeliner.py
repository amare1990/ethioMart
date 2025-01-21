"""Pipeline stages for fine_tuning models."""

from fine_tuning_NER import NERModel

# A method to fine tine the ner model
def fine_tune_ner_model():
    # Initialize the EthioMartNERModel with the dataset path
    ner_model = NERModel(
      model_name='xlm-roberta-base',
      dataset_path = '../data/labeled_dataset.conll'

    )

    # ner_model = NERModel()
    print("NER Model initialized.")

    # Load and prepare the data
    ner_model.load_data()
    print("Data loaded and prepared.")

    # Tokenize the data
    ner_model.tokenize_data()
    print("Data tokenized.")

    # Load the pre-trained model
    ner_model.load_model()
    print("Model loaded.")

    # Fine-tune the model
    ner_model.train_model()
    print("Model fine-tuned.")

    # Evaluate the model
    result = ner_model.evaluate_model()
    print("Model evaluated.")
    print(result)

    # Save the fine-tuned model
    ner_model.save_model()
    print("Model saved.")
