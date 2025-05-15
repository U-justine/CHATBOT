from ChatbotModelTrainer import ChatbotTrainer

# Initialize trainer
trainer = ChatbotTrainer()

# Prepare data
train_dataset, val_dataset = trainer.prepare_data(r'jetbrains://pycharm/navigate/reference?project=CHATBOT&path=ICT_QA.xlsx')

# Initialize model
trainer.initialize_model()

# Train model
trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=16,
    num_epochs=5,
    learning_rate=2e-5,
    save_path='chatbot_model'
)

# Make predictions
response = trainer.predict("How do I reset my password?")
print(response)