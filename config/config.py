class Config:
    class_names = ['pos', 'neu', "neg"]
    PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
    BATCH_SIZE = 16
    MAX_LEN = 160
    EPOCHS = 10
    learning_rate = 2e-5
