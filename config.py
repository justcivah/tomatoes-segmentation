DS_PATH = '/home/aislab/Desktop/tomatoes-segmentation/datasets/enhanced-tomato-greenhouse-dataset/'
#DS_PATH = '/home/aislab/Desktop/tomatoes-segmentation/datasets/ood-test/'
EXPERIMENT_NAME = 'tomatoes-segmentation'
TARGET_CATEGORY = 0
IMG_HEIGHT = 360
IMG_WIDTH = 640
NUM_WORKERS = 8
SEED = 42

TOTAL_EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
CURRICULUM = True
AUGMENT=True
DROPOUT = 0.15
PATIENCE = 50
EARLY_STOP_START_EPOCH = 350