from enum import Enum

class ModelType(Enum):
    EGOCENTRIC = 1
    BIRDSEYE = 2
    BIRDSEYE_DOUBLE = 3
    EGO_AND_BIRDSEYE = 4

EPOCH = 10
MODEL_TYPE = ModelType.EGOCENTRIC

SHOW_AUX_POS = 0

DEVICE = 'mps'

TRAIN_MODEL_MORE = 0
STARTING_EPOCH = 0
EPOCHS_TO_TRAIN = 10

LEARNING_RATE = 0.0002
BATCH_SIZE = 128
USE_LSTM = 1
LSTM_LAYERS = 2

GATHER_DATA = True
GATHER_DATA_MORE = 0
STARTING_EPISODES = 0

NO_EPISODES = 100
IMAGE_SIZE = 128

FPS = 240.
TIME_STEP = 1. / FPS

WAIT_TIME_DROP_LAST = 0.25
WAIT_TIME_DROP = 0.2
WAIT_TIME_OTHER = 0.1
REACHED_TARGET_THRESHOLD = 0.01
ROBOT_SPEED = 0.1

CAMERA_DISTANCE = 1.1
CAMERA_YAW = 0
CAMERA_PITCH = -30
CAMERA_TARGET_POSITION = [0, 0, 0]

MAX_CUBE_RANDOM = 0.15
MAX_START_RANDOM_XZ = 0.03
MAX_START_RANDOM_Y = 0.05

GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0

PANDA_END_EFFECTOR_INDEX = 11
PANDA_DOFS = 7

LL = [-7] * PANDA_DOFS
UL = [7] * PANDA_DOFS
JR = [7] * PANDA_DOFS

STARTING_JOINT_POSITIONS = [1.678, -0.533, -0.047, -2.741, -0.029, 2.207, 0.869, 0.02, 0.02]
RP = STARTING_JOINT_POSITIONS

MODEL_PATH = 'models/model_' + str(MODEL_TYPE.name)+ '-' + str(IMAGE_SIZE) + 'px_' + str(NO_EPISODES) + '_episodes' + ('_lstm_' + str(LSTM_LAYERS) + 'layers' if USE_LSTM else '')
EPOCH_LOSSES_PATH = MODEL_PATH + '_epoch_losses'
VALIDATION_LOSSES_PATH = MODEL_PATH + '_validation_losses'
DETAILS_PATH = MODEL_PATH + '_details'

IMAGES_PATH = 'images/'
SAMPLE_LABEL_PATH = 'labels/sample_labels_' + str(IMAGE_SIZE) + 'px_' + str(NO_EPISODES) + '_episodes'
GATHER_MORE_LABEL_PATH = 'labels/sample_labels_' + str(IMAGE_SIZE) + 'px_' + str(STARTING_EPISODES) + '_episodes'
SAVING_LABEL_PATH = 'labels/sample_labels_' + str(IMAGE_SIZE) + 'px_' + str(STARTING_EPISODES + NO_EPISODES if GATHER_DATA_MORE else NO_EPISODES) + '_episodes'

CAMERA_POSITIONS = {
    "cam1": {
        "distance": 1.1,
        "yaw": 0,
        "pitch": -30,
        "roll": 0,
        "target_position": [0, 0, 0],
    },
    "cam2": {
        "distance": 1.1,
        "yaw": 90,
        "pitch": -40,
        "roll": 0,
        "target_position": [0, 0, -0.4],
    }
}

CAMERA_SETUPS = {
    1: ["ego"],
    2: ["cam1"],
    3: ["cam1", "cam2"],
    4: ["cam1", "ego"]
}

ACTIVE_CAMERAS = CAMERA_SETUPS[MODEL_TYPE.value]