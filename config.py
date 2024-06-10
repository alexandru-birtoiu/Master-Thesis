from enum import Enum
import numpy as np

# Enums for model and task types
class ModelType(Enum):
    EGOCENTRIC = 1
    BIRDSEYE = 2
    BIRDSEYE_MULTIPLE = 3
    EGO_AND_BIRDSEYE = 4

class TaskType(Enum):
    CUBE_TABLE = 1
    INSERT_CUBE = 2
    CUBE_DEPTH = 3
    # INSERT_CUBE_2 = 4
    # Add other tasks as needed

class PredictionType(Enum):
    VELOCITY = 1
    POSITION = 2
    TARGET_POSITION = 3
    # Add other tasks as needed

class ImageType(Enum):
    RGB = 1
    D = 2
    RGBD = 3

# General configurations
EPOCH: int = 12
MODEL_TYPE: ModelType = ModelType.EGO_AND_BIRDSEYE
USE_DEPTH: bool = False
TASK_TYPE: TaskType = TaskType.CUBE_DEPTH
PREDICTION_TYPE: PredictionType = PredictionType.TARGET_POSITION
IMAGE_TYPE: ImageType = ImageType.RGB

SHOW_AUX_POS: int = 0
DEVICE: str = 'mps'

# Training configurations
TRAIN_MODEL_MORE: int = 0
STARTING_EPOCH: int = 0
EPOCHS_TO_TRAIN: int = 20

USE_TASK_LOSS = 0

USE_TRANSFORMERS: bool = True

USE_LSTM: bool = False
LSTM_LAYERS: int = 1

SEQUENCE_LENGTH:int = 4

STEPS_SKIPED:int = 2

MAX_SEQUENCE:int = 150

LEARNING_RATE: float = 0.0002
CLIP_GRADIENTS_VALUE = 1.0
SCHEDULER_STEP_SIZE: int = 3
BATCH_SIZE: int = 64

NETWORK_OUTPUT_SIZE: int = 15 if PREDICTION_TYPE == PredictionType.VELOCITY else 11 + (3 if USE_TASK_LOSS else 0)
NETWORK_IMAGE_LAYER_SIZE: int = 1 if IMAGE_TYPE == ImageType.D else 4 if IMAGE_TYPE == ImageType.RGB else 5

# Data gathering configurations

NO_EPISODES: int = 2000
GATHER_DATA: bool = True
GATHER_DATA_MORE: int = True
STARTING_EPISODES: int = 500

# Image configurations
IMAGE_SIZE: int = 128

# Simulation parameters
FPS: float = 240.0
TIME_STEP: float = 1.0 / FPS

WAIT_TIME_DROP: float = 0.25
WAIT_TIME_GRASP: float = 0.04#0.025
WAIT_TIME_OTHER: float = 0.1
REACHED_TARGET_THRESHOLD: float = 0.01
ROBOT_SPEED: float = 0.1

# Camera configurations
CAMERA_DISTANCE: float = 1.1
CAMERA_YAW: int = 0
CAMERA_PITCH: int = -30
CAMERA_TARGET_POSITION: list[float] = [0, 0, 0]

# Randomization parameters
MAX_CUBE_RANDOM: float = 0.07
MAX_START_RANDOM_XZ: float = 0.01
MAX_START_RANDOM_Y: float = 0.01

# Gripper configurations
GRIPPER_OPEN: float = 0.04
GRIPPER_CLOSE: float = 0.0

# Robot parameters
PANDA_END_EFFECTOR_INDEX: int = 11
PANDA_DOFS: int = 7

LL: list[int] = [-7] * PANDA_DOFS
UL: list[int] = [7] * PANDA_DOFS
JR: list[int] = [7] * PANDA_DOFS

PLANE_START_POSITION: np.ndarray = np.array([0, 0.03, -0.5])
ROBOT_START_POSITION: np.ndarray = np.array([0, 0.3, -0.35])

STARTING_JOINT_POSITIONS: list[float] = [1.678, -0.533, -0.047, -2.741, -0.029, 2.207, 0.869, 0.02, 0.02]
RP: list[float] = STARTING_JOINT_POSITIONS

# Paths
def get_model_path() -> str:
    return f'models/{TASK_TYPE.name.lower()}/'    \
        + f'model_{MODEL_TYPE.name}_{PREDICTION_TYPE.name}_{IMAGE_SIZE}px_{IMAGE_TYPE.name}_{NO_EPISODES}_episodes_{STEPS_SKIPED}step'   \
        + ((f'_lstm_{LSTM_LAYERS}layers' if USE_LSTM else '') + f'_{SEQUENCE_LENGTH}seq')   \
        + ('_transformers' if USE_TRANSFORMERS else '') \
        + ('_taskloss' if USE_TASK_LOSS else '')

MODEL_PATH: str = get_model_path()
DETAILS_PATH: str = f'{MODEL_PATH}_details'

def get_label_path(episodes: int) -> str:
    return f'labels/{TASK_TYPE.name.lower()}/sample_labels_{IMAGE_SIZE}px_' + str(episodes) + '_episodes'

IMAGES_PATH: str = f'images/{TASK_TYPE.name.lower()}/'
SAMPLE_LABEL_PATH: str = get_label_path(NO_EPISODES)
GATHER_MORE_LABEL_PATH: str = get_label_path(STARTING_EPISODES)

# Camera positions and setups
CAMERA_POSITIONS: dict[str, dict[str, float]] = {
    "cam1": {
        "distance": 1,
        "yaw": 0,
        "pitch": -30,
        "roll": 0,
        "target_position": [0, 0, 0],
    },
    "cam2": {
        "distance": 0.9,
        "yaw": 90,
        "pitch": -40,
        "roll": 0,
        "target_position": [0.15, 0, -0.4],
    },
    "cam3": {
        "distance": 0.9,
        "yaw": 270,
        "pitch": -40,
        "roll": 0,
        "target_position": [-0.15, 0, -0.4],
    }
}
CAMERA_FAR_PLANE = 5
CAMERA_NEAR_PLANE = 0.01

CAMERA_SETUPS: dict[int, list[str]] = {
    1: ["ego"],
    2: ["cam1"],
    3: ["cam1", "cam2", "cam3"],
    4: ["cam1", "ego"]
}

ACTIVE_CAMERAS: list[str] = CAMERA_SETUPS[MODEL_TYPE.value]