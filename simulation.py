import time
import numpy as np
import math
import os
from typing import List, Tuple, Dict, Union
from torchvision.io import read_image
from config import *
from network_base import Network as NetworkBase
from network_transformers import Network as NetworkTransformers
from network_transformers_lstm import Network as NetworkTransformersLSTM
import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms
from utils import *
from enum import Enum
from tasks.Task import Task
from tasks.CubeTask import CubeTask
from tasks.InsertTask import InsertTask
from tasks.CubeDepthTask import CubeDepthTask
from tasks.TeddyBearTask import TeddyBearTask

# Define a transformation pipeline for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

class PandaSim:
    def __init__(self, bullet_client, use_network: bool, task_type: Enum):
        self.use_network = use_network
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)

        self.start_position = PLANE_START_POSITION
        self.robot_start = ROBOT_START_POSITION

        # Load URDF models for the plane and the panda robot
        self.bullet_client.loadURDF("plane.urdf", self.start_position, [-0.5, -0.5, -0.5, 0.5])
        self.panda = self.bullet_client.loadURDF("objects/panda.urdf", np.array([0, 0, 0]), PANDA_ORN, useFixedBase=True)
        self.orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., math.pi / 2., 0.])

        self.control_dt = TIME_STEP
        self.finger_target = GRIPPER_OPEN

        # Create constraints for the robot gripper
        c = self.bullet_client.createConstraint(self.panda, 9, self.panda, 10, jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        self.init_starting_poses(STARTING_JOINT_POSITIONS)
        self.initial_state_id = self.bullet_client.saveState()
        self.randomize_arm()

        # Load the task
        self.task: Task = self.load_task(task_type)
        self.task.randomize_environment()

        self.t = 0.
        self.t_target = 0.
        self.prev_pos = []
        self.current_target_1 = np.array([])
        self.current_target_2 = np.array([])
        self.randx = 0
        self.randz = 0
        self.rand_start = 0
        self.text_id_cube = -1
        self.text_id_ee = -1
        self.steps = 0
        self.episode = 0
        self.sample_episode = 0

        if self.use_network:
            # Load the network based on the configuration settings

            self.device = torch.device(DEVICE)
            if USE_TRANSFORMERS and USE_LSTM:
                self.model = NetworkTransformersLSTM(len(ACTIVE_CAMERAS), self.device).to(self.device)
                self.model.reset_hidden_states(1)
            elif USE_TRANSFORMERS:
                self.model = NetworkTransformers(len(ACTIVE_CAMERAS), self.device).to(self.device)
            else:
                self.model = NetworkBase(len(ACTIVE_CAMERAS), self.device).to(self.device)
            self.active_cameras = ACTIVE_CAMERAS
            create_folder('network')
            self.target = None
        else:
            self.active_cameras = ['cam1', "cam2", "cam3", "ego"]
            for camera in self.active_cameras:
                create_folder(camera)

        self.init_cameras()

    def load_model(self, epoch: int) -> None:
        """Load the model weights from a specific epoch."""
        self.model.load_state_dict(torch.load(MODEL_PATH + '_' + str(epoch) + '.pth'))
        self.model.eval()

    def load_task(self, task_type: Enum) -> Union[Task, str]:
        """Load the task based on the provided task type."""
        tasks = {
            TaskType.CUBE_TABLE: CubeTask(self.bullet_client, self.next_episode),
            TaskType.INSERT_CUBE: InsertTask(self.bullet_client, self.next_episode),
            TaskType.CUBE_DEPTH: CubeDepthTask(self.bullet_client, self.next_episode),
            TaskType.TEDDY_BEAR: TeddyBearTask(self.bullet_client, self.next_episode),
        }
        return tasks.get(task_type, f"Unknown task: {task_type.name}")

    def init_starting_poses(self, poses: List[float]) -> None:
        """Initialize the starting joint positions for the robot and moves the arm there."""
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            jointType = info[2]
            if jointType in [self.bullet_client.JOINT_PRISMATIC, self.bullet_client.JOINT_REVOLUTE]:
                self.bullet_client.resetJointState(self.panda, j, poses[index])
                index += 1

    def randomize_arm(self) -> None:
        """Randomize the starting position of the robot arm."""
        self.rand_start = np.array([np.random.uniform(-MAX_START_RANDOM_XZ, MAX_START_RANDOM_XZ),
                                    np.random.uniform(-MAX_START_RANDOM_Y, MAX_START_RANDOM_Y),
                                    np.random.uniform(-MAX_START_RANDOM_XZ, MAX_START_RANDOM_XZ)])
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, PANDA_END_EFFECTOR_INDEX,
                                                                   self.robot_start + self.rand_start, self.orn, LL, UL,
                                                                   JR, RP, maxNumIterations=20)
        for i in range(PANDA_DOFS):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     jointPoses[i], force=5 * 240.)

    def init_cameras(self) -> None:
        """Initialize the cameras used for capturing images."""
        self.images = {}
        self.cameras = {}
        self.pmat = self.bullet_client.computeProjectionMatrixFOV(fov=90, aspect=(128 / 128), nearVal=CAMERA_NEAR_PLANE,
                                                                   farVal=CAMERA_FAR_PLANE)
        for camera in self.active_cameras:
            if camera == 'ego':
                self.pmat_ego = self.bullet_client.computeProjectionMatrixFOV(fov=110, aspect=(128 / 128),
                                                                              nearVal=CAMERA_NEAR_PLANE,
                                                                              farVal=CAMERA_FAR_PLANE)
                self.update_ego_camera()
                self.images["ego"] = []
            else:
                cam1_settings = CAMERA_POSITIONS[camera]
                vmat = self.bullet_client.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=cam1_settings["target_position"],
                    distance=cam1_settings["distance"], yaw=cam1_settings["yaw"], pitch=cam1_settings["pitch"],
                    roll=cam1_settings["roll"], upAxisIndex=1)
                cam = (vmat, self.pmat)
                self.cameras[camera] = cam
                self.images[camera] = []

    def update_ego_camera(self) -> None:
        """Update the ego camera position and orientation."""
        if "ego" in self.active_cameras:
            state, _ = self.get_state()
            ee_pos = state[-3:]
            camera_pos = [ee_pos[0], ee_pos[1] + 0.15, ee_pos[2] - 0.15]
            ground_position = [camera_pos[0], 0, camera_pos[2]]
            vmat = self.bullet_client.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=ground_position,
                cameraUpVector=[0, 0, 1],
            )
            ego = (vmat, self.pmat_ego)
            self.cameras["ego"] = ego

    def get_state(self) -> Tuple[List[float], List[float]]:
        """Retrieve the current state of the robot and environment."""
        current_state = []
        current_positions = []

        for i in range(PANDA_DOFS):
            pos, vel, _, _ = self.bullet_client.getJointState(self.panda, i)
            current_state.append(vel)
            current_positions.append(pos)

        current_state += self.current_target_1.tolist()
        current_state += self.current_target_2.tolist()
        current_state += ([1, 0] if self.task.is_gripper_closed() else [0, 1])
        cube_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.task.bodies[0])
        ee_state = self.bullet_client.getLinkState(self.panda, PANDA_END_EFFECTOR_INDEX)
        current_state += [p for p in cube_pos]
        current_state += ee_state[0]

        return current_state, current_positions

    def show_cube_pos(self, cube: List[float], end_effector: List[float]) -> None:
        """Display the positions of the cube and end effector."""
        if SHOW_AUX_POS:
            if self.text_id_cube != -1 and self.text_id_ee != -1:
                self.bullet_client.removeUserDebugItem(self.text_id_cube)
                self.bullet_client.removeUserDebugItem(self.text_id_ee)
            self.text_id_cube = self.bullet_client.addUserDebugPoints([cube], [[0, 0, 1]], pointSize=100, lifeTime=0.1)
            self.text_id_ee = self.bullet_client.addUserDebugPoints([end_effector], [[0, 1, 0.5]], pointSize=25,
                                                                    lifeTime=0.1)

    def move_robot_network(self, output: List[float]) -> None:
        """Move the robot based on the network's output."""
        self.show_cube_pos(output[0:3], output[-3:])
        self.finger_target = GRIPPER_OPEN if output[-8] < output[-7] else GRIPPER_CLOSE

        if PREDICTION_TYPE in [PredictionType.POSITION, PredictionType.TARGET_POSITION]:
            target_pos = np.array(output[:3])
            pos = self.prev_pos
            diff = (self.prev_pos - target_pos) * ROBOT_SPEED
            self.prev_pos = self.prev_pos - diff

            jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, PANDA_END_EFFECTOR_INDEX, pos.tolist(),
                                                                       self.orn, LL, UL, JR, RP, maxNumIterations=20)

            for i in range(PANDA_DOFS):
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         jointPoses[i], force=5 * 240.)
        else:
            output = output[:7]
            for i in range(PANDA_DOFS):
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.VELOCITY_CONTROL,
                                                         targetVelocity=output[i], force=5 * 240.)

        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=100)

    def remove_oldest_images(self) -> None:
        """Remove the oldest images from the image buffer."""
        for camera in self.active_cameras:
            self.images[camera] = self.images[camera][1:]

    def get_episode(self) -> int:
        """Return the current episode number."""
        return self.episode

    def get_camera_image(self, camera: str) -> Tuple[Image.Image, np.ndarray]:
        """Capture an image from a specified camera."""
        (vmat, pmat) = self.cameras[camera]
        img = self.bullet_client.getCameraImage(
            IMAGE_SIZE, IMAGE_SIZE, viewMatrix=vmat, projectionMatrix=pmat,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL
        )
        rgbBuffer = img[2]
        depthBuffer = np.reshape(img[3], (IMAGE_SIZE, IMAGE_SIZE))
        rgbim = Image.fromarray(rgbBuffer)
        far = CAMERA_FAR_PLANE
        near = CAMERA_NEAR_PLANE
        depth_image = far * near / (far - (far - near) * np.array(depthBuffer))
        return rgbim, depth_image

    def process_and_store_image(self, key: str) -> None:
        """Process and store images captured from the cameras."""
        rgbim, depthim = self.get_camera_image(key)

        if IMAGE_TYPE == ImageType.D:
            depth_data = torch.tensor(depthim)
            depth_data_normalized = normalize_depth_data(depth_data)
            image_to_store = depth_data_normalized.to(self.device, dtype=torch.float)
            depth_pil_image = transforms.ToPILImage()(depth_data_normalized.cpu())
            depth_pil_image.save(IMAGES_PATH + 'network/depth_' + str(key) + '.png')

        elif IMAGE_TYPE == ImageType.RGB:
            rgb_data = transform(rgbim).to(self.device, dtype=torch.float)
            rgb_data_normalized = normalize_image(rgb_data)  # Normalize the RGB image
            image_to_store = rgb_data_normalized
            rgbim.save(IMAGES_PATH + 'network/image_' + str(key) + '.png')

        elif IMAGE_TYPE == ImageType.RGBD:
            depth_data = torch.tensor(depthim)
            depth_data_normalized = normalize_depth_data(depth_data).to(self.device, dtype=torch.float)
            rgb_data = transform(rgbim).to(self.device, dtype(torch.float))
            rgb_data_normalized = normalize_image(rgb_data)  # Normalize the RGB image
            image_to_store = torch.cat((rgb_data_normalized, depth_data_normalized), dim=0)

            rgbim.save(IMAGES_PATH + 'network/image_' + str(key) + '.png')
            depth_pil_image = transforms.ToPILImage()(depth_data_normalized.cpu())
            depth_pil_image.save(IMAGES_PATH + 'network/depth_' + str(key) + '.png')

        # Apply resizing to the image and store it
        resized_image = resize_tensor(image_to_store, (IMAGE_SIZE_TRAIN, IMAGE_SIZE_TRAIN))
        self.images[key].append(resized_image)

    def capture_images(self) -> int:
        """Capture images from the active cameras."""
        self.update_ego_camera()
        if self.use_network:
            # Store the images into memory
            if self.steps % SIMULATION_STEPS == 0:
                if len(self.images[list(self.images.keys())[0]]) < SEQUENCE_LENGTH:
                    for key in self.active_cameras:
                        self.process_and_store_image(key)
            return 1 if len(self.images[list(self.images.keys())[0]]) == SEQUENCE_LENGTH else -1
        else:
            # Store the images on disk, to create the dataset
            if self.steps % SIMULATION_STEPS == 0:
                for key in self.active_cameras:
                    rgbim, depthim = self.get_camera_image(key)
                    image_path = get_image_path(key, self.episode, self.sample_episode)
                    rgbim.save(image_path)
                    torch.save(depthim, image_path.replace('.png', '_depth'))
                self.sample_episode += 1
            return self.sample_episode - 1

    def get_network_output(self, positions: List[float]) -> List[float]:
        """Get the output from the network based on the current positions and images."""
        inputs = []
        for camera in self.active_cameras:
            inputs.append(torch.cat(self.images[camera], dim=0).unsqueeze(0).to(self.device, dtype=torch.float))
        positions_tensor = torch.tensor(positions).to(self.device, dtype=torch.float)
        if USE_TRANSFORMERS and USE_LSTM:
            output = (self.model(inputs, positions_tensor.unsqueeze(0), torch.tensor([True])).squeeze()).tolist()
        else:
            output = (self.model(inputs, positions_tensor.unsqueeze(0)).squeeze()).tolist()
        return output

    def step_network(self) -> None:
        """Step the simulation using the network for control."""
        if self.capture_images() == -1:
            return
        _, positions = self.get_state()
        self.target = self.get_network_output(positions)
        self.move_robot_network(self.target)
        self.remove_oldest_images()

    def next_episode(self) -> None:
        """Start the next episode of the simulation."""
        self.episode += 1
        self.steps = 0
        self.sample_episode = 0
        if self.use_network:
            if USE_TRANSFORMERS and USE_LSTM:
                self.model.reset_hidden_states(1)
        for body in self.task.bodies:
            self.bullet_client.removeBody(body)
        self.bullet_client.restoreState(self.initial_state_id)
        self.randomize_arm()
        self.task.randomize_environment()

    def step_simulation(self) -> None:
        """Step the simulation without using the network for control."""
        state, pos = self.get_state()
        current_pos = np.array(state[-3:])
        if self.task.is_idle_state() or self.task.is_drop_state() or self.task.is_grab_state():
            if self.task.is_grab_state():
                self.finger_target = GRIPPER_CLOSE
            else:
                self.finger_target = GRIPPER_OPEN
            self.t += self.control_dt
            if self.task.is_drop_state():
                wait_time = WAIT_TIME_DROP
            elif self.task.is_grab_state():
                wait_time = WAIT_TIME_GRASP
            else:
                wait_time = WAIT_TIME_OTHER
            if self.t > wait_time:
                self.task.next_state()
                self.t = 0
            self.prev_pos = current_pos
            self.current_target_1 = current_pos
            self.current_target_2 = current_pos
        else:
            if check_distance(current_pos, self.task.target_pos, REACHED_TARGET_THRESHOLD):
                self.task.next_state()
            else:
                pos = self.prev_pos
                self.current_target_1 = pos
                self.current_target_2 = self.task.target_pos
                diff = (self.prev_pos - self.task.target_pos) * ROBOT_SPEED
                self.prev_pos = self.prev_pos - diff
                jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, PANDA_END_EFFECTOR_INDEX,
                                                                           pos.tolist(), self.orn, LL, UL, JR, RP,
                                                                           maxNumIterations=20)
                for i in range(PANDA_DOFS):
                    self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                             jointPoses[i], force=5 * 240.)
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=100)

    def step(self) -> None:
        """Step the simulation based on the network usage setting."""
        if self.use_network:
            if self.steps > START_STEPS_NETWORK:
                self.step_network()
            else:
                state, _ = self.get_state()
                self.prev_pos = np.array(state[-3:])
        else:
            self.step_simulation()
        self.steps += 1
