import time
import numpy as np
import math
import os
from torchvision.io import read_image
from config import *
from network import Network as NetworkBase
from network_transformers import Network as NetorkTransformers
import torch
from PIL import Image
from torchvision import transforms
from utils import get_image_path
from enum import Enum
from CubeTask import CubeTask
from InsertTask import InsertTask
from CubeDepthTask import CubeDepthTask

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x*255
])   

class PandaSim(object):

    def __init__(self, bullet_client, use_network, task_type):
        self.use_network = use_network
        self.bullet_client = bullet_client
        
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)

        self.start_position = PLANE_START_POSITION
        self.robot_start = ROBOT_START_POSITION


        self.bullet_client.loadURDF("plane.urdf", self.start_position, [-0.5, -0.5, -0.5, 0.5])

        orn = [-0.707107, 0.0, 0.0, 0.707107]

        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]), orn,
                                                 useFixedBase=True)
        self.orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., math.pi / 2., 0.])

        self.control_dt = TIME_STEP
        self.finger_target = 0

        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        self.init_starting_poses(STARTING_JOINT_POSITIONS)
        self.initial_state_id =  self.bullet_client.saveState()
        self.randomize_arm()

        self.task = self.load_task(task_type)
        self.task.randomize_environment()
        # self.task.randomize_environment()
        
        self.t = 0.
        self.t_target = 0.

        self.prev_pos = []

        self.randx = 0
        self.randz = 0
        self.rand_start = 0

        self.text_id_cube = -1
        self.text_id_ee = -1

        self.steps = 0
        
        if self.use_network:
            self.device = torch.device(DEVICE)
            if USE_TRANSFORMERS:
                self.model  = NetorkTransformers(len(ACTIVE_CAMERAS), self.device).to(self.device)
            else:
                self.model = NetworkBase(len(ACTIVE_CAMERAS), self.device).to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH + '_' + str(EPOCH) + '.pth'))
            self.model.eval()
            self.active_cameras = ACTIVE_CAMERAS
            self.create_folder('network')

        else:
            self.episode = 0
            self.sample_episode = 0
            self.active_cameras = ['cam1', "cam2", "ego"]
            for camera in self.active_cameras:
                self.create_folder(camera)
        
        self.init_cameras()

    def load_task(self, task_type):
        tasks = {
            TaskType.CUBE_TABLE: CubeTask(self.bullet_client, self.next_episode),
            TaskType.INSERT_CUBE: InsertTask(self.bullet_client, self.next_episode),
            TaskType.CUBE_DEPTH: CubeDepthTask(self.bullet_client, self.next_episode),
        }
        # Get the value corresponding to the key `option` from the dictionary.
        # If the key is not found, return the default value.
        return tasks.get(task_type, f"Unknown task: {task_type.name}")

    def create_folder(self, camera):
        folder_path = IMAGES_PATH + camera
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")

    def init_starting_poses(self, poses):
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            jointType = info[2]

            if jointType == self.bullet_client.JOINT_PRISMATIC or jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, poses[index])
                index = index + 1

    def randomize_arm(self):
        self.rand_start = np.array([np.random.uniform(-MAX_START_RANDOM_XZ, MAX_START_RANDOM_XZ),
                                    np.random.uniform(-MAX_START_RANDOM_Y, MAX_START_RANDOM_Y),
                                    np.random.uniform(-MAX_START_RANDOM_XZ, MAX_START_RANDOM_XZ)])
            
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, PANDA_END_EFFECTOR_INDEX, self.robot_start + self.rand_start,
                                                                self.orn, LL, UL, JR, RP, maxNumIterations=20)

        for i in range(PANDA_DOFS):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                    jointPoses[i], force=5 * 240.)

    def init_cameras(self):
        self.images = {}
        self.cameras = {}
        
        self.pmat = self.bullet_client.computeProjectionMatrixFOV(fov=90, aspect=(128 / 128), nearVal=CAMERA_NEAR_PLANE, farVal=CAMERA_FAR_PLANE) 

        for camera in self.active_cameras:
            if camera == 'ego':
                self.pmat_ego = self.bullet_client.computeProjectionMatrixFOV(fov=110, aspect=(128 / 128), nearVal=CAMERA_NEAR_PLANE, farVal=CAMERA_FAR_PLANE) 
                self.update_ego_camera()
                self.images["ego"] = []
            else:
                cam1_settings = CAMERA_POSITIONS[camera]
                vmat = self.bullet_client.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=cam1_settings["target_position"],
                    distance=cam1_settings["distance"], 
                    yaw=cam1_settings["yaw"], 
                    pitch=cam1_settings["pitch"], 
                    roll=cam1_settings["roll"],
                    upAxisIndex=1
                )
                cam = (vmat, self.pmat)
                self.cameras[camera] = cam
                self.images[camera] = []
 
    def update_ego_camera(self):
        if "ego" in self.active_cameras:
            state, _ = self.get_state()
            ee_pos = state[-3:]
            camera_pos = [ee_pos[0], ee_pos[1] + 0.15, ee_pos[2] - 0.15]
            ground_position = [camera_pos[0], 0, camera_pos[2]]
            vmat = self.bullet_client.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=ground_position, 
                cameraUpVector=[0,0,1], 
            )
            ego = (vmat, self.pmat_ego)
            self.cameras["ego"] = ego

    def get_state(self):
        current_state = []
        current_positions = []

        for i in range(PANDA_DOFS):
            pos, vel, _, _ = self.bullet_client.getJointState(self.panda, i)
            # current_state.append(vel)
            current_positions.append(pos)
        
        
        current_state += ([1, 0] if self.task.is_gripper_closed() else [0, 1])
        cube_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.task.bodies[0])

        ee_state = self.bullet_client.getLinkState(self.panda, PANDA_END_EFFECTOR_INDEX)

        current_state += [p for p in cube_pos]

        current_state += ee_state[0]

        return current_state, current_positions

    def show_cube_pos(self, cube, end_effector):
        if SHOW_AUX_POS:
            if self.text_id_cube != -1 and self.text_id_ee != -1:
                self.bullet_client.removeUserDebugItem(self.text_id_cube)
                self.bullet_client.removeUserDebugItem(self.text_id_ee)

            self.text_id_cube = self.bullet_client.addUserDebugPoints([cube], [[0, 0, 1]], pointSize=100, lifeTime=0.1)
            self.text_id_ee = self.bullet_client.addUserDebugPoints([end_effector], [[0, 1, 0.5]], pointSize=25, lifeTime=0.1)

    def move_robot_network(self, output, state):
        self.show_cube_pos(output[-6:-3], output[-3:])

        self.finger_target = GRIPPER_OPEN if output[7] < output[8] else GRIPPER_CLOSE

        for i in range(PANDA_DOFS):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.VELOCITY_CONTROL,
                                                         targetVelocity=output[i], force=5 * 240.)

        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         self.finger_target, force=100)

    def remove_oldest_images(self):
        for camera in self.active_cameras:
            self.images[camera] = self.images[camera][1:]
        
    def get_episode(self):
        return self.episode

    def get_camera_image(self, camera):
        (vmat, pmat) = self.cameras[camera]

        img = self.bullet_client.getCameraImage(
            IMAGE_SIZE, IMAGE_SIZE,
            viewMatrix=vmat, projectionMatrix=pmat,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL
        )

        rgbBuffer = img[2]
        depthBuffer = np.reshape(img[3], (IMAGE_SIZE, IMAGE_SIZE))
        rgbim = Image.fromarray(rgbBuffer)
        
        far = CAMERA_FAR_PLANE
        near = CAMERA_NEAR_PLANE

        depth_image = far * near / (far - (far - near) * np.array(depthBuffer))
        
        return rgbim, depth_image

    def capture_images(self):
        self.update_ego_camera()

        if self.use_network:
            if self.steps % STEPS_SKIPED == 0:
                if len(self.images[list(self.images.keys())[0]]) < SEQUENCE_LENGTH:
                    for key in self.active_cameras:
                        rgbim, depthim = self.get_camera_image(key)

                        if USE_DEPTH:
                            depth_data = torch.tensor(depthim).view(1, IMAGE_SIZE, IMAGE_SIZE)
                            min_val = depth_data.min()
                            max_val = depth_data.max()
                            
                            depth_data_normalized = (depth_data - min_val) / (max_val - min_val)

                            self.images[key].append(depth_data_normalized.to(self.device, dtype=torch.float))
                        else:
                            self.images[key].append(transform(rgbim).to(self.device, dtype=torch.float))
                            rgbim.save(IMAGES_PATH + 'network/image_' + str(key) + '.png')
                        # depthim.save(IMAGES_PATH + 'network/depth_' + str(key) + '.png')

            return 1 if len(self.images[list(self.images.keys())[0]]) == SEQUENCE_LENGTH else -1
        else: 
            if self.steps % STEPS_SKIPED == 0:
                for key in self.active_cameras:
                    rgbim, depthim = self.get_camera_image(key)
                    image_path = get_image_path(key, self.episode, self.sample_episode)
                    rgbim.save(image_path)
                    torch.save(depthim, image_path.replace('.png', '_depth'))

                self.sample_episode += 1

            return self.sample_episode - 1

    
    def get_network_output(self, positions):
        inputs = []
        for camera in self.active_cameras:
            inputs.append(torch.cat(self.images[camera], dim=0).unsqueeze(0).to(self.device, dtype=torch.float))
        positions = torch.tensor(positions).to(self.device, dtype=torch.float)

        output = (self.model(inputs, positions.unsqueeze(0)).squeeze()).tolist()

        return output
    
    def get_network_output_lstm(self, positions):
        inputs = []
        for camera in self.active_cameras:
            inputs.append(self.images[camera][-1].unsqueeze(0).to(self.device, dtype=torch.float))

        positions = torch.tensor(positions).to(self.device, dtype=torch.float)

        output = (self.model(inputs, positions.unsqueeze(0)).squeeze()).tolist()

        return output

    def step_network(self):
        if self.capture_images() == -1:
            return

        state, positions = self.get_state()
        output = self.get_network_output(positions)
        self.move_robot_network(output, state)

        self.remove_oldest_images()
        
    def check_distance(self, point1, point2):
        distance = np.linalg.norm(point1 - point2)
        return distance < REACHED_TARGET_THRESHOLD

    def next_episode(self):
        self.episode += 1
        self.steps = 0
        self.sample_episode = 0
        for body in self.task.bodies:
            self.bullet_client.removeBody(body)
        self.bullet_client.restoreState(self.initial_state_id)

        self.randomize_arm()
        self.task.randomize_environment()
    
    def update_state(self):
        state, pos = self.get_state()
        
        curent_pos = np.array(state[-3:])

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
                wait_time =  WAIT_TIME_OTHER

            if self.t > wait_time:
                self.task.next_state()
                self.t = 0

            self.prev_pos = curent_pos
        else:
            if self.check_distance(curent_pos, self.task.target_pos):
                self.task.next_state()
            else:
                pos = self.prev_pos

                diff = (self.prev_pos - self.task.target_pos) * ROBOT_SPEED
                self.prev_pos = self.prev_pos - diff

                jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, PANDA_END_EFFECTOR_INDEX, pos.tolist(),
                                                                            self.orn, LL, UL, JR, RP, maxNumIterations=20)

                for i in range(PANDA_DOFS):
                    self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                                jointPoses[i], force=5 * 240.)
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                        self.finger_target, force=100)


    def step(self):
        if self.use_network:
            if self.steps > 300:
                self.step_network()
        else:
            self.update_state()
        self.steps += 1



