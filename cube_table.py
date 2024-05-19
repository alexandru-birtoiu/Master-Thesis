import time
import numpy as np
import math
import os

from torchvision.io import read_image

from config import *
from network import Network
import torch
from PIL import Image
from torchvision import transforms
from utils import get_image_path
from enum import Enum

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x*255
])

class TaskStages(Enum):
    IDLE = 1
    GO_ABOVE_CUBE = 2
    GO_ON_CUBE = 3
    GRASP_CUBE = 4
    GO_TABLE = 5
    DROP_CUBE = 6
    GO_ON_CUBE_2 = 7
    GRASP_CUBE_2 = 8
    GO_ABOVE_CUBE_2 = 9
    GO_TO_RANDOM_CUBE = 10
    DROP_CUBE_2 = 11
    GO_TO_START = 12
     

class PandaSim(object):

    def __init__(self, bullet_client, use_network):
        self.randomize()
        self.use_network = use_network
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)

        self.start_position = np.array([0, 0.03, -0.5])

        self.robot_start = np.array([0, 0.3, -0.35])

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.bullet_client.loadURDF("plane.urdf", self.start_position, [-0.5, -0.5, -0.5, 0.5], flags=flags)

        legoOrientation = self.bullet_client.getQuaternionFromEuler([-1,0,0])
        self.lego = self.bullet_client.loadURDF("lego/lego.urdf", 
            np.array([
                self.start_position[0] + self.randx, 
                self.start_position[1], 
                self.start_position[2] + self.randz]), 
            legoOrientation, 
            globalScaling=1.5, 
            flags=flags
        )

        tableOrientation = self.bullet_client.getQuaternionFromEuler([-math.pi / 2, math.pi / 2, 0])
        self.table = self.bullet_client.loadURDF("table/table.urdf", np.array([-0.5, 0.0, -0.2]), tableOrientation,
                                        globalScaling=0.25, flags=flags)
                                        
        self.bullet_client.changeVisualShape(self.lego, -1, rgbaColor=[1, 0, 0, 1])

        orn = [-0.707107, 0.0, 0.0, 0.707107]

        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]), orn,
                                                 useFixedBase=True, flags=flags)
        self.orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., math.pi / 2., 0.])

        self.state = TaskStages.IDLE

        self.control_dt = TIME_STEP
        self.finger_target = 0

        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            jointType = info[2]
            if jointType == self.bullet_client.JOINT_PRISMATIC or jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, STARTING_JOINT_POSITIONS[index])
                index = index + 1

        self.t = 0.
        self.t_target = 0.

        self.prev_pos = []
        self.target_pos = []

        self.randx = 0
        self.randz = 0
        self.rand_start = 0

        self.text_id_cube = -1
        self.text_id_ee = -1
        
        if use_network:
            self.device = torch.device(DEVICE)
            self.model = Network(len(ACTIVE_CAMERAS), self.device).to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH + '_' + str(EPOCH) + '.pth'))
            self.model.eval()
            self.steps = 0
            self.active_cameras = ACTIVE_CAMERAS
            self.create_folder('network')
        else:
            self.episode = 0
            self.sample_episode = 0
            self.active_cameras = ['cam1', "cam2", "ego"]
            for camera in self.active_cameras:
                self.create_folder(camera)
    
        self.init_cameras()

    def create_folder(self, camera):
        folder_path = IMAGES_PATH + camera
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")

    def init_cameras(self):
        self.images = {}
        self.cameras = {}
        
        self.pmat = self.bullet_client.computeProjectionMatrixFOV(fov=90, aspect=(128 / 128), nearVal=0.01, farVal=5) 

        for camera in self.active_cameras:
            if camera == 'ego':
                self.pmat_ego = self.bullet_client.computeProjectionMatrixFOV(fov=120, aspect=(128 / 128), nearVal=0.01, farVal=5) 
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
            current_state.append(vel)
            current_positions.append(pos)
        
        current_state += ([1, 0] if self.state.value >= 4 and self.state.value < 6 else [0, 1])
        cube_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)

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
        for key in self.images.keys():
            self.images[key] = self.images[key][1:]
        
    def get_episode(self):
        return self.episode

    def get_camera_image(self, camera):
        (vmat, pmat) = self.cameras[camera]

        img = self.bullet_client.getCameraImage(IMAGE_SIZE, IMAGE_SIZE, viewMatrix=vmat, projectionMatrix=pmat,renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)
        rgbBuffer = img[2]
        rgbim = Image.fromarray(rgbBuffer)

        return rgbim

    def capture_images(self):
        self.update_ego_camera()
                
        if self.use_network:
            if len(self.images[list(self.images.keys())[0]]) < 4:
                for key in self.active_cameras:
                    rgbim = self.get_camera_image(key)

                    self.images[key].append(transform(rgbim).to(self.device, dtype=torch.float))
                    rgbim.save(IMAGES_PATH + 'network/image_' + str(key) + '.png')

            return len(self.images[list(self.images.keys())[0]]) == 4
        
        else: 
            for key in self.active_cameras:
                rgbim = self.get_camera_image(key)

                rgbim.save(get_image_path(key, self.episode, self.sample_episode))

            self.sample_episode += 1

            return -1 if self.sample_episode < 3 else self.sample_episode
    
    def get_network_output(self, positions):
        inputs = []
        for camera in self.active_cameras:
            inputs.append(torch.cat(self.images[camera], dim=0).unsqueeze(0).to(self.device, dtype=torch.float))
        
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


    def randomize(self):
        self.randx = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.randz = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)

        self.rand_start = np.array([np.random.uniform(-MAX_START_RANDOM_XZ, MAX_START_RANDOM_XZ),
                                    np.random.uniform(-MAX_START_RANDOM_Y, MAX_START_RANDOM_Y),
                                    np.random.uniform(-MAX_START_RANDOM_XZ, MAX_START_RANDOM_XZ)])
        

    def check_distance(self, point1, point2):
        distance = np.linalg.norm(point1 - point2)
        return distance < REACHED_TARGET_THRESHOLD


    def next_state(self):
        current_stage_index = self.state.value - 1 
        next_stage_index = (current_stage_index + 1) % len(TaskStages)
        self.state = TaskStages(next_stage_index + 1)

        if self.state == TaskStages.GO_ABOVE_CUBE or self.state == TaskStages.GO_ABOVE_CUBE_2:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)
            diff = 0.1 if self.state == TaskStages.GO_ABOVE_CUBE else 0.2
            self.target_pos = np.array([pos[0], pos[1] + diff, pos[2]])
        elif self.state == TaskStages.GO_ON_CUBE or self.state == TaskStages.GO_ON_CUBE_2:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)
            pos = [pos[0], pos[1]- 0.01, pos[2]]
            self.target_pos = np.array(pos)
        elif self.state == TaskStages.GO_TABLE:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.table)
            self.randomize()
            self.target_pos = np.array([pos[0], pos[1] + 0.3, pos[2]])
        elif self.state == TaskStages.GO_TO_RANDOM_CUBE:
            self.target_pos = np.array([
                self.start_position[0] + self.randx, 
                self.start_position[1] + 0.1, 
                self.start_position[2] + self.randz])
    
        elif self.state == TaskStages.GO_TO_START:
            self.target_pos = self.robot_start + self.rand_start
            self.next_episode()

    def next_episode(self):
        self.episode += 1
        self.sample_episode = 0
    
    def is_drop_state(self):
        return self.state == TaskStages.DROP_CUBE or self.state == TaskStages.DROP_CUBE_2

    def is_grab_state(self):
        return self.state == TaskStages.GRASP_CUBE or self.state == TaskStages.GRASP_CUBE_2
    
    def update_state(self):
        state, _ = self.get_state()
        self.show_cube_pos(state[-6:-3], state[-3:])
        curent_pos = np.array(state[-3:])

        if self.state == TaskStages.IDLE or self.is_drop_state() or self.is_grab_state():

            if self.is_grab_state():
                self.finger_target = GRIPPER_CLOSE
            else:
                self.finger_target = GRIPPER_OPEN

            self.t += self.control_dt
            if self.state == TaskStages.DROP_CUBE:
                wait_time = WAIT_TIME_DROP_LAST
            else:
                wait_time = WAIT_TIME_DROP if self.is_drop_state() else WAIT_TIME_OTHER
            if self.t > wait_time:
                self.next_state()
                self.t = 0

            self.prev_pos = curent_pos
        else:
            if self.check_distance(curent_pos, self.target_pos):
                self.next_state()
            else:
                pos = self.prev_pos

                diff = (self.prev_pos - self.target_pos) * ROBOT_SPEED
    
                self.prev_pos = self.prev_pos - diff

                jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, PANDA_END_EFFECTOR_INDEX, pos.tolist(),
                                                                       self.orn,
                                                                       LL, UL, JR, RP, maxNumIterations=20)

                for i in range(PANDA_DOFS):
                    self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         jointPoses[i], force= 5 * 240.)
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force= 100)

    def step(self):
        if self.use_network:
            if self.steps > 500:
                self.step_network()
            else:
                self.steps += 1
        else:
            self.update_state()


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, use_network):
        PandaSim.__init__(self, bullet_client, use_network)

    def is_done(self):
        return self.done

    #TODO move to task
    def is_moving(self):
        return self.state.value > 1 and self.state.value < 7

