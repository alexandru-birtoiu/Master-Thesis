import time
import numpy as np
import math

from torchvision.io import read_image

from config import TIME_STEP, GRIPPER_OPEN, GRIPPER_CLOSE, STARTING_JOINT_POSITIONS, MODEL_PATH, PANDA_DOFS, \
    PANDA_END_EFFECTOR_INDEX, LL, UL, JR, RP, MAX_CUBE_RANDOM, MAX_START_RANDOM_XZ, MAX_START_RANDOM_Y, EPOCH, \
    IMAGE_SIZE, SHOW_AUX_POS, DEVICE
from network import Network
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x*255
])


class PandaSim(object):
    def __init__(self, bullet_client, use_network):
        self.use_network = use_network
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.start_position = np.array([0, 0.03, -0.5])

        self.robot_start = np.array([0, 0.03, -0.3])

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.legos = []

        self.bullet_client.loadURDF("plane.urdf", self.start_position, [-0.5, -0.5, -0.5, 0.5], flags=flags)

        legoOrientation = self.bullet_client.getQuaternionFromEuler([-1,0,0])
        self.legos.append(
            self.bullet_client.loadURDF("lego/lego.urdf", np.array([0, 0.05, -0.4]), legoOrientation, 
                                        globalScaling=1.5, flags=flags))

        tableOrientation = self.bullet_client.getQuaternionFromEuler([-math.pi / 2, math.pi / 2, 0])
        self.table = self.bullet_client.loadURDF("table/table.urdf", np.array([-0.5, 0.0, -0.2]), tableOrientation,
                                        globalScaling=0.25, flags=flags)
                                        
        for i, cubeId in enumerate(self.legos):
            self.bullet_client.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

        orn = [-0.707107, 0.0, 0.0, 0.707107]

        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]), orn,
                                                 useFixedBase=True, flags=flags)
        self.orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., math.pi / 2., 0.])

        self.state = 0
        self.control_dt = TIME_STEP
        self.finger_target = 0
        self.gripper_height = 2
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

        self.randx = 0
        self.randz = 0
        self.rand_start = 0

        self.nextPos = []

        self.text_id_cube = -1
        self.text_id_ee = -1

        if use_network:
            self.device = torch.device(DEVICE)
            self.model = Network().to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH + '_' + str(EPOCH) + '.pth'))
            self.model.eval()
            self.images = []
            self.steps = 0
            self.running_steps = 0
            

    def get_state(self):
        state = []
        positions = []
        for i in range(PANDA_DOFS):
            pos, vel, _, _ = self.bullet_client.getJointState(self.panda, i)
            state.append(vel)
            positions.append(pos)

        state += ([1, 0] if self.state == 6 else [0, 1])
        cube_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.legos[self.target])

        ee_state = self.bullet_client.getLinkState(self.panda, PANDA_END_EFFECTOR_INDEX)

        state += [p for p in cube_pos]

        state += ee_state[0]
        return state, positions

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
                                                         self.finger_target, force=10)

    def step_network(self):
        if len(self.images) < 4:
            img = self.bullet_client.getCameraImage(IMAGE_SIZE, IMAGE_SIZE, renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)
            rgb_buffer = img[2]
            rgbim = Image.fromarray(rgb_buffer)
            self.images.append(transform(rgbim).to(self.device, dtype=torch.float))
            rgbim.save('images/img_' + str(len(self.images) - 1) + '.png')

        if len(self.images) == 4:
            # TODO: Investigate why I have to read the images from disk

            input_tensor = torch.cat(self.images, dim=0).unsqueeze(0).to(self.device, dtype=torch.float)

            state, positions = self.get_state()
            positions = torch.tensor(positions).to(self.device, dtype=torch.float)

            output = (self.model(input_tensor, positions.unsqueeze(0)).squeeze()).tolist()

            self.move_robot_network(output, state)

            self.images = self.images[1:]
            self.running_steps = 0

    def step(self):
        if self.use_network:
            self.steps += 1
            self.running_steps += 1
            if self.steps > 500:
                # if self.running_steps > 5:
                self.step_network()
            return

        if self.state == 6:
            self.finger_target = GRIPPER_CLOSE
        if self.state == 5:
            self.finger_target = GRIPPER_OPEN

        self.update_state()
        alpha = 0.9

        state, _ = self.get_state()
        self.show_cube_pos(state[-6:-3], state[-3:])

        # if self.state == 10:
        #     _, p = self.get_state()
        #     print(p)



        if self.state == 3 or self.state == 4 or self.state == 7 or self.state == 8:

            self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.03
            if self.state == 3 or self.state == 7 or self.state == 8:
                self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.2

            self.t += self.control_dt


            if self.state == 3 or self.state == 4:
                pos, _ = self.bullet_client.getBasePositionAndOrientation(self.legos[self.target])
                pos = [pos[0], self.gripper_height, pos[2]]
                self.prev_pos = pos

                self.randx = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
                self.randz = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)

                self.rand_start = np.array([np.random.uniform(-MAX_START_RANDOM_XZ, MAX_START_RANDOM_XZ),
                                           np.random.uniform(self.prev_pos[1], MAX_START_RANDOM_Y),
                                           np.random.uniform(-MAX_START_RANDOM_XZ, MAX_START_RANDOM_XZ)])

            # GO TO RANDOM CUBE POSITION FOR NEXT MOVE

            if self.state == 7:
                pos = self.prev_pos
                diff = pos - (self.robot_start + self.rand_start)

                self.prev_pos = [self.prev_pos[0] - diff[0] * 0.1,
                                 self.rand_start[1],
                                 self.prev_pos[2] - diff[2] * 0.1]
        
            # GO BACK TO RANDOM STARTING POSITION
            if self.state == 8:
                pos = self.prev_pos
                diffX = pos[0] - (self.start_position[0] + self.randx)
                diffZ = pos[2] - (self.start_position[2] + self.randz)
                self.prev_pos = [self.prev_pos[0] - diffX * 0.1, self.prev_pos[1], self.prev_pos[2] - diffZ * 0.1]


            jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, PANDA_END_EFFECTOR_INDEX, pos,
                                                                       self.orn,
                                                                       LL, UL, JR, RP, maxNumIterations=20)

            for i in range(PANDA_DOFS):
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         jointPoses[i], force= 5 * 240.)
            # target for fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=10)


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, use_network):
        PandaSim.__init__(self, bullet_client, use_network)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 3, 5, 4, 6, 3, 8, 5, 7]
        self.state_durations = [0.2, 0.05, 0.1, 0.1, 0.1, 0.2, 0.5, 0.1, 0.5]
        self.target = 0

    def is_done(self):
        return self.done

    def is_moving(self):
        return self.state != 0 and self.cur_state < 6

    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
                self.target = (self.target + 1) % len(self.legos)
            self.state_t = 0
            self.state = self.states[self.cur_state]
