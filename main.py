import pybullet as p
import time
import pybullet_data

# Sym init
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)

state = -1
planeId = p.loadURDF("plane.urdf")
# urdfRootPath=pybullet_data.getDataPath()

pandaId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, globalScaling=2)
print(f"Panda num joins: {p.getNumJoints(pandaId)}")
for i in range(p.getNumJoints(pandaId)):
    print(f"Joint {i} info: {p.getJointInfo(pandaId, i)}")

# Cubes
holeWidth = 0.25
translation = 1.5
cubeIds = [p.loadURDF("cube.urdf", [holeWidth + translation, 0, 0.1], globalScaling=0.2), 
           p.loadURDF("cube.urdf", [-holeWidth + translation, 0, 0.1], globalScaling=0.2), 
           p.loadURDF("cube.urdf", [0 + translation, holeWidth, 0.1], globalScaling=0.2), 
           p.loadURDF("cube.urdf", [0 + translation, -holeWidth, 0.1], globalScaling=0.2)]
colors = [(0, 0, 1, 1),
          (0, 0, 1, 1),
          (0, 0, 1, 1),
          (0, 0, 1, 1)]
for i, cubeId in enumerate(cubeIds):
    p.changeVisualShape(cubeId, -1, rgbaColor=colors[i])

c = p.createConstraint(pandaId,
                       9,
                       pandaId,
                       10,
                       jointType=p.JOINT_GEAR,
                       jointAxis=[1, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

# Simulation loop
for i in range (10000):
    joint_positions = p.calculateInverseKinematics(pandaId, 8, [1, -11, 3])

    keys = p.getKeyboardEvents()
    if len(keys)>0:
      for k,v in keys.items():
        if v & p.KEY_WAS_TRIGGERED:
          if (k==ord('1')):
            state = 1
          if (k==ord('2')):
            state = 2
        if (k==ord('3')):
            state = 3
        if v & p.KEY_WAS_RELEASED:
            state = 0
    
    finger_target = 0.01 if state == 3 else 0.04
    if(state == 1):
        for j in range(8):
            p.setJointMotorControl2(pandaId, j, p.POSITION_CONTROL, targetPosition=joint_positions[j])
    elif state == 2 or state == 3:
        for i in [10, 11]:
            p.setJointMotorControl2(pandaId, i, p.POSITION_CONTROL, finger_target, force= 10)
    
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
