import numpy as np
from math import sin, cos, sqrt, pi
import time

def dist(x,y):
    return sqrt((((x[0]*x[0])-(y[0]*y[0]))*((x[0]*x[0])-(y[0]*y[0])))+(((x[1]*x[1])-(y[1]*y[1]))*((x[1]*x[1])-(y[1]*y[1])))+(((x[2]*x[2])-(y[2]*y[2]))*((x[2]*x[2])-(y[2]*y[2]))))

class KinematicsSolver:
    #DH matrix in format [[theetas], [alphas], [r/a], [d]]
    def __init__(self, dh_matrix):
        self.dh_matrix = np.array(dh_matrix)

    ###########################################################################################################
    #FK portion of class
    ###########################################################################################################
    #Generates DH matrix, each matrix is the transfer function of the end of the joint relative to the pivot
    def createFKMatrices(self, dh_matrix):
        joint_matrix = []

        #Generates the matrix of DH matrices
        for i in range(self.dh_matrix.shape[1]):
            joint_matrix.append(np.zeros((4,4)))
            
            joint_matrix[-1][0][0] = cos(dh_matrix[0][i])
            joint_matrix[-1][0][1] = -sin(dh_matrix[0][i])*cos(dh_matrix[1][i])
            joint_matrix[-1][0][2] = sin(dh_matrix[0][i])*sin(dh_matrix[1][i])
            joint_matrix[-1][0][3] = dh_matrix[2][i]*cos(dh_matrix[0][i])
                        
            joint_matrix[-1][1][0] = sin(dh_matrix[0][i])
            joint_matrix[-1][1][1] = cos(dh_matrix[0][i])*cos(dh_matrix[1][i])
            joint_matrix[-1][1][2] = -cos(dh_matrix[0][i])*sin(dh_matrix[1][i])
            joint_matrix[-1][1][3] = dh_matrix[2][i]*sin(dh_matrix[0][i])
                        
            joint_matrix[-1][2][0] = 0
            joint_matrix[-1][2][1] = -sin(dh_matrix[1][i])
            joint_matrix[-1][2][2] = cos(dh_matrix[1][i])
            joint_matrix[-1][2][3] = dh_matrix[3][i]
                        
            joint_matrix[-1][3][0] = 0
            joint_matrix[-1][3][1] = 0
            joint_matrix[-1][3][2] = 0
            joint_matrix[-1][3][3] = 1

        return joint_matrix
        
    #Transforms from matrix A to matrix B
    def transformation(self, A, B):
        return np.matmul(A,B)

    #Gives transformation from robot base to selected joint
    def findJointPose(self, joint_matrix, joint):
        if joint == 1:
            return joint_matrix[0]
        else:
            return self.transformation(self.findJointPose(joint_matrix, joint-1),joint_matrix[joint-1])

    def displayJointPose(self, joint):
        joint_matrix = self.createFKMatrices(self.dh_matrix)
        print("The pose of joint", joint,"is:\n", self.findJointPose(joint_matrix, joint),'\n')

    def displayEndEffectorPose(self):
        self.displayJointPose(self.dh_matrix.shape[1])

    ###########################################################################################################
    #IK portion of class
    ###########################################################################################################

    #Creates the IK matrices for the DH-matrix as np arrays. Also converts other which have been inputted as lists to np arrays
    def createIKMatrices(self, axis_matrix, angles_matrix, goal_position):
        axis_matrix = np.array(axis_matrix)

        angles_matrix = np.zeros((len(angles_matrix),1))
        for i in range(len(angles_matrix)):
            if i == 0:
                angles_matrix[i][0] = angles_matrix[i] 
            else:
                angles_matrix[i][0] = angles_matrix[i]+angles_matrix[i-1] 
        angles_matrix = angles_matrix.astype(float)

        goal_position = np.array([goal_position])

        return (axis_matrix, angles_matrix, goal_position)

    #Starts IK function. Takes the IK algorithm as input. Available algorithms are:
    #Jacobian Transpose (JT), Jacobian Psuedo-Inverse (JP)
    def calculateIK(self, axis_matrix, goal_position, error = 10e-5, algorithm = 'JT'):
        t = time.time()
        n=0
        (axis_matrix, angles_matrix, goal_position) = self.createIKMatrices(axis_matrix, self.dh_matrix[1], goal_position)
        print("Goal is: ", goal_position[0],"\n")

        joints = self.createJointMatrices(angles_matrix, self.dh_matrix[2])   #Creates the joint matrix
        self.displayRobotInfo(axis_matrix, angles_matrix, goal_position, joints) #Displays the current robot poistion info

        while (dist(goal_position[0], joints[-1][0]) > error): #Loop for iterations; closes once displacement becomes lower than the required max error
            n += 1
            (angles_matrix, axis_matrix, joints) = self.runIter(angles_matrix, axis_matrix, joints, algorithm, goal_position)  #Updates values from the iteration result
            
        self.displayRobotInfo(axis_matrix, angles_matrix, goal_position, joints)

        for i in range(angles_matrix.shape[0]):
            if i == 0:
                prev_abs_angle = angles_matrix[i][0]
            else:
                temp = angles_matrix[i][0]
                angles_matrix[i][0] = angles_matrix[i][0]-prev_abs_angle
                prev_abs_angle = temp

        self.dh_matrix[0] = np.transpose(angles_matrix)[0]

        print("Number of iterations: ", n)
        print('Time taken: ', time.time() - t,'seconds\n')

    #Runs a single iteration
    def runIter(self, angles_matrix, axis_matrix, joints, algorithm, goal_position):
        J = self.createJacobianMatrix(axis_matrix, angles_matrix, joints) #Calculates the jacobian matrix

        d_angle = self.runIKAlgorithm(goal_position, joints, J, algorithm) #Calculates the delta angle corresponding to the current angle positions
        step_size = self.calculateStepSize(d_angle) #Calculates the step size which at the maximum allows 5 degree rotation
        
        angles_matrix += (step_size*d_angle)   #Updates the angle values

        joints = self.createJointMatrices(angles_matrix, self.dh_matrix[2])   #Updates the joint matrix according to the new angle values

        return (angles_matrix, axis_matrix, joints)

    #Creates the joint matrix that contains position value of each joint
    def createJointMatrices(self, angles_matrix, link_lengths):
        joints = np.zeros((angles_matrix.shape[0]+1,1,3))

        for i in range(angles_matrix.shape[0]):
            joints[i+1][0][0] = link_lengths[i]*cos(angles_matrix[i][0]) + joints[i][0][0]
            joints[i+1][0][1] = link_lengths[i]*sin(angles_matrix[i][0]) + joints[i][0][1]
            joints[i+1][0][2] = 0 + joints[i][0][2]

        return joints

    #Createse the jacobian matrix for the axi of rotation, angle orientation and position values of the joints
    def createJacobianMatrix(self, axis_matrix, angles_matrix, joints):
        J = np.zeros((3,angles_matrix.shape[0]))

        for i in range(angles_matrix.shape[0]):
            v = np.cross(axis_matrix[i],joints[-1]-joints[i])

            J[0][i] = v[0][0]
            J[1][i] = v[0][1]
            J[2][i] = v[0][2]

        return J

    #Calculates the step size, limiting it to a maximum value of 5 degrees
    def calculateStepSize(self, d_angle):
        step_size = 0.1
        d_angle_max = 0
        
        for i in range(d_angle.shape[0]):
            if abs(d_angle[i][0]) > d_angle_max:
                d_angle_max = abs(d_angle[i][0])
        if d_angle_max > (5*pi/180):
            step_size = (5*pi/180)/d_angle_max

        return step_size

    #Runs the actual IK algorithm (JI, JT, JP etc)
    def runIKAlgorithm(self, goal_position, joints, J, algorithm):
        if (algorithm == 'JP'):
            d_angle = np.matmul( (np.matmul(np.transpose(J),np.linalg.inv(np.matmul(J,np.transpose(J))))) , np.transpose(goal_position-joints[-1]) )
        elif (algorithm == 'JT'):
            d_angle = np.matmul(np.transpose(J),np.transpose(goal_position-joints[-1]))

        return d_angle

    #Displays the joint matrix, distance between current and goal and the angle values of the joints
    def displayRobotInfo(self, axis_matrix, angles_matrix, goal_position, joints):
        print("\nJoint's matrix is:\n",joints,"\n")
        print('Distance between goal and curent: ', dist(goal_position[0], joints[-1][0]))
        print('Angles are:\n', angles_matrix*180/pi)
        print('')