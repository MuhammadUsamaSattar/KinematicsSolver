import numpy as np
from math import sin, cos, pi, sqrt
import KinematicsSolver

np.set_printoptions(suppress= True)

axis = [[0,0,1],[1,0,0],[0,0,1]]
angles = [pi/3,2*pi,pi/2]
m = [[0,0,0],[pi*0,pi*0,pi*0],[1,2,2],[0,0,0]]
goal = [1,1,0]

a = KinematicsSolver.KinematicsSolver(m)
a.calculateIK(axis, goal ,10e-10, "JT")

a.displayEndEffectorPose()