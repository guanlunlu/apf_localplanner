#!/usr/bin/env python3
# import __future__
import rospy
import queue
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
from nav_msgs.srv import GetPlan, GetPlanResponse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib import colors
import math
import operator
from scipy import ndimage
import numpy as np
import itertools
pi = math.pi

class pose():
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        
    def __eq__(self, other): 
            if not isinstance(other, pose):
                # don't attempt to compare against unrelated types
                return NotImplemented

            return self.x == other.x and self.y == other.y # and self.theta == other.theta

    def theta_convert(self, input):
        # convert rad domain to [-pi, pi]
        if input >=0:
            input = math.fmod(input, 2*pi)
            if input > pi:
                input -= 2*pi
                output = input
            else:
                output = input
        else:
            input *= -1
            input = math.fmod(input, 2*pi)
            if input > pi:
                input -= 2*pi
                output = input*-1
            else:
                output = input*-1
        return output

    def distance(self, pose1, pose2):
        d_x = pose1.x - pose2.x
        d_y = pose1.y - pose2.y
        return math.sqrt(pow(d_x, 2)+pow(d_y, 2))

    def rep_potential(self, pos):
        # return potential at position
        obstacle_list = getattr(apf, "obstacle_list")
        rep_threshold = getattr(apf, "rep_threshold")
        gain_repulsive = getattr(apf, "gain_repulsive")
        subgoal = getattr(apf, "subgoal")
        n = getattr(apf, "n")

        # distance from robot to target
        d_subgoal = self.distance(subgoal, pos)
        rep_potential = 0
        for obs in obstacle_list:
            # distance from robot to current obstacle
            d_obs = self.distance(obs, pos)
            if d_obs > rep_threshold:
                rep_potential += 0
            else:
                if d_obs == 0:
                    d_obs = 0.00001
                rep_potential += 0.5 * gain_repulsive * (1/d_obs - 1/rep_threshold) * pow(d_subgoal, n)/(1 + pow(d_subgoal, n))
                # rep_potential += 0.5 * gain_repulsive * pow((1/d_obs - 1/rep_threshold), 2)
                # rep_potential += 0.5 * gain_repulsive * pow((1/d_obs), 2)
        return rep_potential
    
    def att_potential(self, pos):
        subgoal = getattr(apf,"subgoal")
        gain = getattr(apf,"gain_attractive")
        d_threshold = getattr(apf,"attr_threshold")
        d_subgoal = self.distance(subgoal, pos)
        att_potential = 0.5 * gain * pow(d_subgoal, 2)
        return att_potential

    def force(self):
        d = getattr(apf,"gradient_step")
        cur_pos_x = pose(self.x-d, self.y, self.theta)
        cur_pos_y = pose(self.x, self.y-d, self.theta)
        cur_posx =  pose(self.x + d, self.y, self.theta)
        cur_posy =  pose(self.x , self.y + d, self.theta)
        # curPos_potential = self.rep_potential(cur_pos) + self.att_potential(cur_pos)
        grad_x = (self.att_potential(cur_posx)+self.rep_potential(cur_posx) - (self.att_potential(cur_pos_x)+self.rep_potential(cur_pos_x)))/d/2
        grad_y = (self.att_potential(cur_posy)+self.rep_potential(cur_posy) - (self.att_potential(cur_pos_y)+self.rep_potential(cur_pos_y)))/d/2
        grad = pose(grad_x, grad_y, 0)
        return grad
    
    def getHeuristic(self):
        goal = getattr(apf, "goal_pose")
        d_x = self.x-goal.x
        d_y = self.y-goal.y
        Mcost = abs(d_x) + abs(d_y)
        #Euclidean distance
        Ecost = math.sqrt(math.pow((d_x),2) + math.pow((d_y),2))
        #Chebyshev's distance
        Ccost = min(d_x, d_y) * math.sqrt(2) + (max(d_x, d_y) - min(d_x, d_y)) * 1
        return Ecost
   
class APF():
    def __init__(self):
        self.mapsub = rospy.Subscriber("map", OccupancyGrid, self.mapCallback)
        self.server = rospy.Service('path', GetPlan, self.apfCallback)
        self.pathpub = rospy.Publisher('/path_rviz', Path, queue_size=10)
        # map param
        self.map_width = 0
        self.map_height = 0
        self.map_origin = []
        self.map_resolution = 0
        self.mapdata = []
        self.processed_mapdata = []
        self.subtract_mapdata = []
        self.edge_mapdata = []
        self.external_mapdata = []
        self.obstacle_list = []
        self.inf_size = 2 #inflation_size
        # apf param
        self.gradient_step = 0.001
        self.force_threshold = 10
        # self.descent_rate = 1.5
        self.descent_rate = 0.001
        
        self.gain_attractive = 0.2
        # self.gain_attractive = 30
        # self.gain_attractive = 3
        self.attr_threshold = 3

        self.gain_repulsive = 30
        # self.gain_repulsive = 0.1
        self.rep_threshold = 6
        self.n = 10

        self.init_pose = pose(0,0,0)
        self.goal_pose = pose(0,0,0)
        self.subgoal = pose(0,0,0)
        self.subgoal_tolerance = 3
        self.path = []
        # rolling window param.
        self.rollingWindow_radius = 0.5
        self.rollingWindow_res = 100 # number of point on curriculum
        # output
        self.output_path = Path()
        self.path_x = []
        self.path_y = []
        self.path_theta = []

    def theta_convert(self, input):
        # convert rad domain to [-pi, pi]
        if input >=0:
            input = math.fmod(input, 2*pi)
            if input > pi:
                input -= 2*pi
                output = input
            else:
                output = input
        else:
            input *= -1
            input = math.fmod(input, 2*pi)
            if input > pi:
                input -= 2*pi
                output = input*-1
            else:
                output = input*-1
        return output

    def apfCallback(self,req):
        # goal request from client
        self.goal_pose.x, self.goal_pose.y = self.real2map([req.goal.pose.position.x, req.goal.pose.position.y])
        gqx = req.goal.pose.orientation.x
        gqy = req.goal.pose.orientation.y
        gqz = req.goal.pose.orientation.z
        gqw = req.goal.pose.orientation.w
        _, _, self.goal_pose.theta = euler_from_quaternion([gqx,gqy,gqz,gqw])
        print("(map) goal requested = "), [self.goal_pose.x, self.goal_pose.y, self.goal_pose.theta]

        # initial pose from client
        self.init_pose.x, self.init_pose.y = self.real2map([req.start.pose.position.x, req.start.pose.position.y])
        iqx = req.start.pose.orientation.x
        iqy = req.start.pose.orientation.y
        iqz = req.start.pose.orientation.z
        iqw = req.start.pose.orientation.w
        _, _, self.init_pose.theta = euler_from_quaternion([iqx,iqy,iqz,iqw])

        # run apf
        # self.apf(self.init_pose, self.goal_pose)
        # run rollingWindow
        self.rollingWindow(self.init_pose, self.goal_pose)
        # return astar_controllerResponse(self.path_x, self.path_y, self.path_theta)
        return GetPlanResponse(self.output_path)

    def mapCallback (self, raw_map_data):
        # map parameter achieved
        self.map_width = raw_map_data.info.width
        self.map_height = raw_map_data.info.height
        self.map_origin = [raw_map_data.info.origin.position.x, raw_map_data.info.origin.position.y]
        self.map_resolution = raw_map_data.info.resolution

        self.mapdata = [[0]*self.map_width for i in range(self.map_height)]
        self.processed_mapdata = [[0]*self.map_width for i in range(self.map_height)]
        self.subtract_mapdata = [[0]*self.map_width for i in range(self.map_height)]
        self.edge_mapdata = [[0]*self.map_width for i in range(self.map_height)]
        raw_map_queue = queue.Queue()

        for i in raw_map_data.data:
            raw_map_queue.put(i)

        for i in range(self.map_height):
            for j in range(self.map_width):
                self.mapdata[i][j] = raw_map_queue.get()
        print("Map data achieved !!")
        self.mapdata = np.array(self.mapdata)    
        # map visualization
        # cmap = colors.ListedColormap(['lavender','midnightblue'])
        # plt.imshow(self.mapdata, cmap = cmap, origin = "lower")
        # plt.show()
        self.obstacle_inflation()

    def real2map(self, realcoord):
        # convert world coordinate to map coordinate
        # realcoord = [x, y]
        realx, realy = realcoord
        orgx, orgy = self.map_origin[0], self.map_origin[1]
        mapx = int((realx - orgx)/self.map_resolution)
        mapy = int((realy - orgy)/self.map_resolution)
        mapcoord = [mapx, mapy]
        return mapcoord

    def map2real(self, mapcoord):
        # convert map coordinate to world coordinate
        # mapcoord = [x, y]
        mapx, mapy = mapcoord
        orgx, orgy = self.map_origin[0], self.map_origin[1]
        realx = mapx * self.map_resolution + orgx
        realy = mapy * self.map_resolution + orgy
        realcoord = [realx, realy]
        return realcoord

    def obstacle_inflation(self):
        # occupy 100
        # empty 0
        # unknown -1        
        structure1 = ndimage.generate_binary_structure(2,2)

        self.processed_mapdata = ndimage.binary_dilation(self.mapdata, structure=structure1, iterations=self.inf_size).astype(self.mapdata.dtype)
        self.subtract_mapdata = ndimage.binary_dilation(self.mapdata, structure=structure1, iterations= self.inf_size-1).astype(self.mapdata.dtype)
        self.external_mapdata =ndimage.binary_dilation(self.mapdata, structure=structure1, iterations= self.inf_size+1).astype(self.mapdata.dtype)
        for i in range(self.map_height):
            for j in range(self.map_width):
                if self.processed_mapdata[i][j] ==1:
                    self.obstacle_list.append(pose(j, i, 0))
                
                # self.edge_mapdata[i][j] = self.processed_mapdata[i][j] - self.subtract_mapdata[i][j]
                # if self.edge_mapdata[i][j] ==1:
                #     self.obstacle_list.append(pose(j, i, 0))
        print("Map data inflated !!")
        # self.rviz_pathshow(self.obstacle_list)
        # map visualization
        # cmap = colors.ListedColormap(['lavender','midnightblue'])
        # plt.imshow(self.processed_mapdata, cmap = cmap, origin = "lower")
        # plt.show()

    def isObstacle(self, maplist, pose):
        x = int(pose.x)
        y = int(pose.y)
        if  maplist[y][x] == 1:
            return True
        else:
            return False
            
    def distance(self, pose1, pose2):
        d_x = pose1.x - pose2.x
        d_y = pose1.y - pose2.y
        return math.sqrt(pow(d_x, 2)+pow(d_y, 2))

    def subgoalReached(self, pose):
        if self.distance(self.subgoal, pose) < self.subgoal_tolerance:
            return True
        else:
            return False

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def rollingWindow(self, init_pose, goal_pose):
        del self.path[:]
        subgoal_list = []
        cur_center = init_pose
        minHcost = 100000000000000000
        subgoal = pose(0,0,0)
        print("init center:"), (cur_center.x, cur_center.y)
        while cur_center != goal_pose:
            if self.distance(cur_center, goal_pose) > self.rollingWindow_radius:
                for i in range(self.rollingWindow_res):
                    theta = i * 2 * pi/self.rollingWindow_res
                    x = cur_center.x + self.rollingWindow_radius * math.cos(theta)
                    y = cur_center.y + self.rollingWindow_radius * math.sin(theta)
                    sub = pose(x,y,0)
                    if self.isObstacle(self.external_mapdata,sub):
                        subHcost = 100000000000000000
                    else:
                        subHcost = sub.getHeuristic()
                    if subHcost < minHcost:
                        minHcost = subHcost
                        subgoal = sub
            else:
                subgoal = goal_pose

            print("current subgoal :", (subgoal.x, subgoal.y))
            localPath = self.apf(cur_center,subgoal)
            self.rviz_pathshow(localPath)
            self.path.append(localPath)
            cur_center = subgoal
            print("current center :", (cur_center.x, cur_center.y))
        # flatten 2d list
        self.path = self.flatten(self.path)
        self.rviz_pathshow(self.path)
        # print (self.path)
                
    def apf(self, init_pose, subgoal_pose):
        path = []
        # self.subgoal = self.goal_pose
        self.subgoal = subgoal_pose
        # self.potential_field_show()
        cur_force = init_pose.force()
        cur_pos = init_pose
        curGrad_mag = math.sqrt(pow(cur_force.x, 2) + pow(cur_force.y, 2))
        # cur_force.x = cur_force.x/curGrad_mag
        # cur_force.y = cur_force.y/curGrad_mag
        path.append(cur_pos)
        # print ("init pose:"), (init_pose.x, init_pose.y)
        # print ("init gradient:"), (cur_force.x, cur_force.y)
        while 1:    
            next_x = cur_pos.x - self.descent_rate * cur_force.x
            next_y = cur_pos.y - self.descent_rate * cur_force.y
            # print("next xy:"), [next_x, next_y]
            theta = math.atan2(next_y - cur_pos.y, next_x - cur_pos.x)
            cur_pos = pose(next_x, next_y, theta)
            cur_force = cur_pos.force()
            curGrad_mag = math.sqrt(pow(cur_force.x, 2) + pow(cur_force.y, 2))
            # cur_force.x = cur_force.x/curGrad_mag
            # cur_force.y = cur_force.y/curGrad_mag
            # print("next grad:"), [cur_force.x, cur_force.y]
            path.append(cur_pos)
            self.rviz_pathshow(path)
            if self.subgoalReached(cur_pos):
                break
        path.append(subgoal_pose)
        # show path in rviz and publish path
        # self.rviz_pathshow(path)
        return path
        
    def rviz_pathshow(self, path):
        # visualize astar path in rviz
        # publish path by navmsgs/path
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        for i in path:
            x, y = self.map2real([i.x, i.y])
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0
            quaternion = quaternion_from_euler(0, 0, i.theta)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            msg.poses.append(pose)
        self.output_path = msg
        self.pathpub.publish(msg)

    # def path_publish

########################################
# for potential field 3D visualization #
########################################

    def att_potential(self, x, y):
        m, n = x.shape
        att = np.zeros((m, n))
        d_goal = np.sqrt((x-self.subgoal.x)**2 + (y - self.subgoal.y)**2)
        att = 0.5 * self.gain_attractive * (d_goal**2)
        return att

    def rep_potential(self, x, y):
            # distance from robot to target
            # d_subgoal = self.distance(self.subgoal, pos)
            # rep_potential = 0
            
            m, n = x.shape
            rep = np.zeros((m, n))
            rep_threshold = np.zeros((m, n))
            rep_threshold += self.rep_threshold

            for obs in self.obstacle_list:
                # distance from robot to current obstacle
                d_obs = np.sqrt((x-obs.x)**2 + (y-obs.y)**2)
                d_obs[d_obs == 0] = 0.00000001
                # rep += 0.5 * self.gain_repulsive * d_obs**(-2)
                d_subgoal = np.sqrt((x-self.subgoal.x)**2 + (y - self.subgoal.y)**2)
                rep += 0.5 * self.gain_repulsive * (1/d_obs - 1/rep_threshold) * (d_subgoal**(self.n))/(1+d_subgoal**(self.n))
                # rep += 0.5 * self.gain_repulsive * (1/d_obs - 1/rep_threshold)
            return rep

    def potential_field_show(self):
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.linspace(0, self.map_width, 50)
        y = np.linspace(0, self.map_height, 50)
        X, Y = np.meshgrid(x, y)
        Z_rep = self.rep_potential(X, Y)
        Z_att = self.att_potential(X, Y)
        Z = Z_rep + Z_att
        # Z = Z_rep
        print (Z_att)
        Z[Z > 50] = 50
        fig = plt.figure()
        ax = Axes3D(fig) #<-- Note the difference from your original code...
        ax.contour3D(X, Y, Z, 100, cmap='binary', origin = "lower")
        # ax.contour3D(X, Y, Z, 50, cmap=cm.coolwarm, origin = "lower")
        # ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

if __name__ == '__main__':
    rospy.init_node('apf', anonymous = True)
    apf = APF()
    rospy.spin()