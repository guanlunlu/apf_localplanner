#!/usr/bin/env python3
# import __future__
from mimetypes import init
import rospy
import queue
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData

from nav_msgs.srv import GetPlan, GetPlanResponse
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray

import matplotlib.pyplot as plt
from matplotlib import colors
import math
import operator
from scipy import ndimage
import numpy as np
from mpl_toolkits import mplot3d
pi = math.pi

class pose():
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    
    def distanceTo(self, pose_):
        d = math.sqrt(pow(pose_.x-self.x, 2) + pow(pose_.y-self.y, 2))
        return d
    
    def repulsivePotetial(self, cur_pose):
        # return potential at position
        static_obstacles = getattr(apf, "static_obstacles")
        dynamic_obstacles = getattr(apf, "dynamic_obstacles")
        repulsive_threshold = getattr(apf, "repulsive_threshold")
        repulsive_gain = getattr(apf, "repulsive_gain")
        goal_pose = getattr(apf, "goal_pose")
        n = getattr(apf, "n")

        # distance from robot to target
        d_goal = goal_pose.distanceTo(cur_pose)
        rep_potential = 0

        for obs in static_obstacles:
            # distance from robot to current obstacle
            d_obs = obs.distanceTo(cur_pose)
            if d_obs > repulsive_threshold:
                rep_potential += 0
            else:
                if d_obs < 0.0001:
                    d_obs = 0.0001
                rep_potential += 0.5 * repulsive_gain * (1/d_obs - 1/repulsive_threshold) * pow(d_goal, n)/(1 + pow(d_goal, n))
                # rep_potential += 0.5 * repulsive_gain * pow((1/d_obs - 1/repulsive_threshold), 2)
                # rep_potential += 0.5 * repulsive_gain * pow((1/d_obs), 2)

        for obs in dynamic_obstacles:
            # distance from robot to current obstacle
            d_obs = obs.distanceTo(cur_pose)
            if d_obs > repulsive_threshold:
                rep_potential += 0
            else:
                if d_obs < 0.0001:
                    d_obs = 0.0001
                rep_potential += 0.5 * repulsive_gain * (1/d_obs - 1/repulsive_threshold) * pow(d_goal, n)/(1 + pow(d_goal, n))

        return rep_potential
    
    def attractivePotential(self, cur_pose):
        goal_pose = getattr(apf,"goal_pose")
        gain_attractive = getattr(apf,"gain_attractive")
        d_threshold = getattr(apf,"attr_threshold")

        d_goal = goal_pose.distanceTo(cur_pose)
        att_potential = 0.5 * gain_attractive * pow(d_goal, 2)

        return att_potential

    def force(self):
        d = getattr(apf,"gradient_step")
        cur_pos_xm = pose(self.x-d, self.y, self.theta)
        cur_pos_ym = pose(self.x, self.y-d, self.theta)
        
        cur_pos_xp =  pose(self.x + d, self.y, self.theta)
        cur_pos_yp =  pose(self.x , self.y + d, self.theta)
        # curPos_potential = self.rep_potential(cur_pos) + self.att_potential(cur_pos)
        grad_x = (self.attractivePotential(cur_pos_xp)+self.repulsivePotetial(cur_pos_xp) - (self.attractivePotential(cur_pos_xm)+self.repulsivePotetial(cur_pos_xm)))/d/2

        grad_y = (self.attractivePotential(cur_pos_yp)+self.repulsivePotetial(cur_pos_yp) - (self.attractivePotential(cur_pos_ym)+self.repulsivePotetial(cur_pos_ym)))/d/2

        grad = pose(grad_x, grad_y, 0)

        return grad

        
class Apf():
    def __init__(self):
        self.server = rospy.Service('/apf_localplanner', GetPlan, self.apfCallback)
        self.mapsub = rospy.Subscriber("/map", OccupancyGrid, self.mapCallback)
        self.pathpub = rospy.Publisher('/local_path', Path, queue_size=10)
        # map param
        self.map_width = 0
        self.map_height = 0
        self.map_origin = []
        self.map_resolution = 0
        self.mapdata = []

        self.processed_mapdata = []
        self.substract_mapdata = [] 
        self.external_mapdata = []

        self.inflation_size = 2 #inflation_size

        # apf param.
        self.static_obstacles = []
        self.dynamic_obstacles = []
        
        self.init_pose = pose(0,0,0)
        self.goal_pose = pose(0,0,0)

        # apf param
        self.gradient_step = 0.01
        self.descent_rate = 0.1

        self.force_threshold = 10

        self.gain_attractive = 0.2
        self.attr_threshold = 3

        self.repulsive_gain = 30
        self.repulsive_threshold = 40
        self.n = 10

        self.goal_tolerance = 0.01

        # output
        self.path = []
        self.pathmsg = Path()

        self.t1 = rospy.get_time()
        self.t2 = rospy.get_time()

    def apfCallback(self,req):
        # goal request from client
        self.goal_pose.x, self.goal_pose.y = self.real_to_map([req.goal.pose.position.x, req.goal.pose.position.y])
        gqx = req.goal.pose.orientation.x
        gqy = req.goal.pose.orientation.y
        gqz = req.goal.pose.orientation.z
        gqw = req.goal.pose.orientation.w
        _, _, yaw = euler_from_quaternion([gqx,gqy,gqz,gqw])
        self.goal_pose.theta = yaw
        print("(map) goal requested = ", [self.goal_pose.x, self.goal_pose.y, self.goal_pose.theta])
        
        # initial pose from client
        self.init_pose.x, self.init_pose.y = self.real_to_map([req.start.pose.position.x, req.start.pose.position.y])
        iqx = req.start.pose.orientation.x
        iqy = req.start.pose.orientation.y
        iqz = req.start.pose.orientation.z
        iqw = req.start.pose.orientation.w
        _, _, yaw = euler_from_quaternion([iqx,iqy,iqz,iqw])
        self.init_pose.theta = yaw
        print("(map) init requested = ", [self.init_pose.x, self.init_pose.y, self.init_pose.theta])

        # run apf
        # reset output
        del self.path[:]
        self.t1 = rospy.get_time()
        print("obstacle list = ", len(self.static_obstacles))
        self.path = self.apf(self.init_pose, self.goal_pose)
        self.pathPublish(self.path)
        self.t2 = rospy.get_time()
        print("d_t = ", self.t2- self.t1)
        return GetPlanResponse(self.pathmsg)

    def mapCallback (self, raw_map_data):
        # map parameter achieved
        self.map_width = raw_map_data.info.width
        self.map_height = raw_map_data.info.height
        self.map_origin = [raw_map_data.info.origin.position.x, raw_map_data.info.origin.position.y]
        self.map_resolution = raw_map_data.info.resolution
        self.mapdata = [[0]*self.map_width for i in range(self.map_height)]
        self.processed_mapdata = [[0]*self.map_width for i in range(self.map_height)]
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
        self.obstacleInflation()
    
    def obstacleInflation(self):
        # occupy 100
        # empty 0
        # unknown -1        
        structure1 = ndimage.generate_binary_structure(2,2)

        self.processed_mapdata = ndimage.binary_dilation(self.mapdata, structure=structure1, iterations=self.inflation_size).astype(self.mapdata.dtype)

        self.substract_mapdata = ndimage.binary_dilation(self.mapdata, structure=structure1, iterations= self.inflation_size-1).astype(self.mapdata.dtype)

        self.external_mapdata =ndimage.binary_dilation(self.mapdata, structure=structure1, iterations= self.inflation_size+1).astype(self.mapdata.dtype)

        self.processed_mapdata -= self.substract_mapdata

        for i in range(self.map_height):
            for j in range(self.map_width):
                if self.processed_mapdata[i][j] ==1:
                    self.static_obstacles.append(pose(j,i,0))

        print("Map data inflated !!")

    def apf(self, init_pose, goal_pose):
        iter_pose = init_pose
        output_path = []
        while not self.goalReached(iter_pose, goal_pose):
            next_x = iter_pose.x - self.descent_rate * iter_pose.force().x
            next_y = iter_pose.y - self.descent_rate * iter_pose.force().y
            
            theta = math.atan2(next_y - iter_pose.y, next_x - iter_pose.x)
            iter_pose = pose(next_x, next_y, theta)
            # print("iter_pose = ",(iter_pose.x, iter_pose.y, iter_pose.theta))
            output_path.append(iter_pose)
        
        return output_path


    def goalReached(self,cur_pose, goal_pose):
        if cur_pose.distanceTo(goal_pose) < self.goal_tolerance:
            return True
        else:
            return False

    def real_to_map(self, realcoord):
        # convert world coordinate to map coordinate
        # realcoord = [x, y]
        realx, realy = realcoord
        orgx, orgy = self.map_origin[0], self.map_origin[1]
        mapx = int((realx - orgx)/self.map_resolution)
        mapy = int((realy - orgy)/self.map_resolution)
        mapcoord = [mapx, mapy]
        return mapcoord

    def map_to_real(self, mapcoord):
        # convert map coordinate to world coordinate
        # mapcoord = [x, y]
        mapx, mapy = mapcoord
        orgx, orgy = self.map_origin[0], self.map_origin[1]
        realx = mapx * self.map_resolution + orgx
        realy = mapy * self.map_resolution + orgy
        realcoord = [realx, realy]
        return realcoord

    def thetaConvert(self, theta):
        while theta > pi:
            theta -= 2 * pi
        while theta < -pi:
            theta += 2 * pi
        return theta

    def pathPublish(self, path):
        # visualize astar
        del self.pathmsg.poses[:]
        self.pathmsg.header.frame_id = "map"
        self.pathmsg.header.stamp = rospy.Time.now()
        for i in path:
            x, y = self.map_to_real([i.x, i.y])
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0
            quaternion = quaternion_from_euler(0, 0, i.theta)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            self.pathmsg.poses.append(pose)
        self.pathpub.publish(self.pathmsg)

if __name__ == '__main__':
    rospy.init_node('apf_local_planner', anonymous = True)
    apf = Apf()
    rospy.spin()
