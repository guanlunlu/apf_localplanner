#!/usr/bin/env python3
# import __future__
import rospy
import math
import numpy as np

from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetPlanResponse
from geometry_msgs.msg import PoseStamped

import static_map
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
        self.pathpub = rospy.Publisher('/local_path', Path, queue_size=10)

        # map param
        self.map_resolution = 0.1
        self.static_map = static_map.MapLayer(self.map_resolution)
        self.inflation_size = 2 #inflation_size

        # apf param.
        self.static_obstacles = self.static_map.obstacle_list
        # print(self.static_obstacles)
        self.dynamic_obstacles = []
        
        self.init_pose = pose(0,0,0)
        self.goal_pose = pose(0,0,0)

        # apf param
        self.gradient_step = 0.01
        self.descent_rate = 0.01

        self.force_threshold = 0.4

        self.gain_attractive = 0.2
        self.attr_threshold = 3

        self.repulsive_gain = 0.15
        self.repulsive_threshold = 0.3
        self.n = 10

        self.goal_tolerance = 0.01

        # output
        self.path = []
        self.pathmsg = Path()

        self.t1 = rospy.get_time()
        self.t2 = rospy.get_time()

    def apfCallback(self,req):
        # goal request from client
        self.goal_pose.x, self.goal_pose.y = [req.goal.pose.position.x, req.goal.pose.position.y]
        gqx = req.goal.pose.orientation.x
        gqy = req.goal.pose.orientation.y
        gqz = req.goal.pose.orientation.z
        gqw = req.goal.pose.orientation.w
        _, _, yaw = euler_from_quaternion([gqx,gqy,gqz,gqw])
        self.goal_pose.theta = yaw
        print("goal requested = ", [self.goal_pose.x, self.goal_pose.y, self.goal_pose.theta])
        
        # initial pose from client
        self.init_pose.x, self.init_pose.y = [req.start.pose.position.x, req.start.pose.position.y]
        iqx = req.start.pose.orientation.x
        iqy = req.start.pose.orientation.y
        iqz = req.start.pose.orientation.z
        iqw = req.start.pose.orientation.w
        _, _, yaw = euler_from_quaternion([iqx,iqy,iqz,iqw])
        self.init_pose.theta = yaw
        print("init requested = ", [self.init_pose.x, self.init_pose.y, self.init_pose.theta])

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

    def apf(self, init_pose, goal_pose):
        iter_pose = init_pose
        output_path = []
        while not self.goalReached(iter_pose, goal_pose) and not rospy.is_shutdown():
            d_len = math.sqrt(pow(iter_pose.force().x, 2)+pow(iter_pose.force().y,2))
            d_x = iter_pose.force().x / d_len
            d_y = iter_pose.force().y / d_len

            next_x = iter_pose.x - self.descent_rate * d_x
            next_y = iter_pose.y - self.descent_rate * d_y
            
            theta = math.atan2(next_y - iter_pose.y, next_x - iter_pose.x)
            iter_pose = pose(next_x, next_y, theta)
            # print("iter_pose = ",(iter_pose.x, iter_pose.y, iter_pose.theta))
            output_path.append(iter_pose)
            # self.pathPublish(output_path)
        return output_path

    def goalReached(self,cur_pose, goal_pose):
        if cur_pose.distanceTo(goal_pose) < self.goal_tolerance:
            return True
        else:
            return False

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
            pose = PoseStamped()
            pose.pose.position.x = i.x
            pose.pose.position.y = i.y
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
