#!/usr/bin/env python3
import numpy as np
from itertools import chain
from apf_planner import pose

class SegmentObject:
    def __init__(self, start_point, end_point, resolution) -> None:
        self.start = start_point
        self.end = end_point

        # resolution
        self.resolution = resolution
        self.d = start_point.distanceTo(end_point)
        self.list_len = int(self.d/resolution) + 1

        self.segment_list = []
        self.interp()


    def interp(self):
        x_list = np.linspace(self.start.x, self.end.x, self.list_len)
        y_list = np.linspace(self.start.y, self.end.y, self.list_len)

        for i in range(self.list_len):
            self.segment_list.append(pose(x_list[i], y_list[i], 0))
    

class MapLayer:
    def __init__(self, resolution) -> None:
        self.resolution = resolution
        self.obstacle_list = [SegmentObject(pose(0.00, 0.00, 0.00), pose(1.49, 0.00, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(1.49, 0.00, 0.00), pose(2.00, 0.51, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(2.00, 0.51, 0.00), pose(2.00, 2.49, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(2.00, 2.49, 0.00), pose(1.49, 3.00, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(1.49, 3.00, 0.00), pose(0.00, 3.00, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.00, 3.00, 0.00), pose(0.00, 2.55, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.00, 2.55, 0.00), pose(0.09, 2.55, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.09, 2.55, 0.00), pose(0.09, 1.83, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.09, 1.83, 0.00), pose(0.00, 1.83, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.00, 1.83, 0.00), pose(0.00, 1.725, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.00, 1.725, 0.00), pose(0.102, 1.725, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.102, 1.725, 0.00), pose(0.102, 1.275, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.102, 1.275, 0.00), pose(0.00, 1.275, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.00, 1.275, 0.00), pose(0.00, 1.17, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.00, 1.170, 0.00), pose(0.09, 1.17, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.09, 1.170, 0.00), pose(0.09, 0.45, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.09, 0.45, 0.00), pose(0.00, 0.45, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.00, 0.45, 0.00), pose(0.00, 0.00, 0.00), self.resolution).segment_list,
                              SegmentObject(pose(0.00, 1.50, 0.00), pose(0.30, 1.50, 0.00), self.resolution).segment_list]
        
        
        self.obstacle_list = self.flatten(self.obstacle_list)

    def flatten(self, arr):
        flatten_list = list(chain.from_iterable(arr))
        return flatten_list