#ifndef DATA_H_
#define DATA_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

typedef struct _dPoint{
    double x;
    double y;

}dPoint;


typedef struct _CarInfo{
    int16_t isValid; //当前GPS是否有效
    double heading;
    double pitch;  //roll & pitch 是RTK中的信息
    double roll; 
    double x;
    double y;
    double height;
} CarInfo;

typedef struct _TrafficSign{
    //double heading; //度，路牌正面朝向，正东为0,逆时针为正
    //double pitch;   //度，路牌俯仰角，上仰为正
    //double roll;    //度

    Point3d box_3d[4]; //牌子四个点坐标，目前无序
    Point2d box_2d[4]; //3D to 2D　结果
    Rect detection_box;

} TrafficSign;

typedef struct _LocationInfo{
    string _frameNumber;
    Mat _frame;
    CarInfo _gps;
    vector<TrafficSign> _trafficsign;
} LocationInfo;


#endif
