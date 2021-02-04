#ifndef GETDXDYDZWITHPNP_H_
#define GETDXDYDZWITHPNP_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "data.h"
#include "main.hpp"
#include "testran.h"
using namespace std;
using namespace cv;

//dPoint GaussProjCal_zhao(dPoint bol);
//高斯投影由大地坐标(Unit:m)反算经纬度(Unit:DD)
//dPoint GaussProjInvCal_zhao(dPoint xoy);
//void convertRotateVec_zhao(double _angle, const dPoint& _midPos, dPoint& vec_pos);
/************************************************************************
* function: 通过视觉计算出来的标识牌的相对坐标和标识牌在地图中的
*           经纬度计算自车位置经纬度
* parameter:
*            _angle:		车头航向角度，单位：弧度，
                            正北方向为0度，顺时针方向增加
*            _midPos:		标识牌的相对坐标，单位：m
*            _targetPoint:	标识牌的经纬度，单位：单位：10^-8度
*            modifypos :	计算出来的自车位置经纬度，单位：10^-8度
* return: void
*********************************************************************/
//void calculateCarPos_zhao(double _angle, const dPoint& _midPos, const dPoint& _targetPoint, dPoint& modifypos);
void set_P_matrix(cv::Mat &_P_matrix, const cv::Mat &R_matrix, const cv::Mat &t_matrix);
void set_P_matrix_f(cv::Mat &_P_matrix, const cv::Mat &R_matrix, const cv::Mat &t_matrix);
cv::Point2d backproject3DPoint(cv::Point3d &cameraPoint, cv::Mat _A_matrix, cv::Mat _P_matrix, const cv::Point3d &point3d);
cv::Point3d backprojectCamera2WorldPoint(cv::Point3d cameraPoint, const cv::Mat R_matrix, const cv::Mat t_matrix);
cv::Point3d getPositionRefSignOri(cv::Point3d cameraPoint, const cv::Mat R_matrix, const cv::Mat t_matrix);
//bool getDxDyDzWithPnP(std::vector<cv::Point3d>& rtk3dPoints, std::vector<cv::Point2d>& backImagePoints, std::vector<cv::Point3d> sign_points, std::vector<cv::Point2d> image_points);
//bool getDxDyDzWithPnP(std::vector<cv::Point3d>& rtk3dPoints, std::vector<cv::Point3d> sign_points, std::vector<cv::Point2d> image_points);
bool getDxDyDzWithPnP(std::vector<cv::Point3d>& rtk3dPoints, std::vector<cv::Point2d>& backImagePoints, cv::Point3d& cameraPosGauss, std::vector<cv::Point3d> sign_points, std::vector<cv::Point2d> image_points, std::string confFile, std::vector<cv::Mat> covarmatrix, int ind, Mat &dx);
bool getDxDyDzWithPnP(std::vector<cv::Point3d>& rtk3dPoints, std::vector<cv::Point2d>& backImagePoints, cv::Point3d& cameraPosRefSignOri, std::vector<cv::Point3d> sign_points, std::vector<cv::Point2d> image_points, cv::Mat _A_matrix, cv::Mat dist_coeffs, cv::Mat rtk2Camera_R_matrix, cv::Mat rtk2Camera_t_matrix, std::vector<cv::Mat> covarmatrix, Mat &dx, int ind = 0);

bool shuffPoints(std::vector<cv::Point3d>& shuffSeleSign, std::vector<cv::Point2d>& shuffSeleImg, std::vector<int>& shuffnum, std::vector<cv::Point3d> seleSign, std::vector<cv::Point2d> seleImg);
//void MySolvePnpPosit(const std::vector<cv::Point3f> &objectPoints, const std::vector<cv::Point2f> &imagePoints,\
                     const cv::Mat_<double> &cameraIntrinsicParams, const cv::Mat_<double> &distCoeffs,\
                     cv::Mat &outRotationEstimated, cv::Mat &outTranslationEstimated);
//cv::Point2f MyImageCoordsToIdealCameraCoords(const cv::Mat_<double> & cameraIntrinsicParams, const cv::Point2f & pt);
//void MyPosit_IdealCameraCoords(const std::vector<cv::Point3f> & objectPoints, const std::vector<cv::Point2f> & imagePoints,\
                           cv::Mat &outRotationEstimated, cv::Mat & outTranslationEstimated);
double DistPoint(std::vector<cv::Point2d> p1s, std::vector<cv::Point2d> p2s);
bool PnP(std::vector<cv::Point3d>& cameraPoints, std::vector<cv::Point2d>& backImagePoints, cv::Mat& R, cv::Mat& T, std::vector<int>& shuffnums, std::vector<cv::Point3d> model_points, std::vector<cv::Point2d> image_points, \
         cv::Mat _A_matrix, cv::Mat dist_coeffs, std::vector<cv::Mat> covarmatrix, Mat &dx, int times);
bool transformDouble3ToFloat3(std::vector<cv::Point3f>& point3f, std::vector<cv::Point3d> point3ds);
bool transformDouble2ToFloat2(std::vector<cv::Point2f>& point2f, std::vector<cv::Point2d> point2ds);

#endif
