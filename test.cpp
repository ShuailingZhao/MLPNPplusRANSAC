#include "main.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <random>
#include <time.h>
#include "getDxDyDzWithPnP.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace type;
using namespace std;

cv::Mat getRx(const double theta_x)
{
    double pi = 3.1415926535898;
    double thetax = theta_x*pi/180.0;

    cv::Mat Rx = cv::Mat(3, 3, CV_64FC1);
    Rx.at<double>(0,0) = 1.0;
    Rx.at<double>(0,1) = 0.0;
    Rx.at<double>(0,2) = 0.0;
    Rx.at<double>(1,0) = 0.0;
    Rx.at<double>(1,1) = cos(thetax);
    Rx.at<double>(1,2) = sin(thetax);
    Rx.at<double>(2,0) = 0.0;
    Rx.at<double>(2,1) = -sin(thetax);
    Rx.at<double>(2,2) = cos(thetax);

    return Rx;
}

cv::Mat getRy(const double theta_y)
{
    double pi = 3.1415926535898;
    double thetay = theta_y*pi/180.0;

    cv::Mat Ry = cv::Mat(3, 3, CV_64FC1);
    Ry.at<double>(0,0) = cos(thetay);
    Ry.at<double>(0,1) = 0.0;
    Ry.at<double>(0,2) = -sin(thetay);
    Ry.at<double>(1,0) = 0.0;
    Ry.at<double>(1,1) = 1.0;
    Ry.at<double>(1,2) = 0.0;
    Ry.at<double>(2,0) = sin(thetay);
    Ry.at<double>(2,1) = 0.0;
    Ry.at<double>(2,2) = cos(thetay);

    return Ry;
}

cv::Mat getRz(const double theta_z)
{
    double pi = 3.1415926535898;
    double thetaz = theta_z*pi/180.0;

    cv::Mat Rz = cv::Mat(3, 3, CV_64FC1);
    Rz.at<double>(0,0) = cos(thetaz);
    Rz.at<double>(0,1) = sin(thetaz);
    Rz.at<double>(0,2) = 0.0;
    Rz.at<double>(1,0) = -sin(thetaz);
    Rz.at<double>(1,1) = cos(thetaz);
    Rz.at<double>(1,2) = 0.0;
    Rz.at<double>(2,0) = 0.0;
    Rz.at<double>(2,1) = 0.0;
    Rz.at<double>(2,2) = 1.0;
    return Rz;
}

cv::Mat getRxyz(const double thetax, const double thetay, const double thetaz)
{

        cv::Mat Rxyz;
        Rxyz = getRx(thetax) * getRy(thetay) * getRz(thetaz);
        return Rxyz;
}


cv::Mat getRzyx(const double thetax, const double thetay, const double thetaz)
{

        cv::Mat Rzyx;
        Rzyx =  getRz(thetaz) * getRy(thetay) * getRx(thetax) ;
        return Rzyx;
}

cv::Mat getRyxz(const double thetax, const double thetay, const double thetaz)
{

        cv::Mat Ryxz;
        Ryxz =  getRy(thetay) * getRx(thetax) * getRz(thetaz);
        return Ryxz;
}

double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

int main()
{
//    bearingVectors_t imgvecs;
//    vector<cv::Point3d> imgvs;
//    points_t p3ds;
//    type::cov3_mats_t covMats;
    std::vector<cv::Point3d> rtk3dPoints;
    std::vector<cv::Point2d> backImagePoints;
    cv::Point3d cameraPosGauss;
    std::vector<cv::Point3d> sign_points;
    std::vector<cv::Point2d> image_points;
    std::vector<double> stdy, hprs;
    double hpr;
    double heading;
    double pitch;
    double roll;

//    cv::Mat _A_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
//        _A_matrix.at<double>(0,0) = 1360.741577148438;
//        _A_matrix.at<double>(0,1) = 0;
//        _A_matrix.at<double>(0,2) = 658.3298488965738;
//        _A_matrix.at<double>(1,0) = 0;
//        _A_matrix.at<double>(1,1) = 1362.7548828125;
//        _A_matrix.at<double>(1,2) = 331.7633069283911;
//        _A_matrix.at<double>(2,0) = 0;
//        _A_matrix.at<double>(2,1) = 0;
//        _A_matrix.at<double>(2,2) = 1;

//        double u = 658.3298488965738;
//        double v = 331.7633069283911;

    std::ifstream hprread;
    hprread.open("/home/wangtao/QTcode/P3Ptest/hprtest.txt");
    if(!hprread)
    {
        cout<<"error read hpr.txt"<<endl;
        return 1;
    }
    while(hprread>>hpr)
    {
        hprs.push_back(hpr);
    }
    hprread.close();

    int picnum;
    std::vector<int> picnums;
    std::ifstream picread;
    picread.open("/home/wangtao/QTcode/P3Ptest/picnumbertest.txt");
    if(!picread)
    {
        cout<<"error read picnumber.txt"<<endl;
        return 1;
    }
    while(picread>>picnum)
    {
        picnums.push_back(picnum);
    }
    picread.close();

    double imgnum;
    std::vector<double> img;
    std::ifstream imgread;
    imgread.open("/home/wangtao/QTcode/P3Ptest/imagenumbertest.txt");
    if(!imgread)
    {
        cout<<"error read imagenumber.txt"<<endl;
        return 1;
    }
    while(imgread>>imgnum)
    {
        img.push_back(imgnum);
    }
    imgread.close();

    double p3Dnum;
    std::vector<double> p3D;
    std::ifstream p3Dread;
    p3Dread.open("/home/wangtao/QTcode/P3Ptest/p3Dnumbertest.txt");
    if(!p3Dread)
    {
        cout<<"error read p3Dnumber.txt"<<endl;
        return 1;
    }
    while(p3Dread>>p3Dnum)
    {
        p3D.push_back(p3Dnum);
    }
    p3Dread.close();

    for(int i=0;i<p3D.size()/3;i++)
    {
        image_points.push_back(cv::Point2d(img.at(2*i),img.at(2*i+1)));
        sign_points.push_back(cv::Point3d(p3D.at(3*i),p3D.at(3*i+1),p3D.at(3*i+2)));
    }

    //***********产生协方差***********//
    cv::Mat errimg = cv::Mat::zeros(2 * image_points.size(),100,CV_64FC1);
    int i0 = 0;
    cv::Mat errimgpoint = cv::Mat::zeros(2,1,CV_64FC1);
    srand((unsigned int)time(NULL));
    for(auto i = image_points.cbegin();i != image_points.cend();++i)
    {
        errimgpoint.at<double>(0) = i->x;
        errimgpoint.at<double>(1) = i->y;
        for(int j=0;j<100;++j)
        {
            errimgpoint.at<double>(0) = errimgpoint.at<double>(0) + gaussrand();
            errimgpoint.at<double>(1) = errimgpoint.at<double>(1) + gaussrand();
            errimg.at<double>(2*i0,j) = errimgpoint.at<double>(0);
            errimg.at<double>(2*i0+1,j) = errimgpoint.at<double>(1);
        }
        i0++;
    }

    std::vector<cv::Mat> covarmatrix;
    cv::Mat covar = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Mat covars = cv::Mat::zeros(2,2,CV_64FC1);
    cv::Mat means = cv::Mat::zeros(2,1,CV_64FC1);
    for(int i=0;i<image_points.size();++i)
    {
        cv::calcCovarMatrix(errimg.rowRange(2 * i,2 * i+2), covars, means, CV_COVAR_NORMAL | CV_COVAR_COLS);
        covar.at<double>(0,0) = covars.at<double>(0,0);
        covar.at<double>(1,0) = covars.at<double>(1,0);
        covar.at<double>(0,1) = covars.at<double>(0,1);
        covar.at<double>(1,1) = covars.at<double>(1,1);
//        cout<<errimg.rowRange(2 * i,2 * i+2)<<endl;
//        cout<<covar<<endl;
        covarmatrix.push_back(covar);
    }
    //***********产生协方差***********//

//  test
    std::string confFile = "/home/wangtao/QTcode/testdaima/duantouImgUndistortsuo.xml";
    int ind = 0;

//    std::vector<cv::Mat> covarmatrix;
//    cv::Mat covar = cv::Mat::eye(3, 3, CV_64FC1);
//    covar.at<double>(2, 2) = 0;

    cv::Mat dx = cv::Mat::zeros(3, 1, CV_64FC1);
    int testN=picnums.at(0);
    int x=0;
    int y=0;
    std::vector<cv::Point2d> imgPoints;
    std::vector<cv::Point3d> colPoints;
    std::vector<cv::Mat> covarmatrixs;
    for(auto j = picnums.begin();j != picnums.end();++j)
    {
        if(testN != *j)
        {
            cout<<testN<<endl;
//            for(auto ins = imgPoints.begin();ins != imgPoints.end();++ins)
//            {
            heading = hprs.at(3*y);
            pitch = hprs.at(3*y+1);
            roll = hprs.at(3*y+2);
            bool pnpOK = getDxDyDzWithPnP(rtk3dPoints, backImagePoints, cameraPosGauss, colPoints, imgPoints, confFile, covarmatrixs, ind, dx);

            Mat R(3,3,CV_64FC1);
            R = getRyxz(pitch, roll, -heading);
            Mat dxdydz(3,1,CV_64FC1);
            dxdydz = R * dx;
            if(pnpOK==1)
            {
                stdy.push_back(dxdydz.at<double>(2));
                cout<<"dy: "<<dx.at<double>(1)<<endl;
                cout<<"dz: "<<dxdydz.at<double>(2)<<endl;
            }
//            }
//            cout<<"\n"<<endl;
            y++;
            imgPoints.clear();
            colPoints.clear();
            covarmatrixs.clear();

            if(j != picnums.end())
            {
                testN = *j;
            }
        }
        if(x<image_points.size())
        {
            for(int s=0;s<4;++s)
            {
                imgPoints.push_back(image_points.at(x));
                colPoints.push_back(sign_points.at(x));
                covarmatrixs.push_back(covarmatrix.at(x));
//                covarmatrix.push_back(covar);
                x++;
            }
        }
    }

    std::ofstream fout;
    fout.open("/home/wangtao/QTcode/P3Ptest/stdnumber.txt");
    for(auto is = stdy.begin();is != stdy.end();++is)
    {
        fout<<*is<<"\n";
    }
    fout.close();

//    double covnum;
//    std::vector<double> cov;
//    std::ifstream covread;
//    covread.open("/home/wangtao/QTcode/P3Ptest/covs.txt");
//    if(!covread)
//    {
//        cout<<"error read covs.txt"<<endl;
//        return 1;
//    }
//    while(covread>>covnum)
//    {
//        cov.push_back(covnum);
//    }
//    covread.close();

//    cov3_mat_t covMat;
//    for(int i=0;i<p3D.size()/3;i++)
//    {
//        p3ds.push_back(point_t(p3D.at(3*i),p3D.at(3*i+1),p3D.at(3*i+2)));
////        double xs = (img.at(2*i)-u)/_A_matrix.at<double>(0,0);
////        double ys = (img.at(2*i+1)-v)/_A_matrix.at<double>(0,0);
////        imgvs.push_back(cv::Point3d(xs,ys,1));
//        imgvecs.push_back(bearingVector_t(img.at(3*i),img.at(3*i+1),img.at(3*i+2)));
//        covMat << cov.at(3*i),cov.at(3*i+1),cov.at(3*i+2),
//                cov.at(3*i+3),cov.at(3*i+4),cov.at(3*i+5),
//                cov.at(3*i+6),cov.at(3*i+7),cov.at(3*i+8);
//        covMats.push_back(covMat);
//    }

//    cv::Mat sigmas = cv::Mat::eye(3, 3, CV_64FC1);
//    sigmas.at<double>(2, 2) = 0;
//    sigmas = _A_matrix.inv() * sigmas * _A_matrix.inv();

//    bearingVector_t feavecs;
//    cov3_mat_t covMat;
//    cv::Point3d fvector;
//    cv::Mat VTV = cv::Mat::zeros(3, 3, CV_64FC1);
//    for(int imgsn = 0;imgsn < imgvs.size();++imgsn)
//    {
//        cv::Mat Jan = cv::Mat::eye(3, 3, CV_64FC1);
//        fvector = imgvs.at(imgsn);
//        double muo = sqrt(fvector.ddot(fvector));
//        fvector.x = fvector.x/muo;
//        fvector.y = fvector.y/muo;
//        fvector.z = fvector.z/muo;
//        VTV.at<double>(0,0) = fvector.x * fvector.x;
//        VTV.at<double>(0,1) = fvector.x * fvector.y;
//        VTV.at<double>(0,2) = fvector.x * fvector.z;
//        VTV.at<double>(1,0) = fvector.y * fvector.x;
//        VTV.at<double>(1,1) = fvector.y * fvector.y;
//        VTV.at<double>(1,2) = fvector.y * fvector.z;
//        VTV.at<double>(2,0) = fvector.z * fvector.x;
//        VTV.at<double>(2,1) = fvector.z * fvector.y;
//        VTV.at<double>(2,2) = fvector.z * fvector.z;
//        feavecs[0] = fvector.x;
//        feavecs[1] = fvector.y;
//        feavecs[2] = fvector.z;
//        imgvecs.push_back(feavecs);
//        Jan = (Jan - VTV)/muo;
//        covMat(0,0) = Jan.at<double>(0,0);
//        covMat(0,1) = Jan.at<double>(0,1);
//        covMat(0,2) = Jan.at<double>(0,2);
//        covMat(1,0) = Jan.at<double>(1,0);
//        covMat(1,1) = Jan.at<double>(1,1);
//        covMat(1,2) = Jan.at<double>(1,2);
//        covMat(2,0) = Jan.at<double>(2,0);
//        covMat(2,1) = Jan.at<double>(2,1);
//        covMat(2,2) = Jan.at<double>(2,2);
//        covMats.push_back(covMat);
//    }

    //*******************************************************************//
//    type::cov3_mats_t covMat2s;
// //    std::vector<double> stdy;
//    std::vector<int> indices;
//    Eigen::VectorXd dx(6);
//    dx << 0,0,0,0,0,0;
//    transformation_t solution;
//    int testN=picnums.at(0);
//    int x=0;
//    int y=0;
//    int z=0;
//    bearingVectors_t imgPoints;
//    points_t colPoints;
//    for(auto j = picnums.begin();j != picnums.end();++j)
//    {
//        if(testN != *j)
//        {
//            cout<<testN<<endl;

//            point_t origin = colPoints.at(0);
//            for(auto cols = colPoints.begin();cols != colPoints.end();++cols)
//            {
//                *cols = *cols - origin;
//            }

//            mlpnp::mlpnp_main(imgPoints, colPoints, covMat2s, indices, solution, dx);
// //           stdy.push_back(dx[6]);
//            cout<<dx<<"\n"<<solution<<endl;

//            y++;
//            imgPoints.clear();
//            colPoints.clear();
//            covMat2s.clear();
//            z=0;

//            if(j != picnums.end())
//            {
//                testN = *j;
//            }
//        }
//        if(x<imgvecs.size())
//        {
//            for(int s=0;s<4;++s)
//            {
//                colPoints.push_back(p3ds.at(x));
//                imgPoints.push_back(imgvecs.at(x));
//                covMat2s.push_back(covMats.at(x));
//                indices.push_back(z);
//                z++;
//                x++;
//            }
//        }
//    }

//    std::ofstream fout;
//    fout.open("/home/wangtao/QTcode/P3Ptest/stdnumber.txt");
//    for(auto is = stdy.begin();is != stdy.end();++is)
//    {
//        fout<<*is<<"\n";
//    }
//    fout.close();
    //*******************************************************************//

    return 0;
}
