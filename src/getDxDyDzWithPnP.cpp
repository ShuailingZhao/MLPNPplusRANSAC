#include "getDxDyDzWithPnP.h"

void selrightpoint(std::vector<cv::Point3d> &model_points, std::vector<cv::Point2d> &image_points, std::vector<cv::Mat> &covarmatrix, Mat _mask_local_inliers)
{
    std::vector<cv::Point3d> model = model_points;
    std::vector<cv::Point2d> image = image_points;
    std::vector<cv::Mat> covars = covarmatrix;
    model_points.clear();
    image_points.clear();
    covarmatrix.clear();
//    cout<<_mask_local_inliers<<endl;
    const uchar* mask = _mask_local_inliers.ptr<uchar>();
    for(int i = 0;i < _mask_local_inliers.rows; ++i)
    {
        if (mask[i])
        {
            model_points.push_back(model.at(i));
            image_points.push_back(image.at(i));
            covarmatrix.push_back(covars.at(i));
        }
    }
}

bool inputs(InputArray _opoints, InputArray _ipoints, Mat &_inliers1, Mat &_inliers2)
{
    _inliers1 = _opoints.getMat();
    _inliers2 = _ipoints.getMat();
}

void set_P_matrix(cv::Mat &_P_matrix, const cv::Mat &R_matrix, const cv::Mat &t_matrix)
{
    // Rotation-Translation Matrix Definition
    _P_matrix.at<double>(0,0) = R_matrix.at<double>(0,0);
    _P_matrix.at<double>(0,1) = R_matrix.at<double>(0,1);
    _P_matrix.at<double>(0,2) = R_matrix.at<double>(0,2);
    _P_matrix.at<double>(1,0) = R_matrix.at<double>(1,0);
    _P_matrix.at<double>(1,1) = R_matrix.at<double>(1,1);
    _P_matrix.at<double>(1,2) = R_matrix.at<double>(1,2);
    _P_matrix.at<double>(2,0) = R_matrix.at<double>(2,0);
    _P_matrix.at<double>(2,1) = R_matrix.at<double>(2,1);
    _P_matrix.at<double>(2,2) = R_matrix.at<double>(2,2);
    _P_matrix.at<double>(0,3) = t_matrix.at<double>(0);
    _P_matrix.at<double>(1,3) = t_matrix.at<double>(1);
    _P_matrix.at<double>(2,3) = t_matrix.at<double>(2);
}

void set_P_matrix_f(cv::Mat &_P_matrix, const cv::Mat &R_matrix, const cv::Mat &t_matrix)
{
    // Rotation-Translation Matrix Definition
    _P_matrix.at<double>(0,0) = R_matrix.at<float>(0,0);
    _P_matrix.at<double>(0,1) = R_matrix.at<float>(0,1);
    _P_matrix.at<double>(0,2) = R_matrix.at<float>(0,2);
    _P_matrix.at<double>(1,0) = R_matrix.at<float>(1,0);
    _P_matrix.at<double>(1,1) = R_matrix.at<float>(1,1);
    _P_matrix.at<double>(1,2) = R_matrix.at<float>(1,2);
    _P_matrix.at<double>(2,0) = R_matrix.at<float>(2,0);
    _P_matrix.at<double>(2,1) = R_matrix.at<float>(2,1);
    _P_matrix.at<double>(2,2) = R_matrix.at<float>(2,2);
    _P_matrix.at<double>(0,3) = t_matrix.at<float>(0);
    _P_matrix.at<double>(1,3) = t_matrix.at<float>(1);
    _P_matrix.at<double>(2,3) = t_matrix.at<float>(2);
}

cv::Point2d backproject3DPoint(cv::Point3d &cameraPoint, cv::Mat _A_matrix, cv::Mat _P_matrix, const cv::Point3d &point3d)
{
    // 3D point vector [x y z 1]'
    cv::Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
    point3d_vec.at<double>(0) = point3d.x;
    point3d_vec.at<double>(1) = point3d.y;
    point3d_vec.at<double>(2) = point3d.z;
    point3d_vec.at<double>(3) = 1;

    // 2D point vector [u v 1]'
    cv::Mat point2d_vec = cv::Mat(3, 1, CV_64FC1);
    cv::Mat cameraP = cv::Mat(3, 1, CV_64FC1);
    cameraP = _P_matrix * point3d_vec;
    point2d_vec = _A_matrix * cameraP;
    cameraPoint.x = cameraP.at<double>(0);
    cameraPoint.y = cameraP.at<double>(1);
    cameraPoint.z = cameraP.at<double>(2);

    // Normalization of [u v]'
    cv::Point2d point2d;
    point2d.x = (point2d_vec.at<double>(0) / point2d_vec.at<double>(2));
    point2d.y = (point2d_vec.at<double>(1) / point2d_vec.at<double>(2));

    return point2d;
}

cv::Point3d backprojectCamera2WorldPoint(cv::Point3d cameraPoint, const cv::Mat R_matrix, const cv::Mat t_matrix)
{
    cv::Mat X_c = cv::Mat::ones(3, 1, CV_64F);
    X_c.at<double>(0) = cameraPoint.x;
    X_c.at<double>(1) = cameraPoint.y;
    X_c.at<double>(2) = cameraPoint.z;
    cv::Mat X_w = R_matrix.inv() * ( X_c - t_matrix );
    cv::Point3d w_p;
    w_p.x = X_w.at<double>(0);
    w_p.y = X_w.at<double>(1);
    w_p.z = X_w.at<double>(2);

    return w_p;

}

cv::Point3d getPositionRefSignOri(cv::Point3d cameraPoint, const cv::Mat R_matrix, const cv::Mat t_matrix)
{
   return backprojectCamera2WorldPoint(cameraPoint, R_matrix, t_matrix);
}


double DistPoint(std::vector<cv::Point2d> p1s, std::vector<cv::Point2d> p2s)
{
    if(p1s.size() != p2s.size())
        return 100000;
    double sum =0;
    for(int i=0; i<p1s.size(); i++)
    {
        sum += abs(p1s[i].x - p2s[i].x) + abs(p1s[i].y - p2s[i].y);
    }

    return sum;
}
/************************************************************************
 * function: Get distance about x,y,z coordinate with solvePnP method,
 * parameter:
 *          rtk3dPoint: Output, x,y,z distance between the first point and the rtk with solvePnP
 *          sign_points: Input, the input signboard longitude(02), latitude(02), altitude eg: (116.22549587 40.08148139 40.2)
 *          image_points: Input, the signboard four corner points(pixel) in the undistort image(200, 301)
 *          confFile: Input, configer file include intrinsic matrix, distortion matrix of the camera, rotation matix, translation matirx from rtk to camera
 * return ok is true, or false
 * ***********************************************************************/
bool getDxDyDzWithPnP(std::vector<cv::Point3d>& rtk3dPoints, std::vector<cv::Point2d>& backImagePoints, cv::Point3d& cameraPosGauss, std::vector<cv::Point3d> sign_points, std::vector<cv::Point2d> image_points, std::string confFile, std::vector<cv::Mat> covarmatrix, int ind, cv::Mat &dx)
{
    //0. check special case

    if(!(sign_points.size() == image_points.size() && sign_points.size()>=0))
    {
        cout << "Num of signboard points and image points are not matching" <<endl;
        return false;
    }


    //1. read config file to get the intrinsic matrix, distortion matrix and the rotation matrix, translation matrix between camera and rtk

    std::string instrinsic_filename = confFile;
    cv::FileStorage fs;
    if(!fs.open(confFile, cv::FileStorage::READ))
    {
        cout << "Filed to open file " << instrinsic_filename<<endl;
        return false;
    }
    cv::Mat _A_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
    cv::Mat dist_coeffs;
    cv::Mat rtk2Camera_R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // rotation matrix
    cv::Mat rtk2Camera_t_matrix = cv::Mat::zeros(3, 1, CV_64FC1);   // translation matrix

    fs["camera_matrix"] >> _A_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    fs["rotation_matrix"] >> rtk2Camera_R_matrix;
    fs["translation_matrix"] >> rtk2Camera_t_matrix;


//    _A_matrix.at<double>(0,0) = 1360.741577148438;
//    _A_matrix.at<double>(0,1) = 0;
//    _A_matrix.at<double>(0,2) = 658.3298488965738;
//    _A_matrix.at<double>(1,0) = 0;
//    _A_matrix.at<double>(1,1) = 1362.7548828125;
//    _A_matrix.at<double>(1,2) = 331.7633069283911;
//    _A_matrix.at<double>(2,0) = 0;
//    _A_matrix.at<double>(2,1) = 0;
//    _A_matrix.at<double>(2,2) = 1;


    cout << "_A_matrix" <<endl<< _A_matrix << endl;
    cout << "dist_coeffs" <<endl<< dist_coeffs << endl;
    cout << "rtk2Camera_R_matrix" <<endl<< rtk2Camera_R_matrix << endl;
    cout << "rtk2Camera_t_matrix" <<endl<< rtk2Camera_t_matrix << endl;


    //std::vector<cv::Point3f> rtk3dPoints;
    //std::vector<cv::Point2f> backImagePoints;

    getDxDyDzWithPnP(rtk3dPoints,backImagePoints, cameraPosGauss, sign_points, image_points, _A_matrix, dist_coeffs, rtk2Camera_R_matrix, rtk2Camera_t_matrix, covarmatrix, dx, ind);
    if(!(ind < rtk3dPoints.size()))
        return false;
    //rtk3dPoint = rtk3dPoints[ind];

    fs.release();
    return true;

}

bool PnP(std::vector<cv::Point3d>& cameraPoints, std::vector<cv::Point2d>& backImagePoints, cv::Mat& R, cv::Mat& T,std::vector<int>& shuffnums, std::vector<cv::Point3d> model_points, std::vector<cv::Point2d> image_points,\
         cv::Mat _A_matrix, cv::Mat dist_coeffs, std::vector<cv::Mat> covarmatrix, cv::Mat &dx, int times)
{
    if(model_points.size() != image_points.size())
    {
        return false;
    }

    if(!cameraPoints.empty())
    {
        cameraPoints.clear();
    }

    if(!backImagePoints.empty())
    {
        backImagePoints.clear();
    }

    if(!R.empty())
    {
        R.release();
    }
    if(!T.empty())
    {
        T.release();
    }

    std::vector<cv::Point3d> model = model_points;
//    std::vector<cv::Point2d> image = image_points;

    cv::Mat A_matrix = _A_matrix.clone();//
    A_matrix.at<double>(0, 2) = 0.0;
    A_matrix.at<double>(1, 2) = 0.0;
    //Get rotation and transform matirx
    cv::Mat _R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // rotation matrix
    cv::Mat _t_matrix = cv::Mat::zeros(3, 1, CV_64FC1);   // translation matrix
    cv::Mat _P_matrix = cv::Mat::zeros(3, 4, CV_64FC1);   // rotation-translation matrix
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
    Eigen::VectorXd dxs(6);
    dxs << 0,0,0,0,0,0;
    transformation_t solution;

    int signNum = model_points.size();
    if(times<=10)
    {
        if(!(signNum>5))
        {//Just one sign board, the strategy is AP3P-> init interRanSan

            if(model_points.size() == 5)
            {
                model_points.pop_back();
                image_points.pop_back();
            }
            std::vector<cv::Point3d> shuffModel_points;
            std::vector<cv::Point2d> shuffImage_points;
            std::vector<int> shuffnum;
            shuffPoints(shuffModel_points, shuffImage_points, shuffnum, model_points, image_points);
            shuffnums=shuffnum;

            std::vector<cv::Point3f> model_pointsf;
            std::vector<cv::Point2f> image_pointsf;
            transformDouble3ToFloat3(model_pointsf, shuffModel_points);
            transformDouble2ToFloat2(image_pointsf, shuffImage_points);

//            transformDouble3ToFloat3(model_pointsf, model_points);
//            transformDouble2ToFloat2(image_pointsf, image_points);

            cv::solvePnP(model_pointsf, image_pointsf, _A_matrix, dist_coeffs, rotation_vector, translation_vector, false, cv::SOLVEPNP_AP3P);

        }else
        {//More than one sign board, the strategy is EPnPRanSan -> interRanSan
            if(model_points.size()>6)
            {
                Mat opoints, ipoints;
                bool s = inputs(model_points, image_points, opoints, ipoints);
                int model_pointes = 5;
                Mat rvec = Mat(3, 1, CV_64FC1);
                Mat tvec = Mat(3, 1, CV_64FC1);
                int ransac_kernel_method = SOLVEPNP_EPNP;

                double param1 = 8.0;                  // reprojection error
                double param2 = 0.99;                 // confidence
                int param3 = 100;                     // number maximum iterations
                Mat _local_model(3, 2, CV_64FC1);
                Mat _mask_local_inliers(1, opoints.rows, CV_8UC1);
                Ptr<PointSetRegistrator::Callback> cb; // pointer to callback
                cb = makePtr<PnPRansacCallback>( _A_matrix, dist_coeffs, ransac_kernel_method, false, rvec, tvec);

                int result = createRANSACPointSetRegistrator(cb, model_pointes,
                    param1, param2, param3)->run(opoints, ipoints, _local_model, _mask_local_inliers);

                if (result == 1)
                {
                    selrightpoint(model_points, image_points, covarmatrix, _mask_local_inliers);
                }
            }

            std::vector<cv::Point3d> shuffModel_points;
            std::vector<cv::Point2d> shuffImage_points;
            std::vector<int> shuffnum;
            shuffPoints(shuffModel_points, shuffImage_points, shuffnum, model_points, image_points);
            shuffnums=shuffnum;

//            cv::Mat inliers;
//            std::vector<cv::Point3f> model_pointsf;
//            std::vector<cv::Point2f> image_pointsf;
//            transformDouble3ToFloat3(model_pointsf, shuffModel_points);
//            transformDouble2ToFloat2(image_pointsf, shuffImage_points);


            bearingVectors_t image_pointes;
            points_t model_pointes;
            cov3_mats_t covMats;
            cov3_mat_t covMat;
            cv::Point3d img_vector;
            double xv,yv,module ;
            double VTVmid;
            cv::Mat VVT = cv::Mat::zeros(3, 3, CV_64FC1);
            cv::Mat Exx;
            cv::Mat jands;
            for(int chnum = 0;chnum != shuffImage_points.size();++chnum)
            {
                //构建3d坐标
                jands = cv::Mat::eye(3, 3, CV_64FC1);
                model_pointes.push_back(point_t(shuffModel_points.at(chnum).x, shuffModel_points.at(chnum).y, shuffModel_points.at(chnum).z));
                //构建2d向量
                xv = (shuffImage_points.at(chnum).x - _A_matrix.at<double>(0, 2))/_A_matrix.at<double>(0, 0);
                yv = (shuffImage_points.at(chnum).y - _A_matrix.at<double>(1, 2))/_A_matrix.at<double>(0, 0);
                module = sqrt(xv * xv + yv *yv + 1);
                img_vector.x = xv/module;
                img_vector.y = yv/module;
                img_vector.z = 1/module;
                image_pointes.push_back(bearingVector_t(img_vector.x,img_vector.y,img_vector.z));
                //构建cov
                Exx = A_matrix.inv() * covarmatrix.at(shuffnum.at(chnum)) * A_matrix.inv();
                VTVmid = img_vector.x * img_vector.x;
                VVT.at<double>(0, 0) = VTVmid;
                VTVmid = img_vector.y * img_vector.y;
                VVT.at<double>(1, 1) = VTVmid;
                VTVmid = img_vector.z * img_vector.z;
                VVT.at<double>(2, 2) = VTVmid;
                VTVmid = img_vector.x * img_vector.y;
                VVT.at<double>(0, 1) = VTVmid;
                VVT.at<double>(1, 0) = VTVmid;
                VTVmid = img_vector.x * img_vector.z;
                VVT.at<double>(0, 2) = VTVmid;
                VVT.at<double>(2, 0) = VTVmid;
                VTVmid = img_vector.y * img_vector.z;
                VVT.at<double>(1, 2) = VTVmid;
                VVT.at<double>(2, 1) = VTVmid;
                jands = (jands - VVT)/module;
                jands = jands * Exx * jands.t();
                covMat << jands.at<double>(0, 0), jands.at<double>(0, 1), jands.at<double>(0, 2),
                        jands.at<double>(1, 0), jands.at<double>(1, 1), jands.at<double>(1, 2),
                        jands.at<double>(2, 0), jands.at<double>(2, 1), jands.at<double>(2, 2);
                covMats.push_back(covMat);
            }

            mlpnp::mlpnp_main(image_pointes, model_pointes, covMats, shuffnum, solution, dxs);

        }

    }else
    {
        if(!(signNum>5))
        {//Just one sign board, the strategy is AP3P-> init interRanSan

            std::vector<cv::Point3d> shuffModel_points;
            std::vector<cv::Point2d> shuffImage_points;
            std::vector<int> shuffnum;
            shuffPoints(shuffModel_points, shuffImage_points, shuffnum, model_points, image_points);
            shuffnums=shuffnum;

            std::vector<cv::Point3f> model_pointsf;
            std::vector<cv::Point2f> image_pointsf;
            transformDouble3ToFloat3(model_pointsf, shuffModel_points);
            transformDouble2ToFloat2(image_pointsf, shuffImage_points);

//            transformDouble3ToFloat3(model_pointsf, model_points);
//            transformDouble2ToFloat2(image_pointsf, image_points);

            cv::solvePnP(model_pointsf, image_pointsf, _A_matrix, dist_coeffs, rotation_vector, translation_vector, false, cv::SOLVEPNP_ITERATIVE);

        }else
        {//More than one sign board, the strategy is EPnPRanSan -> interRanSan
            if(model_points.size()>6)
            {
                Mat opoints, ipoints;
                bool s = inputs(model_points, image_points, opoints, ipoints);
                int model_pointes = 5;
                Mat rvec = Mat(3, 1, CV_64FC1);
                Mat tvec = Mat(3, 1, CV_64FC1);
                int ransac_kernel_method = SOLVEPNP_EPNP;

                double param1 = 8.0;                  // reprojection error
                double param2 = 0.99;                 // confidence
                int param3 = 100;                     // number maximum iterations
                Mat _local_model(3, 2, CV_64FC1);
                Mat _mask_local_inliers(1, opoints.rows, CV_8UC1);
                Ptr<PointSetRegistrator::Callback> cb; // pointer to callback
                cb = makePtr<PnPRansacCallback>( _A_matrix, dist_coeffs, ransac_kernel_method, false, rvec, tvec);

                int result = createRANSACPointSetRegistrator(cb, model_pointes,
                    param1, param2, param3)->run(opoints, ipoints, _local_model, _mask_local_inliers);

                if (result == 1)
                {
                    selrightpoint(model_points, image_points, covarmatrix, _mask_local_inliers);
                }
            }

            std::vector<cv::Point3d> shuffModel_points;
            std::vector<cv::Point2d> shuffImage_points;
            std::vector<int> shuffnum;
            shuffPoints(shuffModel_points, shuffImage_points, shuffnum, model_points, image_points);
            shuffnums=shuffnum;

//            cv::Mat inliers;
//            std::vector<cv::Point3f> model_pointsf;
//            std::vector<cv::Point2f> image_pointsf;
//            transformDouble3ToFloat3(model_pointsf, shuffModel_points);
//            transformDouble2ToFloat2(image_pointsf, shuffImage_points);

            bearingVectors_t image_pointes;
            points_t model_pointes;
            cov3_mats_t covMats;
            cov3_mat_t covMat;
            cv::Point3d img_vector;
            double xv,yv,module ;
            double VTVmid;
            cv::Mat VVT = cv::Mat::zeros(3, 3, CV_64FC1);
            cv::Mat Exx;
            cv::Mat jands;
            for(int chnum = 0;chnum != shuffImage_points.size();++chnum)
            {
                //构建3d坐标
                jands = cv::Mat::eye(3, 3, CV_64FC1);
                model_pointes.push_back(point_t(shuffModel_points.at(chnum).x, shuffModel_points.at(chnum).y, shuffModel_points.at(chnum).z));
                //构建2d向量
                xv = (shuffImage_points.at(chnum).x - _A_matrix.at<double>(0, 2))/_A_matrix.at<double>(0, 0);
                yv = (shuffImage_points.at(chnum).y - _A_matrix.at<double>(1, 2))/_A_matrix.at<double>(0, 0);
                module = sqrt(xv * xv + yv *yv + 1);
                img_vector.x = xv/module;
                img_vector.y = yv/module;
                img_vector.z = 1/module;
                image_pointes.push_back(bearingVector_t(img_vector.x,img_vector.y,img_vector.z));
                //构建cov
                Exx = A_matrix.inv() * covarmatrix.at(shuffnum.at(chnum)) * A_matrix.inv();
                VTVmid = img_vector.x * img_vector.x;
                VVT.at<double>(0, 0) = VTVmid;
                VTVmid = img_vector.y * img_vector.y;
                VVT.at<double>(1, 1) = VTVmid;
                VTVmid = img_vector.z * img_vector.z;
                VVT.at<double>(2, 2) = VTVmid;
                VTVmid = img_vector.x * img_vector.y;
                VVT.at<double>(0, 1) = VTVmid;
                VVT.at<double>(1, 0) = VTVmid;
                VTVmid = img_vector.x * img_vector.z;
                VVT.at<double>(0, 2) = VTVmid;
                VVT.at<double>(2, 0) = VTVmid;
                VTVmid = img_vector.y * img_vector.z;
                VVT.at<double>(1, 2) = VTVmid;
                VVT.at<double>(2, 1) = VTVmid;
                jands = (jands - VVT)/module;
                jands = jands * Exx * jands.t();
                covMat << jands.at<double>(0, 0), jands.at<double>(0, 1), jands.at<double>(0, 2),
                        jands.at<double>(1, 0), jands.at<double>(1, 1), jands.at<double>(1, 2),
                        jands.at<double>(2, 0), jands.at<double>(2, 1), jands.at<double>(2, 2);
                covMats.push_back(covMat);
            }

            mlpnp::mlpnp_main(image_pointes, model_pointes, covMats, shuffnum, solution, dxs);

        }


    }


        if(signNum == 4||signNum == 5)
        {
            Rodrigues(rotation_vector,_R_matrix);
            _t_matrix = translation_vector;
        }
        else
        {
            dx.at<double>(0) = dxs[3];
            dx.at<double>(1) = dxs[4];
            dx.at<double>(2) = dxs[5];
            _R_matrix.at<double>(0, 0) = solution(0, 0);
            _R_matrix.at<double>(0, 1) = solution(0, 1);
            _R_matrix.at<double>(0, 2) = solution(0, 2);
            _t_matrix.at<double>(0) = solution(0, 3);
            _R_matrix.at<double>(1, 0) = solution(1, 0);
            _R_matrix.at<double>(1, 1) = solution(1, 1);
            _R_matrix.at<double>(1, 2) = solution(1, 2);
            _t_matrix.at<double>(1) = solution(1, 3);
            _R_matrix.at<double>(2, 0) = solution(2, 0);
            _R_matrix.at<double>(2, 1) = solution(2, 1);
            _R_matrix.at<double>(2, 2) = solution(2, 2);
            _t_matrix.at<double>(2) = solution(2, 3);
            Rodrigues(_R_matrix, rotation_vector);
            translation_vector = _t_matrix;
        }


    cout <<"------------------------"<<endl;
    cout << _t_matrix<<"\n"<<_R_matrix<<endl;

    set_P_matrix(_P_matrix, _R_matrix, _t_matrix);

    R = _R_matrix.clone();
    T = _t_matrix.clone();

    //3.Get the relative position with the car
    bool isNegative=false;
    for(int i=0; i<model.size();i++)
    {
        cv::Point3d cameraPoint;
        cv::Point2d imagePoint  = backproject3DPoint(cameraPoint, _A_matrix, _P_matrix, model[i]);
        /*
         *z coordinate of the cameraPoint must be larger than 3.0, y must be between -10.0 10.0m, x must be between -50.0 50.0 m
         */
//        if((cameraPoint.z < 3.0 || cameraPoint.z >160) // mazhou data jingcheng
//          || !(cameraPoint.y > -20.0 &&  cameraPoint.y < 20.0)
//          || !(cameraPoint.x > -50.0 && cameraPoint.x< 50.0))
        if((cameraPoint.z < 3.0 || cameraPoint.z >200)
          || !(cameraPoint.y > -30.0 &&  cameraPoint.y < 30.0) // had data jingcheng
          || !(cameraPoint.x > -40.0 && cameraPoint.x< 20.0))
        {
            isNegative = true;
            break;
        }
        cameraPoints.push_back(cameraPoint);
        //backImagePoints.push_back(imagePoint);
    }

    if(isNegative)
    {
        cameraPoints.clear();
        //backImagePoints.clear();
        R.release();
        T.release();
        return false;
    }

    cv::projectPoints(model,rotation_vector,translation_vector,_A_matrix,dist_coeffs,backImagePoints);

    return true;

}

/************************************************************************
 * function: Get distances about x,y,z coordinate with solvePnP method,
 * parameter:
 *          rtk3dPoints: Output, x,y,z distances between points and the rtk with solvePnP
 *          backImagePoints: Output, the points in the image of the back projection
 *          sign_points: Input, the input signboard longitude(02), latitude(02), altitude eg: (116.22549587 40.08148139 40.2)
 *          image_points: Input, the signboard four corner points(pixel) in the undistort image(200, 301)
 *          _A_matrix: instrinsic matrix of the camera
 *          dist_coeffs: distortion matrix
 *          rtk2Camera_R_matrix: The rotation matrix from rtk to camera
 *          rtk2Camera_t_matrix: The translation matrix from rtk to camera
 * return ok is true, or false
 * ***********************************************************************/
bool getDxDyDzWithPnP(std::vector<cv::Point3d>& rtk3dPoints, std::vector<cv::Point2d>& backImagePoints, cv::Point3d& cameraPosGauss, std::vector<cv::Point3d> sign_points, std::vector<cv::Point2d> image_points, cv::Mat _A_matrix, cv::Mat dist_coeffs, cv::Mat rtk2Camera_R_matrix, cv::Mat rtk2Camera_t_matrix, std::vector<cv::Mat> covarmatrix, cv::Mat &dx, int ind)
{
    if(rtk3dPoints.empty() || backImagePoints.empty())
    {
        rtk3dPoints.clear();
        backImagePoints.clear();
    }

    //1.Transform sign_points to relative position
    std::vector<cv::Point3d> model_points;
    std::vector<int> shuffnum;
   //dPoint originalPos=GaussProjCal_zhao({sign_points[ind].x, sign_points[ind].y});
    dPoint originalPos=dPoint{sign_points[ind].x, sign_points[ind].y};
    double originalGaussz = sign_points[ind].z;
    for(int i=0; i<sign_points.size(); i++)
    {
        //dPoint gaussPos=GaussProjCal_zhao({sign_points[i].x, sign_points[i].y});
        dPoint gaussPos=dPoint{sign_points[i].x, sign_points[i].y};
        double gaussz = sign_points[i].z;
        model_points.push_back(cv::Point3d(gaussPos.x - originalPos.x, gaussPos.y - originalPos.y, gaussz - originalGaussz));
    }


    std::vector<cv::Point3d> bestCameraPoints;
    std::vector<cv::Point2d> bestBackImgPoints;

    cv::Mat R, _R;
    cv::Mat T, _T;

    float  bestImgErr=100000.0;
    int times = 50;

    for(int i=0; i<times; i++)
    {
        std::vector<cv::Point3d> curCameraPoints;
        std::vector<cv::Point2d> curBackImgPoints;
        //get cameraPoints and backImagePoints

        if(PnP(curCameraPoints, curBackImgPoints, _R, _T, shuffnum, model_points, image_points, _A_matrix, dist_coeffs, covarmatrix, dx, i))
        {
            //get retImgErr
            float curImgErr = DistPoint(curBackImgPoints, image_points);

            if(curImgErr<bestImgErr)
            {
                //record bestImgErr, cameraPoints and backImagePoints
                bestImgErr = curImgErr;
                bestCameraPoints.assign(curCameraPoints.begin(), curCameraPoints.end());
                bestBackImgPoints.assign(curBackImgPoints.begin(), curBackImgPoints.end());
                R = _R.clone();
                T = _T.clone();

            }

        }
    }

    //****** confidence level solution ******//
//    DLT dlt;
// //    cv::Mat Tdlt = cv::Mat::zeros(3, 1 ,CV_64FC1);
//    bool m = dlt.sloveModel(image_points, model_points, _A_matrix, Tdlt, covarmatrix);
//    cout<<Tdlt<<endl;//是误差波动
//    if(Tdlt.at<double>(2) > maxerror/(1-minconfident)||Tdlt.at<double>(2) < 1e-10)
//        confident = 0;
//    else
//        confident = 1-Tdlt.at<double>(2)/(maxerror/(1-minconfident));
//    //****** confidence level end ******//

    if(bestCameraPoints.empty() || bestBackImgPoints.empty())
    {
        return false;
    }

    backImagePoints.assign(bestBackImgPoints.begin(),bestBackImgPoints.end());


    for(int i=0; i<bestCameraPoints.size();i++)
    {

        cv::Point3d backCamera2WorldPoint = backprojectCamera2WorldPoint(bestCameraPoints[i], rtk2Camera_R_matrix, rtk2Camera_t_matrix);
        rtk3dPoints.push_back(backCamera2WorldPoint);

    }


    if(!R.empty() && !T.empty())
    {
        cv::Point3d cameraPos{0,0,0};
        cameraPosGauss = getPositionRefSignOri(cameraPos, R, T);
        cameraPosGauss.x += sign_points[ind].x;
        cameraPosGauss.y += sign_points[ind].y;
        cameraPosGauss.z += sign_points[ind].z;
    }



    return true;

}


bool shuffPoints(std::vector<cv::Point3d>& shuffSeleSign, std::vector<cv::Point2d>& shuffSeleImg, std::vector<int>& shuffnum, std::vector<cv::Point3d> seleSign, std::vector<cv::Point2d> seleImg)
{
    shuffSeleSign.clear();
    shuffSeleImg.clear();


    std::vector<int> vi;
    for(int i=0; i<seleSign.size(); i++)
    {
        vi.push_back(i);
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(vi.begin(), vi.end(), std::default_random_engine(seed));


    for(int j=0; j<seleSign.size();j++)
    {
        shuffSeleSign.push_back(seleSign[vi[j]]);
        shuffSeleImg.push_back(seleImg[vi[j]]);
        shuffnum.push_back(vi[j]);

    }
    return true;

}


bool transformDouble3ToFloat3(std::vector<cv::Point3f>& point3f, std::vector<cv::Point3d> point3ds)
{
    if(point3ds.size()<1)
        return false;

    for(int i=0; i<point3ds.size();i++)
    {
        point3f.push_back(cv::Point3f{float(point3ds[i].x), float(point3ds[i].y), float(point3ds[i].z)});
    }
    return true;

}

bool transformDouble2ToFloat2(std::vector<cv::Point2f>& point2f, std::vector<cv::Point2d> point2ds)
{
    if(point2ds.size()<1)
        return false;

    for(int i=0; i<point2ds.size();i++)
    {
        point2f.push_back(cv::Point2f{float(point2ds[i].x), float(point2ds[i].y)});
    }
    return true;

}


// How to use
//cv::Point3f sign0tocam2rtk_position
//std::vector<cv::Point3f> sign_points
//std::vector<cv::Point2f> image_points
//std::string confFile
//getDxDyDzWithPnP(sign0tocam2rtk_position, sign_points, image_points, confFile);
