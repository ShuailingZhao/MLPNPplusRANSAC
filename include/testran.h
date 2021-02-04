#ifndef TESTRAN_H
#define TESTRAN_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

namespace cv
{
    class CV_EXPORTS PointSetRegistrator : public Algorithm
    {
    public:
        class CV_EXPORTS Callback
        {
        public:
            virtual ~Callback() {}
            virtual int runKernel(InputArray m1, InputArray m2, OutputArray model) const = 0;
            virtual void computeError(InputArray m1, InputArray m2, InputArray model, OutputArray err) const = 0;
            virtual bool checkSubset(InputArray, InputArray, int) const { return true; }
        };

        virtual void setCallback(const Ptr<PointSetRegistrator::Callback>& cb) = 0;
        virtual bool run(InputArray m1, InputArray m2, OutputArray model, OutputArray mask) const = 0;
    };
    CV_EXPORTS Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& cb, int modelPoints, double threshold, double confidence = 0.99, int maxIters = 1000);
    CV_EXPORTS Ptr<PointSetRegistrator> createLMeDSPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& cb, int modelPoints, double confidence = 0.99, int maxIters = 1000);

    class PnPRansacCallback CV_FINAL : public PointSetRegistrator::Callback
    {

    public:

        PnPRansacCallback(Mat _cameraMatrix=Mat(3,3,CV_64F), Mat _distCoeffs=Mat(4,1,CV_64F), int _flags=SOLVEPNP_ITERATIVE,
                bool _useExtrinsicGuess=false, Mat _rvec=Mat(), Mat _tvec=Mat() )
            : cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), flags(_flags), useExtrinsicGuess(_useExtrinsicGuess),
              rvec(_rvec), tvec(_tvec) {}

        /* Pre: True */
        /* Post: compute _model with given points and return number of found models */
        int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const CV_OVERRIDE
        {
            Mat opoints = _m1.getMat(), ipoints = _m2.getMat();

            bool correspondence = solvePnP( _m1, _m2, cameraMatrix, distCoeffs,
                                                rvec, tvec, useExtrinsicGuess, flags );

            Mat _local_model;
            hconcat(rvec, tvec, _local_model);
            _local_model.copyTo(_model);

            return correspondence;
        }

        /* Pre: True */
        /* Post: fill _err with projection errors */
        void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const CV_OVERRIDE
        {

            Mat opoints = _m1.getMat(), ipoints = _m2.getMat(), model = _model.getMat();

            int i, count = opoints.checkVector(3);
            Mat _rvec = model.col(0);
            Mat _tvec = model.col(1);


            Mat projpoints(count, 2, CV_32FC1);
            projectPoints(opoints, _rvec, _tvec, cameraMatrix, distCoeffs, projpoints);

            const Point2f* ipoints_ptr = ipoints.ptr<Point2f>();
            const Point2f* projpoints_ptr = projpoints.ptr<Point2f>();

            _err.create(count, 1, CV_32FC1);
            float* err = _err.getMat().ptr<float>();

            for ( i = 0; i < count; ++i)
                err[i] = (float)norm( Matx21f(ipoints_ptr[i] - projpoints_ptr[i]), NORM_L2SQR );

        }


        Mat cameraMatrix;
        Mat distCoeffs;
        int flags;
        bool useExtrinsicGuess;
        Mat rvec;
        Mat tvec;
    };
}

#endif // TESTRAN_H
