/******************************************************************************
 * Author:   Laurent Kneip                                                    *
 * Contact:  kneip.laurent@gmail.com                                          *
 * License:  Copyright (c) 2013 Laurent Kneip, ANU. All rights reserved.      *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 * * Redistributions of source code must retain the above copyright           *
 *   notice, this list of conditions and the following disclaimer.            *
 * * Redistributions in binary form must reproduce the above copyright        *
 *   notice, this list of conditions and the following disclaimer in the      *
 *   documentation and/or other materials provided with the distribution.     *
 * * Neither the name of ANU nor the names of its contributors may be         *
 *   used to endorse or promote products derived from this software without   *
 *   specific prior written permission.                                       *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"*
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE  *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE *
 * ARE DISCLAIMED. IN NO EVENT SHALL ANU OR THE CONTRIBUTORS BE LIABLE        *
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL *
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER *
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT         *
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY  *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF     *
 * SUCH DAMAGE.                                                               *
 ******************************************************************************/


#include <Eigen/NonLinearOptimization>
#include <Eigen/NumericalDiff>
#include <Eigen/StdVector>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "main.hpp"

using namespace cays;
using namespace type;
/////////////////////
// MLPnP
/////////////////////
void
mlpnp::mlpnp_main(
	const bearingVectors_t& f,
	const points_t& p,
	const cov3_mats_t& covMats,
	const std::vector<int>& indices,
    transformation_t& result, Eigen::VectorXd& dx)
{
	
	size_t numberCorrespondences = indices.size();
	assert(numberCorrespondences > 5);

	bool planar = false;
	// compute the nullspace of all vectors
	std::vector<Eigen::MatrixXd> nullspaces(numberCorrespondences);
	Eigen::MatrixXd points3(3, numberCorrespondences);
    points_t points3v(numberCorrespondences);
    points4_t points4v(numberCorrespondences);
	for (size_t i = 0; i < numberCorrespondences; i++)
	{
		bearingVector_t f_current = f[indices[i]];
        points3.col(i) = p[indices[i]];
		// nullspace of right vector
		Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner>
			svd_f(f_current.transpose(), Eigen::ComputeFullV);
		nullspaces[i] = svd_f.matrixV().block(0, 1, 3, 2);
        points3v[i] = p[indices[i]];
	}

	//////////////////////////////////////
	// 1. test if we have a planar scene
	//////////////////////////////////////

	Eigen::Matrix3d planarTest = points3*points3.transpose();
    Eigen::FullPivHouseholderQR<Eigen::Matrix3d> rankTest(planarTest);
	//int r, c;
	//double minEigenVal = abs(eigen_solver.eigenvalues().real().minCoeff(&r, &c));
	Eigen::Matrix3d eigenRot;
	eigenRot.setIdentity();

	// if yes -> transform points to new eigen frame
	//if (minEigenVal < 1e-3 || minEigenVal == 0.0)
	//rankTest.setThreshold(1e-10);
	if (rankTest.rank() == 2)
	{
		planar = true;
		// self adjoint is faster and more accurate than general eigen solvers
		// also has closed form solution for 3x3 self-adjoint matrices
		// in addition this solver sorts the eigenvalues in increasing order
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(planarTest);
		eigenRot = eigen_solver.eigenvectors().real();
		eigenRot.transposeInPlace();
		for (size_t i = 0; i < numberCorrespondences; i++)
			points3.col(i) = eigenRot * points3.col(i);
	}
	//////////////////////////////////////
	// 2. stochastic model
	//////////////////////////////////////
	Eigen::SparseMatrix<double> P(2 * numberCorrespondences,
		2 * numberCorrespondences);
	bool use_cov = false;
	P.setIdentity(); // standard

	// if we do have covariance information 
	// -> fill covariance matrix
	if (covMats.size() == numberCorrespondences)
	{
		use_cov = true;
		int l = 0;
		for (size_t i = 0; i < numberCorrespondences; ++i)
		{
			// invert matrix
			cov2_mat_t temp = nullspaces[i].transpose() * covMats[i] * nullspaces[i];
			temp = temp.inverse().eval();
			P.coeffRef(l, l) = temp(0, 0);
			P.coeffRef(l, l + 1) = temp(0, 1);
			P.coeffRef(l + 1, l) = temp(1, 0);
			P.coeffRef(l + 1, l + 1) = temp(1, 1);
			l += 2;
		}
	}

	//////////////////////////////////////
	// 3. fill the design matrix A
	//////////////////////////////////////
	const int rowsA = 2 * numberCorrespondences;
	int colsA = 12;
	Eigen::MatrixXd A;
	if (planar)
	{
		colsA = 9;
        A = Eigen::MatrixXd(rowsA, 9);
	}
	else
        A = Eigen::MatrixXd(rowsA, 12);
	A.setZero();

	// fill design matrix
	if (planar)
	{
		for (size_t i = 0; i < numberCorrespondences; ++i)
		{
			point_t pt3_current = points3.col(i);

			// r12
			A(2 * i, 0) = nullspaces[i](0, 0) * pt3_current[1];
			A(2 * i + 1, 0) = nullspaces[i](0, 1) * pt3_current[1];
			// r13
			A(2 * i, 1) = nullspaces[i](0, 0) * pt3_current[2];
			A(2 * i + 1, 1) = nullspaces[i](0, 1) * pt3_current[2];
			// r22
			A(2 * i, 2) = nullspaces[i](1, 0) * pt3_current[1];
			A(2 * i + 1, 2) = nullspaces[i](1, 1)* pt3_current[1];
			// r23
			A(2 * i, 3) = nullspaces[i](1, 0) * pt3_current[2];
			A(2 * i + 1, 3) = nullspaces[i](1, 1) * pt3_current[2];
			// r32
			A(2 * i, 4) = nullspaces[i](2, 0) * pt3_current[1];
			A(2 * i + 1, 4) = nullspaces[i](2, 1) * pt3_current[1];
			// r33
			A(2 * i, 5) = nullspaces[i](2, 0) * pt3_current[2];
			A(2 * i + 1, 5) = nullspaces[i](2, 1) * pt3_current[2];
			// t1
			A(2 * i, 6) = nullspaces[i](0, 0);
			A(2 * i + 1, 6) = nullspaces[i](0, 1);
			// t2
			A(2 * i, 7) = nullspaces[i](1, 0);
			A(2 * i + 1, 7) = nullspaces[i](1, 1);
			// t3
			A(2 * i, 8) = nullspaces[i](2, 0);
			A(2 * i + 1, 8) = nullspaces[i](2, 1);
		}
	}
	else
	{
		for (size_t i = 0; i < numberCorrespondences; ++i)
		{
			point_t pt3_current = points3.col(i);

			// r11
			A(2 * i, 0) = nullspaces[i](0, 0) * pt3_current[0];
			A(2 * i + 1, 0) = nullspaces[i](0, 1) * pt3_current[0];
			// r12
			A(2 * i, 1) = nullspaces[i](0, 0) * pt3_current[1];
			A(2 * i + 1, 1) = nullspaces[i](0, 1) * pt3_current[1];
			// r13
			A(2 * i, 2) = nullspaces[i](0, 0) * pt3_current[2];
			A(2 * i + 1, 2) = nullspaces[i](0, 1) * pt3_current[2];
			// r21
			A(2 * i, 3) = nullspaces[i](1, 0) * pt3_current[0];
			A(2 * i + 1, 3) = nullspaces[i](1, 1) * pt3_current[0];
			// r22
			A(2 * i, 4) = nullspaces[i](1, 0) * pt3_current[1];
			A(2 * i + 1, 4) = nullspaces[i](1, 1)* pt3_current[1];
			// r23
			A(2 * i, 5) = nullspaces[i](1, 0) * pt3_current[2];
			A(2 * i + 1, 5) = nullspaces[i](1, 1) * pt3_current[2];
			// r31
			A(2 * i, 6) = nullspaces[i](2, 0) * pt3_current[0];
			A(2 * i + 1, 6) = nullspaces[i](2, 1) * pt3_current[0];
			// r32
			A(2 * i, 7) = nullspaces[i](2, 0) * pt3_current[1];
			A(2 * i + 1, 7) = nullspaces[i](2, 1) * pt3_current[1];
			// r33
			A(2 * i, 8) = nullspaces[i](2, 0) * pt3_current[2];
			A(2 * i + 1, 8) = nullspaces[i](2, 1) * pt3_current[2];
			// t1
			A(2 * i, 9) = nullspaces[i](0, 0);
			A(2 * i + 1, 9) = nullspaces[i](0, 1);
			// t2
			A(2 * i, 10) = nullspaces[i](1, 0);
			A(2 * i + 1, 10) = nullspaces[i](1, 1);
			// t3
			A(2 * i, 11) = nullspaces[i](2, 0);
			A(2 * i + 1, 11) = nullspaces[i](2, 1);
		}
	}

	//////////////////////////////////////
	// 4. solve least squares
	//////////////////////////////////////
	Eigen::MatrixXd AtPA;
	if (use_cov)
		AtPA = A.transpose() * P * A; // setting up the full normal equations seems to be unstable
	else
		AtPA = A.transpose() * A;

	Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(AtPA, Eigen::ComputeFullV);
	Eigen::MatrixXd result1 = svd_A.matrixV().col(colsA - 1);

	////////////////////////////////
	// now we treat the results differently,
	// depending on the scene (planar or not)
	////////////////////////////////
	//transformation_t T_final;
	rotation_t Rout;
	translation_t tout;
	if (planar) // planar case
	{
		rotation_t tmp;
		// until now, we only estimated 
		// row one and two of the transposed rotation matrix
		tmp << 0.0, result1(0, 0), result1(1, 0),
			0.0, result1(2, 0), result1(3, 0),
			0.0, result1(4, 0), result1(5, 0);
		//double scale = 1 / sqrt(tmp.col(1).norm() * tmp.col(2).norm());
		// row 3
		tmp.col(0) = tmp.col(1).cross(tmp.col(2));
		tmp.transposeInPlace();

		double scale = 1.0 / std::sqrt(std::abs(tmp.col(1).norm()* tmp.col(2).norm()));
		// find best rotation matrix in frobenius sense
		Eigen::JacobiSVD<Eigen::MatrixXd> svd_R_frob(tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
		rotation_t Rout1 = svd_R_frob.matrixU() * svd_R_frob.matrixV().transpose();
		// test if we found a good rotation matrix
		if (Rout1.determinant() < 0)
			Rout1 *= -1.0;
		// rotate this matrix back using the eigen frame
		Rout1 = eigenRot.transpose() * Rout1;

		translation_t t = scale * translation_t(result1(6, 0), result1(7, 0), result1(8, 0));
		Rout1.transposeInPlace();
		Rout1 *= -1;
		if (Rout1.determinant() < 0.0)
			Rout1.col(2) *= -1;
		// now we have to find the best out of 4 combinations
		rotation_t R1, R2;
		R1.col(0) = Rout1.col(0); R1.col(1) = Rout1.col(1); R1.col(2) = Rout1.col(2);
		R2.col(0) = -Rout1.col(0); R2.col(1) = -Rout1.col(1); R2.col(2) = Rout1.col(2);

        std::vector<transformation_t,Eigen::aligned_allocator<transformation_t>> Ts(4);
		Ts[0].block<3, 3>(0, 0) = R1; Ts[0].block<3, 1>(0, 3) = t;
		Ts[1].block<3, 3>(0, 0) = R1; Ts[1].block<3, 1>(0, 3) = -t;
		Ts[2].block<3, 3>(0, 0) = R2; Ts[2].block<3, 1>(0, 3) = t;
		Ts[3].block<3, 3>(0, 0) = R2; Ts[3].block<3, 1>(0, 3) = -t;

        std::vector<double> normVal(4);
		for (int i = 0; i < 4; ++i)
		{
			point_t reproPt;
			double norms = 0.0;
			for (int p = 0; p < 6; ++p)
			{
				reproPt = Ts[i].block<3, 3>(0, 0)*points3v[p] + Ts[i].block<3, 1>(0, 3);
				reproPt = reproPt / reproPt.norm();
				norms += (1.0 - reproPt.transpose()*f[indices[p]]);
			}
			normVal[i] = norms;
		}
		std::vector<double>::iterator
			findMinRepro = std::min_element(std::begin(normVal), std::end(normVal));
		int idx = std::distance(std::begin(normVal), findMinRepro);
		Rout = Ts[idx].block<3, 3>(0, 0);
		tout = Ts[idx].block<3, 1>(0, 3);
	}
	else // non-planar
	{
		rotation_t tmp;
		tmp << result1(0, 0), result1(3, 0), result1(6, 0),
			result1(1, 0), result1(4, 0), result1(7, 0),
			result1(2, 0), result1(5, 0), result1(8, 0);
		// get the scale
		double scale = 1.0 / 
			std::pow(std::abs(tmp.col(0).norm() * tmp.col(1).norm()* tmp.col(2).norm()), 1.0 / 3.0);
	    //double scale = 1.0 / std::sqrt(std::abs(tmp.col(0).norm() * tmp.col(1).norm()));
		// find best rotation matrix in frobenius sense
		Eigen::JacobiSVD<Eigen::MatrixXd> svd_R_frob(tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Rout = svd_R_frob.matrixU() * svd_R_frob.matrixV().transpose();
		// test if we found a good rotation matrix
		if (Rout.determinant() < 0)
			Rout *= -1.0;
		// scale translation
		tout = Rout * (scale * translation_t(result1(9, 0), result1(10, 0), result1(11, 0)));

		// find correct direction in terms of reprojection error, just take the first 6 correspondences
        std::vector<double> error(2);
        std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> Ts(2);
		for (int s = 0; s < 2; ++s)
		{
			error[s] = 0.0;
			Ts[s] = Eigen::Matrix4d::Identity();
			Ts[s].block<3, 3>(0, 0) = Rout;
			if (s == 0)
				Ts[s].block<3, 1>(0, 3) = tout;
			else
				Ts[s].block<3, 1>(0, 3) = -tout;
			Ts[s] = Ts[s].inverse().eval();
			for (int p = 0; p < 6; ++p)
			{
				bearingVector_t v = Ts[s].block<3, 3>(0, 0)* points3v[p] + Ts[s].block<3, 1>(0, 3);
				v = v / v.norm();
				error[s] += (1.0 - v.transpose() * f[indices[p]]);
			}
		}
		if (error[0] < error[1])
			tout = Ts[0].block<3, 1>(0, 3);
		else
			tout = Ts[1].block<3, 1>(0, 3);
		Rout = Ts[0].block<3, 3>(0, 0);

	}

	//////////////////////////////////////
	// 5. gauss newton
	//////////////////////////////////////
//    rodrigues_t omega = rot2rodrigues(Rout);
    cv::Mat Rs = cv::Mat::zeros(3, 3, CV_64FC1);
    Rs.at<double>(0, 0) = Rout(0, 0);
    Rs.at<double>(0, 1) = Rout(0, 1);
    Rs.at<double>(0, 2) = Rout(0, 2);
    Rs.at<double>(1, 0) = Rout(1, 0);
    Rs.at<double>(1, 1) = Rout(1, 1);
    Rs.at<double>(1, 2) = Rout(1, 2);
    Rs.at<double>(2, 0) = Rout(2, 0);
    Rs.at<double>(2, 1) = Rout(2, 1);
    Rs.at<double>(2, 2) = Rout(2, 2);

    cv::Mat omega = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Rodrigues(Rs, omega);
	Eigen::VectorXd minx(6);
    minx[0] = omega.at<double>(0);
    minx[1] = omega.at<double>(1);
    minx[2] = omega.at<double>(2);
	minx[3] = tout[0];
	minx[4] = tout[1];
	minx[5] = tout[2];

    mlpnp::mlpnp_gn(minx,
        points3v, nullspaces, P, dx, use_cov);

//    if((minx[3] == tout[0])&&(minx[4] == tout[1])&&(minx[5] == tout[2]))
//    {
//        dx << 0,0,0,0,0,0;
//    }

//    Rout = rodrigues2rot(rodrigues_t(minx[0], minx[1], minx[2]));
    omega.at<double>(0) = minx[0];
    omega.at<double>(1) = minx[1];
    omega.at<double>(2) = minx[2];
    cv::Rodrigues(omega,Rs);
    Rout(0, 0) = Rs.at<double>(0, 0);
    Rout(0, 1) = Rs.at<double>(0, 1);
    Rout(0, 2) = Rs.at<double>(0, 2);
    Rout(1, 0) = Rs.at<double>(1, 0);
    Rout(1, 1) = Rs.at<double>(1, 1);
    Rout(1, 2) = Rs.at<double>(1, 2);
    Rout(2, 0) = Rs.at<double>(2, 0);
    Rout(2, 1) = Rs.at<double>(2, 1);
    Rout(2, 2) = Rs.at<double>(2, 2);

	tout = translation_t(minx[3], minx[4], minx[5]);
	// result inverse as opengv uses this convention
    result.block<3, 3>(0, 0) = Rout;
    result.block<3, 1>(0, 3) = tout;
}
