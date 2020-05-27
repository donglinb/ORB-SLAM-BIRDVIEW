#ifndef ODOM_G2O_TYPE_H_
#define ODOM_G2O_TYPE_H_

// #include "Frame.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <eigen3/Eigen/Core>

namespace ORB_SLAM2
{

using namespace Eigen;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

class VertexSE3Quat : public g2o::BaseVertex<6, g2o::SE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	bool read(std::istream& is){return false;}
	bool write(std::ostream& os) const{return false;}
	virtual void setToOriginImpl()
	{
		_estimate = g2o::SE3Quat();
	}
	virtual void oplusImpl(const double* update_)
	{
		Eigen::Map<const Vector6d> update(update_);
		setEstimate(g2o::SE3Quat::exp(update)*estimate());
	}
};

class EdgeSE3ProjectXYZOnlyPoseQuat: public g2o::BaseUnaryEdge<2, Vector2d, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	EdgeSE3ProjectXYZOnlyPoseQuat(){}
	bool read(std::istream& is){return false;}
	bool write(std::ostream& os) const{return false;}

	void computeError()
	{
		const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
		Vector3d p = v->estimate().map(Xw);
		_error = Vector2d(_measurement)-cam_project(p);
	}

	bool isDepthPositive()
	{
		const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
		return (v->estimate().map(Xw))(2)>0.0;
	}

	virtual void linearizeOplus();

	Vector2d cam_project(const Vector3d & trans_xyz) const;
	Matrix3d skew(Vector3d phi);

	Vector3d Xw;
	double fx, fy, cx, cy;
};

class EdgeSE3ProjectXYZ2XYZOnlyPoseQuat: public g2o::BaseUnaryEdge<3, Vector3d, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	EdgeSE3ProjectXYZ2XYZOnlyPoseQuat(){}
	bool read(std::istream& is){return false;}
	bool write(std::ostream& os) const{return false;}

	void computeError()
	{
		const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
		Vector3d p = v->estimate().map(Xw);
		_error = Xc-p;
	}

	virtual void linearizeOplus();

	Matrix3d skew(Vector3d phi);

	Vector3d Xw,Xc;
};

class EdgePointTransformSE3Quat :public g2o::BaseBinaryEdge<3, Vector3d, VertexSE3Quat, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	virtual bool read(std::istream& is) { return false; }
	virtual bool write(std::ostream& os) const { return false; }
	virtual void computeError()
	{
		//pose1:reference frame; pose2:current frame
		g2o::SE3Quat pose1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
		g2o::SE3Quat pose2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
		g2o::SE3Quat pose21 = pose2 * pose1.inverse();
		//_error = pc2 - (pose21.rotation_matrix()*pc1+pose21.translation());
		_error = pc2 - pose21 * pc1;
	}
	virtual void linearizeOplus();

	Matrix3d skew(Vector3d phi);

	Vector3d pc1,pc2;
};

class EdgeSE3ProjectXYZ2UVQuat :public g2o::BaseBinaryEdge<2, Vector2d, g2o::VertexSBAPointXYZ, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	virtual bool read(std::istream& is) { return false; }
	virtual bool write(std::ostream& os) const { return false; }
	virtual void computeError()
	{
		g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
		Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

		Vector3d p = pose * point;
		_error = Vector2d(_measurement) - cam_project(p);
	}
	virtual void linearizeOplus();

	Matrix3d skew(Vector3d phi);
	Vector2d cam_project(const Vector3d & trans_xyz) const;
	bool isDepthPositive();
	
	double fx, fy, cx, cy;
};

class EdgeSE3ProjectXYZ2XYZQuat :public g2o::BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	virtual bool read(std::istream& is) { return false; }
	virtual bool write(std::ostream& os) const { return false; }
	virtual void computeError()
	{
		g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
		Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

		Vector3d p = pose * point;
		_error = Vector3d(_measurement) - p;
	}
	virtual void linearizeOplus();

	Matrix3d skew(Vector3d phi);
};

//templete params: BaseBinaryEdge<ErrorDimension,MeasurementType,Vertex1,Vertex2>
class EdgeSE3Quat : public g2o::BaseBinaryEdge<6, g2o::SE3Quat, VertexSE3Quat, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	bool read(std::istream& is){return false;}
	bool write(std::ostream& os) const{return false;}
	virtual void computeError()
	{
		g2o::SE3Quat v1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
		g2o::SE3Quat v2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	//   _error = (_measurement.inverse()*v1.inverse()*v2).log();
		_error = (_measurement.inverse()*v1*v2.inverse()).log();
	}
	virtual void linearizeOplus();
	Matrix6d JRInv(Vector6d e);
	Matrix3d skew(Vector3d phi);
};

}  //namespace ORB_SLAM2

#endif  //ODOM_G2O_TYPE_H_
