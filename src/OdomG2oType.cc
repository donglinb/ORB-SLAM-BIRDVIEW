#include"OdomG2oType.h"

namespace ORB_SLAM2
{

void EdgeSE3ProjectXYZOnlyPoseQuat::linearizeOplus() 
{
	const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
	Vector3d p = v->estimate().map(Xw);

  	double x = p[0], y = p[1], z = p[2];
	double z2 = z * z;

	Matrix<double, 2, 3> jacobian_e_p;
	jacobian_e_p << fx / z, 0, -fx * x / z2,
		0, fy / z, -fy * y / z2;
	jacobian_e_p = -jacobian_e_p;

	Matrix<double, 3, 6> jacobian_p_ksi;
	jacobian_p_ksi << -skew(p), Matrix3d::Identity();

	_jacobianOplusXi = jacobian_e_p * jacobian_p_ksi;
}
Vector2d EdgeSE3ProjectXYZOnlyPoseQuat::cam_project(const Vector3d & trans_xyz) const
{
	Vector2d res;
	res[0] = fx * trans_xyz[0] / trans_xyz[2] + cx;
	res[1] = fy * trans_xyz[1] / trans_xyz[2] + cy;
	return res;
}
Matrix3d EdgeSE3ProjectXYZOnlyPoseQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}


void EdgeSE3ProjectXYZ2XYZOnlyPoseQuat::linearizeOplus() 
{
	const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
	Vector3d p = v->estimate().map(Xw);

	Matrix<double, 3, 6> jacobian_p_ksi;
	jacobian_p_ksi << -skew(p), Matrix3d::Identity();

	_jacobianOplusXi = -jacobian_p_ksi;
}
Matrix3d EdgeSE3ProjectXYZ2XYZOnlyPoseQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}


Matrix3d EdgePointTransformSE3Quat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}
void EdgePointTransformSE3Quat::linearizeOplus()
{
	g2o::SE3Quat pose1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
	g2o::SE3Quat pose2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	g2o::SE3Quat pose21 = pose2 * pose1.inverse();
	// _error = pc2 - (pose21.rotation_matrix()*pc1+pose21.translation());
	//Vector3d pc2p = pose21.rotation_matrix()*pc1+pose21.translation();
	Vector3d pc2p = pose21 * pc1;

	Matrix<double, 3, 6> jacobian_p_ksi2;
	jacobian_p_ksi2 << -skew(pc2p), Matrix3d::Identity();

	Matrix<double, 3, 6> jacobian_p_ksi1;
	jacobian_p_ksi1 << -skew(pc1), Matrix3d::Identity();

	_jacobianOplusXi = pose21.rotation().toRotationMatrix()*jacobian_p_ksi1;
	_jacobianOplusXj = -jacobian_p_ksi2;
}


void EdgeSE3ProjectXYZ2UVQuat::linearizeOplus()
{
	g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

	Vector3d p = pose * point;

	double x = p[0], y = p[1], z = p[2];
	double z2 = z * z;

	Matrix<double, 2, 3> jacobian_e_p;
	jacobian_e_p << fx / z, 0, -fx * x / z2,
		0, fy / z, -fy * y / z2;
	jacobian_e_p = -jacobian_e_p;

	Matrix<double, 3, 6> jacobian_p_ksi;
	jacobian_p_ksi <<-skew(p), Matrix3d::Identity();

	_jacobianOplusXj = jacobian_e_p * jacobian_p_ksi;
	_jacobianOplusXi = jacobian_e_p * pose.rotation().toRotationMatrix();
}
Matrix3d EdgeSE3ProjectXYZ2UVQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}
Vector2d EdgeSE3ProjectXYZ2UVQuat::cam_project(const Vector3d & trans_xyz) const
{
	Vector2d res;
	res[0] = fx * trans_xyz[0] / trans_xyz[2] + cx;
	res[1] = fy * trans_xyz[1] / trans_xyz[2] + cy;
	return res;
}
bool EdgeSE3ProjectXYZ2UVQuat::isDepthPositive()
{
	g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

	Vector3d p = pose * point;

	return (p[2] > 0.0);
}



void EdgeSE3ProjectXYZ2XYZQuat::linearizeOplus()
{
	g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

	Vector3d p = pose * point;

	Matrix<double, 3, 6> jacobian_p_ksi;
	jacobian_p_ksi <<-skew(p), Matrix3d::Identity();

	_jacobianOplusXj = -jacobian_p_ksi;
	_jacobianOplusXi = -pose.rotation().toRotationMatrix();
}
Matrix3d EdgeSE3ProjectXYZ2XYZQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}


Matrix6d EdgeSE3Quat::JRInv(Vector6d e)
{
	Matrix6d J;
	J.block(0, 0, 3, 3) = skew(e.head(3));
	J.block(3, 3, 3, 3) = skew(e.head(3));
	J.block(3, 0, 3, 3) = skew(e.tail(3));
	J.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero();
	
	J = 0.5 * J + Matrix6d::Identity();
	return J;
}
void EdgeSE3Quat::linearizeOplus()
{
	g2o::SE3Quat v1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
	g2o::SE3Quat v2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
//  Sophus::SE3 e = _measurement.inverse() * v1.inverse() * v2;
	g2o::SE3Quat e = _measurement.inverse() * v1 * v2.inverse();
	Matrix6d J = JRInv(e.log());

//   _jacobianOplusXi = -J * v2.inverse().Adj();
//   _jacobianOplusXj = J * v2.inverse().Adj();

_jacobianOplusXi = J*v2.adj()*v1.inverse().adj();
_jacobianOplusXj = -J;
}
Matrix3d EdgeSE3Quat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}

}  //namespace ORB_SLAM2
