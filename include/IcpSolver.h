#ifndef ICPSOLVER_H
#define ICPSOLVER_H

#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>

namespace ORB_SLAM2
{

using namespace std;

class IcpSolver
{
public:

    typedef pair<int,int> Match;

    IcpSolver(int MaxIterations=100)
    {
        mMaxIterations = MaxIterations;
    }

    void FindRtICP(const vector<cv::Point3f> &vKeysXYZ1, const vector<cv::Point3f> &vKeysXYZ2, const vector<Match> &vMatches, 
                vector<bool> &vbMatchesInliers, cv::Mat &R, cv::Mat &t, float &score, float sigma=1);
    void FindRtICP2D(const vector<cv::Point2f> &vKeysXY1, const vector<cv::Point2f> &vKeysXY2, const vector<Match> &vMatches, 
                vector<bool> &vbMatchesInliers, cv::Mat &R, cv::Mat &t, float &score, float sigma);

private:

    bool ComputeRtICP(const vector<cv::Point3f> &vP1, const vector<cv::Point3f> &vP2, cv::Mat &R, cv::Mat &t);
    bool ComputeRtICP2D(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2, cv::Mat &R, cv::Mat &t);

    int CheckRtICP(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point3f> &vP3D1, const vector<cv::Point3f> &vP3D2,
                const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers, float sigma);
    int CheckRtICP2D(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point2f> &vP2D1, const vector<cv::Point2f> &vP2D2,
                const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers, float sigma);

    
    int mMaxIterations;
};


}  // namespace ORB_SLAM2

#endif  //ICPSOLVER_H