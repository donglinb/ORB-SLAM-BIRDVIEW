#include "IcpSolver.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{

/********************* Modified Here *********************/
// Assume P1 = R * P2 + t, then R = U*Vt (selected)
// Assume P2 = R * P1 + t, then R = V*Ut
// Need at least 3 point matches.
bool IcpSolver::ComputeRtICP(const vector<cv::Point3f> &vP1, const vector<cv::Point3f> &vP2, cv::Mat &R, cv::Mat &t)
{
    int N = vP1.size();

    // centroid
    cv::Point3f p1=cv::Point3f(),p2=cv::Point3f();
    for(int k=0;k<N;k++)
    {
        p1+=vP1[k];
        p2+=vP2[k];
    }
    p1/=N;
    p2/=N;

    // minus centroid
    vector<cv::Point3f> vQ1(N), vQ2(N);
    for(int k=0;k<N;k++)
    {
        vQ1[k] = vP1[k]-p1;
        vQ2[k] = vP2[k]-p2;
    }

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    for(int k=0;k<N;k++)
    {
        W+=cv::Mat(vQ1[k])*cv::Mat(vQ2[k]).t();
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(W,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    R = u*vt;
    if(cv::determinant(R)<0)
    {
        u.col(2) = -u.col(2);
        R = u*vt;
    }

    t = cv::Mat(p1)-R*cv::Mat(p2);

    return true;
}

int IcpSolver::CheckRtICP(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point3f> &vP3D1, const vector<cv::Point3f> &vP3D2,
                    const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = vMatches12.size();

    vbMatchesInliers.resize(N,false);
    const float th = 7.815;
    const float invSigmaSquare = 1.0/(sigma*sigma);
    int nNumInliers = 0;

    for(int i=0;i<N;i++)
    {
        bool bIn = true;

        cv::Mat e = cv::Mat(vP3D1[vMatches12[i].first])-(R*cv::Mat(vP3D2[vMatches12[i].second])+t);
        float ex = e.at<float>(0), ey = e.at<float>(1), ez = e.at<float>(2);
        float squareDist = ex*ex+ey*ey+ez*ez;
        float chiSquare = squareDist*invSigmaSquare;

        if(chiSquare>th)
        {
            bIn = false;
        }
        
        if(bIn)
        {
            nNumInliers++;
            vbMatchesInliers[i] = true;
        }
        else
        {
            vbMatchesInliers[i] = false;
        }
    }
    // cout<<"Current Inliers = "<<nNumInliers<<"/"<<N<<endl; 
    return nNumInliers;
}

void IcpSolver::FindRtICP(const vector<cv::Point3f> &vKeysXYZ1, const vector<cv::Point3f> &vKeysXYZ2, const vector<Match> &vMatches, vector<bool> &vbMatchesInliers, cv::Mat &R, cv::Mat &t, float &score, float sigma)
{
    const int N = vMatches.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 3 points for each RANSAC iteration
    vector<vector<size_t> > vSets(mMaxIterations,vector<size_t>(3,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<3; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            vSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Best Results variables
    int bestNumInliers=0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point3f> vP1i(3);
    vector<cv::Point3f> vP2i(3);
    cv::Mat Ri,ti;
    vector<bool> vbCurrentInliers(N,false);
    // float currentScore;
    int nNumInliers=0;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<3; j++)
        {
            int idx = vSets[it][j];

            vP1i[j] = vKeysXYZ1[vMatches[idx].first];
            vP2i[j] = vKeysXYZ2[vMatches[idx].second];
        }

        ComputeRtICP(vP1i,vP2i,Ri,ti);

        nNumInliers = CheckRtICP(Ri,ti,vKeysXYZ1,vKeysXYZ2,vMatches,vbCurrentInliers,sigma);

        if(nNumInliers>bestNumInliers)
        {
            R = Ri.clone();
            t = ti.clone();
            vbMatchesInliers = vbCurrentInliers;
            bestNumInliers = nNumInliers;
        }
    }

    score = bestNumInliers;
}

// Assume P1 = R * P2 + t, then R = U*Vt (selected)
// Assume P2 = R * P1 + t, then R = V*Ut
// Need at least 2 point matches.
bool IcpSolver::ComputeRtICP2D(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2, cv::Mat &R, cv::Mat &t)
{
    int N = vP1.size();

    // centroid
    cv::Point2f p1=cv::Point2f(),p2=cv::Point2f();
    for(int k=0;k<N;k++)
    {
        p1+=vP1[k];
        p2+=vP2[k];
    }
    p1/=N;
    p2/=N;

    // minus centroid
    vector<cv::Point2f> vQ1(N), vQ2(N);
    for(int k=0;k<N;k++)
    {
        vQ1[k] = vP1[k]-p1;
        vQ2[k] = vP2[k]-p2;
    }

    cv::Mat W(2,2,CV_32F,cv::Scalar(0));
    for(int k=0;k<N;k++)
    {
        W+=cv::Mat(vQ1[k])*cv::Mat(vQ2[k]).t();
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(W,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    R = u*vt;
    if(cv::determinant(R)<0)
    {
        u.col(1) = -u.col(1);
        R = u*vt;
    }

    t = cv::Mat(p1)-R*cv::Mat(p2);

    return true;
}

int IcpSolver::CheckRtICP2D(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point2f> &vP2D1, const vector<cv::Point2f> &vP2D2,
                const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = vMatches12.size();

    vbMatchesInliers.resize(N,false);
    const float th = 5.991;
    const float invSigmaSquare = 1.0/(sigma*sigma);
    int nNumInliers = 0;

    for(int i=0;i<N;i++)
    {
        bool bIn = true;

        cv::Mat e = cv::Mat(vP2D1[vMatches12[i].first])-(R*cv::Mat(vP2D2[vMatches12[i].second])+t);
        float ex = e.at<float>(0), ey = e.at<float>(1);
        float squareDist = ex*ex+ey*ey;
        float chiSquare = squareDist*invSigmaSquare;

        if(chiSquare>th)
        {
            bIn = false;
        }
        
        if(bIn)
        {
            nNumInliers++;
            vbMatchesInliers[i] = true;
        }
        else
        {
            vbMatchesInliers[i] = false;
        }
    }
    // cout<<"Current Inliers = "<<nNumInliers<<"/"<<N<<endl; 
    return nNumInliers;
}

void IcpSolver::FindRtICP2D(const vector<cv::Point2f> &vKeysXY1, const vector<cv::Point2f> &vKeysXY2, const vector<Match> &vMatches, vector<bool> &vbMatchesInliers, cv::Mat &R, cv::Mat &t, float &score, float sigma)
{
    const int N = vMatches.size();
    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 2 points for each RANSAC iteration
    vector<vector<size_t> > vSets(mMaxIterations,vector<size_t>(2,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<2; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            vSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }


    // Best Results variables
    int bestNumInliers=0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vP1i(2);
    vector<cv::Point2f> vP2i(2);
    cv::Mat Ri,ti;
    vector<bool> vbCurrentInliers(N,false);
    int nNumInliers=0;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<2; j++)
        {
            int idx = vSets[it][j];

            vP1i[j] = vKeysXY1[vMatches[idx].first];
            vP2i[j] = vKeysXY2[vMatches[idx].second];
        }

        ComputeRtICP2D(vP1i,vP2i,Ri,ti);
        nNumInliers = CheckRtICP2D(Ri,ti,vKeysXY1,vKeysXY2,vMatches,vbCurrentInliers,sigma);

        if(nNumInliers>bestNumInliers)
        {
            R = Ri.clone();
            t = ti.clone();
            vbMatchesInliers = vbCurrentInliers;
            bestNumInliers = nNumInliers;
        }
    }

    score = bestNumInliers;
}

}  // namespace ORB_SLAM2