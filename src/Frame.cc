/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;
/********************* Modified Here *********************/
float Frame::mfGridElementWidthInvBirdview, Frame::mfGridElementHeightInvBirdview;
cv::Mat Frame::Tbc,Frame::Tcb;
int Frame::birdviewRows, Frame::birdviewCols;

const double correction = 1.7;
const double Frame::pixel2meter = 0.03984*correction;
const double Frame::meter2pixel = 25.1/correction;
const double Frame::rear_axle_to_center = 1.393;
const double Frame::vehicle_length = 4.63;
const double Frame::vehicle_width = 1.901;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     mbHaveBirdview(frame.mbHaveBirdview),mvKeysBird(frame.mvKeysBird),mDescriptorsBird(frame.mDescriptorsBird.clone()),
    mvKeysBirdCamXYZ(frame.mvKeysBirdCamXYZ),mnBirdviewRefFrameId(frame.mnBirdviewRefFrameId),mvbBirdviewInliers(frame.mvbBirdviewInliers),
    mvnBirdviewMatches21(frame.mvnBirdviewMatches21),mBirdviewImg(frame.mBirdviewImg.clone()),mBirdviewMask(frame.mBirdviewMask.clone()),
    mvpMapPointsBird(frame.mvpMapPointsBird),mvKeysBirdBaseXY(frame.mvKeysBirdBaseXY),mpReferenceKFBird(frame.mpReferenceKFBird),Nbird(frame.Nbird)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);

    /********************* Modified Here *********************/
    if(frame.mbHaveBirdview)
    {
        for(int i=0;i<FRAME_GRID_COLS;i++)
            for(int j=0; j<FRAME_GRID_ROWS; j++)
                mGridBirdview[i][j]=frame.mGridBirdview[i][j];
    }
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        /********************* Modified Here *********************/
        CalculateExtrinsics();

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        /********************* Modified Here *********************/
        CalculateExtrinsics();

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        /********************* Modified Here *********************/
        CalculateExtrinsics();

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
}


/********************* Modified Here *********************/
Frame::Frame(const cv::Mat &imGray, const cv::Mat &birdviewGray, const cv::Mat &birdviewMask, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),mpReferenceKFBird(static_cast<KeyFrame*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),mbHaveBirdview(true)
{
    // Frame ID
    mnId=nNextId++;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        cout<<"Image Bounds:"<<endl;
        cout<<"minX = "<<mnMinX<<" , maxX = "<<mnMaxX<<endl;
        cout<<"minY = "<<mnMinY<<" , maxY = "<<mnMaxY<<endl;

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        mfGridElementWidthInvBirdview=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(birdviewGray.cols);
        mfGridElementHeightInvBirdview=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(birdviewGray.rows);

        birdviewCols = birdviewGray.cols;
        birdviewRows = birdviewGray.rows;

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        CalculateExtrinsics();

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);
    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    mBirdviewImg = birdviewGray.clone();
    mBirdviewMask = birdviewMask.clone();

    // preprocess mask, ignore footprint
    double boundary = 15.0;
    double x = birdviewCols / 2 - (vehicle_width / 2 / pixel2meter) - boundary;
    double y = birdviewRows / 2 - (vehicle_length / 2 / pixel2meter) - boundary;
    double width = vehicle_width / pixel2meter + 2 * boundary;
    double height = vehicle_length / pixel2meter + 2 * boundary;
    cv::rectangle(mBirdviewMask, cv::Rect(x, y, width, height), cv::Scalar(0),-1);
    // extract ORB for birdview
    cv::Ptr<cv::ORB> extractorBird = cv::ORB::create(2000);
    extractorBird->detect(mBirdviewImg,mvKeysBird,mBirdviewMask);
    vector<cv::Point2f> vKeysBird(mvKeysBird.size());
    for(int k=0;k<mvKeysBird.size();k++)
    {
        vKeysBird[k] = mvKeysBird[k].pt;
    }
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER,40,0.001);
    cv::cornerSubPix(mBirdviewImg,vKeysBird,cv::Size(5,5),cv::Size(-1,-1),criteria);
    for(int k=0;k<mvKeysBird.size();k++)
    {
        mvKeysBird[k].pt = vKeysBird[k];
    }
    extractorBird->compute(mBirdviewImg,mvKeysBird,mDescriptorsBird);
    
    Nbird = mvKeysBird.size();
    
    mvpMapPointsBird = vector<MapPointBird*>(mvKeysBird.size(),static_cast<MapPointBird*>(NULL));  


    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mvKeysBirdCamXYZ.resize(mvKeysBird.size());
    mvKeysBirdBaseXY.resize(mvKeysBird.size());
    for(int k=0;k<mvKeysBird.size();k++)
    {
        cv::Point3f p3d = BirdviewKP2XYZ(mvKeysBird[k]);
        cv::Point2f p2d;
        p2d.x = p3d.x;
        p2d.y = p3d.y;
        mvKeysBirdBaseXY[k] = p2d;
        mvKeysBirdCamXYZ[k] = TransformPoint3fWithMat(Tcb,p3d);
    }
    mvnBirdviewMatches21 = vector<int>(mvKeysBird.size(),-1);
    mvbBirdviewInliers = vector<bool>(mvKeysBird.size(),true);

    // cout<<"Frame "<<mnId<<" Created."<<endl;
}


void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }

    /********************* Modified Here *********************/
    if(mbHaveBirdview)
    {
        int Nbird = mvKeysBird.size();
        nReserve = 0.5f*Nbird/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
        for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
            for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
                mGridBirdview[i][j].reserve(nReserve);

        for(int i=0;i<Nbird;i++)
        {
            const cv::KeyPoint &kp = mvKeysBird[i];

            int nGridPosX, nGridPosY;
            if(PosInGridBirdview(kp,nGridPosX,nGridPosY))
                mGridBirdview[nGridPosX][nGridPosY].push_back(i);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    /********************* Modified Here *********************/
    // cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    cv::fisheye::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        /********************* Modified Here *********************/
        // cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        cv::fisheye::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        // mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        // mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        // mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

        mnMinX = std::numeric_limits<float>::max();
        mnMaxX = std::numeric_limits<float>::min();
        mnMinY = std::numeric_limits<float>::max();
        mnMaxY = std::numeric_limits<float>::min();

        for (int i = 0; i < 4; ++i)
        {
          if (mat.at<float>(i,0) < mnMinX)
            mnMinX = mat.at<float>(i,0);

          if (mat.at<float>(i,0) > mnMaxX)
            mnMaxX = mat.at<float>(i,0);

          if (mat.at<float>(i,1) < mnMinY)
            mnMinY = mat.at<float>(i,1);

          if (mat.at<float>(i,1) > mnMaxY)
            mnMaxY = mat.at<float>(i,1);
        }
#ifdef DEBUG
        std::cout << "mnMinX: " << mnMinX << std::endl;
        std::cout << "mnMaxX: " << mnMaxX << std::endl;
        std::cout << "mnMinY: " << mnMinY << std::endl;
        std::cout << "mnMaxY: " << mnMaxY << std::endl;
#endif
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

/********************* Modified Here *********************/
bool Frame::PosInGridBirdview(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round(kp.pt.x*mfGridElementWidthInvBirdview);
    posY = round(kp.pt.y*mfGridElementHeightInvBirdview);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
} 

vector<size_t> Frame::GetFeaturesInAreaBirdview(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(mvKeysBird.size());

    const int nMinCellX = max(0,(int)floor((x-r)*mfGridElementWidthInvBirdview));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x+r)*mfGridElementWidthInvBirdview));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-r)*mfGridElementHeightInvBirdview));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y+r)*mfGridElementHeightInvBirdview));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGridBirdview[ix][iy];
            if(vCell.empty())
                continue;
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kp = mvKeysBird[vCell[j]];

                if(bCheckLevels)
                {
                    if(kp.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kp.octave>maxLevel)
                            continue;
                }

                const float distx = kp.pt.x-x;
                const float disty = kp.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

// void Frame::CalculateExtrinsics()
// {
//     // from front camera to base footprint
//     cv::Mat tbc=(cv::Mat_<float>(3,1)<<-0.012, 2.746, -2.654);
//     //static cv::Mat qbc=(cv::Mat_<float>(4,1)<<0.631, -0.623, 0.325, 0.330);
//     float qx=0.631,qy=-0.623,qz=0.325,qw=0.330;
//     cv::Mat Rcb=(cv::Mat_<float>(3,3)<<1-2*(qy*qy+qz*qz),  2*(qx*qy-qw*qz),    2*(qx*qz+qw*qy),
//                                                 2*(qx*qy+qw*qz),  1-2*(qx*qx+qz*qz),  2*(qy*qz-qw*qx),
//                                                 2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx),    1-2*(qx*qx+qy*qy));
//     // Rcb=Rcb.t();
//     //Extrinsics : odometer and front camera
//     Tbc=cv::Mat::eye(4,4,CV_32F);
//     Tcb=cv::Mat::eye(4,4,CV_32F);
//     Rcb.copyTo(Tcb.rowRange(0,3).colRange(0,3));
//     tbc.copyTo(Tcb.rowRange(0,3).col(3));
//     cv::Mat Rbc=Rcb.t();
//     cv::Mat tcb=-Rbc*tbc;
//     Rbc.copyTo(Tbc.rowRange(0,3).colRange(0,3));
//     tcb.copyTo(Tbc.rowRange(0,3).col(3));
//     cout<<"extrinsics: "<<endl;
//     cout<<"Tbc = "<<endl<<Tbc<<endl;
//     cout<<"Tcb = "<<endl<<Tcb<<endl;
// }

void Frame::CalculateExtrinsics()
{
    // from front camera to base footprint
    cv::Mat tbc = (cv::Mat_<float>(3,1)<<3.747, 0.040, 0.736);
    float qx=0.631,qy=-0.623,qz=0.325,qw=-0.330;
    cv::Mat Rbc=(cv::Mat_<float>(3,3)<<1-2*(qy*qy+qz*qz),  2*(qx*qy-qw*qz),    2*(qx*qz+qw*qy),
                                                2*(qx*qy+qw*qz),  1-2*(qx*qx+qz*qz),  2*(qy*qz-qw*qx),
                                                2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx),    1-2*(qx*qx+qy*qy));
    Tbc = cv::Mat::eye(4,4,CV_32F);
    Rbc.copyTo(Tbc.rowRange(0,3).colRange(0,3));
    tbc.copyTo(Tbc.rowRange(0,3).col(3));

    Tcb = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Rcb = Rbc.t();
    cv::Mat tcb = -Rcb*tbc;
    Rcb.copyTo(Tcb.rowRange(0,3).colRange(0,3));
    tcb.copyTo(Tcb.rowRange(0,3).col(3));

    cout<<"extrinsics: "<<endl;
    cout<<"Tbc = "<<endl<<Tbc<<endl;
    cout<<"Tcb = "<<endl<<Tcb<<endl;
}

cv::Point3f Frame::BirdviewKP2XYZ(const cv::KeyPoint &kp)
{
    cv::Point3f p;
    p.x = (birdviewRows/2-kp.pt.y)*pixel2meter+rear_axle_to_center;
    p.y = (birdviewCols/2-kp.pt.x)*pixel2meter;
    p.z = 0;
    // p.z = -0.32115;

    return p;
}

cv::Point2f Frame::ProjectXYZ2Birdview(const cv::Point3f &p)
{
    cv::Point2f pt;
    pt.x = birdviewCols/2-p.y*meter2pixel;
    pt.y = birdviewRows/2-(p.x-rear_axle_to_center)*meter2pixel;
    return pt;
}

cv::Point3f Frame::TransformPoint3fWithMat(cv::Mat T, cv::Point3f p)
{
    cv::Mat p_src(p);
    cv::Mat p_dst = T.rowRange(0,3).colRange(0,3)*p_src+T.rowRange(0,3).col(3);
    return cv::Point3f(p_dst.at<float>(0),p_dst.at<float>(1),p_dst.at<float>(2));
}

cv::Mat Frame::InverseTransformSE3(cv::Mat T12)
{
    cv::Mat R12 = T12.rowRange(0,3).colRange(0,3);
    cv::Mat t12 = T12.rowRange(0,3).col(3);

    cv::Mat R21 = R12.t();
    cv::Mat t21 = -R21*t12;

    cv::Mat T21 = cv::Mat::eye(4,4,CV_32F);
    R21.copyTo(T21.rowRange(0,3).colRange(0,3));
    t21.copyTo(T21.rowRange(0,3).col(3));

    return T21.clone();
}

} //namespace ORB_SLAM
