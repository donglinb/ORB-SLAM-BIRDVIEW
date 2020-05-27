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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>
#include<fstream>

#include<mutex>

// #define DRAW_MATCH

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


/********************* Modified Here *********************/
cv::Mat Tracking::GrabImageMonocularWithBirdview(const cv::Mat &im, const cv::Mat &birdview, const cv::Mat &birdviewmask, const double &timestamp)
{
    mImGray = im;
    mBirdviewGray = birdview;
    mbHaveBirdview=true;

    // Convert front view to grayscale
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    //Convert bird view to grayscale
    if(mBirdviewGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mBirdviewGray,mBirdviewGray,CV_RGB2GRAY);
        else
            cvtColor(mBirdviewGray,mBirdviewGray,CV_BGR2GRAY);
    }
    else if(mBirdviewGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mBirdviewGray,mBirdviewGray,CV_RGBA2GRAY);
        else
            cvtColor(mBirdviewGray,mBirdviewGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,mBirdviewGray,birdviewmask,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,mBirdviewGray,birdviewmask,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    // in the initialization stage, feature match is done in MonocularInitialization().
    // if(mState==NO_IMAGES_YET||mState==NOT_INITIALIZED)
    // {
        // mvPrevMatchedBirdview.resize(mCurrentFrame.mvKeysBird.size());
        // for(int k=0;k<mCurrentFrame.mvKeysBird.size();k++)
        // {
        //     mvPrevMatchedBirdview[k]=mCurrentFrame.mvKeysBird[k].pt;
        // }
        // mBirdviewRefFrame = mCurrentFrame;
        // mnRefNumMatches = std::numeric_limits<float>::max();
        // mCurrentFrame.mnBirdviewRefFrameId=0;
        if(!mpmatcherBirdview)
            mpmatcherBirdview = new ORBmatcher(0.99,true);
        if(!mIcp)
            mIcp = new IcpSolver(200);
    // }
//     else
//     {
//         int nmatches = mpmatcherBirdview->BirdviewMatch(mBirdviewRefFrame,mCurrentFrame,mvnBirdviewMatches12,mvPrevMatchedBirdview,10);
//         if(nmatches<0.5*mnRefNumMatches)
//         {
//             // cout<<"Change RefFrame."<<endl;
//             // cout<<"RefMatches : "<<mnRefNumMatches<<" , CurrentMatches : "<<nmatches<<endl;
//             mBirdviewRefFrame = Frame(mLastFrame);
//             mvPrevMatchedBirdview.resize(mLastFrame.mvKeysBird.size());
//             for(int k=0;k<mLastFrame.mvKeysBird.size();k++)
//             {
//                 mvPrevMatchedBirdview[k]=mLastFrame.mvKeysBird[k].pt;
//             }
//             nmatches = mpmatcherBirdview->BirdviewMatch(mBirdviewRefFrame,mCurrentFrame,mvnBirdviewMatches12,mvPrevMatchedBirdview,10);
//             mnRefNumMatches = nmatches;
//             // cout<<"After change : "<<nmatches<<" Matches."<<endl;
//         }
//         mCurrentFrame.mnBirdviewRefFrameId = mBirdviewRefFrame.mnId;
// #ifdef DRAW_MATCH
//         cv::Mat matchesImg;
//         vector<cv::DMatch> vMatches12;
//         for(int k=0;k<mvnBirdviewMatches12.size();k++)
//         {
//             int idx2 = mvnBirdviewMatches12[k];
//             if(idx2<0)
//             {
//                 continue;
//             }
//             cv::Mat d1 =  mBirdviewRefFrame.mDescriptorsBird.row(k);
//             cv::Mat d2 =  mCurrentFrame.mDescriptorsBird.row(idx2);
//             int distance = mpmatcherBirdview->DescriptorDistance(d1,d2);
//             vMatches12.push_back(cv::DMatch(k,idx2,distance));
//         }
//         cv::drawMatches(mBirdviewRefFrame.mBirdviewImg,mBirdviewRefFrame.mvKeysBird,mCurrentFrame.mBirdviewImg,mCurrentFrame.mvKeysBird,vMatches12,matchesImg);
//         // cout<<"Matches between "<<mBirdviewRefFrame.mnId<<" and "<<mCurrentFrame.mnId<<endl;
//         cv::imshow("birdview matches",matchesImg);
// #endif
//     }

    Track();

    // if(mState==NOT_INITIALIZED||mState==NO_IMAGES_YET)
    // {
    //     mLastFrame = Frame(mCurrentFrame);
    // }

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        // mLastFrame = Frame(mCurrentFrame);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                // Match Birdview Keypoints
                // MatchAndRetriveBirdMP();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                    if(!bOK)
                    {
                        cout<<"No Motion Model Track Reference KeyFrame failed, tracking lost."<<endl;
                    }
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                    {
                        cout<<"Track Motion Model failed, Track with Reference KeyFrame."<<endl;
                        bOK = TrackReferenceKeyFrame();
                        if(!bOK)
                        {
                            cout<<"Track Reference KeyFrame failed, tracking lost."<<endl;
                        }
                    }
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }


        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

             /********************* Modified Here *********************/
            //  mBirdviewRefFrame = Frame(mCurrentFrame);
             mvPrevMatchedBirdview.resize(mCurrentFrame.mvKeysBird.size());
             for(size_t i=0;i<mCurrentFrame.mvKeysBird.size();i++)
             {
                 mvPrevMatchedBirdview[i] = mCurrentFrame.mvKeysBird[i].pt;
             }

            return;
        }
    }
    else
    {
        // Try to initialize
        /********************* Modified Here *********************/
        if((int)mCurrentFrame.mvKeys.size()<=100||(int)mCurrentFrame.mvKeysBird.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvnBirdviewMatches12.begin(),mvnBirdviewMatches12.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
    
        /********************* Modified Here *********************/
        // vector<int>  vnMatchesBird12;
        // mvnBirdviewMatches12 = vector<int>(mInitialFrame.mvKeysBird.size(),-1);
        int nmatchesBird = mpmatcherBirdview->BirdviewMatch(mInitialFrame,mCurrentFrame,mvnBirdviewMatches12,mvPrevMatchedBirdview,15);
        // Check if there are enough correspondences
        if(nmatches<100||nmatchesBird<50)
        {
            cout<<"too few matches: front = "<<nmatches<<" , birdview = "<<nmatchesBird<<endl;
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }
#ifdef DRAW_INITIALIZE
        // draw matches
        cv::Mat matchimg = mCurrentFrame.mBirdviewImg.clone();
        cv::Mat matchesImg;
        vector<cv::DMatch> vMatches12;
        for(int k=0;k<mvnBirdviewMatches12.size();k++)
        {
            int idx2 = mvnBirdviewMatches12[k];
            if(idx2<0)
            {
                continue;
            }

            cv::Point2f p1,p2;
            p1 = mInitialFrame.mvKeysBird[k].pt;
            p2 = mCurrentFrame.mvKeysBird[idx2].pt;
            cv::circle(matchimg,p1,2,cv::Scalar(255,0,0),1);
            cv::circle(matchimg,p2,2,cv::Scalar(0,255,0),1);
            cv::line(matchimg,p1,p2,cv::Scalar(0,0,255),1);

            cv::Mat d1 =  mInitialFrame.mDescriptorsBird.row(k);
            cv::Mat d2 =  mCurrentFrame.mDescriptorsBird.row(idx2);
            int distance = mpmatcherBirdview->DescriptorDistance(d1,d2);
            vMatches12.push_back(cv::DMatch(k,idx2,distance));
        }
        cv::drawMatches(mInitialFrame.mBirdviewImg,mInitialFrame.mvKeysBird,mCurrentFrame.mBirdviewImg,mCurrentFrame.mvKeysBird,
                vMatches12,matchesImg,cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // cout<<"Matches between "<<mBirdviewRefFrame.mnId<<" and "<<mCurrentFrame.mnId<<endl;  
        cv::imshow("matches on one img",matchimg);
        cv::imshow("matches",matchesImg);
#endif
        
        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, mvnBirdviewMatches12, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }


            // // draw matches
            mvbMatchesInliersBird12 = mpInitializer->GetMatchesInliersBird();
            // cv::Mat matchesImg;
            // vector<cv::DMatch> vMatches12;
            // // ORBmatcher matcher(0.99,true);
            // // cout<<"Match size = "<<vbMatchesInliersBird12.size()<<endl;
            // for(int k=0;k<mvbMatchesInliersBird12.size();k++)
            // {
            //     if(!mvbMatchesInliersBird12[k])
            //         continue;
            //     if(mvnBirdviewMatches12[k]<0)
            //         continue;
            //     int idx1 = k;
            //     int idx2 = mvnBirdviewMatches12[k];
            //     cv::Mat d1 =  mInitialFrame.mDescriptorsBird.row(idx1);
            //     cv::Mat d2 =  mCurrentFrame.mDescriptorsBird.row(idx2);
            //     int distance = matcher.DescriptorDistance(d1,d2);
            //     vMatches12.push_back(cv::DMatch(idx1,idx2,distance));
            // }
            // cout<<"Inlier size = "<<vMatches12.size()<<endl;
            // cv::drawMatches(mInitialFrame.mBirdviewImg,mInitialFrame.mvKeysBird,mCurrentFrame.mBirdviewImg,mCurrentFrame.mvKeysBird,
            //         vMatches12,matchesImg,cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            // cv::imshow("initial matches inliers",matchesImg);
            // cv::imwrite("initial_matches.jpg",matchesImg);


            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();

            /********************* Modified Here *********************/
            mBirdviewRefFrame = Frame(mInitialFrame);
            mnRefNumMatches = nmatchesBird;
            mCurrentFrame.mnBirdviewRefFrameId = mInitialFrame.mnId;

            // // print initialization results
            // cv::Mat Tbw=Frame::Tbc*Tcw*Frame::Tcb;
            // cv::Mat R = Tbw.rowRange(0,3).colRange(0,3).t();
            // vector<float> q = Converter::toQuaternion(R);
            // cv::Mat t = -R*Tbw.rowRange(0,3).col(3);
            // ofstream f("Initialization_result.txt");
            // f<<fixed;
            // f << setprecision(6) << mInitialFrame.mTimeStamp << " " << mCurrentFrame.mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " 
            //     << t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
            // f<<"Length = "<<cv::norm(t)<<endl;
            // f.close();

        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    /********************* Modified Here *********************/
    cout<<"mvnBirdviewMatches12.size() = "<<mvnBirdviewMatches12.size()<<endl;
    cout<<"mvbMatchesInliersBird12.size() = "<<mvbMatchesInliersBird12.size()<<endl;

    for(int k=0;k<mvnBirdviewMatches12.size();k++)
    {
        if(mvnBirdviewMatches12[k]<0)
            continue;
        if(!mvbMatchesInliersBird12[k])
            continue;
        
        // Create MapPointBird
        cv::Mat worldPosBird(mInitialFrame.mvKeysBirdCamXYZ[k]);
        MapPointBird *pMPBird = new MapPointBird(worldPosBird,pKFcur,mpMap);

        pKFini->AddMapPointBird(pMPBird,k);
        pKFcur->AddMapPointBird(pMPBird,mvnBirdviewMatches12[k]);

        pMPBird->AddObservation(pKFini,k);
        pMPBird->AddObservation(pKFcur,mvnBirdviewMatches12[k]);

        pMPBird->ComputeDistinctiveDescriptors();

        mpMap->AddMapPointBird(pMPBird);

        mCurrentFrame.mvbBirdviewInliers[mvnBirdviewMatches12[k]]=true;
        mCurrentFrame.mvpMapPointsBird[mvnBirdviewMatches12[k]] = pMPBird;
    }
    cout<<"Insert "<<mpMap->MapPointsBirdInMap()<<" Birdview Points."<<endl;

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    if(mbHaveBirdview)
    {
        Optimizer::GlobalBundleAdjustemntWithBirdview(mpMap,20);
        // Optimizer::GlobalBundleAdjustemnt(mpMap,20);
    }
    else
    {
        Optimizer::GlobalBundleAdjustemnt(mpMap,20);
    }
    

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        cout<<"medianDepth = "<<medianDepth<<" , TrackedMapPoint = "<<pKFcur->TrackedMapPoints(1)<<endl;
        Reset();
        return;
    }

    // // Scale initial baseline
    // cv::Mat Tc2w = pKFcur->GetPose();
    // Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    // pKFcur->SetPose(Tc2w);

    // // Scale points
    // vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    // for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    // {
    //     if(vpAllMapPoints[iMP])
    //     {
    //         MapPoint* pMP = vpAllMapPoints[iMP];
    //         pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
    //     }
    // }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();

    /********************* Modified Here *********************/
    mvpLocalKeyFramesBird.push_back(pKFcur);
    mvpLocalKeyFramesBird.push_back(pKFini);
    mvpLocalMapPointsBird = mpMap->GetAllMapPointsBird();

    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    cout<<"Track Reference KeyFrame, "<<nmatches<<" Matches."<<endl;

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    if(mbHaveBirdview)
    {
        cout<<"Search Birdview Matches."<<endl;
        vector<MapPointBird*> vpMapPointMatchesBird;
        int nmatchesbird = mpmatcherBirdview->SearchByMatchBird(mpReferenceKF,mCurrentFrame,vpMapPointMatchesBird,15);
        if(nmatchesbird<15)
        {
            nmatchesbird = mpmatcherBirdview->SearchByMatchBird(mpReferenceKF,mCurrentFrame,vpMapPointMatchesBird,20);
        }
        mCurrentFrame.mvpMapPointsBird = vpMapPointMatchesBird;
        cout<<"Track Reference KeyFrame, "<<nmatchesbird<<" Birdview Matches."<<endl;

        Optimizer::PoseOptimizationWithBirdview(&mCurrentFrame);
        // MatchAndRetriveBirdMP();
        // Optimizer::PoseOptimizationWithBirdview(&mCurrentFrame);
        // Optimizer::PoseOptimization(&mCurrentFrame);
        // DrawMatchesInliersBird();
        // Optimizer::PoseOptimizationWithBirdview(&mCurrentFrame,&mBirdviewRefFrame);
    }
    else
    {
        Optimizer::PoseOptimization(&mCurrentFrame);
    }
    

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    cout<<"After Optimization, "<<nmatches<<" Matches, "<<nmatchesMap<<" Matches in Map."<<endl;

    /********************* Modified Here *********************/
    int nmatchesBirdMap=0;
    if(mbHaveBirdview)
    {
        // Discard outliers for birdview
        // cv::Mat Twc = mCurrentFrame.mTcw.inv();
        for(int k=0;k<mCurrentFrame.mvKeysBird.size();k++)
        {
            if(mCurrentFrame.mvpMapPointsBird[k])
            {
                if(!mCurrentFrame.mvbBirdviewInliers[k])
                {
                    // cv::Mat localPos = cv::Mat(mCurrentFrame.mvKeysBirdCamXYZ[k]);
                    // cv::Mat worldPos = Twc.rowRange(0,3).colRange(0,3)*localPos+Twc.rowRange(0,3).col(3);
                    // MapPointBird *pMPBird = new MapPointBird(worldPos,&mCurrentFrame,mpMap,k);
                    // mpMap->AddMapPointBird(pMPBird);
                    // mCurrentFrame.mvpMapPointsBird[k] = pMPBird;

                    mCurrentFrame.mvpMapPointsBird[k] = static_cast<MapPointBird*>(NULL);
                    mCurrentFrame.mvbBirdviewInliers[k]=true;
                }
                else if(mCurrentFrame.mvpMapPointsBird[k]->Observations()>0)
                {
                    nmatchesBirdMap++;
                }
                
            }
        }
    }


    return nmatchesMap+nmatchesBirdMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    /********************* Modified Here *********************/
    if(mbHaveBirdview)
    {
        // if(mbTcrBirdUpdated)
        // {
        //     mCurrentFrame.SetPose(mTcrBirdc*mBirdviewRefFrame.mTcw);
        //     mbTcrBirdUpdated = false;
        //     mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        // }
        // else
        {
            mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        }
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    }
    

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    int nmatchesBird = 0;
    if(mbHaveBirdview)
    {
        // match birdview points.
        fill(mCurrentFrame.mvpMapPointsBird.begin(),mCurrentFrame.mvpMapPointsBird.end(),static_cast<MapPointBird*>(NULL));
        nmatchesBird = mpmatcherBirdview->SearchByMatchBird(mCurrentFrame,mLastFrame,15);
        if(nmatchesBird<20)
        {
            fill(mCurrentFrame.mvpMapPointsBird.begin(),mCurrentFrame.mvpMapPointsBird.end(),static_cast<MapPointBird*>(NULL));
            nmatchesBird = mpmatcherBirdview->SearchByMatchBird(mCurrentFrame,mLastFrame,20);
        }
        cout<<"Track with Motion Model, "<<nmatches<<" Matches and "<<nmatchesBird<<" Birdview Matchces."<<endl;
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    if(mbHaveBirdview)
    {
        Optimizer::PoseOptimizationWithBirdview(&mCurrentFrame);
        // MatchAndRetriveBirdMP();
        // Optimizer::PoseOptimizationWithBirdview(&mCurrentFrame);
        // Optimizer::PoseOptimization(&mCurrentFrame);
        // DrawMatchesInliersBird();
        // Optimizer::PoseOptimizationWithBirdview(&mCurrentFrame,&mBirdviewRefFrame);
    }
    else
    {
        Optimizer::PoseOptimization(&mCurrentFrame);
    }
    

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    cout<<"After Optimization, "<<nmatches<<" Matches and "<<nmatchesMap<<" Matches in Map."<<endl;

    if(mbHaveBirdview)
    {
        // Discard outliers for birdview
        // cv::Mat Twc = mCurrentFrame.mTcw.inv();
        for(int k=0;k<mCurrentFrame.mvKeysBird.size();k++)
        {
            if(mCurrentFrame.mvpMapPointsBird[k])
            {
                if(!mCurrentFrame.mvbBirdviewInliers[k])
                {
                    // cv::Mat localPos = cv::Mat(mCurrentFrame.mvKeysBirdCamXYZ[k]);
                    // cv::Mat worldPos = Twc.rowRange(0,3).colRange(0,3)*localPos+Twc.rowRange(0,3).col(3);
                    // MapPointBird *pMPBird = new MapPointBird(worldPos,&mCurrentFrame,mpMap,k);
                    // mpMap->AddMapPointBird(pMPBird);
                    // mCurrentFrame.mvpMapPointsBird[k] = pMPBird;

                    mCurrentFrame.mvpMapPointsBird[k] = static_cast<MapPointBird*>(NULL);
                    mCurrentFrame.mvbBirdviewInliers[k]=true;
                    nmatchesBird--;
                }
            }
        }
        cout<<"After Optimization, "<<nmatchesBird<<" Birdview Matches."<<endl;    
    }
    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    cout<<"Start Track Local Map."<<endl;
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    /********************* Modified Here *********************/
    if(mbHaveBirdview)
    {
        // generate more birdview mappoints
        MatchAndRetriveBirdMP();
        Optimizer::PoseOptimizationWithBirdview(&mCurrentFrame);
        DrawMatchesInliersBird();
        // Optimizer::PoseOptimization(&mCurrentFrame);
    }
    else
    {
        Optimizer::PoseOptimization(&mCurrentFrame);
    }
    
    
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    cout<<"Track Local Map, MatchesInliers = "<<mnMatchesInliers<<endl;

    int nMatchesInliersBird=0;
    if(mbHaveBirdview)
    { 
        // Discard outliers for birdview
        // cv::Mat Twc = mCurrentFrame.mTcw.inv();
        for(int k=0;k<mCurrentFrame.mvKeysBird.size();k++)
        {
            if(mCurrentFrame.mvpMapPointsBird[k])
            {
                if(!mCurrentFrame.mvbBirdviewInliers[k])
                {
                    // cv::Mat localPos = cv::Mat(mCurrentFrame.mvKeysBirdCamXYZ[k]);
                    // cv::Mat worldPos = Twc.rowRange(0,3).colRange(0,3)*localPos+Twc.rowRange(0,3).col(3);
                    // MapPointBird *pMPBird = new MapPointBird(worldPos,&mCurrentFrame,mpMap,k);
                    // mpMap->AddMapPointBird(pMPBird);
                    // mCurrentFrame.mvpMapPointsBird[k] = pMPBird;

                    mCurrentFrame.mvpMapPointsBird[k] = static_cast<MapPointBird*>(NULL);
                    mCurrentFrame.mvbBirdviewInliers[k]=true;
                }
                else
                {
                    nMatchesInliersBird++;
                }
            }
        }  

        cout<<"Track Local Map, Birdview MatchesInliers = "<<nMatchesInliersBird<<endl;
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers+nMatchesInliersBird<50)
        return false;

    if(mnMatchesInliers+nMatchesInliersBird<30)
    {
        cout<<"Inliers less than 30 , tracking lost."<<endl;
        return false;
    } 
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    // if(mSensor==System::MONOCULAR)
    //     thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    // cout<<"c1a = "<<c1a<<" , c1b = "<<c1b<<" , c1c = "<<c1c<<" , c2 = "<<c2<<", result = "<<((c1a||c1b||c1c)&&c2)<<endl;

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    if(mbHaveBirdview)
    {
        int nMPBird=0;
        for(int k=0;k<mCurrentFrame.mvpMapPointsBird.size();k++)
        {
            MapPointBird *pMPBird = mCurrentFrame.mvpMapPointsBird[k];
            if(pMPBird)
            {
                pMPBird->AddObservation(pKF,k);
                pKF->AddMapPointBird(pMPBird,k);
                pMPBird->ComputeDistinctiveDescriptors();
                mpMap->AddMapPointBird(pMPBird);
                nMPBird++;
            }
        }
        cout<<"Insert "<<nMPBird<<" Birdview MapPoints to Map."<<endl;
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    cout<<"Start Search Local Points."<<endl;
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }

    if(mbHaveBirdview)
    {
        for(vector<MapPointBird*>::iterator vit=mCurrentFrame.mvpMapPointsBird.begin(), vend=mCurrentFrame.mvpMapPointsBird.end(); vit!=vend; vit++)
        {
            MapPointBird* pMPBird = *vit;
            if(pMPBird)
            {
                pMPBird->mnLastFrameSeen = mCurrentFrame.mnId;
            }
        }
        mpmatcherBirdview->SearchByProjectionBird(mCurrentFrame,mvpLocalMapPointsBird);
    }
}

void Tracking::UpdateLocalMap()
{
    cout<<"Start Update Local Map."<<endl;
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    cout<<"Start Update Local Point."<<endl;
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }

    if(mbHaveBirdview)
    {
        // cout<<"Update Birdview Local Points."<<endl;
        mvpLocalMapPointsBird.clear();
        for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFramesBird.begin(), itEndKF=mvpLocalKeyFramesBird.end(); itKF!=itEndKF; itKF++)
        {
            KeyFrame* pKF = *itKF;
            const vector<MapPointBird*> vpMPBirds = pKF->GetMapPointMatchesBird();
            // cout<<"KeyFrame "<<pKF->mnId<<" has "<<vpMPBirds.size()<<" Birdview MapPoints."<<endl;
            for(vector<MapPointBird*>::const_iterator itMPBird=vpMPBirds.begin(), itEndMPBird=vpMPBirds.end(); itMPBird!=itEndMPBird; itMPBird++)
            {
                MapPointBird* pMPBird = *itMPBird;
                if(!pMPBird)
                    continue;
                if(pMPBird->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                    continue;

                mvpLocalMapPointsBird.push_back(pMPBird);
                pMPBird->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    cout<<"Start Update Local KeyFrames."<<endl;
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }

    if(mbHaveBirdview)
    {
        // Each map point vote for the keyframes in which it has been observed
        // cout<<"Update Birdview Local KeyFrames."<<endl;
        map<KeyFrame*,int> keyframeCounterBird;
        for(int i=0; i<mCurrentFrame.Nbird; i++)
        {
            MapPointBird* pMPBird = mCurrentFrame.mvpMapPointsBird[i];
            if(pMPBird)
            {
                const map<KeyFrame*,size_t> observations = pMPBird->GetObservations();
                // cout<<"Birdview MapPoint "<<pMPBird->mnId<<" has "<<observations.size()<<" observations."<<endl;
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounterBird[it->first]++;
            }
        }

        if(keyframeCounterBird.empty())
        {
            cout<<"No Local KeyFrames Found."<<endl;
            return;
        }
        // cout<<"Found "<<keyframeCounterBird.size()<<" Local KeyFrames."<<endl;

        int maxBird=0;
        KeyFrame* pKFmaxBird= static_cast<KeyFrame*>(NULL);

        mvpLocalKeyFramesBird.clear();
        mvpLocalKeyFramesBird.reserve(keyframeCounterBird.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for(map<KeyFrame*,int>::const_iterator it=keyframeCounterBird.begin(), itEnd=keyframeCounterBird.end(); it!=itEnd; it++)
        {
            KeyFrame* pKF = it->first;

            if(pKF->isBad())
                continue;

            if(it->second>maxBird)
            {
                maxBird=it->second;
                pKFmaxBird=pKF;
            }

            mvpLocalKeyFramesBird.push_back(it->first);
        }

        if(pKFmaxBird)
        {
            mpReferenceKFBird = pKFmaxBird;
            mCurrentFrame.mpReferenceKFBird = mpReferenceKFBird;
        }
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}


/********************* Modified Here *********************/
void Tracking::MatchAndRetriveBirdMP()
{
    cout<<"Match and Retrive Bird MapPoint."<<endl;

    mvnBirdviewMatches12.clear();
    int nmatches = mpmatcherBirdview->BirdviewMatch(mLastFrame,mCurrentFrame,mvnBirdviewMatches12,10);
    if(nmatches<100)
    {
        nmatches = mpmatcherBirdview->BirdviewMatch(mLastFrame,mCurrentFrame,mvnBirdviewMatches12,20);
    }
    cout<<"Matched "<<nmatches<<" birdview points."<<endl;

    cv::Mat Twc1 = Frame::InverseTransformSE3(mLastFrame.mTcw);
    cv::Mat Tb2w = Frame::Tbc*mCurrentFrame.mTcw;
    if(mCurrentFrame.mTcw.empty())
    {
        cout<<"Tcw empty, failed."<<endl;
        return;
    }

    int nMap=0;
    for(int k=0;k<mvnBirdviewMatches12.size();k++)
    {
        int idx2 = mvnBirdviewMatches12[k];
        if(idx2<0)
            continue;
        
        MapPointBird *pMPBird = mLastFrame.mvpMapPointsBird[k];
        if(pMPBird)
        {
            mCurrentFrame.mvpMapPointsBird[idx2] = pMPBird;
        }
        else
        {
            cv::Mat localPos1(mLastFrame.mvKeysBirdCamXYZ[k]);
            cv::Mat worldPos = Twc1.rowRange(0,3).colRange(0,3)*localPos1+Twc1.rowRange(0,3).col(3);
            cv::Mat localPos2 = Tb2w.rowRange(0,3).colRange(0,3)*worldPos+Tb2w.rowRange(0,3).col(3);
            cv::Point2f pt = Frame::ProjectXYZ2Birdview(cv::Point3f(localPos2.at<float>(0),localPos2.at<float>(1),localPos2.at<float>(2)));
            cv::KeyPoint kp = mCurrentFrame.mvKeysBird[idx2];
            cv::Point2f ptm = kp.pt;
            double e = (pt.x-ptm.x)*(pt.x-ptm.x)+(pt.y-ptm.y)*(pt.y-ptm.y);
            double chi2 = e*mCurrentFrame.mvInvLevelSigma2[kp.octave];
            // cout<<"localPos1 = "<<localPos1.t()<<" , localPos2 = "<<localPos2.t()<<endl;
            // cout<<"pt = "<<cv::Mat(pt).t()<<" , ptm = "<<cv::Mat(ptm).t()<<endl;
            // cout<<"e = "<<e<<" , chi2 = "<<chi2<<endl;
            if(chi2>5.991)
                continue;
            pMPBird = new MapPointBird(worldPos,&mCurrentFrame,mpMap,idx2);
            mLastFrame.mvpMapPointsBird[k] = pMPBird;
            mCurrentFrame.mvpMapPointsBird[idx2] = pMPBird;
            // mpMap->AddMapPointBird(pMPBird);
            nMap++;
        }
    }
    cout<<"Added "<<nMap<<" map points bird"<<endl;
}
/*
void Tracking::MatchAndRetriveBirdMP()
{
    mvnBirdviewMatches12.clear();
    int nmatches = mpmatcherBirdview->BirdviewMatch(mBirdviewRefFrame,mCurrentFrame,mvnBirdviewMatches12,mvPrevMatchedBirdview,75);
    // if matches less than 30, choose a new reference frame.
    if(nmatches<75)
    {
        mBirdviewRefFrame = Frame(mLastFrame);
        mvPrevMatchedBirdview.resize(mLastFrame.mvKeysBird.size());
        for(int k=0;k<mLastFrame.mvKeysBird.size();k++)
        {
            mvPrevMatchedBirdview[k]=mLastFrame.mvKeysBird[k].pt;
        }
        mvnBirdviewMatches12.clear();
        nmatches = mpmatcherBirdview->BirdviewMatch(mBirdviewRefFrame,mCurrentFrame,mvnBirdviewMatches12,mvPrevMatchedBirdview,75);
        mvbMatchesInliersBird12.resize(mvnBirdviewMatches12.size(),false);
        cout<<"Change Reference : "<<mBirdviewRefFrame.mnId<<" , "<<nmatches<<" matches."<<endl;
    }
    // select matches inliers by ICP
    // if(mCurrentFrame.mnId-mBirdviewRefFrame.mnId>2)
    // {
        vector<Match> vMatchesBird;
        for(int k=0;k<mvnBirdviewMatches12.size();k++)
        {
            if(mvnBirdviewMatches12[k]<0)
                continue;
            vMatchesBird.push_back(make_pair(k,mvnBirdviewMatches12[k]));
        }
        vector<bool> vbMatchesInliers(vMatchesBird.size(),false);
        cv::Mat R12,t12;
        float score;
        float sigma = Frame::pixel2meter;
        mIcp->FindRtICP2D(mBirdviewRefFrame.mvKeysBirdBaseXY,mCurrentFrame.mvKeysBirdBaseXY,vMatchesBird,vbMatchesInliers,R12,t12,score,sigma);
        mvbMatchesInliersBird12.resize(mvnBirdviewMatches12.size(),false);
        for(int k=0;k<vMatchesBird.size();k++)
        {
            if(vbMatchesInliers[k])
            {
                int idx1 = vMatchesBird[k].first;
                int idx2 = vMatchesBird[k].second;
                mvbMatchesInliersBird12[idx1]=true;
                mCurrentFrame.mvbBirdviewInliers[idx2] = true;
            }
        }
        cv::Mat T12b = cv::Mat::eye(4,4,CV_32F);
        R12.copyTo(T12b.rowRange(0,2).colRange(0,2));
        t12.copyTo(T12b.rowRange(0,2).col(3));
        cv::Mat T12c = Frame::Tcb*T12b*Frame::Tbc;
        mTcrBirdc = Frame::InverseTransformSE3(T12c);
        mbTcrBirdUpdated = true;
        cout<<"Reference = "<<mBirdviewRefFrame.mnId<<" , Current = "<<mCurrentFrame.mnId<<" , score = "<<score<<endl;
    // }

    cv::Mat Twc = Frame::InverseTransformSE3(mBirdviewRefFrame.mTcw);
    for(int k=0;k<mvnBirdviewMatches12.size();k++)
    {
        if(mvnBirdviewMatches12[k]<0)
            continue;

        MapPointBird *pMPBird=NULL;

        if(mBirdviewRefFrame.mvpMapPointsBird[k])
        {
            pMPBird = mBirdviewRefFrame.mvpMapPointsBird[k];
        }
        else
        {
            if(mvbMatchesInliersBird12[k])
            {
                cv::Mat localPos(mBirdviewRefFrame.mvKeysBirdCamXYZ[k]);
                cv::Mat worldPos = Twc.rowRange(0,3).colRange(0,3)*localPos+Twc.rowRange(0,3).col(3);
                pMPBird = new MapPointBird(worldPos,&mCurrentFrame,mpMap,mvnBirdviewMatches12[k]);
                mBirdviewRefFrame.mvpMapPointsBird[k] = pMPBird;
            }  
        }

        mCurrentFrame.mvpMapPointsBird[mvnBirdviewMatches12[k]] = pMPBird;

        mpMap->AddMapPointBird(pMPBird);
    }

    DrawMatchesInliersBird();
}
*/
void Tracking::DrawMatchesInliersBird()
{
    // draw matches
    cv::Mat matchesImg,matchesinliersImg;
    vector<cv::DMatch> vMatches12,vMatchesInliers12;
    for(int k=0;k<mvnBirdviewMatches12.size();k++)
    {
        int idx2 = mvnBirdviewMatches12[k];
        if(idx2<0)
        {
            continue;
        }
        cv::Mat d1 =  mLastFrame.mDescriptorsBird.row(k);
        cv::Mat d2 =  mCurrentFrame.mDescriptorsBird.row(idx2);
        int distance = mpmatcherBirdview->DescriptorDistance(d1,d2);
        vMatches12.push_back(cv::DMatch(k,idx2,distance));
        if(mCurrentFrame.mvbBirdviewInliers[idx2])
        {
            vMatchesInliers12.push_back(cv::DMatch(k,idx2,distance));
        }
    }
    cv::drawMatches(mLastFrame.mBirdviewImg,mLastFrame.mvKeysBird,mCurrentFrame.mBirdviewImg,mCurrentFrame.mvKeysBird,
            vMatches12,matchesImg,cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(mLastFrame.mBirdviewImg,mLastFrame.mvKeysBird,mCurrentFrame.mBirdviewImg,mCurrentFrame.mvKeysBird,
            vMatchesInliers12,matchesinliersImg,cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // cout<<"Matches between "<<mBirdviewRefFrame.mnId<<" and "<<mCurrentFrame.mnId<<endl; 
    cout<<vMatches12.size()<<" matches, "<<vMatchesInliers12.size()<<" inliers."<<endl; 
    cv::imshow("birdview matches",matchesImg);
    cv::imshow("birdview matches inliers",matchesinliersImg);
}

} //namespace ORB_SLAM
