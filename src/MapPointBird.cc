#include "MapPointBird.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2
{

long unsigned int MapPointBird::nNextId=0;

MapPointBird::MapPointBird(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap)
:mWorldPos(Pos.clone()),mpRefKF(pRefKF),mpMap(pMap),nObs(0),mnBALocalForKF(0),
mnTrackReferenceForFrame(0),mnLastFrameSeen(0)
{
    // cout<<"Construct Birdview MapPoint with KeyFrame "<<pRefKF->mnId<<endl;
    mnId = nNextId++;
}

MapPointBird::MapPointBird(const cv::Mat &Pos, Frame* pFrame, Map* pMap, const int &idxF)
:mWorldPos(Pos.clone()),mpMap(pMap),mpRefKF(static_cast<KeyFrame*>(NULL)),nObs(0),mnBALocalForKF(0),
mnTrackReferenceForFrame(0),mnLastFrameSeen(0)
{
    // cout<<"Construct Birdview MapPoint with Frame "<<pFrame->mnId<<endl;
    mnId = nNextId++;
    mDescriptor = pFrame->mDescriptorsBird.row(idxF).clone();
    mnLastFrameSeen = pFrame->mnId;
    // mnScaleLevel = pFrame->mvKeysBird[idxF].octave;
}

void MapPointBird::AddObservation(KeyFrame* pKF, size_t idx)
{
    // cout<<"Birdview MapPoint "<<mnId<<" Add Observation for KeyFrame "<<pKF->mnId<<endl;
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    nObs++;
    // cout<<"Add Observation Successful."<<endl;
}

void MapPointBird::EraseObservation(KeyFrame* pKF)
{
    // cout<<"Birdview MapPoint "<<mnId<<"Erase Observation for KeyFrame "<<pKF->mnId<<endl;
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
    {
        nObs--;
        mObservations.erase(pKF);
        if(mpRefKF==pKF)
            mpRefKF=mObservations.begin()->first;
    }
    // cout<<"Erase Observation Successful."<<endl;
}

map<KeyFrame*, size_t> MapPointBird::GetObservations()
{
    // cout<<"Get MapPoint Bird Observations..."<<endl;
    unique_lock<mutex> lock(mMutexFeatures);
    // cout<<"Get MapPoint Bird Observations Successful."<<endl;
    return mObservations;
}

cv::Mat MapPointBird::GetWorldPos()
{
    // cout<<"Get MapPoint Bird WorldPos..."<<endl;
    unique_lock<mutex> lock(mMutexPos);
    // cout<<"Get MapPoint Bird WorldPos Successful."<<endl;
    return mWorldPos.clone();
}

void MapPointBird::SetWorldPos(const cv::Mat &Pos)
{
    // cout<<"Set MapPoint Bird WorldPos..."<<endl;
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
    // cout<<"Set MapPoint Bird WorldPos Successful."<<endl;
}

int MapPointBird::GetIndexInKeyFrame(KeyFrame *pKF)
{
    // cout<<"Get Index of MapPoint Bird "<<mnId<<" in KeyFrame "<<pKF->mnId<<endl;
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

void MapPointBird::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    if(!mpRefKF&&!mDescriptor.empty())
        vDescriptors.push_back(mDescriptor);

    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }   
}

cv::Mat MapPointBird::GetDescriptor()
{
    // cout<<"Get MapPoint Bird Descriptor..."<<endl;
    unique_lock<mutex> lock(mMutexFeatures);
    // cout<<"Get MapPoint Bird Descriptor Successful."<<endl;
    return mDescriptor.clone();
}

int MapPointBird::Observations()
{
    // cout<<"Get MapPoint Bird Num Observations..."<<endl;
    unique_lock<mutex> lock(mMutexFeatures);
    // cout<<"Get MapPoint Bird Num Observations Successful."<<endl;
    return nObs;
}

}  // namespace ORB_SLAM2