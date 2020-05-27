#ifndef MAPPOINTBIRD_H
#define MAPPOINTBIRD_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include <opencv2/opencv.hpp>

#include <mutex>

namespace ORB_SLAM2
{

class Map;
class Frame;
class KeyFrame;

class MapPointBird
{
public:
    MapPointBird(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPointBird(const cv::Mat &Pos, Frame* pFrame, Map* pMap, const int &idxF);

    void AddObservation(KeyFrame* pKF, size_t idx);
    void EraseObservation(KeyFrame* pKF);
    std::map<KeyFrame*,size_t> GetObservations();
    cv::Mat GetWorldPos();
    void SetWorldPos(const cv::Mat &Pos);
    int GetIndexInKeyFrame(KeyFrame* pKF);
    void ComputeDistinctiveDescriptors();
    cv::Mat GetDescriptor();
    int Observations();

public:
    long unsigned int mnId;
    static long unsigned int nNextId;

    // Position in absolute coordinates
    cv::Mat mWorldPos;

    KeyFrame* mpRefKF;
    long unsigned int mnBALocalForKF;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame*,size_t> mObservations;
    int nObs;

    long unsigned int mnTrackReferenceForFrame;

    long unsigned int mnLastFrameSeen;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;
    // int mnScaleLevel;

    Map* mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;

};



}  //  namespace ORB_SLAM2






#endif  // MAPPOINTBIRD_H