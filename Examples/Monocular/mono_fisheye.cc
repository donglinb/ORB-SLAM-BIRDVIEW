#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include<opencv2/core/core.hpp>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Geometry>
#include<System.h>

using namespace std;

const double pixel2meter = 0.03984;//*1.737;
const double meter2pixel = 25.1;///1.737;
const double rear_axle_to_center = 1.393;
const double vehicle_length = 4.63;
const double vehicle_width = 1.901;

void LoadDataset(const string &strFile, vector<string> &vstrImageFilenames, vector<string> &vstrBirdviewFilenames, 
                vector<string> &vstrBirdviewMaskFilenames, vector<cv::Vec3d> &vodomPose, vector<double> &vTimestamps);
void applyMask(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask);
void applyMaskBirdview(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask);
void ConvertMaskBirdview(const cv::Mat& src, cv::Mat& dst);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }
    
    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<string> vstrBirdviewFilenames;
    vector<string> vstrBirdviewMaskFilenames;
    vector<double> vTimestamps;
    vector<cv::Vec3d> vodomPose;

    string strFile = string(argv[3])+"/associate.txt";
    // string strFile = string(argv[3])+"/groundtruth.txt";

    //LoadDataset(strFile, vstrImageFilenames, vTimestamps, vodomPos, vodomOri);
    // LoadDataset(strFile, vstrImageFilenames, vTimestamps, vodomPose);
    LoadDataset(strFile, vstrImageFilenames, vstrBirdviewFilenames, vstrBirdviewMaskFilenames, vodomPose, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cv::Mat mask_img=cv::imread("Examples/Monocular/mask_new_front.png");
    if(mask_img.empty())
    {
        cerr<<"failed to read mask image."<<endl;
    }

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    cv::Mat birdview;
    cv::Mat birdviewmask;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        birdview = cv::imread(string(argv[3])+"/"+vstrBirdviewFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        // birdview = cv::imread(string(argv[3])+"/"+vstrBirdviewFilenames[ni], CV_LOAD_IMAGE_GRAYSCALE);
        birdviewmask = cv::imread(string(argv[3])+"/"+vstrBirdviewMaskFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];
        // cv::Vec3d odomframe=vodomPose[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }
        // if(birdview.empty())
        // {
        //     cerr<<endl<<"Failed to load birdview image at: "
        //         <<string(argv[3])<<"/"<<vstrBirdviewFilenames[ni]<<endl
        //         <<"skip image."<<endl;
        //     continue;
        // }

        if(!birdviewmask.empty())
        {
            // cv::Mat birdview_masked;
            // applyMaskBirdview(birdview,birdview_masked,birdviewmask);
            // birdview=birdview_masked.clone();
            ConvertMaskBirdview(birdviewmask,birdviewmask);
        }

        // apply mask
        if(!mask_img.empty())
        {
            cv::Mat im_masked;
            applyMask(im,im_masked,mask_img);
            im=im_masked.clone();
        }
        // crop image
        {
            int crop_origin_x_=0,crop_origin_y_=0,crop_width_=1900,crop_height_=800;
            cv::Mat im_croped=im(cv::Rect(crop_origin_x_, crop_origin_y_, crop_width_, crop_height_));
            im=im_croped.clone();
        }
        // down sample
        cv::resize(im,im,cv::Size(0,0),0.5,0.5);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        //SLAM.TrackMonocular(im,tframe);
        SLAM.TrackMonocularWithBirdview(im,birdview,birdviewmask,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }
    cv::waitKey(0);
    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectoryTUM.txt");
    // SLAM.SaveTrajectoryTUM("TrajectoryTUM.txt");
    SLAM.SaveKeyFrameTrajectoryOdomTUM("KeyFrameTrajectoryOdomTUM.txt");

    return 0;
}

void LoadDataset(const string &strFile, vector<string> &vstrImageFilenames, vector<string> &vstrBirdviewFilenames, 
                vector<string> &vstrBirdviewMaskFilenames, vector<cv::Vec3d> &vodomPose, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            double x,y,theta;
            string image;
            ss >> t;
            vTimestamps.push_back(t);
            ss>>x>>y>>theta;
            vodomPose.push_back(cv::Vec3d(x,y,theta));
            ss >> image;
            vstrImageFilenames.push_back("image/"+image);
            vstrBirdviewFilenames.push_back("birdview/"+image);
            vstrBirdviewMaskFilenames.push_back("mask/"+image);
        }
    }
    // double t0=vTimestamps[0];
    // for_each(vTimestamps.begin(),vTimestamps.end(),[t0](double &t){t-=t0;});
}

void applyMask(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask)
{
  dst = src.clone();
  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j)
    {
      cv::Vec3b pixel = mask.at<cv::Vec3b>(i, j);
      if (pixel[1] > 250)
        dst.at<cv::Vec3b>(i, j) = 0;
    }
}

void applyMaskBirdview(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask)
{
  dst = src.clone();
  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j)
    {
      cv::Vec3b pixel = mask.at<cv::Vec3b>(i, j);
      if (pixel[1] < 20)
        dst.at<cv::Vec3b>(i, j) = 0;
    }
}

// void ConvertMaskBirdview(const cv::Mat& src, cv::Mat& dst)
// {
//     if(src.empty())
//         return;

//     cv::Mat dst_out = cv::Mat(src.rows,src.cols,CV_8UC1);
//     for (int i = 0; i < src.rows; ++i)
//         for (int j = 0; j < src.cols; ++j)
//         {
//             cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
//             if (pixel[1] < 20)
//                 dst_out.at<uchar>(i, j) = 0;
//             else
//                 dst_out.at<uchar>(i, j) = 250;
//         }

//     dst = dst_out.clone();
// }
void ConvertMaskBirdview(const cv::Mat& src, cv::Mat& dst)
{
    if(src.empty())
        return;

    cv::Mat dst_out = cv::Mat(src.rows,src.cols,CV_8UC1);
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
        {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            if (pixel[1] < 20)
                dst_out.at<uchar>(i, j) = 0;
            else
                dst_out.at<uchar>(i, j) = 250;
        }

    // preprocess mask, ignore footprint
    int birdviewCols=src.cols;
    int birdviewRows=src.rows;
    double boundary = 15.0;
    double x = birdviewCols / 2 - (vehicle_width / 2 / pixel2meter) - boundary;
    double y = birdviewRows / 2 - (vehicle_length / 2 / pixel2meter) - boundary;
    double width = vehicle_width / pixel2meter + 2 * boundary;
    double height = vehicle_length / pixel2meter + 2 * boundary;
    cv::rectangle(dst_out, cv::Rect(x, y, width, height), cv::Scalar(0),-1);

    dst = dst_out.clone();
}