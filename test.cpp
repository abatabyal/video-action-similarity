//**********************Testing Videos***************************
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <math.h>
#include <vector>
 
using namespace cv;
using namespace std;

Mat frame,frame2, frameg, frameg2, dictionary, v, bowDescriptors, bowDescriptors2;
double score;

vector<KeyPoint> keypoints;
Mat bowimgdesc, bowimgdesc2, vh1, vh2;

//Function to Calculate Histogram
Mat hist (Mat Sample)
{
	FileStorage fs("dictionary.yml", FileStorage::READ);
    	fs["vocabulary"] >> dictionary;
    	fs.release(); 
    	
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    	    	
        Ptr<FeatureDetector> detector(new SiftFeatureDetector());
                
        Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
                    
        BOWImgDescriptorExtractor bowDE(extractor,matcher);
                
        bowDE.setVocabulary(dictionary);
        
	detector->detect(Sample,keypoints);
        bowDE.compute(Sample,keypoints,bowDescriptors);
        
	return bowDescriptors;
}

int main( int argc, char** argv )
{	
	
	
	FileStorage fs("dictionary.yml", FileStorage::READ);
    	fs["vocabulary"] >> dictionary;
    	fs.release(); 
    	
    	         	    	
    	VideoCapture capture(argv[1]), capture2(argv[2]);
    	
	if ( !capture.isOpened()||!capture2.isOpened())
	    {
	         cout << "Cannot open the video file" << endl;
	         return -1;
	    }
	    
	for(int i=0;;i++) //reading frames from the video
	{
		
		
		capture >> frame;
		cvtColor(frame, frameg, CV_BGR2GRAY);
		
		capture2 >> frame2;
		cvtColor(frame2, frameg2, CV_BGR2GRAY);
		
		if(frameg.empty()||frameg2.empty())  
		{
		break;
		}
		
		if(i%20==0)
		{	
		bowDescriptors.push_back(frameg);
                bowDescriptors2.push_back(frameg2);
               
               
              	}
	
								
	}
	
	bowimgdesc = hist(bowDescriptors);
	bowimgdesc2 = hist(bowDescriptors2);
	
	
	
	score = bowimgdesc.dot(bowimgdesc2);
	cout << "The Similarity Score of the videos is"<<" "<< score;
	
	
		
	
}
