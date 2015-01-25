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

Mat frame,frame2, frameg, frameg2,frameh, desc, op, dictionary, v,uncfeat;

ofstream of, cl;
ifstream in;
vector<KeyPoint> keypoints;

//int main( int argc, char** argv )
int main()
{
//***************************************Training Videos******************************************************
     //Part I of the code
	/*     
	VideoCapture capture(argv[1]);

	SIFT sift;
	
	if ( !capture.isOpened() )
	    {
	         cout << "Cannot open the video file" << endl;
	         return -1;
	    }
	    
	of.open("descriptors.txt",ios::app);
	
	for(int i=0;;i++) //reading frames from the video
	{
		capture >> frame;
		cvtColor(frame, frameg, CV_BGR2GRAY);
		

		if(frameg.empty())  
		{
		break;
		}

                if (i%20==0)
		{
		
		sift (frameg, Mat(), keypoints, desc);
		
		for(int i=0; i<desc.rows; i++)
   		 {
        		for(int j=0; j<desc.cols; j++)
       			 {
           		 of<<desc.at<float>(i,j)<<" ";
        	 	 }
        		of<<endl;
   	 	}	
		}
	
	}
	
	of.close();		
	*/	
	
	//cout<<desc.size();
	   
        //Part II of the code
	in.open("descriptors.txt");
	
	string line;
	int nrows =0;
	while(getline(in, line))
	{
		istringstream iss(line);
		nrows ++;
		int x; 
		while(iss >> x){
			v.push_back(x);
		}
		
	}
			
	v = v.reshape(1, nrows);
	v.convertTo(uncfeat, CV_32FC1);
		
	int clusterCount = 1500, attempts = 3, flags = KMEANS_PP_CENTERS;
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);

	BOWKMeansTrainer bowtrainer( clusterCount, tc , attempts, flags);
	
	dictionary = bowtrainer.cluster(uncfeat);
	cout << dictionary.size();
	
        FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
			
}


