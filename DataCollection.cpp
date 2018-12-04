#include "OcrRecognition.h"
//#include"TrainDataProcess.h"
#include <opencv2\contrib\contrib.hpp>
#include <vector>
#include<time.h>

using namespace std;

//从视频中截取字符图片
int main()
{
	string dir_path = "Z:\\数据整理集\\6A项目\\训练集\\训练的视频集\\按公司-车次字符分类\\";
	cv::Directory DIR;
	vector<string> filenames = DIR.GetListFiles(dir_path, "*.mp4", true);
	for (int k = 0; k < filenames.size(); k++)
	{
		string video_filename = filenames[k];
		string video_path = dir_path + video_filename;

		//测试视频
		cv::VideoCapture capture(video_path);
		cv::Mat frame;
		long frameNumber = 0;

		if (!capture.isOpened())
		{
			cout << "读取视频失败，请确认视频路径是否有误";
			return 0;
		}
		int FrameRate = capture.get(CV_CAP_PROP_FPS);
		int FrameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
		int FrameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
		int WordHeight = 25 * FrameHeight / PICHEIGHT;
		int WordWidth = 8 * FrameWidth / PICWIDTH;
		OcrRecognition myocr(WordHeight, 8);

		std::vector<int> Seconds;
		std::vector<cv::Mat> chars_imgs;
		while (capture.read(frame))
		{
			frameNumber++;
			if (frameNumber % 200 != 5)//隔200帧抽一帧
			{
				continue;
			}
			//if (frameNumber < 6080)continue;
			chars_imgs = myocr.Get_CharfromFrame(frame);
			for (int i = 0; i < chars_imgs.size(); i++)
			{
				char char_img_name[30];
				sprintf(char_img_name,"%05d_%05d_%05d.jpg", k + 10, frameNumber, i + 1);
				cv::imwrite(dir_path + "chars_/" + char_img_name, chars_imgs[i]);
			}			
		}
	}
	//测试视频	
	//string video_path = "F:/workspace/videos/HXD1C6251_株洲所_10_二端司机室_20180625_083001.mp4";
	//string video_path = "F:/workspace/videos/速度检查/HXD1C0748_株洲所_10_二端司机室_20180818_213000.mp4";
	
	return 0;
}
