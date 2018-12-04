#pragma once
//#include<time.h>
//#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"


#define  CHARSUM 80
#define  AREABORD  26     //识别区域高度
#define  PICHEIGHT  288
#define  PICWIDTH  352

typedef struct UpInfor
{
	std::string str_date;
	std::string str_time;
	std::string str_km;
};
typedef struct DownInfor
{
	std::string str_result;
};
typedef struct FrameWord
{
	UpInfor infor_up;
	DownInfor infor_down;
};
/*typedef struct OCR_Date
{
	int years;
	int monthes;
	int days;
};
typedef struct OCR_Time
{
	int hours;
	int minutes;
	int seconds;
};*/
typedef struct OCR_Result
{
	//OCR_Date Date;//日期
	//OCR_Time currentTime;//时间
	//time_t Date_Time;//日期和时间
	tm Date_Time;//日期和时间
	int Velocity;//速度
	double Mileage;//里程
	std::string Train_Number;//车次
	std::string VIN;//车号
	std::string Camera_num;//摄像机编号
	double Confidence_Time=1.0;//时间置信度
	double Confidence_Velocity=1.0;//速度置信度
	double Confidence_Mileage=1.0;//里程置信度
};
typedef struct OCR_Results_video
{
	long frame_Num = 0;//视频总帧数
	double Frame_Rate = 0;
	//OCR_Date Date;//日期
	//std::vector<std::string> Train_Number;//车次
	std::string VIN;//车号
	std::string Camera_num;//摄像机编号
	std::vector<OCR_Result> MTV;//Mileage_Time_Velocity_Train_Number
};
/*typedef struct OCR_Results_video_final
{
	long frame_Num = 0;//视频总帧数
	std::string Train_Number;//车次
	std::string VIN;//车号
	std::string Camera_num;//摄像机编号	
	std::vector<time_t> Date_Time;//Mileage_Time_Velocity
	std::vector<double> Mileage;
	std::vector<int> Velocity;
};*/
typedef struct Str_Long {
	std::string str_value = "";
	long str_count = 0;
};
typedef struct Int_Long {
	int int_value = 0;
	long int_count = 0;
};

void otsu(unsigned char *picPtr, int startRows, int picHeight, int picCols);
int NumCut(unsigned char *picPtr, int startRows, int picHeight, int picCols);
int NumCut1(unsigned char *picPtr, int startRows, int picHeight, int picCols);
void CharsCut(cv::Mat im, cv::Mat orig_im, std::vector<cv::Mat> &char_imgs,int startRows, int picHeight, int picWidth);
void Cut_PreProcess(cv::Mat src, cv::Mat &des);
bool Crop_up(cv::Mat src, cv::Mat &left,cv:: Mat &right);//根据文字分布向图像分成左右两块
bool Crop_down(cv::Mat src, cv::Mat &left, cv::Mat &right);
void RowCharacter(int charNums, int picCols);
void ColCharacter(int charNums, int picCols);
void  DivideVector(int charNums);
std::vector<Str_Long> StrVectorHistCal(std::vector<std::string> s);
std::vector<Int_Long> IntVectorHistCal(std::vector<int> a);

char NumCharIdentify(int k, int picCols);
char LetterIdentify(int k, int picCols);
char MixIdentify(int k, int picCols);


typedef struct {
	int row, col;
}Data;
int ConnectDomain(int k, int picCols);
int Check(int i, int j, int flag, volatile Data* Dp);


int PointScanMidCol(int k, int picCols);
int DomainScan(int k, int rowStart, int rowEnd, int colStart, int colEnd, int picCols);
int PointScanRow(int k, int rowStart, int rowEnd, int picCols, int pointSize);
int PointScanCol(int k, int colStart, int colEnd, int picCols, int pointSize);
int PointCoordRL(int k, int rowStart, int rowEnd, int picCols, int divideCol);
int PointCoordLR(int k, int rowStart, int rowEnd, int picCols, int divideCol);
int PointCoordUD(int k, int colStart, int colEnd, int picCols, int divideCol);
int PointCoordDU(int k, int colStart, int colEnd, int picCols, int divideCol);
char IdentifyOQ(int k, int picCols);
char IdentifyDOQ(int k, int picCols);
char IdentifyDQ0(int k, int picCols);
char Identify2Z(int k, int picCols);
char Identify5S(int k, int picCols);
char Identify8B(int k, int picCols);


class OcrRecognition
{
public:
	OcrRecognition(int wordHeigh,int wordWidth);
	~OcrRecognition();
	void recognize(cv::Mat im, FrameWord &wordInfor);
	void recognizeByNet(cv::Mat im, FrameWord &wordInfor);
	void ImageOcr_up(cv::Mat im, UpInfor &str_infor);
	void ImageOcr_up1(cv::Mat im, UpInfor &str_infor);
	void ImageOcr_up2(cv::Mat im, UpInfor &str_infor);
	void ImageOcr_down(cv::Mat im, DownInfor &str_infor);
	void word_frame2OCR(FrameWord &wordInfor, OCR_Result &ocr_result);
	void Suf_Process_OCR(OCR_Results_video &OCR_Original, OCR_Results_video &OCR_Processed);
	void Suf_Process_OCR_BasedFPS(OCR_Results_video &OCR_Original, OCR_Results_video &OCR_Processed);
	int GetVideoFrame(std::vector<int> seconds);
	std::vector<cv::Mat> Get_CharfromFrame(cv::Mat im);
	void Get_CharfromFrame(cv::Mat im, std::vector<cv::Mat> &up1, std::vector<cv::Mat>&up2, std::vector<cv::Mat>&down1);
private:
	int m_wordHeigh;
	int m_wordWidth;	
};

