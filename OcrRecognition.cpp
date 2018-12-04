#include "OcrRecognition.h"
//#include"TrainDataProcess.h"
#include<algorithm>
#include<math.h>

using namespace cv;
using namespace std;

int numWidth[CHARSUM] = { 0 };
int numHeight[CHARSUM] = { 0 };
unsigned char numData[CHARSUM][AREABORD * PICWIDTH] = { 0 };

int g_flagNum[2];
unsigned char g_connectBuf[CHARSUM + 2][PICWIDTH + 2];
int StandardWordWidth = 8;
int g_upleft_width = 65535;//用来记录时间和速度的分隔点
int g_downleft_width = 65535;//用来记录车号和后面字符的分割点
int g_upright_x = 0;
OcrRecognition::OcrRecognition(int wordHeigh, int wordWidth)
{
	m_wordHeigh = wordHeigh;
	m_wordWidth = wordWidth;
	//g_upleft_width = 65535;
	//g_downleft_width = 65535;
}

OcrRecognition::~OcrRecognition()
{
}

void OcrRecognition::recognize(cv::Mat im, FrameWord &wordInfor) {
	Mat imGray;
	if (im.channels() == 3) {
		cvtColor(im, imGray, CV_BGR2GRAY);
	}
	else {
		imGray = im.clone();
	}
	StandardWordWidth = m_wordWidth;
	Mat im_up, im_up1, im_up2, im_up3, im_down, im_down1;
	Rect rect(0,0,imGray.cols,m_wordHeigh);
	im_up = imGray(rect).clone();
	Cut_PreProcess(im_up, im_up1);
	rect.y = imGray.rows - m_wordHeigh - 1;
	rect.x = 30;//直接将车次两个字去掉，9.28增加
	rect.width = rect.width - 30;
	im_down= imGray(rect).clone();
	Cut_PreProcess(im_down, im_down1);
	if (m_wordHeigh > 25 || im_down1.cols > PICWIDTH)
	{
		resize(im_up1, im_up1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_up1, im_up1, 127, 255, CV_THRESH_BINARY);//缩放时会导致边缘像素值小于255，影响识别效果
		resize(im_down1, im_down1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_down1, im_down1, 127, 255, CV_THRESH_BINARY);
	}
	//ImageOcr_up(im_up1, wordInfor.infor_up);

	numWidth[CHARSUM] = { 0 };
	numHeight[CHARSUM] = { 0 };
	bool find = false;
	find = Crop_up(im_up1, im_up2, im_up3);//判断有没有找到字段间的分界线
	if (find)
	{
		ImageOcr_up1(im_up2, wordInfor.infor_up);
		ImageOcr_up2(im_up3, wordInfor.infor_up);
	}
	else
	{
		ImageOcr_up(im_up1, wordInfor.infor_up);
	}
	//9.28增加
	for (int i = 0; i < 10; i++)
	{
		int acc = 0;
		for (int j = 0; j < im_down1.rows; j++)
		{
			if (im_down1.data[j*im_down1.cols+i] > 0)
			{
				acc++;
				im_down1.data[j*im_down1.cols + i] = 0;
			}
		}
		if (acc == 0)//若出现全黑列
		{
			break;
		}
	}
	ImageOcr_down(im_down1, wordInfor.infor_down);
}

void OcrRecognition::recognizeByNet(cv::Mat im, FrameWord &wordInfor)
{
	vector<cv::Mat> chars_img1, chars_img2, chars_img3;
	Mat imGray;
	if (im.channels() == 3) {
		cvtColor(im, imGray, CV_BGR2GRAY);
	}
	else {
		imGray = im.clone();
	}
	StandardWordWidth = m_wordWidth;
	Mat im_up, im_up1, im_up2, im_up3, im_down, im_down1, im_down2, im_down3;
	Rect rect(0, 0, imGray.cols, m_wordHeigh);
	im_up = imGray(rect).clone();
	Cut_PreProcess(im_up, im_up1);
	rect.y = imGray.rows - m_wordHeigh - 1;
	rect.x = 30;//直接将车次两个字去掉，9.28增加
	rect.width = rect.width - 30;
	im_down = imGray(rect).clone();
	Cut_PreProcess(im_down, im_down1);
	if (m_wordHeigh > 25 || im_down1.cols > PICWIDTH)
	{
		resize(im_up1, im_up1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_up1, im_up1, 127, 255, CV_THRESH_BINARY);//缩放时会导致边缘像素值小于255，影响识别效果
		resize(im_up, im_up, Size(PICWIDTH, 25), (0, 0), (0, 0));
		resize(im_down1, im_down1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_down1, im_down1, 127, 255, CV_THRESH_BINARY);
		resize(im_down, im_down, Size(PICWIDTH, 25), (0, 0), (0, 0));
	}
	bool find = false;
	find = Crop_up(im_up1, im_up2, im_up3);//判断有没有找到字段间的分界线
	if (find)
	{
		cv::Mat orig_up1, orig_up2;
		orig_up1 = im_up(cv::Rect(0, 0, im_up2.cols, im_up2.rows)).clone();
		CharsCut(im_up2, orig_up1, chars_img1, 0, im_up2.rows, im_up2.cols);
		//cv::Mat input = qu::imgs2Mat(chars_img1);
		orig_up2 = im_up(cv::Rect(im_up2.cols, 0, im_up3.cols, im_up3.rows)).clone();
		CharsCut(im_up3, orig_up2, chars_img2, 0, im_up3.rows, im_up3.cols);
	}
	else
	{
		CharsCut(im_up1, im_up, chars_img1, 0, im_up1.rows, im_up1.cols);
	}
	find = Crop_up(im_down1, im_down2, im_down3);//判断有没有找到字段间的分界线
	if (find)
	{
		cv::Mat orig_down1, orig_down2;
		//只处理左边的
		orig_down1 = im_down(cv::Rect(0, 0, im_down2.cols, im_down2.rows)).clone();
		CharsCut(im_down2, orig_down1, chars_img3, 0, im_down2.rows, im_down2.cols);
	}
	else
	{
		CharsCut(im_down1, im_down, chars_img3, 0, im_down1.rows, im_down1.cols);
	}
}

vector<cv::Mat> OcrRecognition::Get_CharfromFrame(cv::Mat im)
{
	vector<cv::Mat> chars_img;
	Mat imGray;
	if (im.channels() == 3) {
		cvtColor(im, imGray, CV_BGR2GRAY);
	}
	else {
		imGray = im.clone();
	}
	StandardWordWidth = m_wordWidth;
	Mat im_up, im_up1, im_up2, im_up3, im_down, im_down1, im_down2, im_down3;
	Rect rect(0, 0, imGray.cols, m_wordHeigh);
	im_up = imGray(rect).clone();
	Cut_PreProcess(im_up, im_up1);
	rect.y = imGray.rows - m_wordHeigh - 1;
	rect.x = 30;//直接将车次两个字去掉，9.28增加
	rect.width = rect.width - 30;
	im_down = imGray(rect).clone();
	Cut_PreProcess(im_down, im_down1);
	if (m_wordHeigh > 25 || im_down1.cols > PICWIDTH)
	{
		resize(im_up1, im_up1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_up1, im_up1, 127, 255, CV_THRESH_BINARY);//缩放时会导致边缘像素值小于255，影响识别效果
		resize(im_up, im_up, Size(PICWIDTH, 25), (0, 0), (0, 0));//从原图截取字符
		resize(im_down1, im_down1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_down1, im_down1, 127, 255, CV_THRESH_BINARY);
		resize(im_down, im_down, Size(PICWIDTH, 25), (0, 0), (0, 0));//从原图截取字符
	}
	bool find = false;
	find = Crop_up(im_up1, im_up2, im_up3);//判断有没有找到字段间的分界线
	if (find)
	{
		cv::Mat orig_up1, orig_up2;
		orig_up1 = im_up(cv::Rect(0, 0, im_up2.cols, im_up2.rows)).clone();
		CharsCut(im_up2, orig_up1, chars_img, 0, im_up2.rows, im_up2.cols);
		orig_up2 = im_up(cv::Rect(im_up2.cols, 0, im_up3.cols,im_up3.rows)).clone();
		//CharsCut(im_up3, orig_up2, chars_img, 0, im_up3.rows, im_up3.cols);
		CharsCut(im_up3, orig_up2, chars_img, 0, im_up3.rows, im_up3.cols);
	}
	else
	{
		CharsCut(im_up1, im_up, chars_img, 0, im_up1.rows, im_up1.cols);
	}
	find = Crop_up(im_down1, im_down2, im_down3);//判断有没有找到字段间的分界线
	if (find)
	{
		cv::Mat orig_down1, orig_down2;
		//只处理左边的
		orig_down1 = im_down(cv::Rect(0, 0, im_down2.cols, im_down2.rows)).clone();
		//CharsCut(im_down2, orig_down1, chars_img, 0, im_down2.rows, im_down2.cols);
		CharsCut(im_down2, orig_down1, chars_img, 0, im_down2.rows, im_down2.cols);
	}
	else
	{
		//CharsCut(im_down1, im_down, chars_img, 0, im_down1.rows, im_down1.cols);
		CharsCut(im_down1, im_down1, chars_img, 0, im_down1.rows, im_down1.cols);
	}	
	return chars_img;
}

void OcrRecognition::Get_CharfromFrame(cv::Mat im, std::vector<cv::Mat> &up1, std::vector<cv::Mat>&up2, std::vector<cv::Mat>&down1)
{
	up1.clear();
	up2.clear();
	down1.clear();
	Mat imGray;
	if (im.channels() == 3) {
		cvtColor(im, imGray, CV_BGR2GRAY);
	}
	else {
		imGray = im.clone();
	}
	StandardWordWidth = m_wordWidth;
	Mat im_up, im_up1, im_up2, im_up3, im_down, im_down1, im_down2, im_down3;
	Rect rect(0, 0, imGray.cols, m_wordHeigh);
	im_up = imGray(rect).clone();
	Cut_PreProcess(im_up, im_up1);
	rect.y = imGray.rows - m_wordHeigh - 1;
	rect.x = 30;//直接将车次两个字去掉，9.28增加
	rect.width = rect.width - 30;
	im_down = imGray(rect).clone();
	Cut_PreProcess(im_down, im_down1);
	if (m_wordHeigh > 25 || im_down1.cols > PICWIDTH)
	{
		resize(im_up1, im_up1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_up1, im_up1, 127, 255, CV_THRESH_BINARY);//缩放时会导致边缘像素值小于255，影响识别效果
		resize(im_up, im_up, Size(PICWIDTH, 25), (0, 0), (0, 0));//从原图截取字符
		resize(im_down1, im_down1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_down1, im_down1, 127, 255, CV_THRESH_BINARY);
		resize(im_down, im_down, Size(PICWIDTH, 25), (0, 0), (0, 0));//从原图截取字符
	}
	bool find = false;
	find = Crop_up(im_up1, im_up2, im_up3);//判断有没有找到字段间的分界线
	if (find)
	{
		cv::Mat orig_up1, orig_up2;
		orig_up1 = im_up(cv::Rect(0, 0, im_up2.cols, im_up2.rows)).clone();
		CharsCut(im_up2, orig_up1, up1, 0, im_up2.rows, im_up2.cols);
		orig_up2 = im_up(cv::Rect(im_up2.cols, 0, im_up3.cols, im_up3.rows)).clone();
		//CharsCut(im_up3, orig_up2, chars_img, 0, im_up3.rows, im_up3.cols);
		CharsCut(im_up3, im_up3, up2, 0, im_up3.rows, im_up3.cols);
		cv::imwrite("F:/workspace/videos/binary.jpg", im_up3);
	}
	else
	{
		CharsCut(im_up1, im_up, up1, 0, im_up1.rows, im_up1.cols);
	}
	find = Crop_down(im_down1, im_down2, im_down3);//判断有没有找到字段间的分界线
	if (find)
	{
		cv::Mat orig_down1, orig_down2;
		//只处理左边的
		orig_down1 = im_down(cv::Rect(0, 0, im_down2.cols, im_down2.rows)).clone();
		//CharsCut(im_down2, orig_down1, chars_img, 0, im_down2.rows, im_down2.cols);
		CharsCut(im_down2, im_down2, down1, 0, im_down2.rows, im_down2.cols);
	}
	else
	{
		//CharsCut(im_down1, im_down, chars_img, 0, im_down1.rows, im_down1.cols);
		CharsCut(im_down1, im_down1, down1, 0, im_down1.rows, im_down1.cols);
	}
}

void OcrRecognition::ImageOcr_up(Mat im, UpInfor &str_infor) {
	
	imwrite("F:/workspace/videos/up.jpg", im);
	int colNum = im.cols;
	int charNums1 = NumCut(im.data, 0, im.rows, im.cols);
	RowCharacter(charNums1, colNum); //计算横线特征
	ColCharacter(charNums1, colNum); //计算竖线特征
	DivideVector(charNums1); //计算特征向量


	int flag = 0;
	int flag1 = 0;
	string str_date;
	string str_time,str_km,str_s;
	int tag = 5;
	for (int i = 0; i < charNums1; i++)
	{
		if (i < 8)
		{
			str_date += NumCharIdentify(i, colNum);
		}
		else if (i >= 8 && i < 16) //时间一共16个字符
		{
			str_time += NumCharIdentify(i, colNum);
		}
		else
		{
			if (numHeight[i] > 12 || numWidth[i] > 13) //字符高度大于12或者宽度大于13，考虑为汉字，暂不识别
			//if (numWidth[i] > 13) //字符宽度大于13，考虑为汉字，暂不识别
			{
				str_km += '?';
			}
			else
			{
				str_km += NumCharIdentify(i, colNum);
				if (str_km[str_km.size()-1] == 'k' && flag == 0)
				{
					str_km += 'm';
					str_km += '/';
					str_km += 'h';
					i += 3;
					flag = 1;
				}
				else if (str_km[str_km.size() - 1] == 'm' && flag1 == 0)
				{
					if (str_km.size() > 2)
					{
						str_km[str_km.size() - 2] = 'k';
					}
					str_km += '/';
					str_km += 'h';
					i += 2;
					flag1 = 1;
				}
				if (str_km[str_km.size() - 1] == 'k' && flag == 1)
				{
					str_km += 'm';
					break;
				}
				else if (str_km[str_km.size() - 1] == 'm' && flag1 == 1)
				{
					str_km[str_km.size() - 2] = 'k';
					break;
				}
			}

		}
	}
	//后处理
	/*string tmp,tmp1;
	if (str_date[4] == '.')
	{
		str_time = str_date[7] + str_time;
		str_km = str_time[7] + str_km;
		for (int i = 0; i < 7; i++)
		{
			tmp[i] = str_date[i];
			tmp1[i] = str_time[i];
		}
		str_date = tmp;
		str_time = tmp1;
	}
	if (str_time[4] == ':')
	{
		str_km = str_time[7] + str_km;
		for (int i = 0; i < 7; i++)
		{
			tmp1[i] = str_time[i];
		}
		str_time = tmp1;
	}*/
	//

	str_infor.str_date = str_date;
	str_infor.str_time = str_time;
	str_infor.str_km = str_km;
}

void OcrRecognition::ImageOcr_up1(Mat im, UpInfor &str_infor) {
	imwrite("F:/workspace/videos/up1.jpg", im);
	int colNum = im.cols;
	int charNums1 = NumCut(im.data, 0, im.rows, im.cols);
	RowCharacter(charNums1, colNum); //计算横线特征
	ColCharacter(charNums1, colNum); //计算竖线特征
	DivideVector(charNums1); //计算特征向量


	string str_date, str_time;
	for (int i = 0; i < charNums1; i++)
	{
		if (i < 8)
		{
			if (i == 7 && str_date[4] == '.')//日期可能缺位
			{
				str_time += NumCharIdentify(i, colNum);
			}
			else
			{
				str_date += NumCharIdentify(i, colNum);
			}			
		}
		else if (i >= 8 && i < 16) //时间一共16个字符
		{
			str_time += NumCharIdentify(i, colNum);
		}
		
	}
	str_infor.str_date = str_date;
	str_infor.str_time = str_time;
}

void OcrRecognition::ImageOcr_up2(Mat im, UpInfor &str_infor) {
	imwrite("F:/workspace/videos/up2.jpg", im);
	int colNum = im.cols;
	int charNums1 = NumCut1(im.data, 0, im.rows, im.cols);
	RowCharacter(charNums1, colNum); //计算横线特征
	ColCharacter(charNums1, colNum); //计算竖线特征
	DivideVector(charNums1); //计算特征向量


	int flag = 0;
	int flag1 = 0;
	string str_km;
	int tag = 5;
	for (int i = 0; i < charNums1; i++)
	{
		if (numHeight[i] > 12 || numWidth[i] > 13) //字符高度大于12或者宽度大于13，考虑为汉字，暂不识别
		//if (numWidth[i] > 13) //字符宽度大于13，考虑为汉字，暂不识别
		{
			str_km += '?';
		}
		else
		{
			str_km += NumCharIdentify(i, colNum);
			if (str_km[str_km.size() - 1] == 'k' && flag == 0)
			{
				str_km += 'm';
				str_km += '/';
				str_km += 'h';
				i += 3;
				flag = 1;
			}
			else if (str_km[str_km.size() - 1] == 'm' && flag1 == 0)
			{
				if (str_km.size() > 2)
				{
					str_km[str_km.size() - 2] = 'k';
				}
				str_km += '/';
				str_km += 'h';
				i += 2;
				flag1 = 1;
			}
			if (str_km[str_km.size() - 1] == 'k' && flag == 1)
			{
				str_km += 'm';
				break;
			}
			else if (str_km[str_km.size() - 1] == 'm' && flag1 == 1)
			{
				str_km[str_km.size() - 2] = 'k';
				break;
			}
			/*if (str_km[str_km.size() - 1] == 'k' && flag == 0)
			{
				str_km += 'm';
				str_km += '/';
				str_km += 'h';
				i += 3;
				flag = 1;
			}
			if (str_km[str_km.size() - 1] == 'k' && flag == 1)
			{
				str_km += 'm';
				break;
			}*/
		}
	}
	str_infor.str_km = str_km;
}

void OcrRecognition::ImageOcr_down(Mat im, DownInfor &str_infor) {
	imwrite("F:/workspace/videos/down.jpg", im);
	int colNum = im.cols;
	int charNums2 = NumCut1(im.data, 0, im.rows, im.cols);
	RowCharacter(charNums2, colNum); //计算横线特征
	ColCharacter(charNums2, colNum); //计算竖线特征
	DivideVector(charNums2); //计算特征向量

	for (int i = 0; i < charNums2; i++)
	{
		if (numHeight[i] > 12 || numWidth[i] > 13) //字符高度大于12或者宽度大于13，考虑为汉字，暂不识别
		//if (numWidth[i] > 13) //字符宽度大于13，考虑为汉字，暂不识别
		{
			str_infor.str_result += '?';
		}
		else
		{
			str_infor.str_result += MixIdentify(i, colNum);
		}
	}
}


/*****************************************************************
*	函数名:	otsu
*	功能描述：算法对输入的灰度图进行二值化
*			  将直方图分成两个部分，使得两部分之间的距离最大。划分点就乔蟮玫你兄?
*	形参数:  picPtr    //指向图像区域首地址的指针
*			   picRows   //图像区域行数
*			   picCols   //图像区域列数
*	返回值：二值化阈值
*   全局变量:  无
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	     Revised by	  Item Description
*   V1.0	2018/4/12    张光华      原始版本
******************************************************************/
void otsu(unsigned char *picPtr, int startRows, int picHeight, int picCols)
{
	/*	unsigned char *np;  // 图像指针
	int ihist[256] = {0};   // 图像直方图，256个点
	int i, j, k;  // various counters
	int n, n1, n2, gmin, gmax, sum, csum;
	double m1, m2, fmax, sb;
	int thresholdValue = 0;


	//	memset(ihist, 0, sizeof(ihist));

	gmin = 255; gmax = 0;

	// 生成直方图
	for (i = 0; i < picRows; i++)
	{
	np = picPtr + i*picCols;
	for (j = 0; j < picCols; j++)
	{
	ihist[*np]++;

	if (*np > gmax)
	gmax = *np;
	if (*np < gmin)
	gmin = *np;
	np++;
	}
	}


	// set up everything
	sum = csum = 0;
	n = 0;

	for (k = 0; k < 255; k++)
	{
	sum += k * ihist[k];    // x*f(x)质量矩
	n += ihist[k];       // f(x)质量
	}

	// do the otsu global thresholding method
	fmax = -1.0;
	n1 = 0;
	for (k = 0; k < 255; k++)
	{
	n1 += ihist[k];
	if (!n1)
	continue;
	n2 = n - n1;
	if (n2 == 0)
	break;
	csum += k *ihist[k];
	m1 = (float)(csum / n1);
	m2 = (float)((sum - csum) / n2);
	sb = (double)n1 *(double)n2 *(m1 - m2) * (m1 - m2);
	//bbg: note: can be optimized.
	if (sb > fmax)
	{
	fmax = sb;
	thresholdValue = k;  // at this point we have our thresholding value
	}
	}*/
	//	Mat otsuTest(startRows + picHeight, picCols, CV_8U);

	for (int i = startRows; i < startRows + picHeight; i++)
	{
		for (int j = 0; j < picCols; j++)  //前段
		{
			picPtr[i*picCols + j] = (picPtr[i*picCols + j]  > 230) ? 255 : 0;
			//	otsuTest.data[i*picCols + j] = picPtr[i*picCols + j];

		}
	}

	//return thresholdValue;
}

int NumCut(unsigned char *picPtr, int startRows, int picHeight, int picCols)
{
	int i, j, k = 0;
	int oneNum = 0, upTag = 0, leftRightFlag = 0, upDownFlag = 0; //标记下次要扫描的边界，0表示左边界，1表示右边界
	int areaStart = 0, areaUpBord = 0, areaDownBord = startRows + picHeight, upBord = 0, downBord = 0, leftBord = 0, rightBord = 0;

	//	IplImage*img2 = cvCreateImage(cvSize(picCols, picRows), IPL_DEPTH_8U, 1);
	Mat cutTest(startRows + picHeight, picCols, CV_8U);
	Mat cutAreaTest(startRows + picHeight, picCols, CV_8U);

	for (i = startRows; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picCols; j++)
		{
			cutAreaTest.data[(i - startRows)*picCols + j] = picPtr[i*picCols + j];
		}
	}

	for (i = startRows + 2; i < startRows + picHeight; i++)
	{
		oneNum = 0;

		for (j = 0; j < picCols / 2; j++)
		{
			oneNum += picPtr[i*picCols + j]; //记录每列中255像素点的个数
		}

		if (i == startRows + 10)
		{
			areaStart = 1; //找了10行还未找到，说明区域上方无噪音
			break;
		}
		if (oneNum <= 255 * 2) //允许2个点的噪音
		{
			areaStart = i;  //剔除区域上方噪音
			break;
		}
	}

	for (i = areaStart; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picCols / 2; j++)
		{
			if (picPtr[i*picCols + j] == 255 && picPtr[i*picCols + j + 1] == 255)
			{
				areaUpBord = i; //数字区域上边界
				upTag = 1;
				break;
			}
		}
		if (upTag == 1)
		{
			break;
		}

	}

	for (i = areaUpBord + 6; i < startRows + picHeight; i++)
	{
		oneNum = 0;

		for (j = 0; j < picCols; j++)
		{
			oneNum += picPtr[i*picCols + j]; //记录每列中255像素点的个数
		}

		if (oneNum <= 255 * 4) //允许4个点的噪音
		{
			areaDownBord = i; //将字符下边界存放到数组中 
			break;
		}
		if (i == startRows + picHeight - 1)
		{
			areaDownBord = startRows + picHeight - 1; //最终没找到取末行
		}
	}

	//	numHeight = areaDownBord - areaUpBord;

	for (j = 0; j < picCols; j++)
	{
		if (leftRightFlag == 0)  //leftRightFlag=0表示要扫描字符的左边界
		{
			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				if (picPtr[i*picCols + j] == 255 && picPtr[(i + 1)*picCols + j] == 255)
				{
					leftBord = j; //数字左边界
					leftRightFlag = 1; //将leftRightFlag置1，表示下次要扫描右边界
					if (j < picCols - 2)
					{
						j += 2; //右边界从后3列开始扫描
					}
					break;
				} //if (picPtr[i*outlinesize + j] == 255)...end			
			} //for (i = 0; i <= picRows; i++)...end
		}  //if (leftRightFlag == 0)...end


		if (leftRightFlag == 1) //leftRightFlag=1表示要扫描字符的右边界
		{
			oneNum = 0;

			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				oneNum += picPtr[i*picCols + j]; //记录每列中255像素点的个数
			}

			if (oneNum <= 0) //找到全黑的一列（之前允许1个点噪音后取消）
			{
				rightBord = j; //将字符右边界存放到数组中 
				leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
				numWidth[k] = rightBord - leftBord;
			}
			else if (j == leftBord + 16) //扫描16列(汉字14列)后仍未找到字符右边界，考虑2个字符未分开
			{
				rightBord = leftBord + 8; //字符宽度一般小于12，将其定为右边界
				j = leftBord + 8; //下个字符左边界起始点
				leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
				numWidth[k] = rightBord - leftBord;
			}
			else if (j == picCols - 1) //字符右边界刚好是区域右边界
			{
				rightBord = j; //将字符右边界存放到数组中 
				leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
				numWidth[k] = rightBord - leftBord;
			}

			if (leftRightFlag == 0) //找到右边界，进行上下分割
			{
				for (i = areaUpBord; i <= areaDownBord; i++)
				{
					if (upDownFlag == 0)
					{
						for (j = leftBord; j <= rightBord; j++)
						{
							if (picPtr[i*picCols + j] == 255)
							{
								upBord = i;
								upDownFlag = 1;
								if (i + 6 <= areaDownBord)
								{
									i += 6;
								}
								break;

							}
						}
					}
					if (upDownFlag == 1)
					{
						oneNum = 0;
						for (j = leftBord; j <= rightBord; j++)
						{
							oneNum += picPtr[i*picCols + j]; //记录每列中255像素点的个数
						}

						if (oneNum == 0)
						{
							downBord = i;
							numHeight[k] = downBord - upBord;
							upDownFlag = 0;
							break;
						}
						if (i == areaDownBord)
						{
							downBord = areaDownBord;
							numHeight[k] = downBord - upBord;
							upDownFlag = 0;
							oneNum = 0;
							break;
						}

					}
				}

				for (i = upBord; i < downBord; i++)
				{
					for (j = leftBord; j < rightBord; j++)
					{
						numData[k][(i - upBord)*picCols + j - leftBord] = picPtr[i*picCols + j];
						cutTest.data[i*picCols + j] = numData[k][(i - upBord)*picCols + j - leftBord];
					}
				}
				k++;
			}
		} //if (leftRightFlag == 1)...end
	} //for (j=0;j<picCols; j++)...end	

	return k;
}

int NumCut1(unsigned char *picPtr, int startRows, int picHeight, int picCols)
{
	int i, j, k = 0;
	int oneNum = 0, upTag = 0, leftRightFlag = 0, upDownFlag = 0; //标记下次要扫描的边界，0表示左边界，1表示右边界
	int areaStart = 0, areaUpBord = 0, areaDownBord = startRows + picHeight, upBord = 0, downBord = 0, leftBord = 0, rightBord = 0;

	//	IplImage*img2 = cvCreateImage(cvSize(picCols, picRows), IPL_DEPTH_8U, 1);
	Mat cutTest(startRows + picHeight, picCols, CV_8U);
	Mat cutAreaTest(startRows + picHeight, picCols, CV_8U);

	for (i = startRows; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picCols; j++)
		{
			cutAreaTest.data[(i - startRows)*picCols + j] = picPtr[i*picCols + j];
		}
	}

	for (i = startRows + 2; i < startRows + picHeight; i++)
	{
		oneNum = 0;

		for (j = 0; j < picCols / 2; j++)
		{
			oneNum += picPtr[i*picCols + j]; //记录每列中255像素点的个数
		}

		if (i == startRows + 10)
		{
			areaStart = 1; //找了10行还未找到，说明区域上方无噪音
			break;
		}
		if (oneNum <= 255 * 2) //允许2个点的噪音
		{
			areaStart = i;  //剔除区域上方噪音
			break;
		}
	}

	for (i = areaStart; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picCols / 2; j++)
		{
			if (picPtr[i*picCols + j] == 255 && picPtr[i*picCols + j + 1] == 255)
			{
				areaUpBord = i; //数字区域上边界
				upTag = 1;
				break;
			}
		}
		if (upTag == 1)
		{
			break;
		}

	}

	for (i = areaUpBord + 6; i < startRows + picHeight; i++)
	{
		oneNum = 0;

		for (j = 0; j < picCols; j++)
		{
			oneNum += picPtr[i*picCols + j]; //记录每列中255像素点的个数
		}

		if (oneNum <= 255 * 4) //允许4个点的噪音
		{
			areaDownBord = i; //将字符下边界存放到数组中 
			break;
		}
		if (i == startRows + picHeight - 1)
		{
			areaDownBord = startRows + picHeight - 1; //最终没找到取末行
		}
	}

	//	numHeight = areaDownBord - areaUpBord;

	for (j = 0; j < picCols; j++)
	{
		if (leftRightFlag == 0)  //leftRightFlag=0表示要扫描字符的左边界
		{
			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				if (picPtr[i*picCols + j] == 255 && picPtr[(i + 1)*picCols + j] == 255)
				{
					leftBord = j; //数字左边界
					leftRightFlag = 1; //将leftRightFlag置1，表示下次要扫描右边界
					if (j < picCols - 2)
					{
						j += 2; //右边界从后3列开始扫描
					}
					break;
				} //if (picPtr[i*outlinesize + j] == 255)...end			
			} //for (i = 0; i <= picRows; i++)...end
		}  //if (leftRightFlag == 0)...end


		if (leftRightFlag == 1) //leftRightFlag=1表示要扫描字符的右边界
		{
			oneNum = 0;

			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				oneNum += picPtr[i*picCols + j]; //记录每列中255像素点的个数
			}

			if (oneNum <= 0) //找到全黑的一列（之前允许1个点噪音后取消）
			{
				rightBord = j; //将字符右边界存放到数组中 
				leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
				//numWidth[k] = rightBord - leftBord;
			}
			/*else if (j == leftBord + 16) //扫描16列(汉字14列)后仍未找到字符右边界，考虑2个字符未分开
			{
				rightBord = leftBord + 10; //字符宽度一般小于12，将其定为右边界
				j = leftBord + 10; //下个字符左边界起始点
				leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
				//numWidth[k] = rightBord - leftBord;
			}*/
			else if (j == picCols - 1) //字符右边界刚好是区域右边界
			{
				rightBord = j; //将字符右边界存放到数组中 
				leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
				//numWidth[k] = rightBord - leftBord;
			}

			if (leftRightFlag == 0) //找到右边界，进行上下分割
			{
				for (i = areaUpBord; i <= areaDownBord; i++)
				{
					if (upDownFlag == 0)
					{
						for (j = leftBord; j <= rightBord; j++)
						{
							if (picPtr[i*picCols + j] == 255)
							{
								upBord = i;
								upDownFlag = 1;
								if (i + 6 <= areaDownBord)
								{
									i += 6;
								}
								break;

							}
						}
					}
					if (upDownFlag == 1)
					{
						oneNum = 0;
						for (j = leftBord; j <= rightBord; j++)
						{
							oneNum += picPtr[i*picCols + j]; //记录每列中255像素点的个数
						}

						if (oneNum == 0)
						{
							downBord = i;
							numHeight[k] = downBord - upBord;
							upDownFlag = 0;
							break;
						}
						if (i == areaDownBord)
						{
							downBord = areaDownBord;
							numHeight[k] = downBord - upBord;
							upDownFlag = 0;
							oneNum = 0;
							break;
						}

					}
				}

				int strnum = round((double)(rightBord - leftBord) / 8.0);//设定一般字符宽为10
				//CvRect rect;
				if (numHeight[k]<13 && strnum > 1)//字符出现了粘连，强制分割,但中文字符不管
				{
					int strwidth = (rightBord - leftBord) / strnum;
					int strheight = numHeight[k];
					for (int r = 0; r < strnum; r++)
					{
						int rightBordTmp = leftBord + strwidth;

						for (i = upBord; i < downBord; i++)
						{
							for (j = leftBord; j <= rightBordTmp; j++)
							{
								numData[k][(i - upBord)*picCols + j - leftBord] = picPtr[i*picCols + j];
								cutTest.data[i*picCols + j] = numData[k][(i - upBord)*picCols + j - leftBord];
							}
						}
						numHeight[k] = strheight;
						numWidth[k] = strwidth;
						leftBord = rightBordTmp + 1;
						k++;
					}
					j = rightBord;
				}
				else
				{
					for (i = upBord; i < downBord; i++)
					{
						for (j = leftBord; j < rightBord; j++)
						{
							numData[k][(i - upBord)*picCols + j - leftBord] = picPtr[i*picCols + j];
							cutTest.data[i*picCols + j] = numData[k][(i - upBord)*picCols + j - leftBord];
						}
					}
					numWidth[k] = rightBord - leftBord;
					k++;
				}				
			}
		} //if (leftRightFlag == 1)...end
	} //for (j=0;j<picCols; j++)...end	

	return k;
}

void CharsCut(cv::Mat im, cv::Mat orig_im,vector<cv::Mat> &char_imgs, int startRows, int picHeight, int picWidth)
{
	int i, j, k = 0;
	int oneNum = 0, upTag = 0, leftRightFlag = 0, upDownFlag = 0; //标记下次要扫描的边界，0表示左边界，1表示右边界
	int areaStart = 0, areaUpBord = 0, areaDownBord = startRows + picHeight, upBord = 0, downBord = 0, leftBord = 0, rightBord = 0;

	Mat cutTest(startRows + picHeight, picWidth, CV_8U);
	Mat cutAreaTest(startRows + picHeight, picWidth, CV_8U);
	unsigned char *picPtr = im.data;

	for (i = startRows; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picWidth; j++)
		{
			cutAreaTest.data[(i - startRows)*picWidth + j] = picPtr[i*picWidth + j];
		}
	}

	for (i = startRows + 2; i < startRows + picHeight; i++)
	{
		oneNum = 0;
		for (j = 0; j < picWidth / 2; j++)
		{
			oneNum += picPtr[i*picWidth + j]; //记录每行中255像素点的个数
		}

		if (i == startRows + 10)
		{
			areaStart = 1; //找了10行还未找到，说明区域上方无噪音
			break;
		}
		if (oneNum <= 255 * 2) //允许2个点的噪音
		{
			areaStart = i;  //剔除区域上方噪音
			break;
		}
	}

	for (i = areaStart; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picWidth / 2; j++)
		{
			if (picPtr[i*picWidth + j] == 255 && picPtr[i*picWidth + j + 1] == 255)
			{
				areaUpBord = i; //数字区域上边界（汉字上边界与数字一致）
				upTag = 1;
				break;
			}
		}
		if (upTag == 1)
		{
			break;
		}

	}

	for (i = areaUpBord + 6; i < startRows + picHeight; i++)
	{
		oneNum = 0;
		for (j = 0; j < picWidth; j++)
		{
			oneNum += picPtr[i*picWidth + j]; //记录每行中255像素点的个数
		}

		if (oneNum <= 255 * 4) //允许4个点的噪音
		{
			areaDownBord = i; //将字符下边界存放到数组中 
			break;
		}
		if (i == startRows + picHeight - 1)
		{
			areaDownBord = startRows + picHeight - 1; //最终没找到取末行
		}
	}

	for (j = 0; j < picWidth; j++)
	{
		if (leftRightFlag == 0)  //leftRightFlag=0表示要扫描字符的左边界
		{
			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				if (picPtr[i*picWidth + j] == 255 && picPtr[(i + 1)*picWidth + j] == 255)//扫描列
				{
					leftBord = j; //数字左边界
					leftRightFlag = 1; //将leftRightFlag置1，表示下次要扫描右边界
					if (j < picWidth - 2)
					{
						j += 2; //右边界从后3列开始扫描
					}
					break;
				} //if (picPtr[i*outlinesize + j] == 255)...end			
			} //for (i = 0; i <= picRows; i++)...end
		}  //if (leftRightFlag == 0)...end


		if (leftRightFlag == 1) //leftRightFlag=1表示要扫描字符的右边界
		{
			oneNum = 0;

			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				oneNum += picPtr[i*picWidth + j]; //记录每列中255像素点的个数
			}

			if (oneNum <= 0) //找到全黑的一列（之前允许1个点噪音后取消）
			{
				rightBord = j; //将字符右边界存放到数组中 
				leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
								   //numWidth[k] = rightBord - leftBord;
			}
			/*else if (j == leftBord + 16) //扫描16列(汉字14列)后仍未找到字符右边界，考虑2个字符未分开
			{
			rightBord = leftBord + 10; //字符宽度一般小于12，将其定为右边界
			j = leftBord + 10; //下个字符左边界起始点
			leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
			//numWidth[k] = rightBord - leftBord;
			}*/
			else if (j == picWidth - 1) //字符右边界刚好是区域右边界
			{
				rightBord = j; //将字符右边界存放到数组中 
				leftRightFlag = 0; //将leftRightFlag置0，表示下次要扫描左边界
			}

			if (leftRightFlag == 0) //找到右边界，进行上下分割
			{
				for (i = areaUpBord; i <= areaDownBord; i++)
				{
					if (upDownFlag == 0)
					{
						for (j = leftBord; j <= rightBord; j++)
						{
							if (picPtr[i*picWidth + j] == 255)
							{
								upBord = i;
								upDownFlag = 1;
								if (i + 6 <= areaDownBord)
								{
									i += 6;
								}
								break;

							}
						}
					}
					if (upDownFlag == 1)
					{
						oneNum = 0;
						for (j = leftBord; j <= rightBord; j++)
						{
							oneNum += picPtr[i*picWidth + j]; //记录每列中255像素点的个数
						}

						if (oneNum == 0)
						{
							downBord = i;
							numHeight[k] = downBord - upBord;
							upDownFlag = 0;
							break;
						}
						if (i == areaDownBord)
						{
							downBord = areaDownBord;
							numHeight[k] = downBord - upBord;
							upDownFlag = 0;
							oneNum = 0;
							break;
						}
					}
				}

				int charnum;
				//int charnum = round((double)(rightBord - leftBord) / 7.0);//设定一般字符宽为7，结果四舍五入
				if (rightBord - leftBord <= 16)
				{
					charnum = round((double)(rightBord - leftBord) / 8.6);//1026更改，解决m不全的问题
				}
				else
				{
					charnum = round((double)(rightBord - leftBord) / 8.0);//可能出现多个字符粘连
				}
																		 
				if (numHeight[k]<13 && numHeight[k] > 5 && charnum >= 1)//字符出现了粘连，强制分割,但中文字符不管
				{
					int strwidth = (rightBord - leftBord) / charnum;
					//int strwidth = 7;
					int strheight = numHeight[k];
					for (int r = 0; r < charnum; r++)
					{
						int rightBordTmp = leftBord + strwidth;

						cv::Rect roi;
						if (leftBord - 1 < 0)
						{
							roi.x = 0;
						}
						else
						{
							roi.x = leftBord - 1;
						}
						if (upBord - 1 < 0)
						{
							roi.y = 0;
						}
						else
						{
							roi.y = upBord - 1;
						}
						if (rightBordTmp - leftBord + 2 > im.cols - 1 - roi.x)
						{
							roi.width = im.cols - 1 - roi.x;
						}
						else
						{
							roi.width = rightBordTmp - leftBord + 2;
						}
						if (downBord - upBord + 2 > im.rows - 1 - roi.y)
						{
							roi.height = im.rows - 1 - roi.y;
						}
						else
						{
							roi.height = downBord - upBord + 2;
						}
						//cv::Mat char_img = im(roi);
						cv::Mat char_img = orig_im(roi);
						char_imgs.push_back(char_img);
						//cv::imwrite("F:/workspace/videos/tmp.jpg", char_img);//保存字符图像
						leftBord = rightBordTmp;//10.25改不加1
						k++;
					}
					j = rightBord;
				}
				else if(charnum<1)//可能是点.和:
				{
					cv::Rect roi;
					if (leftBord - 2 < 0)
					{
						roi.x = 0;
					}
					else
					{
						roi.x = leftBord - 2;
					}
					if (upBord - 2 < 0)
					{
						roi.y = 0;
					}
					else
					{
						roi.y = upBord - 2;
					}
					if (rightBord - leftBord + 4 > im.cols - 1 - roi.x)
					{
						roi.width = im.cols - 1 - roi.x;
					}
					else
					{
						roi.width = rightBord - leftBord + 4;
					}
					if (downBord - upBord + 4 > im.rows - 1 - roi.y)
					{
						roi.height = im.rows - 1 - roi.y;
					}
					else
					{
						roi.height = downBord - upBord + 4;
					}
					//cv::Rect roi = cv::Rect(leftBord - 2, upBord - 2, rightBord - leftBord + 4, downBord - upBord + 4);
					//cv::Mat char_img = im(roi);
					cv::Mat char_img = orig_im(roi);
					char_imgs.push_back(char_img);
					//cv::imwrite("F:/workspace/videos/tmp.jpg", char_img);//保存字符图像
					k++;
				}
			}
		} //if (leftRightFlag == 1)...end
	} //for (j=0;j<picCols; j++)...end	
}

float colLineCharacter[30][2]; //存储字符竖线特征
float rowLineCharacter[30][2]; //存储字符横线特征
int charColVector[30]; //存储前4个字符的竖线特征向量
int charRowVector[30]; //存储前4个字符的横线特征向量
					   /*****************************************************************
					   *	函数名:	ColCharacter
					   *	功能描述: 用2X1的框记录竖线特征值
					   *	形式参数: 无
					   *	返回值：无
					   *   全局变量：
					   *   文件静态变量：无
					   *   函数静态变量：无
					   *------------------------------------------------------------------
					   *	Revision History
					   *	No.	    Date	     Revised by	  Item Description
					   *   V1.0	2018/04/17    ZhangGH	  	原始版本
					   ******************************************************************/
void ColCharacter(int charNums, int picCols)
{
	int k, i;
	int continueZeroNum = 0; //记录连续0像素点的个数
	int maxContinueZeroNum = 0; //记录最大的连续0像素点的个数

	memset(colLineCharacter[0], 0, sizeof(colLineCharacter));

	for (k = 0; k < charNums; k++)
	{

		//左竖线
		continueZeroNum = 0;
		maxContinueZeroNum = 0;
		for (i = 0; i < numHeight[k]; i++)
		{
			if (numData[k][i*picCols] == 255 || numData[k][i*picCols + 1] == 255)  //2X1的窗口
			{
				continueZeroNum++; //连续出现2?袼氐悖continueZeroNum+1
				if (continueZeroNum > maxContinueZeroNum)
				{
					maxContinueZeroNum = continueZeroNum; //记录最大的连续0像素点个数
				}
			}
			else
			{
				continueZeroNum = 0; //如果0像素点后面那个像素点不为0，则将连续0像素点鍪辶?
			}
		}
		//录竖线特征值
		colLineCharacter[k][0] = maxContinueZeroNum / (float)numHeight[k]; //记录竖线特征百分比

		continueZeroNum = 0;
		maxContinueZeroNum = 0;
		for (i = 0; i < numHeight[k]; i++)
		{
			if (numData[k][i*picCols + numWidth[k] - 2] == 255 || numData[k][i*picCols + numWidth[k] - 1] == 255)   //2X1的窗口
			{
				continueZeroNum++; //连续鱿?个0像素点，continueZeroNum+1
				if (continueZeroNum > maxContinueZeroNum)
				{
					maxContinueZeroNum = continueZeroNum; //记甲畲蟮牧?像素点个数
				}
			}
			else
			{
				continueZeroNum = 0; //如果0像素点后面那个像素点不为0，则将连续0像素点个数清
			}
		}
		//记录竖线特征值
		colLineCharacter[k][1] = maxContinueZeroNum / (float)numHeight[k]; //记录竖线特征百分比
	}
}

/*****************************************************************
*	函数名:	RowCharacter
*	功能描述: 用2X1的框记录横线特征值
*	形式参数: 无
*	返回值：无
*   全局变量：g_charBreadth[CHAR_NUM]、g_charHeight[CHAR_NUM]
g_charSection[CHAR_NUM][CHAR_ROWSIZE][CHAR_COLSIZE]
g_perRowLineCharacter[CHAR_NUM][CHAR_ROWSIZE]
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	     Revised by	  Item Description
*   V1.0	2018/04/17     ZhangGH	   		原始版本
******************************************************************/
void RowCharacter(int charNums, int picCols)
{
	int k, j;
	int continueZeroNum = 0; //记录连续0像素点的个数
	int maxContinueZeroNum = 0; //记录最大的连续0像素点的个数
								//	Mat rowTest(20, picCols, CV_8U);

	memset(rowLineCharacter[0], 0, sizeof(rowLineCharacter));

	//上横线
	for (k = 0; k < charNums; k++) //只计算前面4个字符的横竖线特征
	{

		/*	for (i = 0; i < numHeight[k]; i++)
		{
		for (j = 0; j < numWidth[k]; j++)
		{
		rowTest.data[i*picCols + j] = numData[k][i * picCols + j];
		}
		}*/

		continueZeroNum = 0;
		maxContinueZeroNum = 0;
		for (j = 0; j < numWidth[k]; j++)
		{
			if (numData[k][0 * picCols + j] == 255 || numData[k][1 * picCols + j] == 255)  //2X1的窗口
			{
				continueZeroNum++; //连续出现2个0像素点，continueZeroNum加1
				if (continueZeroNum > maxContinueZeroNum)
				{
					maxContinueZeroNum = continueZeroNum; //记录最大的连续0像素点个数
				}
			}
			else
			{
				continueZeroNum = 0; //如果0像素点后面那个像素点不为0，则将连续0像素点个数清零
			}
		}
		//记录横线特征值
		rowLineCharacter[k][0] = maxContinueZeroNum / (float)numWidth[k]; //记录横线特征百分比

		continueZeroNum = 0;
		maxContinueZeroNum = 0;
		for (j = 0; j < numWidth[k]; j++)
		{
			if (numData[k][(numHeight[k] - 2) * picCols + j] == 255 || numData[k][(numHeight[k] - 1) * picCols + j] == 255) //2X1的窗口
			{
				continueZeroNum++; //连续出现2个0像素点，continueZeroNum加1

				if (continueZeroNum > maxContinueZeroNum)
				{
					maxContinueZeroNum = continueZeroNum; //记录最大的连续0像素点个数
				}
			}
			else
			{
				continueZeroNum = 0; //如果0像素点后面那个像素点不为0，则将连续0像素点个数清零
			}
		}
		//记录横线特征值
		rowLineCharacter[k][1] = maxContinueZeroNum / (float)numWidth[k]; //记录横线特征百分比
	}
}



/*****************************************************************
*	函数名:	DivideVector
*	功能描述: 划分字符集
*	形式参数: 无
*	返回值：无
*   全局变量：charRowVector[CHAR_NUM]
charColVector[CHAR_NUM]
g_perRowLineCharacter[CHAR_NUM][CHAR_ROWSIZE]
g_perColLineCharacter[CHAR_NUM][CHAR_ROWSIZE]
g_leftRightBord[BORD_NUM]，g_upDownBord[BORD_NUM]
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	     Revised by	  Item Description
*   V1.0	2018/04/17     ZhangGH	       原始版本
******************************************************************/
void  DivideVector(int charNums)

{
	for (int k = 0; k < charNums; k++)
	{
		if (colLineCharacter[k][1] >= 0.65) //若字符右竖线特征百分比大于等于0.65，表示此字符有右竖线  
		{
			if (colLineCharacter[k][0] >= 0.65) //若字符左竖线特征百分比大于等于0.65，表示此字符有左竖线  
			{
				charColVector[k] = 4; //若字符即有右竖线，也有左竖线，将此字符竖线特征向量置4
			}
			else
			{
				charColVector[k] = 2; //若字符有右竖线，无左竖线，将此字符竖线特征向量置2
			}
		}
		else
		{
			if (colLineCharacter[k][0] >= 0.65)
			{
				charColVector[k] = 1; //若字符无右竖线，有左竖线，将此字符竖线特征向量置1
			}
			else
			{
				charColVector[k] = 0; //如果字符没有任何竖线特征，将此字符竖线特征向量置0
			}
		}

		if (rowLineCharacter[k][1] >= 0.65)
		{
			if (rowLineCharacter[k][0] >= 0.65)
			{
				charRowVector[k] = 4;
			}
			else
			{
				charRowVector[k] = 2;
			}
		}
		else
		{
			if (rowLineCharacter[k][0] >= 0.65)
			{
				charRowVector[k] = 1;
			}
			else
			{
				charRowVector[k] = 0;
			}
		}
	}
}

char NumCharIdentify(int k, int picCols)
{
	int i, j, dataNums = 0;
	int flag = 0, points = 0;
	int connectDomainNum;
	int continueZeroNum = 0, maxContinueZeroNum = 0;
	/*	Mat numTest9(numHeight[k], picCols, CV_8U);

	for (i = 0; i < numHeight[k]; i++)
	{
	for (j = 0; j < numWidth[k]; j++)
	{
	numTest9.data[i*picCols + j] = numData[k][i*picCols + j];
	}
	}*/

	connectDomainNum = ConnectDomain(k, picCols);  //获取连通域
	if (numWidth[k] < 6 && connectDomainNum == 1) //宽度小于7,可能是1和点
	{
		dataNums = DomainScan(k, 0, numHeight[k], 0, numWidth[k], picCols); //统计前景像素点个数，识别1和点
		if (dataNums < 5)
		{
			return '.';
		}
		if (dataNums < 12)
		{
			return ':';
		}
		return '1';
	}
	else if (numHeight[k] < 8&& numHeight[k] >= 5)
	{
		return 'm';
	}
	else
	{
		if (connectDomainNum == 1)
		{
			for (i = 0; i < numHeight[k]; i++)
			{
				if (numData[k][i*picCols + 1] == 255 || numData[k][i*picCols + 2] == 255)  //2X1的窗口
				{
					continueZeroNum++;
					if (continueZeroNum > maxContinueZeroNum)
					{
						maxContinueZeroNum = continueZeroNum;
					}
				}
				else
				{
					continueZeroNum = 0;
				}
			}
			if (maxContinueZeroNum >= numHeight[k] - 1) //中竖线明显判为1
			{
				flag = PointCoordUD(k, 2, 3, picCols, 4);
				//每列从上往下扫描,如果遇到前景像素点且其横坐标大于等于4，则为k
				if (flag == 1)
				{
					return 'k'; //这里是方便识别km才加进来 一般只识别数字
				}
				return '1';  //受噪音影响，1宽度偶尔超限
			}

			//每列从下往上扫描,如果遇到前景像素点且其横坐标小于等于4，则为7
			flag = PointCoordDU(k, 1, numWidth[k] - 2, picCols, 4);
			if (flag == 1)
			{
				return '7';
			}


			/*	flag = PointScanMidCol(k);
			if (flag > 3)
			{
			return Identify235(k);
			}*/

			//在2-4行中,每行从右往左扫描,如果遇到前景像素点且其纵坐标小于等于numWidth[k]-4，即为5
			flag = PointCoordRL(k, 2, 3, picCols, numWidth[k] - 4);
			if (flag == 1)
			{
				return '5';
			}

			//在倒数3-5行中,每行从右往左扫描,如果遇到前景像素点且其纵坐标小于等于numWidth[k]-3,即为2
			flag = PointCoordRL(k, numHeight[k] - 5, numHeight[k] - 3, picCols, numWidth[k] - 3);
			return '2'*(flag == 1) + '3'*(flag == 0);

		}
		if (connectDomainNum == 2)
		{
			int continueZeroNum = 0, maxContinueZeroNum = 0;

			if (g_flagNum[0] > 12)
			{
				return '0';  //环内像素点大于12个,则肯定是0
			}

			points = PointScanMidCol(k, picCols);
			if (points > 1) //中间几行交点为3的列数大于4,则判为69
			{
				for (i = 0; i < numHeight[k] + 2; i++)
				{
					for (j = 0; j < numWidth[k] + 2; j++)
					{
						if (g_connectBuf[i][j] == 3)
						{
							if (i <= 4) //判断环的上下位置识别6和9
							{
								return '9';
							}
							else
							{
								return '6';
							}
						}
					}
				}
			}


			/*		for (i = 0; i < numHeight[k]; i++)
			{
			if (numData[k][i*picCols + 1] == 255 || numData[k][i*picCols + 2] == 255)  //2X1的窗口
			{
			continueZeroNum++;
			if (continueZeroNum > maxContinueZeroNum)
			{
			maxContinueZeroNum = continueZeroNum;
			}
			}
			else
			{
			continueZeroNum = 0;
			}
			}
			if (maxContinueZeroNum <= 6) //判断竖线特征
			{
			return '4';
			}

			return '0';*/
			return '4';
		}
		if (connectDomainNum == 3)
		{

			return '8';
		}
		return '?';
	}
}  //数字识别结束 


   /*****************************************************************
   *	函数名:	MixIdentify
   *	功能描述: 识别字母和数字
   *	形式参数: Uint8 k //表示第几个字符
   *	返回值：
   *   全局变量：
   *  文件静态变量：无
   *  函数静态变量：无
   *------------------------------------------------------------------
   *	Revision History
   *   No.	     Date	      Revised by	  Item Description
   *	V1.0   	2018/04/17	    ZhangGH 		 原始版本
   ******************************************************************/
char MixIdentify(int k, int picCols)
{
	int pointNums = 0; // 交点个数
	int oneNum1 = 0, oneNum2 = 0;
	int flag; //交点坐标标识位
	int connectDomainNum; //连通域数量
	Mat numTest8(numHeight[k], picCols, CV_8U);

	for (int i = 0; i < numHeight[k]; i++)
	{
		for (int j = 0; j < numWidth[k]; j++)
		{
			numTest8.data[i*picCols + j] = numData[k][i*picCols + j];
		}
	}
	connectDomainNum = ConnectDomain(k, picCols);
	if (numWidth[k] < 6 && connectDomainNum == 1) //字符宽度小于6,初步识别为1和I
	{
		oneNum1 = DomainScan(k, 1, 4, 0, numWidth[k], picCols); //计算第1-4行前景像素点个数
		oneNum2 = DomainScan(k, numHeight[k] - 5, numHeight[k] - 2, 0, numWidth[k], picCols); //计算倒数2-5行前景像素点个数
		return '1'*(oneNum1 - oneNum2 > 1) + 'I'*(oneNum1 - oneNum2 <= 1);
	}
	else
	{
		if (connectDomainNum == 2)
		{
			if (g_flagNum[0] > 15)
			{
				return IdentifyDQ0(k, picCols);
			}
		}

		if (charColVector[k] == 0 && charRowVector[k] == 0)  //若特征向量为00,初步识别为A,V,X,Y,4
		{
			if (connectDomainNum == 1)  //若连通域等于1,则可能是V,X,Y
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //扫描末3行交点数
				if (pointNums > 1)  //末3行中交点为2的行数大于1，则为X
				{
					return 'X';
				}

				flag = PointCoordUD(k, numWidth[k] / 2 - 1, numWidth[k] / 2, picCols, 3);
				return  'V'*(flag == 1) + 'Y'*(flag != 1);
			}
			else if (connectDomainNum == 2)
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //扫描末3行交点数
				if (pointNums > 1)  //末3行中交点为2的行数大于1，则为A
				{
					return 'A';
				}
				return '4';
			}
			else
			{
				return '8';  //正常数字8的特征向量不会是00，这里是考虑到噪音影响，且只有8有3个连通域，方便识别加进来的 
			}

		}

		if (charColVector[k] == 0 && charRowVector[k] == 1)  //若特征向量为01,则识别为T,J，7
		{
			if (connectDomainNum == 1)
			{
				//在第2-3行中,每行从左往右扫描,如果遇到前景像素点且其纵坐标大于等于4,即为7、T
				flag = PointCoordLR(k, 2, 3, picCols, 4);
				if (flag == 1)
				{
					pointNums = PointScanCol(k, 2, 4, picCols, 2); //扫描前3列交点数
					return '7'*(pointNums > 1) + 'T'*(pointNums <= 1);
				}
				pointNums = PointScanMidCol(k, picCols);
				if (pointNums > 1)
				{
					return '5';
				}
				pointNums = PointScanCol(k, 0, 3, picCols, 2); //扫描前3列交点数
				if (pointNums > 1)  //前3列中交点为2的列数大于1，则为J，否则为T
				{
					return 'J';
				}

				return 'T';
			}
			else if (connectDomainNum == 2)
			{
				return 'A';
			}
			else
			{
				return '8';
			}
		}

		if (charColVector[k] == 0 && charRowVector[k] == 2)  //若特征向量为02,则识别为A、1、2、8
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanMidCol(k, picCols);
				return '2'*(pointNums > 1) + '1'* (pointNums <= 1);
			}
			else if (connectDomainNum == 2)
			{
				return 'A';
			}
			else
			{
				return '8';
			}
		}

		if (charColVector[k] == 0 && charRowVector[k] == 4)  //若特征向量为04,则初步识别为A,S,Z,X,J,2,3,5
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanMidCol(k, picCols); //扫描中间几列交点为3的列数
				if (pointNums > 1)
				{
					//在倒数第3-5行中,每行从右往左扫描,如果遇到前景像素点且其纵坐标小于等于numWidth[k] - 3,即为2Z
					flag = PointCoordRL(k, numHeight[k] - 4, numHeight[k] - 2, picCols, numWidth[k] - 4);
					if (flag == 1)
					{
						return Identify2Z(k, picCols);
					}
					return Identify5S(k, picCols);
				}
				pointNums = PointScanRow(k, 0, 3, picCols, 2); //扫描前3行交点数
				if (pointNums > 1)  //末3行中交点为2的行数大于1，则为X，否则为J
				{
					return 'X';
				}

				pointNums = PointScanCol(k, 0, 3, picCols, 2); //扫描前3列交点数
				if (pointNums > 1)  //前3列中交点为2的列数大于1，则为J，否则为T
				{
					return 'J';
				}

				return 'T';
			}
			else if (connectDomainNum == 2)
			{
				return 'A';
			}
			else
			{
				return '8';
			}

		}

		if (charColVector[k] == 1 && charRowVector[k] == 0)  //若特征向量为10,则识别为K,
		{
			if (connectDomainNum == 1)
			{
				return 'K';
			}
			else
			{
				return Identify8B(k, picCols);//考虑异常和噪音
			}
		}

		if (charColVector[k] == 1 && charRowVector[k] == 1)  //若特征向量为11,则初步识别为P,R,F
		{
			if (connectDomainNum == 1)
			{
				return 'F';
			}
			else if (connectDomainNum == 2)
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //扫描末3行交点数
				if (pointNums > 1)  //末3行中交点为2的行数大于1，则为R,否则为P
				{
					return 'R';
				}
				return 'P';
			}
			else
			{
				return Identify8B(k, picCols);//考虑异常和噪音	
			}

		}

		if (charColVector[k] == 1 && charRowVector[k] == 2)  //若特征向量为12,初步识别为L,6
		{
			if (connectDomainNum == 1)
			{
				return 'L';
			}
			else if (connectDomainNum == 2)
			{
				return '6';
			}
			else
			{
				return Identify8B(k, picCols);//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 1 && charRowVector[k] == 4)  //若特征向量为14,则初步识别为B,C,E,G,O,Q,6
		{
			if (connectDomainNum == 1)
			{
				//在后3行中每行从右往左扫描,如果遇到前景像素点且纵坐标小于等于numWidth[k] - 4,则可能为C、E
				flag = PointCoordRL(k, numHeight[k] - 4, numHeight[k] - 2, picCols, numWidth[k] - 4);
				if (flag == 1)
				{
					pointNums = PointScanCol(k, 2, 4, picCols, 3); //扫描2-4列交点数
					if (pointNums > 1)  //3列中交点为3的列数大于1，则为E，否则为C
					{
						return 'E';
					}
					return 'C';
				}
				pointNums = PointScanCol(k, numWidth[k] - 4, numWidth[k] - 2, picCols, 3); //扫描后3列交点数
				if (pointNums > 1)  //后3列中交点为3的列数大于1，则为G，否则为C
				{
					return 'G';
				}
				return 'C';
			}
			else if (connectDomainNum == 2)
			{
				for (int i = 0; i < numHeight[k] + 2; i++)
				{
					for (int j = 0; j < numWidth[k] + 2; j++)
					{
						if (g_connectBuf[i][j] == 3)
						{
							if (i > 4) //判断环的上下位置识别6
							{
								return '6';
							}
						}
					}
				}
				return IdentifyDQ0(k, picCols);
			}
			else
			{
				return Identify8B(k, picCols);//连通域为3，则为8、B
			}
		}

		if (charColVector[k] == 2 && charRowVector[k] == 0)  //若特征向量为20,则识别为4
		{
			if (connectDomainNum == 1)
			{
				return '3';
			}
			else if (connectDomainNum == 2)
			{
				return '4';
			}
			else
			{
				return Identify8B(k, picCols);//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 2 && (charRowVector[k] == 1 || charRowVector[k] == 2 || charRowVector[k] == 4))
		{   //若特征向量为21、22、24,则识别为3、9
			if (connectDomainNum == 1)
			{
				return '3';
			}
			else if (connectDomainNum == 2)
			{
				return '9';
			}
			else
			{
				return Identify8B(k, picCols);//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 0)  //若特征向量为40,则初步识别为H,M,N,W
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanRow(k, 0, 3, picCols, 3); //扫描前3行交点数
				if (pointNums > 1)
				{
					return 'W';  //前3行中交点为2的行数大于1，则为R,否则为P
				}
				//在第2-3列中,每列从上往下扫描,如果遇到前景像素点且其横坐标大于等于3,即为H
				flag = PointCoordUD(k, 2, 3, picCols, 3);
				if (flag == 1)
				{
					return 'H';

				}
				//在倒数第2-3列中,每列从上往下扫描,如果遇到前景像素点且其横坐标大于等于4,即为N,否则为M
				flag = PointCoordUD(k, numWidth[k] - 4, numWidth[k] - 3, picCols, 4);
				return 'N' * (flag == 1) + 'M' * (flag != 1);
			}
			else if (connectDomainNum == 2)
			{
				return 'R'; //连通域为2，则为R
			}
			else
			{
				return Identify8B(k, picCols);//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 1)  //若特征向量为41,则识别为R
		{
			if (connectDomainNum == 2)
			{
				return 'R'; //连通域为2，则为R
			}
			else
			{
				return Identify8B(k, picCols); //考虑异常和噪音	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 2)  //若特征向量为42,则识别为U
		{
			if (connectDomainNum == 1)
			{
				return 'U'; //连通域为2，则为R
			}
			else
			{
				return Identify8B(k, picCols);//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 4)  //若特征向量为44,则初步识别为B,D,O,Q
		{
			if (connectDomainNum == 2)
			{
				return IdentifyDQ0(k, picCols);
			}
			else
			{
				return Identify8B(k, picCols);
			}
		}

		return '?';  //字母识别结束,若无法识别，返回'？'   
	}  //"if (numWidth[k] <= SHORTCHAR)...else..." 结束
}


/*****************************************************************
*	函数名:	LetterIdentify
*	功能描述: 识别字母
*	形式参数: Uint8 k //表示第几个字符
*	返回值：
*   全局变量：
*  文件静态变量：无
*  函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*   No.	     Date	      Revised by	  Item Description
*	V1.0   	2018/04/17	    ZhangGH 		 原始版本
******************************************************************/

char LetterIdentify(int k, int picCols)
{
	int pointNums = 0; // 交点个数
	int flag; //交点坐标标识位
	int connectDomainNum; //连通域数量

	connectDomainNum = ConnectDomain(k, picCols);
	if (numWidth[k] < 5 && connectDomainNum == 1) //字符宽度小于等于SHORTCHAR表示是短字符
	{
		return 'I';
	}
	else
	{
		if (connectDomainNum == 2)
		{
			if (g_flagNum[0] > 15)
			{
				return IdentifyDOQ(k, picCols);
			}
		}

		if (charColVector[k] == 0 && charRowVector[k] == 0)  //若特征向量为00,初步识别为A,V,X,Y
		{
			if (connectDomainNum == 1)  //若连通域等于1,则可能是V,X,Y
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //扫描末3行交点数
				if (pointNums > 1)  //末3行中交点为2的行数大于1，则为X
				{
					return 'X';
				}

				flag = PointCoordUD(k, numWidth[k] / 2 - 1, numWidth[k] / 2, picCols, 3);
				return  'V'*(flag == 1) + 'Y'*(flag != 1);
			}
			else if (connectDomainNum == 2)
			{
				return 'A';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}

		}

		if (charColVector[k] == 0 && charRowVector[k] == 1)  //若特征向量为01,则识别为T,J，A
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanCol(k, 0, 3, picCols, 2); //扫描前3列交点数
				if (pointNums > 1)  //前3列中交点为2的列数大于1，则为J，否则为T
				{
					return 'J';
				}

				return 'T';
			}
			else if (connectDomainNum == 2)
			{
				return 'A';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 0 && charRowVector[k] == 2)  //若特征向量为02,则识别为A
		{
			if (connectDomainNum == 2)
			{
				return 'A';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 0 && charRowVector[k] == 4)  //若特征向量为04,则初步识别为A,S,Z,X,J,2,3,5
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanMidCol(k, picCols); //扫描中间几列交点为3的列数
				if (pointNums > 1)
				{
					//在倒数第3-5行中,每行从右往左扫描,如果遇到前景像素点且其纵坐标小于等于numWidth[k] - 3,即为2Z
					flag = PointCoordRL(k, numHeight[k] - 4, numHeight[k] - 2, picCols, numWidth[k] - 3);
					return  'Z'*(flag == 1) + 'S'*(flag != 1);
				}
				pointNums = PointScanRow(k, 0, 3, picCols, 2); //扫描前3行交点数
				if (pointNums > 1)  //末3行中交点为2的行数大于1，则为X，否则为J
				{
					return 'X';
				}

				return 'J';
			}
			else if (connectDomainNum == 2)
			{
				return 'A';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}

		}

		if (charColVector[k] == 1 && charRowVector[k] == 0)  //若特征向量为10,则识别为K,
		{
			if (connectDomainNum == 1)
			{
				return 'K';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 1 && charRowVector[k] == 1)  //若特征向量为11,则初步识别为P,R,F
		{
			if (connectDomainNum == 1)
			{
				return 'F';
			}
			else if (connectDomainNum == 2)
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //扫描末3行交点数
				if (pointNums > 1)  //末3行中交点为2的行数大于1，则为R,否则为P
				{
					return 'R';
				}
				return 'P';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}

		}

		if (charColVector[k] == 1 && charRowVector[k] == 2)  //若特征向量为12,初步识别为L
		{
			if (connectDomainNum == 1)
			{
				return 'L';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 1 && charRowVector[k] == 4)  //若特征向量为14,则初步识别为B,C,E,G,O,Q,6
		{
			if (connectDomainNum == 1)
			{
				//在后3行中每行从右往左扫描,如果遇到前景像素点且纵坐标小于等于numWidth[k] - 4,则可能为C、E
				flag = PointCoordRL(k, numHeight[k] - 4, numHeight[k] - 2, picCols, numWidth[k] - 4);
				if (flag == 1)
				{
					pointNums = PointScanCol(k, 2, 4, picCols, 3); //扫描2-4列交点数
					if (pointNums > 1)  //3列中交点为3的列数大于1，则为E，否则为C
					{
						return 'E';
					}
					return 'C';
				}
				pointNums = PointScanCol(k, numWidth[k] - 4, numWidth[k] - 2, picCols, 3); //扫描后3列交点数
				if (pointNums > 1)  //后3列中交点为3的列数大于1，则为G，否则为C
				{
					return 'G';
				}
				return 'C';
			}
			else if (connectDomainNum == 2)
			{
				return IdentifyDOQ(k, picCols);
			}
			else
			{
				return 'B';
			}

		}

		if (charColVector[k] == 4 && charRowVector[k] == 0)  //若特征向量为40,则初步识别为H,M,N,W，R
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanRow(k, 0, 3, picCols, 3); //扫描前3行交点数
				if (pointNums > 1)
				{
					return 'W';  //前3行中交点为2的行数大于1，则为R,否则为P
				}
				//在第2-3列中,每列从上往下扫描,如果遇到前景像素点且其横坐标大于等于3,即为H
				flag = PointCoordUD(k, 2, 3, picCols, 3);
				if (flag == 1)
				{
					return 'H';

				}
				//在倒数第2-3列中,每列从上往下扫描,如果遇到前景像素点且其横坐标大于等于4,即为N,否则为M
				flag = PointCoordUD(k, numWidth[k] - 4, numWidth[k] - 3, picCols, 4);
				return 'N' * (flag == 1) + 'M' * (flag != 1);
			}
			else if (connectDomainNum == 2)
			{
				return 'R';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 1)  //若特征向量为41,则识别为R
		{
			if (connectDomainNum == 2)
			{
				return 'R';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 2)  //若特征向量为42,则识别为U
		{
			if (connectDomainNum == 1)
			{
				return 'U';
			}
			else
			{
				return 'B';//考虑异常和噪音	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 4)  //若特征向量为44,则初步识别为B,D,O,Q
		{
			if (connectDomainNum == 2)
			{
				return IdentifyDOQ(k, picCols);
			}
			else
			{
				return 'B';
			}
		}

		return '?';  //字母识别结束,若无法识别，返回'？'   
	}  //"if (numWidth[k] <= SHORTCHAR)...else..." 结束
}


/*****************************************************************
*	函数名:	IdentifyDOQ
*	功能描述: 识别字母D、O和Q
*	形式参数: int k //表示第几个字符
*	返回值：D、O或Q
*   全局变量:
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             原始版本
******************************************************************/
char IdentifyDOQ(int k, int picCols)
{
	int leftUpNum = 0, leftDownNum = 0;

	if (colLineCharacter[k][0] == 1)
	{
		return 'D'; //第1列竖线特征明显,判为D  
	}

	leftUpNum = DomainScan(k, 0, 2, 0, 2, picCols);
	leftDownNum = DomainScan(k, numHeight[k] - 2, numHeight[k], 0, 2, picCols);
	if (leftUpNum + leftDownNum >= 6)
	{
		return 'D';  //左上，左下有前景像素点判为D，否则为O、Q
	}

	return IdentifyOQ(k, picCols);
}

/*****************************************************************
*	函数名:	IdentifyDQ0
*	功能描述: 识别字母D、Q和数字0
*	形式参数: int k //表示第几个字符
*	返回值：D、Q或0
*   全局变量:
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             原始版本
******************************************************************/
char IdentifyDQ0(int k, int picCols)
{
	int leftUpNum = 0, leftDownNum = 0;

	if (colLineCharacter[k][0] == 1)
	{
		return 'D'; //第1列竖线特征明显,判为D  
	}

	leftUpNum = DomainScan(k, 0, 2, 0, 2, picCols);
	leftDownNum = DomainScan(k, numHeight[k] - 2, numHeight[k], 0, 2, picCols);
	if (leftUpNum + leftDownNum >= 6)
	{
		return 'D';  //左上，左下有前景像素点判为D，否则为O、Q
	}

	if (IdentifyOQ(k, picCols) == 'O')
	{
		return '0'; //由于数字0和字母O无法区分，这里统一判为数字0
	}
	else
	{
		return 'Q';
	}
}

/*****************************************************************
*	函数名:	IdentifyOQ
*	功能描述: 识别字母O和Q
*	形式参数: int k //表示第几个字符
*	返回值：O或Q
*   全局变量:
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             原始版本
******************************************************************/
char IdentifyOQ(int k, int picCols)
{
	int i, j;
	int yRecord = 0, xRecord = 0;  //基点坐标
	int goTimes = 0; //游走次数
	int maxSum = 0;

	//在右下角2X2的框找基点
	for (j = numWidth[k] - 1; j >= numWidth[k] - 2; j--)
	{
		for (i = numHeight[k] - 1; i >= numHeight[k] - 2; i--)
		{
			if (numData[k][i*picCols + j] == 255)  //得是白点才能作基点	
			{
				if (i + j > maxSum)
				{
					maxSum = i + j;	 //计算当前列最底部黑点的坐标和
					yRecord = i;  //记录坐标
					xRecord = j;
					break;
				}
			}  // "if (g_charSection[k][i][j] == 0)"...end
		}  // "for (j..)"...end	
	}  // "for (i..)"...end	

	if (xRecord == 0)  //若是在2X2的框内没找到黑点，说明没"尾巴"，判为O
	{
		return 'O';
	}

	//判别右下角斜线游走次数
	while (numData[k][(yRecord - 1)*picCols + xRecord - 1] == 255 && yRecord > 1 && xRecord > 1)
	{
		goTimes++;  //游走次数++	

		yRecord--;
		xRecord--;
	}


	if (goTimes > 2)
	{
		return 'Q';  //斜线游走次数大于2判为Q，否则为O
	}
	else
	{
		return 'O';
	}
}


/*****************************************************************
*	函数名:	Identify2Z
*	功能描述: 识别字母2和Z
*	形式参数: int k //表示第几个字符
*	返回值：2或Z
*   全局变量:
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             原始版本
******************************************************************/
char Identify2Z(int k, int picCols)
{
	if (numData[k][0 * picCols + 0] == 0 && numData[k][0 * picCols + numWidth[k] - 1] == 0)
	{
		return '2'; //第1行第1个像素点以及第1行最后1个像素点为背景,则判为2
	}

	return 'Z';
}

/*****************************************************************
*	函数名:	Identify5S
*	功能描述: 识别字母5和S
*	形式参数: int k //表示第几个字符
*	返回值：5或S
*   全局变量:
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             原始版本
******************************************************************/
char Identify5S(int k, int picCols)
{
	int flag = 0;
	//在第2-3行中,每行从右往左扫描,如果遇到前景像素点且其纵坐标小于等于numWidth[k] - 4,即为5S
	flag = PointCoordRL(k, 2, 3, picCols, numWidth[k] - 4);
	if (flag == 1)
	{
		if (numData[k][0 * picCols + 0] == 0 && numData[k][0 * picCols + numWidth[k] - 1] == 0)
		{
			return 'S'; //第1行第1个像素点以及第1行最后1个像素点为背景,则判为S，否则为5
		}
		return '5';
	}

	return '3';
}


/*****************************************************************
*	函数名:	Identify8B
*	功能描述: 识别字母8和B
*	形式参数: int k //表示第几个字符
*	返回值：8或B
*   全局变量:
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             原始版本
******************************************************************/
char Identify8B(int k, int picCols)
{
	if (numData[k][0 * picCols + 0] == 0 && numData[k][(numHeight[k] - 1)* picCols + 0] == 0)
	{
		return '8'; //第1列第1个像素点以及第1列最后1个像素点为背景,则判为8，否则为B
	}

	return 'B';
}


int dom;
int ConnectDomain(int k, int picCols)
{
	Data Dp[800];
	int i, j, xi, xj, n = 0;
	int flag = 2;
	int tag = 0;
	int connectDomainNum;
	Mat numTest(32, picCols + 2, CV_8U);

	memset(g_connectBuf, 0, sizeof(g_connectBuf));

	for (i = 0; i < numHeight[k] + 2; i++)
	{
		for (j = 0; j < numWidth[k] + 2; j++)  //给字符周围加一圈背景袼氐悖奖懔ㄓ颞?
		{
			if (i == 0 || j == 0 || i == numHeight[k] + 1 || j == numWidth[k] + 1)
			{
				g_connectBuf[i][j] = 0; //给每个字符四周加一圈0
			}

			if (i < numHeight[k] && j < numWidth[k])
			{
				g_connectBuf[i + 1][j + 1] = numData[k][i*picCols + j]; //原像素点向右下移动
			}

		}
	}

	/*
	for (i = 0; i < numHeight[k] + 2; i++)
	{
	for (j = 0; j < numWidth[k] + 2; j++)  //给字符周围加一圈背景袼氐悖奖懔ㄓ颞?
	{
	numTest.data[i*(picCols + 2) + j] = g_connectBuf[i][j];
	}
	}*/

	g_flagNum[0] = 1;
	g_flagNum[1] = 1;

	for (i = 0; i < numHeight[k] + 2; i++)
	{


		for (j = 0; j < numWidth[k] + 2; j++)
		{

			if (g_connectBuf[i][j] == 0)
			{
				dom = 0;
				g_connectBuf[i][j] = flag; //用flag标记像素点,方便后面计算
				n++;
				Dp[dom].row = i;
				Dp[dom].col = j;
				dom++;


				while (dom != 0) //判定栈是否为空
				{
					xi = Dp[dom - 1].row;
					xj = Dp[dom - 1].col;
					dom--;
					//检查该像素点周围像素点,如果是嘲像素点,将其坐标入栈,之后再继续检查,依次入栈循环
					if (xi == 0) //第一行
					{
						if (xj == 0)  //绻堑谝恍惺赘鱿袼氐阒恍枧卸舷路胶陀曳较袼氐?
						{
							Check(xi + 1, xj, flag, Dp); //检查该像素点右方像素点,如果是背景像氐悖渥耆胝?
							Check(xi, xj + 1, flag, Dp); //检查该像素点下方像素点,如果是背景像氐悖渥耆胝?
						}
						else
							if (xj == numWidth[k] + 1)  //如果堑谝恍凶钅└鱿袼氐阒恍枧卸舷路胶妥蠓较袼氐?
							{
								Check(xi + 1, xj, flag, Dp);
								Check(xi, xj - 1, flag, Dp);
							}
							else //如果是第一行其他的像素点需判断下、左、右三方袼氐?
							{
								Check(xi + 1, xj, flag, Dp);
								Check(xi, xj - 1, flag, Dp);
								Check(xi, xj + 1, flag, Dp);
							}
					}

					if (xi == numHeight[k] + 1) //最后一行
					{
						if (xj == 0)  //上，右
						{
							Check(xi - 1, xj, flag, Dp);
							Check(xi, xj + 1, flag, Dp);
						}
						else
							if (xj == numWidth[k] + 1)  //上、左
							{
								Check(xi - 1, xj, flag, Dp);
								Check(xi, xj - 1, flag, Dp);
							}
							else //上⒆蟆⒂?
							{
								Check(xi - 1, xj, flag, Dp);
								Check(xi, xj - 1, flag, Dp);
								Check(xi, xj + 1, flag, Dp);
							}
					}

					if (xj == 0 && xi != 0 && xi != numHeight[k] + 1) //最罅?右、上、下
					{
						Check(xi, xj + 1, flag, Dp);
						Check(xi - 1, xj, flag, Dp);
						Check(xi + 1, xj, flag, Dp);
					}

					if (xj == numWidth[k] + 1 && xi != 0 && xi != numHeight[k] + 1) //最右列 左⑸稀⑾?
					{
						Check(xi, xj - 1, flag, Dp);
						Check(xi - 1, xj, flag, Dp);
						Check(xi + 1, xj, flag, Dp);
					}

					if (xi != 0 && xi != numHeight[k] + 1 && xj != 0 && xj != numWidth[k] + 1) //屑?上、下、左、右
					{
						Check(xi + 1, xj, flag, Dp);
						Check(xi - 1, xj, flag, Dp);
						Check(xi, xj - 1, flag, Dp);
						Check(xi, xj + 1, flag, Dp);
					}
				} //while end

				if (n == 1)
				{
					flag++;
				}
				if (n == 2)
				{
					if (g_flagNum[0] > 1) //环内1个像素点以下填黑,不记连通域
					{
						flag++;  //循环一次后,flag加1,即连通域加1
					}
					else   //如果环内只有1个像素点,表示是噪音,不算连通域
					{
						n--;

						numData[k][(i - 1)*picCols + j - 1] = 255; //环内只有1个像素点,填黑
						g_flagNum[0] = 1;
					}
				}
				if (n == 3)
				{
					if (g_flagNum[1] > 1)
					{
						flag++; //循环一次后,flag加1,即ㄓ蚣?
					}
					else
					{
						n--;

						numData[k][(i - 1)*picCols + j - 1] = 255; //环内只有1个像素点,填黑
						g_flagNum[1] = 1;
					}
				}
			} //if end		 

			if (flag == 5)
			{
				tag = 1;
				break;
			}

		} //for j end

		if (tag == 1)
		{
			break;
		}
	} //for i end

	connectDomainNum = flag - 2; //最后连通域的值为flag -2


								 /*		for (i = 0; i < numHeight[k] + 2; i++)
								 {
								 for (j = 0; j < numWidth[k] + 2; j++)  //给字符周围加一圈背景袼氐悖奖懔ㄓ颞?
								 {
								 numTest.data[i*(picCols + 2) + j] = g_connectBuf[i][j];
								 }
								 }*/

	return connectDomainNum;
} // " ConnectDomain( int k )..end "


  //判断函数
int Check(int i, int j, int flag, volatile Data* Dp)
{
	if (g_connectBuf[i][j] == 0)
	{
		g_connectBuf[i][j] = flag;
		if (flag == 3)
		{
			g_flagNum[0]++;
		}
		if (flag == 4)
		{
			g_flagNum[1]++;
		}
		Dp[dom].row = i;
		Dp[dom].col = j;
		dom++;
	}
	return 0;
}


/*****************************************************************
*	函数名:	PointScanMidCol
*	功能描述: 计算2-6列中交点数等于3的列数
*	形式参数: Uint8 k //表示第几个字符
*	返回值：交点数等于3的列数
*   全局变量： 无
*   文件静态变量：无
*   函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/16     ZhangGH            原始版本
******************************************************************/
int PointScanMidCol(int k, int picCols)
{
	return PointScanCol(k, 2, numWidth[k] - 2, picCols, 3); //扫描2-6列交点数
}

/*****************************************************************
*	函数名:	DomainScan
*	功能描述: 计算某区域内前景像素点个数
*	形式参数: Uint8 k //表示第几个字符
Uint8 rowStart //区域的起始行
Uint8 rowEnd//区域的终止行
Uint8 colStart //区域的起始列
Uint8 colEnd//虻闹罩沽?
*	返回值：zeroNum //区域内前景像素点个数
*   全局变量： g_charSection[CHAR_NUM][CHAR_ROWSIZE][CHAR_COLSIZE]
*   文件静态变量: 无
*   函数静态变量: 无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0   2018/04/16    ZhangGH             原始版本
******************************************************************/
int DomainScan(int k, int rowStart, int rowEnd, int colStart, int colEnd, int picCols)
{
	int i, j;
	int zeroNum = 0;

	for (i = rowStart; i < rowEnd; i++)
	{
		for (j = colStart; j < colEnd; j++)
		{
			if (numData[k][i*picCols + j] == 255)
			{
				zeroNum++;  //计算区域内前景像素点总个数
			}
		}
	}
	return zeroNum;
}

/*****************************************************************
*	函数名:	PointScan
*	功能描述: 计算某些行的交点个数
*	形式参数: Uint8 k //表示第几个字?
Uint8 rowStart //区域的起始行
Uint8 rowEnd   //区域的终止行
*	返回值：g_pointNum[m++]  //区域内交点个数
*   全局变量：numWidth[k]
g_charSection[CHAR_NUM][CHAR_ROWSIZE][CHAR_COLSIZE]
*  文件静态变量：无
*  函蔡淞浚何?*------------------------------------------------------------------
*	Revision History
*	No.	    Date	      Revised by	  Item	Description
*	V1.0    2018/04/16      ZhangGH            原始版本
******************************************************************/
int PointScanRow(int k, int rowStart, int rowEnd, int picCols, int pointSize)
{
	int i, j, flag, num, m = 0;
	int pointNum[30] = { 0 }, pointNums = 0;

	for (i = rowStart; i < rowEnd; i++)
	{
		flag = 0;
		num = 0;

		for (j = 0; j < numWidth[k]; j++)
		{
			if (flag == 0)
			{
				if (numData[k][i*picCols + j] == 255)
				{
					num++;
					flag = 1;
				}
			}

			if (flag == 1)
			{
				if (numData[k][i*picCols + j] == 0)
				{
					flag = 0;
				}
			}
		}
		pointNum[m++] = num;
	}

	for (i = 0; i < m; i++)
	{
		if (pointSize == 1)
		{
			if (pointNum[i] == pointSize)
			{
				pointNums++;
			}
		}
		else
		{
			if (pointNum[i] >= pointSize)
			{
				pointNums++;
			}
		}
	}

	return pointNums;
}


/*****************************************************************
*	函数名:	PointScanCol
*	功能描述: 计算某列交点个数
*	形式参数: Uint8 k //表示第几个字符
Uint8 colStart //区域的起始列
Uint8 colEnd//区域的终止列
*	返回值：g_pointNum[m++] //区域内交点个数
*   全局变量： numHeight[k]
*  文件静态变量：无
*  函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/16    ZhangGH             原始版本
******************************************************************/
int PointScanCol(int k, int colStart, int colEnd, int picCols, int pointSize)
{
	int i, j, flag, num, m = 0;
	int pointNum[30] = { 0 }, pointNums = 0;

	for (j = colStart; j < colEnd; j++)
	{
		flag = 0;
		num = 0;

		for (i = 0; i < numHeight[k]; i++)
		{
			if (flag == 0)
			{
				if (numData[k][i*picCols + j] == 255)
				{
					num++;
					flag = 1;
				}
			}

			if (flag == 1)
			{
				if (numData[k][i*picCols + j] == 0)
				{
					flag = 0;
				}
			}
		}
		pointNum[m++] = num;
	}

	for (i = 0; i < m; i++)
	{
		if (pointSize == 1)
		{
			if (pointNum[i] == pointSize)
			{
				pointNums++;
			}
		}
		else
		{
			if (pointNum[i] >= pointSize)
			{
				pointNums++;
			}
		}
	}

	return pointNums;
}

/*****************************************************************
*	函数名:	PointCoordRL
*	功能描述: 从右往左扫描，判断第一次遇到前景像素点时的位置
*	形式参数: Uint8 k //表示第几个字符
int rowStart //区域的起始行
int rowEnd//区域的起始列
Uint8 divideCol//边界值
*	返回值：flag
*   全局变量：numWidth[k]
*  文件静态变量：无
*  函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/16    ZhangGH              原始版本
******************************************************************/
int PointCoordRL(int k, int rowStart, int rowEnd, int picCols, int divideCol)
{
	int i, j;
	int flag = 0;

	for (i = rowEnd; i >= rowStart; i--)
	{
		for (j = numWidth[k] - 1; j >= 0; j--)
		{
			if (numData[k][i*picCols + j] == 255)
			{
				if (j <= divideCol)
				{
					flag = 1;
					return flag;
				}
				break;
			}
		}
	}
	return flag;
}


/*****************************************************************
*	函数名:	PointCoordLR
*	功能描述: 从左往右扫描，判断第一次遇到前景像素点时的位置
*	形式参数: Uint8 k //表示第几个字符
int rowStart //区域的起始行
int rowEnd//区域的起始列
Uint8 divideCol//边界值
*	返回值：flag
*   全局变量：numWidth[k]
*  文件静态变量：无
*  函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/18    ZhangGH              原始版本
******************************************************************/
int PointCoordLR(int k, int rowStart, int rowEnd, int picCols, int divideCol)
{
	int i, j;
	int flag = 0;

	for (i = rowEnd; i >= rowStart; i--)
	{
		for (j = 0; j < numWidth[k]; j++)
		{
			if (numData[k][i*picCols + j] == 255)
			{
				if (j >= divideCol)
				{
					flag = 1;
					return flag;
				}
				break;
			}
		}
	}
	return flag;
}

/*****************************************************************
*	函数名:	PointCoordUD
*	功能描述: 从上往下扫描，判断第一次遇到前景像素点时的位置
*	形式参数: Uint8 k //表示第几个字符
int rowStart //区域的起始行
int rowEnd//区域的起始列
Uint8 divideCol//边界值
*	返回值：flag
*   全局变量：numWidth[k]
*  文件静态变量：无
*  函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/18    ZhangGH              原始版本
******************************************************************/
int PointCoordUD(int k, int colStart, int colEnd, int picCols, int divideCol)
{
	int i, j;
	int flag = 0;

	for (j = colStart; j <= colEnd; j++)
	{
		for (i = 0; i < numHeight[k]; i++)
		{
			if (numData[k][i*picCols + j] == 255)
			{
				if (i >= divideCol)
				{
					flag = 1;
					return flag;
				}
				break;
			}
		}
	}
	return flag;
}

/*****************************************************************
*	函数名:	PointCoordDU
*	功能描述: 从下往上扫描，判断第一次遇到前景像素点时的位置
*	形式参数: Uint8 k //表示第几个字符
int rowStart //区域的起始行
int rowEnd//区域的起始列
Uint8 divideCol//边界值
*	返回值：flag
*   全局变量：numWidth[k]
*  文件静态变量：无
*  函数静态变量：无
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/18    ZhangGH              原始版本
******************************************************************/
int PointCoordDU(int k, int colStart, int colEnd, int picCols, int divideCol)
{
	int i, j;
	int flag = 0;

	for (j = colStart; j <= colEnd; j++)
	{
		for (i = numHeight[k] - 1; i > 0; i--)
		{
			if (numData[k][i*picCols + j] == 255)
			{
				if (i <= divideCol)
				{
					flag = 1;
					return flag;
				}
				break;
			}
		}
	}
	return flag;
}

void Cut_PreProcess(Mat src, Mat &des)
{
	//找黑边
	des = src.clone();
	des += 1;
	cv::Mat imedge;
	float k[2] = { -1,10 };
	cv::Mat kernel = cv::Mat(1, 2, CV_32FC1, k);
	cv::filter2D(src, imedge, -1, kernel);
	
	//一个黑色的像素，如果上下左右都没有出现白色，则认为不是文字边框
	cv::Mat imedge0 = imedge.clone();
	int thresh = 200;
	for (int j = 1; j < imedge0.cols-1; j++)
	{
		for (int i = 1; i < imedge0.rows-1; i++)
		{
			if (imedge0.at<uchar>(i - 1, j - 1) > thresh || imedge0.at<uchar>(i - 1, j) > thresh || imedge0.at<uchar>(i - 1, j + 1) > thresh || imedge0.at<uchar>(i, j - 1) > thresh || imedge0.at<uchar>(i, j + 1) > thresh || imedge0.at<uchar>(i + 1, j - 1) > thresh || imedge0.at<uchar>(i + 1, j) > thresh || imedge0.at<uchar>(i + 1, j + 1) > thresh)
			{
				continue;
			}
			else
			{
				imedge.at<uchar>(i, j) = 255;
			}
		}
	}

	//若原图中像素为“黑”，且与黑色像素相连，则认为是它可能是文字边框
	int thresh1 = 50;
	for (int j = 1; j < imedge0.cols - 1; j++)
	{
		for (int i = 1; i < imedge0.rows - 1; i++)
		{
			if (des.at<uchar>(i, j) < thresh1)
			{
				if (imedge0.at<uchar>(i - 1, j - 1) < thresh1 || imedge0.at<uchar>(i - 1, j) < thresh1 || imedge0.at<uchar>(i - 1, j + 1) < thresh1 || imedge0.at<uchar>(i, j - 1) < thresh1 || imedge0.at<uchar>(i, j + 1) < thresh1 || imedge0.at<uchar>(i + 1, j - 1) < thresh1 || imedge0.at<uchar>(i + 1, j) < thresh1 || imedge0.at<uchar>(i + 1, j + 1) < thresh1)
				{
					imedge.at<uchar>(i, j) = des.at<uchar>(i, j);
				}
			}			
		}
	}
	
	//将图像边缘像素赋值255	
	for (int j = 0; j < imedge.cols; j++)
	{
		imedge.at<uchar>(0, j) = 255;
		imedge.at<uchar>(imedge.rows - 1, j) = 255;
	}
	for (int i = 0; i < imedge.rows; i++)
	{
		imedge.at<uchar>(i, 0) = 255;
		imedge.at<uchar>(i, imedge.cols - 1) = 255;
	}

	//寻找上边界和下边界
	/*int up = 0, down = 30;
	int NUM[30][30];
	int step = imedge.cols / 30;
	for (int i = 0; i < imedge.rows; i++)
	{
		int tag = 0;
		for (int k = 0; k < 30; k++)
		{
			int acc = 0;
			for (int j = k*step+1; j <= (k+1)*step; j++)
			{
				if (imedge.at<uchar>(i, j) < 50)
				{
					acc++;
				}
			}
			if (acc > 2)
			{
				tag++;
			}
			NUM[i][k] = acc;
		}
		if (tag > 10)
		{
			if (up == 0)
			{
				up = i;
			}
			else
			{
				down = i;
			}
		}		
	}*/

	//将上边界和下边界之外区域赋值为0
	/*for (int i = 0; i < imedge.rows; i++)
	{
		if (i<up-1 || i>down+1)
		{
			for (int j = 0; j < imedge.cols; j++)
			{
				des.at<uchar>(i, j) = 0;
				imedge.at<uchar>(i, j) = 255;
			}
		}
		
	}*/
	imwrite("F:/workspace/videos/tmp0.jpg", imedge);

	//使用填充法将黑边连通之外区域填充为0
	des.at<uchar>(0, 0) = 0;
	des.at<uchar>(0, des.cols - 1) = 0;
	des.at<uchar>(des.rows - 1, 0) = 0;
	des.at<uchar>(des.rows - 1, des.cols - 1) = 0;

	Mat des1 = des.clone();
	for (int i = 0; i < des1.rows - 1; i++)
	{
		for (int j = 0; j < des1.cols - 1; j++)
		{
			if (des1.at<uchar>(i, j) == 0)
			{
				if (imedge.at<uchar>(i + 1, j) > 200)
				{
					des1.at<uchar>(i + 1, j) = 0;
				}
				if (imedge.at<uchar>(i, j + 1) > 200)
				{
					des1.at<uchar>(i, j + 1) = 0;
				}
			}
		}
	}

	//将图像沿X轴翻转180
	Mat imedge1;
	flip(des1, des1, 0);
	flip(imedge, imedge1, 0);
	for (int i = 0; i < des1.rows - 1; i++)
	{
		for (int j = 0; j < des1.cols - 1; j++)
		{
			if (des1.at<uchar>(i, j) == 0)
			{
				if (imedge1.at<uchar>(i + 1, j) > 200)
				{
					des1.at<uchar>(i + 1, j) = 0;
				}
				if (imedge1.at<uchar>(i, j + 1) > 200)
				{
					des1.at<uchar>(i, j + 1) = 0;
				}
			}
		}
	}
	flip(des1, des1, 0);

	//Y轴翻转
	Mat des2,imedge2;;
	flip(des, des2, 1);
	flip(imedge, imedge2, 1);
	
	for (int i = 0; i < des2.rows-1; i++)
	{
		for (int j = 0; j < des2.cols-1; j++)
		{
			if (des2.at<uchar>(i, j) == 0)
			{
				if (imedge2.at<uchar>(i + 1, j) > 200)
				{
					des2.at<uchar>(i + 1, j) = 0;
				}
				if (imedge2.at<uchar>(i, j+1) > 200)
				{
					des2.at<uchar>(i, j+1) = 0;
				}
			}
		}
	}

	//将图像沿X轴翻转180
	Mat des3, imedge3;
	flip(des2, des3, 0);
	flip(imedge2, imedge3, 0);
	for (int i = 0; i < des3.rows - 1; i++)
	{
		for (int j = 0; j < des3.cols - 1; j++)
		{
			if (des3.at<uchar>(i, j) == 0)
			{
				if (imedge3.at<uchar>(i + 1, j) > 200)
				{
					des3.at<uchar>(i + 1, j) = 0;
				}
				if (imedge3.at<uchar>(i, j + 1) > 200)
				{
					des3.at<uchar>(i, j + 1) = 0;
				}
			}
		}
	}
	flip(des3, des3, 0);
	flip(des3, des3, 1);
	des = max(des1, des3);

	Mat imBinary;
	threshold(des, imBinary, 230, 255, CV_THRESH_BINARY);
	
	for (int i = 1; i < imBinary.rows-1; i++)
	{
		int tag = 0,acc = 0;
		for (int j = 1; j < imBinary.cols-1; j++)
		{
			if (imBinary.at<uchar>(i, j) == 255)
			{
				tag = j;
				acc++;
			}
		}
		if ((tag > 10 && tag < imBinary.cols / 3)||(acc>0 && acc <= 10))//
		{
			for (int j = 0; j < imBinary.cols; j++)
			{
				imBinary.at<uchar>(i, j) = 0;
			}
		}
	}
	des = imBinary.clone();
}


bool Crop_up(Mat src, Mat &left, Mat &right)//根据文字分布向图像分成左右两块
{
	Rect rect(0, 0, src.cols, src.rows);
	if (g_upleft_width > src.cols)
	{
		int acc = 0;
		for (int j = 0; j < src.cols; j++)
		{
			int tag = 0;
			for (int i = 0; i < src.rows; i++)
			{
				if (src.at<uchar>(i, j) > 250)
				{
					tag++;
				}
			}
			if (tag == 0)
			{
				acc++;
			}
			else
			{
				acc = 0;
			}
			if (acc > 30)
			{
				rect.width = j;
				g_upleft_width = j;
				break;
			}
		}
	}
	else
	{
		rect.width = g_upleft_width;
	}	
	left = src(rect).clone();

	if (rect.width < src.cols - 1)
	{
		int tmp = rect.width;
		//rect.x = tmp + 1;
		rect.x = tmp;//从0开始，这里不用加1
		rect.width = src.cols - tmp - 1;
		right = src(rect).clone();
		return true;
	}
	else
	{
		return false;
	}
}

bool Crop_down(cv::Mat src, cv::Mat &left, cv::Mat &right)//根据文字分布向图像分成左右两块
{
	Rect rect(0, 0, src.cols, src.rows);
	if (g_downleft_width > src.cols)
	{
		int acc = 0;
		for (int j = 0; j < src.cols; j++)
		{
			int tag = 0;
			for (int i = 0; i < src.rows; i++)
			{
				if (src.at<uchar>(i, j) > 250)
				{
					tag++;
				}
			}
			if (tag == 0)
			{
				acc++;
			}
			else
			{
				acc = 0;
			}
			if (acc > 30)
			{
				rect.width = j;
				g_downleft_width = j;
				break;
			}
		}
	}
	else
	{
		rect.width = g_downleft_width;
	}	
	left = src(rect).clone();
	if (rect.width < src.cols - 1)
	{
		int tmp = rect.width;
		//rect.x = tmp + 1;
		rect.x = tmp;//从0开始，这里不用加1
		rect.width = src.cols - tmp - 1;
		right = src(rect).clone();
		return true;
	}
	else
	{
		return false;
	}
}

//计算字符串向量的统计直方图
std::vector<Str_Long> StrVectorHistCal(std::vector<std::string> s)
{
	long acc = 0;
	long Num = s.size();	
	vector<Str_Long> result;	
	sort(s.begin(),s.end());
	Str_Long tmp;	
	for (int k = 0; k < Num;)
	{
		tmp.str_value = s[k];
		tmp.str_count = count(s.begin(), s.end(), tmp.str_value);
		k = k + tmp.str_count;
		result.push_back(tmp);
	}
	return result;
}

//计算Int向量的统计直方图
std::vector<Int_Long> IntVectorHistCal(std::vector<int> a)
{
	long acc = 0;
	long Num = a.size();
	vector<Int_Long> result;
	sort(a.begin(), a.end());
	Int_Long tmp;
	for (int k = 0; k < Num;)
	{
		tmp.int_value = a[k];
		tmp.int_count = count(a.begin(), a.end(), tmp.int_value);
		k = k + tmp.int_count;
		result.push_back(tmp);
	}
	return result;
}

void OcrRecognition::word_frame2OCR(FrameWord &wordInfor, OCR_Result &ocr_recgInfor)
{
	//转换日期
	char tmp[2], tmp0[4];
	string mydate = "18.01.01";//初始赋值
	int tag = 0;
	for (int i = 0; i < wordInfor.infor_up.str_date.length(); i++)
	{
		if ((i+tag) > 7)break;
		if (wordInfor.infor_up.str_date[i] >= '0'&&wordInfor.infor_up.str_date[i] <= '9')
		{
			mydate[i + tag] = wordInfor.infor_up.str_date[i];
		}		
		if (i==2 && wordInfor.infor_up.str_date[i] != '.')
		{
			tag++;
		}
		if (tag==1 && i == 4 && wordInfor.infor_up.str_date[i] != '.')
		{
			tag++;
		}
		if (tag == 0 && i == 5 && wordInfor.infor_up.str_date[i] != '.')
		{
			tag++;
		}
	}
	tmp0[0] = '2';
	tmp0[1] = '0';
	tmp0[2] = mydate[0];
	tmp0[3] = mydate[1];
	//ocr_recgInfor.Date.years = atoi(tmp);
	ocr_recgInfor.Date_Time.tm_year = atoi(tmp0) - 1900;
	tmp[0] = mydate[3];
	tmp[1] = mydate[4];
	//ocr_recgInfor.Date.monthes = atoi(tmp);
	ocr_recgInfor.Date_Time.tm_mon = atoi(tmp) - 1;//[0-11]
	tmp[0] = mydate[6];
	tmp[1] = mydate[7];
	//ocr_recgInfor.Date.days = atoi(tmp);
	ocr_recgInfor.Date_Time.tm_mday = atoi(tmp);

	//转换时间
	string mytime = "08:01:01";//初始赋值
	int tag1 = 0;
	for (int i = 0; i < wordInfor.infor_up.str_time.length(); i++)
	{
		if ((i + tag1) > 7)break;
		if (wordInfor.infor_up.str_time[i] >= '0'&&wordInfor.infor_up.str_time[i] <= '9')
		{
			mytime[i + tag1] = wordInfor.infor_up.str_time[i];
		}
		
		if (i == 2 && wordInfor.infor_up.str_time[i] != ':')
		{
			tag1++;
		}
		if (tag == 1 && i == 4 && wordInfor.infor_up.str_time[i] != ':')
		{
			tag1++;
		}
		if (tag == 0 && i == 5 && wordInfor.infor_up.str_time[i] != ':')
		{
			tag++;
		}
	}
	tmp[0] = mytime[0];
	tmp[1] = mytime[1];
	//ocr_recgInfor.currentTime.hours = atoi(tmp);
	ocr_recgInfor.Date_Time.tm_hour = atoi(tmp);
	tmp[0] = mytime[3];
	tmp[1] = mytime[4];
	//ocr_recgInfor.currentTime.minutes = atoi(tmp);
	ocr_recgInfor.Date_Time.tm_min = atoi(tmp);
	tmp[0] = mytime[6];
	tmp[1] = mytime[7];
	//ocr_recgInfor.currentTime.seconds = atoi(tmp);
	ocr_recgInfor.Date_Time.tm_sec = atoi(tmp);

	//识别速度和里程
	int len1 = wordInfor.infor_up.str_km.length();
	int flag = 0;
	string V, S;
	bool IsPoint = false;
	for (int i = 0; i < len1; i++)
	{
		if (flag == 0 && V.size() <= 4)
		{
			if (wordInfor.infor_up.str_km[i] >= '0'&&wordInfor.infor_up.str_km[i] <= '9')
			{
				V += wordInfor.infor_up.str_km[i];
			}
			else//不是数字则强制设置为0
			{
				V += '0';
			}
			
			if (wordInfor.infor_up.str_km[i+1] == 'k'|| wordInfor.infor_up.str_km[i + 2] == 'm'|| wordInfor.infor_up.str_km[i + 3] == '/')//
			{
				flag = 1;
				i = i + 4;//km/h
			}
		}
		else
		{
			if (wordInfor.infor_up.str_km[i] == 'k')break;
			if (wordInfor.infor_up.str_km[i] >= '0' && wordInfor.infor_up.str_km[i] <= '9')
			{
				S += wordInfor.infor_up.str_km[i];
			}
			else if (wordInfor.infor_up.str_km[i] == '.'&&S != "")
			{
				S += '.';
				IsPoint = true;
			}
			else
			{
				S += '0';
			}			
		}
			
	}
	//转换速度
	char* s;
	if (V == "")
	{
		ocr_recgInfor.Velocity = 0;//如果V为空，则速度视为0
	}
	/*else if (V[0] == '0')//如果速度第一位为0，则速度为0
	{
		ocr_recgInfor.Velocity = 0;//发现有些视频中速度是三位数
	}*/
	else
	{
		if (V.length() > 4)
		{
			string v = V.substr(V.length() - 4, 4);
			V = v;
		}
		for (int i = 0; i < V.length(); i++)
		{
			if (V[i]<'0' || V[i]>'9')
			{
				V[i] = '0';
				ocr_recgInfor.Confidence_Velocity = 0.5;
			}
		}
		if (V.length() > 0)
		{
			s = &V[0];
			ocr_recgInfor.Velocity = atoi(s);
		}
		else//为空
		{
			ocr_recgInfor.Velocity = 0;
		}
	}
	

	//转换里程
	if(S.find_first_of('.')<0 && (!S.empty()))//点可能缺失
	{
		string ms = S.substr(0, S.length() - 4);
		ms += '.';
		ms += S.substr(S.length() - 3, 3);
		S = ms;
	}
	if (S.length() > 0)
	{
		s = &S[0];
		ocr_recgInfor.Mileage = atof(s);
	}
	else//为空
	{
		ocr_recgInfor.Mileage = 0;
	}
	
	//识别车次和车号
	int len2 = wordInfor.infor_down.str_result.length();
	flag = 0;
	int acc2 = 0;//记录车次中的数字长
	bool VIN_start = false;
	/*for (int j = 2; j < len2; j++)
	{
		if (flag == 0)
		{
			if ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9')|| wordInfor.infor_down.str_result[j] == 'K' || wordInfor.infor_down.str_result[j] == 'T' || wordInfor.infor_down.str_result[j] == 'Z' || wordInfor.infor_down.str_result[j] == 'L' || wordInfor.infor_down.str_result[j] == 'D' || wordInfor.infor_down.str_result[j] == 'G' || wordInfor.infor_down.str_result[j] == 'X')
			{
				if (ocr_recgInfor.Train_Number.size() < 1)//第一位只可能是数字或KTZGDLX
				{
					ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
				}
				else if ((ocr_recgInfor.Train_Number[0] > '9' || ocr_recgInfor.Train_Number[0] < '0'))//如果第一个不是数字，数字最多5位
				{
					if (acc2 < 5)
					{						
						if ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9'))
						{
							acc2++;
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
						}
					}					
				}
				else //若第一位是数字
				{
					if (acc2 < 5)//若第一位是数字，数字长度最大为5
					{						
						if ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9'))
						{
							acc2++;
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
						}
						else//数字之后出现KTZGDLX
						{
							ocr_recgInfor.Train_Number = "";
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
							acc2 = 0;
						}
					}					
				}				
			}
			if (wordInfor.infor_down.str_result[j + 1] == '?'&& j <= 3)//若前四位遇到？说明之前的不是车次
			{
				ocr_recgInfor.Train_Number = "";
				acc2 = 0;
			}
			if ((j >= 3&&j<len2-3)&&(wordInfor.infor_down.str_result[j] == '?'||(wordInfor.infor_down.str_result[j+1]=='H'&&wordInfor.infor_down.str_result[j+2] == 'X'&&wordInfor.infor_down.str_result[j+3] == 'D') || (wordInfor.infor_down.str_result[j + 1] == 'H'&&wordInfor.infor_down.str_result[j + 2] == 'X') || (wordInfor.infor_down.str_result[j + 1] == 'H'&&wordInfor.infor_down.str_result[j + 3] == 'D') || (wordInfor.infor_down.str_result[j + 2] == 'X'&&wordInfor.infor_down.str_result[j + 3] == 'D')))
			{
				flag = 1;
			}
		}*/
	//9.28瞿修改
	for (int j = 0; j < len2; j++)
	{
		if (VIN_start == 0)//VIN还未开始
		{
			if (flag==0 && ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9') || wordInfor.infor_down.str_result[j] == 'K'|| wordInfor.infor_down.str_result[j] == 'k' || wordInfor.infor_down.str_result[j] == 'T' || wordInfor.infor_down.str_result[j] == 'Z' || wordInfor.infor_down.str_result[j] == 'L' || wordInfor.infor_down.str_result[j] == 'D' || wordInfor.infor_down.str_result[j] == 'G' || wordInfor.infor_down.str_result[j] == 'X'))
			{
				if (ocr_recgInfor.Train_Number.size() < 1)//第一位只可能是数字或KTZGDLX
				{
					ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
				}
				else if ((ocr_recgInfor.Train_Number[0] > '9' || ocr_recgInfor.Train_Number[0] < '0'))//如果第一个不是数字，数字最多5位
				{
					if (acc2 < 5)
					{
						if ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9'))
						{
							acc2++;
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
						}
					}
				}
				else //若第一位是数字
				{
					if (acc2 < 4)//若第一位是数字，数字长度最大为5
					{
						if ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9'))
						{
							acc2++;
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
						}
						else//数字之后出现KTZGDLX
						{
							ocr_recgInfor.Train_Number = "";
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
							acc2 = 0;
						}
					}
				}
			}
			if ((j >= 1 && j < len2 - 3) && wordInfor.infor_down.str_result[j] == '?')
			{
				flag = 1;
			}
			if ((wordInfor.infor_down.str_result[j + 1] == 'H'&&wordInfor.infor_down.str_result[j + 2] == 'X'&&wordInfor.infor_down.str_result[j + 3] == 'D')||((j >= 1 && j<len2 - 3) && ((wordInfor.infor_down.str_result[j + 1] == 'H'&&wordInfor.infor_down.str_result[j + 2] == 'X') || (wordInfor.infor_down.str_result[j + 1] == 'H'&&wordInfor.infor_down.str_result[j + 3] == 'D') || (wordInfor.infor_down.str_result[j + 2] == 'X'&&wordInfor.infor_down.str_result[j + 3] == 'D'))))
			{
				flag = 1;
				VIN_start = true;
			}
		}
		else//车号为9位
		{
			if (wordInfor.infor_down.str_result[j] != '?')
			{
				ocr_recgInfor.VIN += wordInfor.infor_down.str_result[j];
				//VIN_len++;
			}
		}
	}
}

void OcrRecognition::Suf_Process_OCR(OCR_Results_video &OCR_Original, OCR_Results_video &OCR_Processed)
{
	long frame_Num = OCR_Original.frame_Num;
	bool IsNextYear = false;
	bool IsNextMonth = false;
	bool IsNextDay = false;
	bool IsNextHours = false;
	bool IsNextMinutes = false;
	OCR_Result frame_result;
	std::vector<string> Train_Num;//车次
	std::vector<string> Train_VIN;//车号
	std::vector<double> Mileage;//里程
	std::vector<int> Velocity;//时速
	std::vector<int> Years;//时速
	std::vector<int> Monthes;//时速
	std::vector<int> Days;//时速
	std::vector<int> Hours;//时速
	std::vector<int> Minutes;//时速
	std::vector<int> Seconds;//时速
	OCR_Processed = OCR_Original;
	for (int k = 0; k < frame_Num; k++)
	{
		frame_result = OCR_Original.MTV.at(k);
		//读取车次和车号
		Train_Num.push_back(frame_result.Train_Number);
		Train_VIN.push_back(frame_result.VIN);
		//读取时速和里程
		Velocity.push_back(frame_result.Velocity);
		Mileage.push_back(frame_result.Mileage);
		//读取日期
		Years.push_back(frame_result.Date_Time.tm_year);
		Monthes.push_back(frame_result.Date_Time.tm_mon);
		Days.push_back(frame_result.Date_Time.tm_mday);
		//读取时间
		Hours.push_back(frame_result.Date_Time.tm_hour);
		Minutes.push_back(frame_result.Date_Time.tm_min);
		Seconds.push_back(frame_result.Date_Time.tm_sec);
	}
	//处理车次,假设最多变化一次	
	std::vector<Str_Long> TNHC = StrVectorHistCal(Train_Num);
	long tmp1 = 0;
	std::string PrimaryTN = "";
	std::string SecondTN = "";
	std::vector<long> P2S;//PrimaryTN变成SecondTN
	std::vector<long> S2P;//SecondTN变成PrimaryTN
	long PrimaryTN_start = frame_Num-1;
	//long PrimaryTN_end = 0;
	long SecondTN_start = frame_Num-1;
	//long SecondTN_end = 0;
	for (int k = 0; k < TNHC.size(); k++)//排名第一车次
	{
		if (TNHC[k].str_count > tmp1)
		{
			tmp1 = TNHC[k].str_count;
			PrimaryTN = TNHC[k].str_value;
		}
	}
	long tmp2 = 0;
	for (int k = 0; k < TNHC.size(); k++)//排名第二车次
	{
		if (TNHC[k].str_count > tmp2 && TNHC[k].str_count > 250 && TNHC[k].str_count != tmp1)
		{
			tmp2 = TNHC[k].str_count;
			SecondTN = TNHC[k].str_value;
		}
	}
	if (SecondTN == "")
	{
		for (long j = 0; j < frame_Num; j++)
		{
			//Train_Num[j] = PrimaryTN;
			OCR_Processed.MTV[j].Train_Number = PrimaryTN;
		}
	}
	else
	{
		for (long j = 0; j < frame_Num - 1; j++)
		{
			if (Train_Num[j] == PrimaryTN&&Train_Num[j + 1] == SecondTN)
			{
				P2S.push_back(j);
			}
			else if (Train_Num[j] == SecondTN&&Train_Num[j + 1] == PrimaryTN)
			{
				S2P.push_back(j);
			}
			else if (Train_Num[j] == PrimaryTN&&PrimaryTN_start == frame_Num - 1)//如果转化处出错，检测出现其他错误车号
			{
				PrimaryTN_start = j;
			}
			else if (Train_Num[j] == SecondTN&&SecondTN_start == frame_Num - 1)
			{
				SecondTN_start = j;
			}
		}

		if (P2S.size() < 1 && S2P.size() == 1)//仅有一次S2P变化
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < PrimaryTN_start)
				{
					OCR_Processed.MTV[j].Train_Number = SecondTN;
				}
				else
				{
					OCR_Processed.MTV[j].Train_Number = PrimaryTN;
				}
			}
		}
		else if (S2P.size() < 1 && P2S.size() == 1)//仅有一次P2S变化
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < SecondTN_start)
				{
					OCR_Processed.MTV[j].Train_Number = PrimaryTN;
				}
				else
				{
					OCR_Processed.MTV[j].Train_Number = SecondTN;
				}
			}
		}
		else
		{
			for (long j = 0; j < frame_Num; j++)
			{
				//Train_Num[j] = PrimaryTN;
				OCR_Processed.MTV[j].Train_Number = PrimaryTN;
			}
		}
	}	

	//处理车号，同一视频中车号不变
	int index_round = (int)(frame_Num*0.1);
	long acc1 = count(Train_VIN.begin(), Train_VIN.end(), Train_VIN.at(index_round));
	if (acc1 > frame_Num*0.7)
	{
		OCR_Processed.VIN = Train_VIN.at(index_round);
	}
	else
	{
		std::vector<Str_Long> VINHC = StrVectorHistCal(Train_VIN);
		long tmp = 0;
		for (int k = 0; k < VINHC.size(); k++)
		{
			if (VINHC[k].str_count > tmp)
			{
				tmp = VINHC[k].str_count;
				OCR_Processed.VIN = VINHC[k].str_value;
			}
		}
	}

	//校验日期
	//月
	std::vector<Int_Long> MHC = IntVectorHistCal(Monthes);
	long tmp0 = 0;
	long PrimaryFrame = frame_Num;//记录跨年时刻的关键帧
	int CurrentMonth = 0;
	for (int k = 0; k < MHC.size(); k++)
	{
		if (MHC[k].int_count > tmp0)
		{
			tmp0 = MHC[k].int_count;
			CurrentMonth = MHC[k].int_value;
		}
	}
	if (CurrentMonth > 11 || CurrentMonth < 0)
	{
		CurrentMonth = 0;//强制设置为0
	}
	
	int premonth = 0;
	int nextmonth = 0;
	long firstmonth_start = frame_Num;//只可能出现2月
	long secondmonth_start = 0;
	//long nextmonth_start = 0;
	//long ind_end = 0;
	if (CurrentMonth == 11)
	{
		premonth = CurrentMonth - 1;
		nextmonth = 0;
	}
	else if (CurrentMonth == 0)
	{
		premonth = 11;
		nextmonth = CurrentMonth + 1;
	}
	else
	{
		premonth = CurrentMonth - 1;
		nextmonth = CurrentMonth + 1;
	}

	//判断是否出现月份变化
	if (count(Monthes.begin(), Monthes.end(), premonth) >= count(Monthes.begin(), Monthes.end(), nextmonth))//根据统计确定出现的月份
	{
		int acc1 = 0, acc2 = 0;
		for (long j = 0; j < frame_Num; j++)
		{
			if (Monthes[j] == premonth)
			{
				acc1++;
				if (acc1 > 90)
				{
					firstmonth_start = j - 90;
				}
			}
			else if (Monthes[j] == CurrentMonth)
			{
				acc2++;
				if (acc2 > 90)
				{
					secondmonth_start = j - 90;
					break;
				}
			}
		}

		if (secondmonth_start > firstmonth_start)
		{			
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < secondmonth_start)
				{
					Monthes[j] = premonth;
					OCR_Processed.MTV[j].Date_Time.tm_mon = premonth;
				}
				else
				{
					Monthes[j] = CurrentMonth;
					OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentMonth;
				}
			}
		}
		else//视为没有跨月
		{
			for (long j = 0; j < frame_Num; j++)
			{
				Monthes[j] = CurrentMonth;
				OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentMonth;
			}
		}
		if (CurrentMonth == 0)
		{
			IsNextYear = true;
			PrimaryFrame = secondmonth_start;//第二年起始帧即为关键帧
		}
	}
	else //if (count(Monthes.begin(), Monthes.end(), premonth) < count(Monthes.begin(), Monthes.end(), nextmonth))
	{
		int acc1 = 0, acc2 = 0;
		for (long j = frame_Num-1; j >= 0; j--)//从后往前遍历
		{
			if (Monthes[j] == nextmonth)
			{
				acc1++;
				if (acc1 > 90)
				{
					secondmonth_start = j + 90;
				}
			}
			else if (Monthes[j] == CurrentMonth)
			{
				acc2++;
				if (acc2 > 90)
				{
					firstmonth_start = j + 90;
					break;
				}
			}
		}

		if (secondmonth_start > firstmonth_start)
		{
			for (long j = frame_Num-1; j >=0 ; j--)
			{
				if (j > firstmonth_start)
				{
					Monthes[j] = nextmonth;
					OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentMonth;
				}
				else
				{
					Monthes[j] = CurrentMonth;
					OCR_Processed.MTV[j].Date_Time.tm_mon = nextmonth;
				}
			}
		}
		else//视为没有跨月
		{
			for (long j = 0; j < frame_Num; j++)
			{
				Monthes[j] = CurrentMonth;
				OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentMonth;
			}
		}
		if (CurrentMonth == 11)
		{
			IsNextYear = true;
			PrimaryFrame = firstmonth_start + 1;//第一年结尾帧即为关键帧
		}
	}

	//年
	std::vector<Int_Long> YHC = IntVectorHistCal(Years);
	tmp0 = 0;
	int currentyear = 2018;
	for (int k = 0; k < YHC.size(); k++)
	{
		if (YHC[k].int_count > tmp0)
		{
			tmp0 = YHC[k].int_count;
			currentyear = YHC[k].int_value;
		}
	}
	if (currentyear > 2050 || currentyear < 0)
	{
		currentyear = 2018 - 1900;//强制设置为2018年
	}

	int preyear = currentyear - 1;
	int nextyear = currentyear + 1;
	
	if (IsNextYear == false)//如果没有跨年
	{
		for (long j = 0; j < frame_Num; j++)
		{
			Years[j] = currentyear;
			OCR_Processed.MTV[j].Date_Time.tm_year = currentyear;
		}
	}
	else//跨年
	{
		int acc1 = 0;
		long currentyear_start = 0;
		for (long j = 0; j < frame_Num; j++)//确定currentyear是第一年还是第二年
		{
			if (Years[j] == currentyear)
			{
				acc1++;
				if (acc1 > 100)
				{
					currentyear_start = j - 100;
					break;
				}
			}
		}

		if (currentyear_start < PrimaryFrame)//currentyear为第一年
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < PrimaryFrame)
				{
					Years[j] = preyear;
					OCR_Processed.MTV[j].Date_Time.tm_year = preyear;
				}
				else
				{
					Years[j] = currentyear;
					OCR_Processed.MTV[j].Date_Time.tm_year = currentyear;
				}
				
			}
		}
		else//currentyear为第二年
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < PrimaryFrame)
				{
					Years[j] = currentyear;
					OCR_Processed.MTV[j].Date_Time.tm_year = currentyear;
				}
				else
				{
					Years[j] = nextyear;
					OCR_Processed.MTV[j].Date_Time.tm_year = nextyear;
				}

			}
		}
		
	}

	/*ind_start = 0;
	ind_end = 0;
	int preyear = currentyear - 1;
	int nextyear = currentyear + 1;
	//前一年
	if (count(Years.begin(), Years.end(), preyear) > 300)
	{
		int acc1 = 0, acc2 = 0;
		for (long j = 0; j < frame_Num; j++)
		{
			if (Years[j] == preyear)
			{
				acc1++;
				if (acc1 > 10)
				{
					ind_start = j - 10;
				}
			}
			else if (Years[j] == currentyear)
			{
				acc2++;
				if (acc2 > 10)
				{
					ind_end = j - 10;
					break;
				}
			}
		}


		for (long j = 0; j < frame_Num; j++)
		{
			if (j < ind_end)
			{
				Years[j] = result - 1;
				OCR_Processed.MTV[j].Date_Time.tm_year = result - 1;
			}
			else
			{
				Years[j] = result;
				OCR_Processed.MTV[j].Date_Time.tm_year = result;
			}
		}
	}
	//后一年
	if (count(Years.begin(), Years.end(), result + 1) > 300)
	{
		int acc1 = 0, acc2 = 0;
		for (long j = 0; j < frame_Num; j++)
		{
			if (Years[j] == result)
			{
				acc1++;
				if (acc1 > 10)
				{
					ind_start = j - 10;
				}
			}
			else if (Years[j] == result + 1)
			{
				acc2++;
				if (acc2 > 10)
				{
					ind_end = j - 10;
					break;
				}
			}
		}

		for (long j = 0; j < frame_Num; j++)
		{
			if (j < ind_end)
			{
				Years[j] = result - 1;
				OCR_Processed.MTV[j].Date_Time.tm_year = result - 1;
			}
			else
			{
				Years[j] = result;
				OCR_Processed.MTV[j].Date_Time.tm_year = result;
			}
		}
	}*/

	/*//日
	std::vector<Int_Long> DHC = IntVectorHistCal(Days);
	tmp0 = 0;
	int CurrentDay = 0;
	for (int k = 0; k < DHC.size(); k++)
	{
	if (DHC[k].int_count > tmp0)
	{
	tmp0 = DHC[k].int_count;
	CurrentDay = DHC[k].int_value;
	}
	}
	ind_start = 0;
	ind_end = 0;
	int preday = 0;
	int nextday = 0;
	if (CurrentDay == 11)
	{
	premonth = CurrentDay - 1;
	nextmonth = 0;
	}
	else if (CurrentDay == 0)
	{
	premonth = 11;
	nextmonth = CurrentDay + 1;
	}
	else
	{
	premonth = CurrentDay - 1;
	nextmonth = CurrentDay + 1;
	}
	//前一天
	if (count(Monthes.begin(), Monthes.end(), premonth) > 300)
	{
	int acc1 = 0, acc2 = 0;
	for (long j = 0; j < frame_Num; j++)
	{
	if (Monthes[j] == premonth)
	{
	acc1++;
	if (acc1 > 10)
	{
	ind_start = j - 10;
	}
	}
	else if (Monthes[j] == CurrentDay)
	{
	acc2++;
	if (acc2 > 10)
	{
	ind_end = j - 10;
	break;
	}
	}
	}

	for (long j = 0; j < frame_Num; j++)
	{
	if (j < ind_end)
	{
	Monthes[j] = premonth;
	OCR_Processed.MTV[j].Date_Time.tm_mon = premonth;
	}
	else
	{
	Years[j] = CurrentDay;
	OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentDay;
	}
	}
	}
	//后一天
	if (count(Monthes.begin(), Monthes.end(), nextmonth) > 300)
	{
	int acc1 = 0, acc2 = 0;
	for (long j = 0; j < frame_Num; j++)
	{
	if (Monthes[j] == CurrentDay)
	{
	acc1++;
	if (acc1 > 10)
	{
	ind_start = j - 10;
	}
	}
	else if (Monthes[j] == nextmonth)
	{
	acc2++;
	if (acc2 > 10)
	{
	ind_end = j - 10;
	break;
	}
	}
	}

	for (long j = 0; j < frame_Num; j++)
	{
	if (j < ind_end)
	{
	Monthes[j] = CurrentDay;
	OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentDay;
	}
	else
	{
	Monthes[j] = nextmonth;
	OCR_Processed.MTV[j].Date_Time.tm_mon = nextmonth;
	}
	}
	}*/

	//校验时间
	int step = OCR_Original.Frame_Rate;//帧率
	bool SecondsReliable = false;//标记当前时间是否可信
	bool MinutesReliable = false;//标记当前时间是否可信
	bool HoursReliable = false;//标记当前时间是否可信
	long start_tag = 0;//记录跳变起始位置
	long end_tag = 0;//记录跳变终止位置
	bool IsNextSecond = false;
	long FramesPerDay = 0;//如果跨天的话，记录每天的帧数
	vector<int> OneDay;//存放一天
	
	/*//根据秒的变化规律求视频帧率
	int fps_acc1 = 0;
	int second_tmp = 0;
	int fps_tmp1 = 0;
	std::vector<int> FPS_TMP;//默认为25帧每秒
	for (long k = 0; k < frame_Num - 4; k++)
	{
		if (Seconds[k] == second_tmp)
		{
			fps_acc1++;
		}
		else if (Seconds[k + 3] == second_tmp)
		{
			fps_acc1 += 3;
		}
		else
		{
			if ((second_tmp == 59 && Seconds[k] == 0)|| (Seconds[k] - second_tmp == 1))//视为正常跳变
			{
				fps_tmp1 = fps_acc1;
				fps_acc1 = 1;
				second_tmp = Seconds[k];
				if (fps_tmp1 > 0)
				{
					FPS_TMP.push_back(fps_tmp1);
				}
			}
			else//非正常跳变
			{
				second_tmp = Seconds[k];
				fps_acc1 = 1;
			}
		}
	}
	//根据统计的最佳FPS
	std::vector<Int_Long> FPSHC = IntVectorHistCal(FPS_TMP);
	tmp0 = 0;
	int best_fps = 25;
	for (int k = 0; k < FPSHC.size(); k++)
	{
		if (FPSHC[k].int_count > tmp0)
		{
			tmp0 = FPSHC[k].int_count;
			best_fps = FPSHC[k].int_value;
		}
	}
	OCR_Processed.Frame_Rate = best_fps;
	step = best_fps;*/

	//开始校验时间
	for (long k = 0; k < frame_Num - 1; k++)
	{
		IsNextMinutes = false;
		IsNextHours = false;
		IsNextDay = false;
		IsNextMonth = false;
		IsNextYear = false;

		//秒
		IsNextSecond = false;
		if (Seconds[k] == 59 && Seconds[k + 1] == 0)//下一分钟
		{
			IsNextMinutes = true;
			IsNextSecond = true;
		}
		else if ((Seconds[k + 1] - Seconds[k] > 1) || (Seconds[k + 1] - Seconds[k] < 0))//非正常跳变
		{
			if (k - start_tag > step - 1 && k - start_tag < step + 1 && SecondsReliable)//跳变间隔在28-32之间，跳变到下一秒
			{
				IsNextSecond = true;
				if (Seconds[k] == 59)
				{
					//OCR_Processed.MTV[k + 1].Date_Time.tm_sec = 0;
					Seconds[k + 1] = 0;
					IsNextMinutes = true;
				}
				else
				{
					//OCR_Processed.MTV[k + 1].Date_Time.tm_sec = Seconds[k] + 1;
					Seconds[k + 1] = Seconds[k] + 1;
				}

			}
			else if (k - start_tag < step - 1 && SecondsReliable)//跳变间隔小于28,时间不变
			{
				//OCR_Processed.MTV[k + 1].Date_Time.tm_sec = Seconds[k];
				Seconds[k + 1] = Seconds[k];
			}
			OCR_Processed.MTV[k + 1].Confidence_Time = 0;//由于对时间进行了校正，置信度减小
		}

		if ((Seconds[k + 1] - Seconds[k] == 1))//正常跳变
		{
			if ((end_tag - start_tag > step - 1) && (end_tag - start_tag < step + 1))
			{
				SecondsReliable = true;
				start_tag = end_tag + 1;
			}
			else if (end_tag - start_tag > step + 3)
			{
				start_tag = 0;
				SecondsReliable = false;
			}
			IsNextSecond = true;
		}

		if (Seconds[k] > 59 || Seconds[k] < 0)
		{
			Seconds[k] = 0;//秒强制设置为0
			OCR_Processed.MTV[k].Confidence_Time = 0;
			SecondsReliable = false;
		}

		if (IsNextSecond)
		{
			if (start_tag == 0)
			{
				start_tag = k + 1;
			}
			else
			{
				end_tag = k;
			}

			
		}
		OCR_Processed.MTV[k].Date_Time.tm_sec = Seconds[k];		

		//分钟
		if (Minutes[k]==59 && Minutes[k + 1] == 0 && IsNextMinutes)//下一小时
		{
			IsNextHours = true;
		}
		else if ((Minutes[k+1] - Minutes[k] >1)|| (Minutes[k + 1] - Minutes[k] <0))//非正常跳变
		{
			if (IsNextMinutes == false)
			{
				if (MinutesReliable == true)
				{
					//OCR_Processed.MTV[k + 1].Date_Time.tm_min = Minutes[k];
					Minutes[k + 1] = Minutes[k];
				}				
			}
			else if(Minutes[k]==59)//根据秒推断跳转到下一分钟
			{
				Minutes[k + 1] = 0;
				IsNextHours = true;
				MinutesReliable = true;
			}
			else
			{
				Minutes[k + 1] = Minutes[k] + 1;
				MinutesReliable = true;
			}
		}
		else if ((Minutes[k + 1] - Minutes[k] == 1) && IsNextMinutes==false)//分钟和秒跳变不一致
		{
			OCR_Processed.MTV[k + 1].Confidence_Time = 0;
		}
		if (Minutes[k] > 59 || Minutes[k] < 0)
		{
			OCR_Processed.MTV[k].Date_Time.tm_min = 0;//分钟强制设置为0
		}
		OCR_Processed.MTV[k].Date_Time.tm_min = Minutes[k];

		//小时
		if (Hours[k]==23 && Hours[k + 1] == 0 && IsNextHours)//下一天
		{
			IsNextDay =true;
		}
		else if ((Hours[k + 1] - Hours[k] > 1) || (Hours[k + 1] - Hours[k] < 0))//非正常跳变
		{
			if (IsNextHours == false)
			{
				//OCR_Processed.MTV[k + 1].Date_Time.tm_hour = Hours[k];
				Hours[k + 1] = Hours[k];
			}
			else if (Hours[k] == 23)
			{
				Hours[k + 1] = 0;
				IsNextDay = true;
			}
			else
			{
				Hours[k + 1] = Hours[k] = 1;
			}
			//OCR_Processed.MTV[k + 1].Confidence_Time -= 0.3;
		}
		else if (Hours[k + 1] - Hours[k] == 1 && IsNextHours==false)//小时和分钟跳变不一致
		{
			OCR_Processed.MTV[k + 1].Confidence_Time  = 0;
		}

		if (Hours[k] > 23 || Hours[k] < 0)
		{
			OCR_Processed.MTV[k].Date_Time.tm_hour = 0;//小时强制设置为0
		}

		OCR_Processed.MTV[k].Date_Time.tm_hour = Hours[k];

		//最后一帧强制等于倒数第二帧
		if (k == frame_Num - 2)
		{
			OCR_Processed.MTV[k + 1].Date_Time.tm_sec = Seconds[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_min = Minutes[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_hour = Hours[k];
		}

		//天
		if (IsNextDay)
		{
			std::vector<Int_Long> DHC = IntVectorHistCal(OneDay);
			long tmp_day = 0;
			int CurrentDay = 0;
			for (long j = 0; j < DHC.size(); j++)
			{
				if (DHC[j].int_count > tmp_day)
				{
					tmp_day = DHC[j].int_count;
					CurrentDay = DHC[j].int_value;
				}
			}
			if (CurrentDay > 31 || CurrentDay < 0)
			{
				CurrentDay = 0;
			}
			for (long j = FramesPerDay; j < OneDay.size(); j++)
			{
				Days[j] = CurrentDay;
				OCR_Processed.MTV[k].Date_Time.tm_mday = CurrentDay;
			}
			FramesPerDay += OneDay.size();
			OneDay.clear();
		}
		else
		{
			OneDay.push_back(Days[k]);
		}
	}

	//校验时速
	std::vector<Int_Long> VHC = IntVectorHistCal(Velocity);
	long tmp_veloccity = 0;
	int CurrentDay = 0;
	for (long j = 0; j < VHC.size(); j++)
	{
		if (VHC[j].int_count > tmp_veloccity)
		{
			tmp_veloccity = VHC[j].int_count;
			//CurrentDay = VHC[j].int_value;
		}
	}
	if (tmp_veloccity > frame_Num*0.8)
	{
		for (long j = 0; j < frame_Num; j++)
		{
			OCR_Processed.MTV[j].Velocity = 0;
		}
	}
}

//根据帧率校正时间和日期
void OcrRecognition::Suf_Process_OCR_BasedFPS(OCR_Results_video &OCR_Original, OCR_Results_video &OCR_Processed)
{
	long frame_Num = OCR_Original.frame_Num;
	OCR_Result frame_result;
	std::vector<string> Train_Num;//车次
	std::vector<string> Train_VIN;//车号
	std::vector<double> Mileage;//里程
	std::vector<int> Velocity;//时速
	std::vector<int> Years;//时速
	std::vector<int> Monthes;//时速
	std::vector<int> Days;//时速
	std::vector<int> Hours;//时速
	std::vector<int> Minutes;//时速
	std::vector<int> Seconds;//时速
	OCR_Processed = OCR_Original;
	for (int k = 0; k < frame_Num; k++)
	{
		frame_result = OCR_Original.MTV.at(k);
		//读取车次和车号
		Train_Num.push_back(frame_result.Train_Number);
		Train_VIN.push_back(frame_result.VIN);
		//读取时速和里程
		Velocity.push_back(frame_result.Velocity);
		Mileage.push_back(frame_result.Mileage);
		//读取日期
		Years.push_back(frame_result.Date_Time.tm_year);
		Monthes.push_back(frame_result.Date_Time.tm_mon);
		Days.push_back(frame_result.Date_Time.tm_mday);
		//读取时间
		Hours.push_back(frame_result.Date_Time.tm_hour);
		Minutes.push_back(frame_result.Date_Time.tm_min);
		Seconds.push_back(frame_result.Date_Time.tm_sec);
	}

	//处理车次,假设最多变化一次	
	std::vector<Str_Long> TNHC = StrVectorHistCal(Train_Num);
	long tmp1 = 0;
	std::string PrimaryTN = "";
	std::string SecondTN = "";
	std::vector<long> P2S;//PrimaryTN变成SecondTN
	std::vector<long> S2P;//SecondTN变成PrimaryTN
	long PrimaryTN_start = frame_Num - 1;
	//long PrimaryTN_end = 0;
	long SecondTN_start = frame_Num - 1;
	//long SecondTN_end = 0;
	for (int k = 0; k < TNHC.size(); k++)//排名第一车次
	{
		if (TNHC[k].str_count > tmp1)
		{
			tmp1 = TNHC[k].str_count;
			PrimaryTN = TNHC[k].str_value;
		}
	}
	long tmp2 = 0;
	for (int k = 0; k < TNHC.size(); k++)//排名第二车次
	{
		if (TNHC[k].str_count > tmp2 && TNHC[k].str_count > 250 && TNHC[k].str_count != tmp1)
		{
			tmp2 = TNHC[k].str_count;
			SecondTN = TNHC[k].str_value;
		}
	}
	if (SecondTN == "")
	{
		for (long j = 0; j < frame_Num; j++)
		{
			//Train_Num[j] = PrimaryTN;
			OCR_Processed.MTV[j].Train_Number = PrimaryTN;
		}
	}
	else
	{
		for (long j = 0; j < frame_Num - 1; j++)
		{
			if (Train_Num[j] == PrimaryTN&&Train_Num[j + 1] == SecondTN)
			{
				P2S.push_back(j);
			}
			else if (Train_Num[j] == SecondTN&&Train_Num[j + 1] == PrimaryTN)
			{
				S2P.push_back(j);
			}
			else if (Train_Num[j] == PrimaryTN&&PrimaryTN_start == frame_Num - 1)//如果转化处出错，检测出现其他错误车号
			{
				PrimaryTN_start = j;
			}
			else if (Train_Num[j] == SecondTN&&SecondTN_start == frame_Num - 1)
			{
				SecondTN_start = j;
			}
		}

		if (P2S.size() < 1 && S2P.size() == 1)//仅有一次S2P变化
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < PrimaryTN_start)
				{
					OCR_Processed.MTV[j].Train_Number = SecondTN;
				}
				else
				{
					OCR_Processed.MTV[j].Train_Number = PrimaryTN;
				}
			}
		}
		else if (S2P.size() < 1 && P2S.size() == 1)//仅有一次P2S变化
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < SecondTN_start)
				{
					OCR_Processed.MTV[j].Train_Number = PrimaryTN;
				}
				else
				{
					OCR_Processed.MTV[j].Train_Number = SecondTN;
				}
			}
		}
		else
		{
			for (long j = 0; j < frame_Num; j++)
			{
				//Train_Num[j] = PrimaryTN;
				OCR_Processed.MTV[j].Train_Number = PrimaryTN;
			}
		}
	}

	//处理车号，同一视频中车号不变
	int index_round = (int)(frame_Num*0.1);
	long acc1 = count(Train_VIN.begin(), Train_VIN.end(), Train_VIN.at(index_round));
	if (acc1 > frame_Num*0.7)
	{
		OCR_Processed.VIN = Train_VIN.at(index_round);
	}
	else
	{
		std::vector<Str_Long> VINHC = StrVectorHistCal(Train_VIN);
		long tmp = 0;
		for (int k = 0; k < VINHC.size(); k++)
		{
			if (VINHC[k].str_count > tmp)
			{
				tmp = VINHC[k].str_count;
				OCR_Processed.VIN = VINHC[k].str_value;
			}
		}
	}

	//校验时速
	//检测突变点（速度变化超过5）1014增加
	int v0_acc = 0;//速度保持为0
	//从前往后
	for (long j = 1; j < Velocity.size(); j++)
	{
		//校验速度
		if (Velocity[j] - Velocity[j - 1] > 5)//速度变化超过5
		{
			if (v0_acc > 10)
			{
				Velocity[j] = 0;
			}
		}
		if (Velocity[j] == 0)
		{
			v0_acc++;
		}
		else
		{
			v0_acc = 0;
		}
	}
	//从后往前
	v0_acc = 0;//速度保持为0
	for (long j = Velocity.size()- 2; j > 0; j--)
	{
		//校验速度
		if (Velocity[j] - Velocity[j+1] > 5)//速度变化超过5
		{
			if (v0_acc > 10)
			{
				Velocity[j] = 0;
			}
		}
		if (Velocity[j] == 0)
		{
			v0_acc++;
		}
		else
		{
			v0_acc = 0;
		}
		OCR_Processed.MTV[j].Velocity = Velocity[j];
	}

	/*std::vector<Int_Long> VHC = IntVectorHistCal(Velocity);
	long tmp_veloccity = 0;
	int CurrentDay = 0;
	for (long j = 0; j < VHC.size(); j++)
	{
		if (VHC[j].int_count > tmp_veloccity)
		{
			tmp_veloccity = VHC[j].int_count;
			//CurrentDay = VHC[j].int_value;
		}
	}
	if (tmp_veloccity > frame_Num*0.8)
	{
		for (long j = 0; j < frame_Num; j++)
		{
			OCR_Processed.MTV[j].Velocity = 0;
		}
	}*/

	//校验日期
	bool IsNextYear = false;
	bool IsNextMonth = false;
	bool IsNextDay = false;
	bool IsNextHours = false;
	bool IsNextMinutes = false;
	
	//校验时间
	int FPS = OCR_Original.Frame_Rate;//帧率
	bool SecondsReliable = false;//标记当前时间是否可信
	bool MinutesReliable = false;//标记当前时间是否可信
	bool HoursReliable = false;//标记当前时间是否可信
	long start_tag = 0;//记录跳变起始位置
	long end_tag = 0;//记录跳变终止位置
	bool IsNextSecond = false;
	long FramesPerDay = 0;//如果跨天的话，记录每天的帧数

	//开始校验时间		
	int secondACC1 = 0;//用于记录连续正常跳转数

	//寻找一个秒数可靠的校验基准，连续正常跳变数大于3
	for (long k = FPS-1; k < frame_Num - FPS - 1; k++)
	{
		if ((Seconds[k] == 59 && Seconds[k + 1] == 0) || Seconds[k + 1] - Seconds[k] == 1)
		{
			int secondACC0 = 1;//
			int tmp1 = Seconds[k], tmp2 = Seconds[k + 1];
			for (long j = k - FPS + 1; j < k; j++)
			{
				if (Seconds[j] != tmp1)
				{
					secondACC0 = 0;
				}
			}
			if (secondACC0 > 0)
			{
				for (long j = k + 2; j <= k + FPS; j++)
				{
					if (Seconds[j] != tmp2)
					{
						secondACC0 = 0;
					}
				}
			}
			if (secondACC0 > 0)
			{
				secondACC1++;
			}
			else
			{
				secondACC1 = 0;
			}			
		}
		if (secondACC1 >= 3)//找到可靠的基准点
		{
			long BaseIndex = k;
			int BaseValue = Seconds[k];
			int tmpACC = 0;
			int tmpValue = BaseValue;
			for (long j = k; j >= 0;j--)
			{
				tmpACC++;
				Seconds[j] = tmpValue;
				if (tmpACC == FPS)
				{
					tmpACC = 0;
					if (tmpValue == 0)
					{
						tmpValue = 59;
					}
					else
					{
						tmpValue = tmpValue - 1;
					}
				}								
			}
			break;//跳出循环
		}
	}
	
	int secondACC2 = 0;//用于记录同一秒的累积帧数
	long minute_start = 0, minute_end, hour_start = 0, hour_end,day_start = 0,day_end;
	for (long k = 0; k < frame_Num - 1; k++)
	{
		IsNextMinutes = false;
		IsNextHours = false;
		IsNextDay = false;
		IsNextMonth = false;
		IsNextYear = false;

		//秒
		if (IsNextSecond)//根据帧率判断跳转到下一秒
		{
			if (Seconds[k] == 59 && Seconds[k + 1] == 0)//下一分钟
			{
				IsNextMinutes = true;//下一分钟开始
				IsNextSecond = false;//下一帧
				secondACC2 = 0;//重新计数
			}
			else if (Seconds[k + 1] == Seconds[k] && secondACC2 <= FPS)//计数已到，但检测下一帧没有跳秒,容忍滞后一帧跳秒
			{
				secondACC2++;
				IsNextSecond = true;//强制下一帧跳转下一秒
				OCR_Processed.MTV[k + 1].Confidence_Time = 0;//时间校验可能有误
			}
			else//本帧强制跳转
			{
				IsNextSecond = false;
				secondACC2 = 0;//重新计数
				if (Seconds[k] == 59)
				{
					Seconds[k + 1] = 0;
					IsNextMinutes = true;//下一分钟开始
				}
				else
				{
					Seconds[k + 1] = Seconds[k] + 1;
				}
			}			
		}
		else//根据计数未跳转下一秒
		{
			secondACC2++;
			//判断计数是否满足跳秒
			if (secondACC2 >= FPS-1)
			{
				if (((Seconds[k] == 59 && Seconds[k + 1] == 0) || Seconds[k + 1] - Seconds[k] == 1))//计数不足，但呈现出正确跳变的迹象，容忍提前一帧跳秒
				{
					OCR_Processed.MTV[k + 1].Confidence_Time = 0;//时间校验可能有误
					secondACC2 = 0;//重新计数
					if (Seconds[k] == 59)
					{
						IsNextMinutes = true;
					}
				}
				else
				{
					Seconds[k + 1] = Seconds[k];
					IsNextSecond = true;
				}				
			}
			else
			{
				Seconds[k + 1] = Seconds[k];
			}
		}

		if (Seconds[k] > 59 || Seconds[k] < 0)//异常数据
		{
			Seconds[k] = 0;//秒强制设置为0
			OCR_Processed.MTV[k].Confidence_Time = 0;
			SecondsReliable = false;
		}
		OCR_Processed.MTV[k].Date_Time.tm_sec = Seconds[k];

		//分钟
		if (IsNextMinutes)
		{
			std::vector<int> OneMinute;
			//std::copy(Minutes.begin() + minute_start, Minutes.begin() + k, OneMinute.begin());
			OneMinute.assign(Minutes.begin() + minute_start, Minutes.begin() + k);
			std::vector<Int_Long> MHC = IntVectorHistCal(OneMinute);
			long tmp0 = 0;
			int CurrentMinute = 0;
			for (int j = 0; j < MHC.size(); j++)
			{
				if (MHC[j].int_count > tmp0)
				{
					tmp0 = MHC[j].int_count;
					CurrentMinute = MHC[j].int_value;					
				}
			}

			if (CurrentMinute > 59 || CurrentMinute < 0)
			{
				if (minute_start > 0)
				{
					if (Minutes[minute_start - 1] == 59)
					{
						CurrentMinute = 0;
					}
					else
					{
						CurrentMinute = Minutes[minute_start - 1] + 1;
					}
				}
				else
				{
					CurrentMinute = 0;
				}
			}

			//修改分钟
			for (long j = minute_start; j <= k; j++)
			{
				Minutes[j] = CurrentMinute;
				OCR_Processed.MTV[j].Date_Time.tm_min = CurrentMinute;
			}

			//判断小时是否跳变
			if (Minutes[k] == 59 && Minutes[k + 1] == 0)//下一小时
			{
				IsNextHours = true;
			}

			//恢复全局参数
			IsNextMinutes = false;
			minute_start = k + 1;
		}

		//小时
		if (IsNextHours)
		{
			std::vector<int> OneHour;
			//std::copy(Hours.begin() + hour_start, Hours.begin() + k, OneHour.begin());
			OneHour.assign(Hours.begin() + hour_start, Hours.begin() + k);
			std::vector<Int_Long> HHC = IntVectorHistCal(OneHour);
			long tmp0 = 0;
			int CurrentHour = 0;
			for (int j = 0; j < HHC.size(); j++)
			{
				if (HHC[j].int_count > tmp0)
				{
					tmp0 = HHC[j].int_count;
					CurrentHour = HHC[j].int_value;					
				}
			}

			if (CurrentHour > 23 || CurrentHour < 0)
			{
				if (hour_start > 0)
				{
					if (Hours[hour_start - 1] == 23)
					{
						CurrentHour = 0;
					}
					else
					{
						CurrentHour = Hours[hour_start - 1] + 1;
					}
				}
				else
				{
					CurrentHour = 0;
				}
			}

			//修改小时
			for (long j = hour_start; j <= k; j++)
			{
				Hours[j] = CurrentHour;
				OCR_Processed.MTV[j].Date_Time.tm_hour = CurrentHour;
			}

			//判断小时是否跳变
			if (Hours[k] == 23 && Hours[k + 1] == 0)//下一小时
			{
				IsNextDay = true;
			}

			//恢复全局参数
			IsNextHours = false;
			hour_start = k + 1;
		}

		//天
		if (IsNextDay)
		{
			std::vector<int> OneDay;//存放一天
			//std::copy(Days.begin() + day_start, Days.begin() + k, OneDay.begin());
			OneDay.assign(Days.begin() + day_start, Days.begin() + k);
			std::vector<Int_Long> DHC = IntVectorHistCal(OneDay);
			long tmp0 = 0;
			int CurrentDay = 0;
			for (int j = 0; j < DHC.size(); j++)
			{
				if (DHC[j].int_count > tmp0)
				{
					tmp0 = DHC[j].int_count;
					CurrentDay = DHC[j].int_value;
				}
			}

			if (CurrentDay > 31 || CurrentDay < 0)
			{
				if (day_start > 0)
				{
					CurrentDay = Days[day_start - 1];
				}
				else
				{
					CurrentDay = 0;
				}
			}

			//修改天
			for (long j = day_start; j <= k; j++)
			{
				Days[j] = CurrentDay;
				OCR_Processed.MTV[j].Date_Time.tm_mday = CurrentDay;
			}

			//恢复全局参数
			IsNextDay = false;
			day_start = k + 1;
		}

		//对最后未满足跳转的分、小时、天进行校正
		if (k == frame_Num - 2)
		{
			//分
			long tmp0;
			std::vector<int> OneMinute;
			if (minute_start < k)
			{
				//std::copy(Minutes.begin() + minute_start, Minutes.begin() + k, OneMinute.begin());
				OneMinute.assign(Minutes.begin() + minute_start, Minutes.begin() + k);
				std::vector<Int_Long> MHC = IntVectorHistCal(OneMinute);
				tmp0 = 0;
				int CurrentMinute = 0;
				for (int j = 0; j < MHC.size(); j++)
				{
					if (MHC[j].int_count > tmp0)
					{
						tmp0 = MHC[j].int_count;
						CurrentMinute = MHC[j].int_value;						
					}
				}

				if (CurrentMinute > 59 || CurrentMinute < 0)
				{
					if (minute_start > 0)
					{
						if (Minutes[minute_start - 1] == 59)
						{
							CurrentMinute = 0;
						}
						else
						{
							CurrentMinute = Minutes[minute_start - 1] + 1;
						}
					}
					else
					{
						CurrentMinute = 0;
					}
				}

				//修改分钟
				for (long j = minute_start; j <= k; j++)
				{
					Minutes[j] = CurrentMinute;
					OCR_Processed.MTV[j].Date_Time.tm_min = CurrentMinute;
				}
			}			

			//小时
			std::vector<int> OneHour;
			if (hour_start < k)
			{
				//std::copy(Hours.begin() + hour_start, Hours.begin() + k, OneHour.begin());
				OneHour.assign(Hours.begin() + hour_start, Hours.begin() + k);
				std::vector<Int_Long> HHC = IntVectorHistCal(OneHour);
				tmp0 = 0;
				int CurrentHour = 0;
				for (int j = 0; j < HHC.size(); j++)
				{
					if (HHC[j].int_count > tmp0)
					{
						tmp0 = HHC[j].int_count;
						CurrentHour = HHC[j].int_value;
					}
				}

				if (CurrentHour > 23 || CurrentHour < 0)
				{
					if (hour_start > 0)
					{
						if (Hours[hour_start - 1] == 23)
						{
							CurrentHour = 0;
						}
						else
						{
							CurrentHour = Hours[hour_start - 1] + 1;
						}
					}
					else
					{
						CurrentHour = 0;
					}
				}

				//修改小时
				for (long j = hour_start; j <= k; j++)
				{
					Hours[j] = CurrentHour;
					OCR_Processed.MTV[j].Date_Time.tm_hour = CurrentHour;
				}
			}			

			//天
			std::vector<int> OneDay;//存放一天
			if (day_start < k)
			{
				//std::copy(Days.begin() + day_start, Days.begin() + k, OneDay.begin());
				OneDay.assign(Days.begin() + day_start, Days.begin() + k);
				std::vector<Int_Long> DHC = IntVectorHistCal(OneDay);
				tmp0 = 0;
				int CurrentDay = 0;
				for (int j = 0; j < DHC.size(); j++)
				{
					if (DHC[j].int_count > tmp0)
					{
						tmp0 = DHC[j].int_count;
						CurrentDay = DHC[j].int_value;
						if (CurrentDay > 31 || CurrentDay < 0)
						{
							if (day_start > 0)
							{
								CurrentDay = Days[day_start - 1];
							}
							else
							{
								CurrentDay = 0;
							}
						}
					}
				}
				//修改天
				for (long j = day_start; j <= k; j++)
				{
					Days[j] = CurrentDay;
					OCR_Processed.MTV[j].Date_Time.tm_mday = CurrentDay;
				}
			}
			

			//倒数第一帧强制等于倒数第二帧
			OCR_Processed.MTV[k + 1].Date_Time.tm_sec = Seconds[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_min = Minutes[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_hour = Hours[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_mday = Days[k];
		}
	}

	//月
	std::vector<Int_Long> MonHC = IntVectorHistCal(Monthes);
	long tmp0 = 0;
	long PrimaryFrame = frame_Num;//记录跨年时刻的关键帧
	int CurrentMonth = 0;
	for (int k = 0; k < MonHC.size(); k++)
	{
		if (MonHC[k].int_count > tmp0)
		{
			tmp0 = MonHC[k].int_count;
			CurrentMonth = MonHC[k].int_value;
		}
	}
	if (CurrentMonth > 11 || CurrentMonth < 0)
	{
		CurrentMonth = 0;//强制设置为0
	}

	int premonth = 0;
	int nextmonth = 0;
	long firstmonth_start = frame_Num;//只可能出现2月
	long secondmonth_start = 0;
	//long nextmonth_start = 0;
	//long ind_end = 0;
	if (CurrentMonth == 11)
	{
		premonth = CurrentMonth - 1;
		nextmonth = 0;
	}
	else if (CurrentMonth == 0)
	{
		premonth = 11;
		nextmonth = CurrentMonth + 1;
	}
	else
	{
		premonth = CurrentMonth - 1;
		nextmonth = CurrentMonth + 1;
	}

	//判断是否出现月份变化
	if (count(Monthes.begin(), Monthes.end(), premonth) >= count(Monthes.begin(), Monthes.end(), nextmonth))//根据统计确定出现的月份
	{
		int acc1 = 0, acc2 = 0;
		for (long j = 0; j < frame_Num; j++)
		{
			if (Monthes[j] == premonth)
			{
				acc1++;
				if (acc1 > 90)
				{
					firstmonth_start = j - 90;
				}
			}
			else if (Monthes[j] == CurrentMonth)
			{
				acc2++;
				if (acc2 > 90)
				{
					secondmonth_start = j - 90;
					break;
				}
			}
		}

		if (secondmonth_start > firstmonth_start)
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < secondmonth_start)
				{
					Monthes[j] = premonth;
					OCR_Processed.MTV[j].Date_Time.tm_mon = premonth;
				}
				else
				{
					Monthes[j] = CurrentMonth;
					OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentMonth;
				}
			}
		}
		else//视为没有跨月
		{
			for (long j = 0; j < frame_Num; j++)
			{
				Monthes[j] = CurrentMonth;
				OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentMonth;
			}
		}
		if (CurrentMonth == 0)
		{
			IsNextYear = true;
			PrimaryFrame = secondmonth_start;//第二年起始帧即为关键帧
		}
	}
	else //if (count(Monthes.begin(), Monthes.end(), premonth) < count(Monthes.begin(), Monthes.end(), nextmonth))
	{
		int acc1 = 0, acc2 = 0;
		for (long j = frame_Num - 1; j >= 0; j--)//从后往前遍历
		{
			if (Monthes[j] == nextmonth)
			{
				acc1++;
				if (acc1 > 90)
				{
					secondmonth_start = j + 90;
				}
			}
			else if (Monthes[j] == CurrentMonth)
			{
				acc2++;
				if (acc2 > 90)
				{
					firstmonth_start = j + 90;
					break;
				}
			}
		}

		if (secondmonth_start > firstmonth_start)
		{
			for (long j = frame_Num - 1; j >= 0; j--)
			{
				if (j > firstmonth_start)
				{
					Monthes[j] = nextmonth;
					OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentMonth;
				}
				else
				{
					Monthes[j] = CurrentMonth;
					OCR_Processed.MTV[j].Date_Time.tm_mon = nextmonth;
				}
			}
		}
		else//视为没有跨月
		{
			for (long j = 0; j < frame_Num; j++)
			{
				Monthes[j] = CurrentMonth;
				OCR_Processed.MTV[j].Date_Time.tm_mon = CurrentMonth;
			}
		}
		if (CurrentMonth == 11)
		{
			IsNextYear = true;
			PrimaryFrame = firstmonth_start + 1;//第一年结尾帧即为关键帧
		}
	}

	//年
	std::vector<Int_Long> YHC = IntVectorHistCal(Years);
	tmp0 = 0;
	int currentyear = 2018;
	for (int k = 0; k < YHC.size(); k++)
	{
		if (YHC[k].int_count > tmp0)
		{
			tmp0 = YHC[k].int_count;
			currentyear = YHC[k].int_value;
		}
	}
	if (currentyear > 2050 || currentyear < 0)
	{
		currentyear = 2018 - 1900;//强制设置为2018年
	}

	int preyear = currentyear - 1;
	int nextyear = currentyear + 1;

	if (IsNextYear == false)//如果没有跨年
	{
		for (long j = 0; j < frame_Num; j++)
		{
			Years[j] = currentyear;
			OCR_Processed.MTV[j].Date_Time.tm_year = currentyear;
		}
	}
	else//跨年
	{
		int acc1 = 0;
		long currentyear_start = 0;
		for (long j = 0; j < frame_Num; j++)//确定currentyear是第一年还是第二年
		{
			if (Years[j] == currentyear)
			{
				acc1++;
				if (acc1 > 100)
				{
					currentyear_start = j - 100;
					break;
				}
			}
		}

		if (currentyear_start < PrimaryFrame)//currentyear为第一年
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < PrimaryFrame)
				{
					Years[j] = preyear;
					OCR_Processed.MTV[j].Date_Time.tm_year = preyear;
				}
				else
				{
					Years[j] = currentyear;
					OCR_Processed.MTV[j].Date_Time.tm_year = currentyear;
				}

			}
		}
		else//currentyear为第二年
		{
			for (long j = 0; j < frame_Num; j++)
			{
				if (j < PrimaryFrame)
				{
					Years[j] = currentyear;
					OCR_Processed.MTV[j].Date_Time.tm_year = currentyear;
				}
				else
				{
					Years[j] = nextyear;
					OCR_Processed.MTV[j].Date_Time.tm_year = nextyear;
				}

			}
		}

	}	
}

//根据视频中秒的变化规律求视频帧率
int OcrRecognition::GetVideoFrame(std::vector<int> seconds)
{	
	int fps_acc1 = 0;
	int second_tmp = 0;
	int fps_tmp1 = 0;
	std::vector<int> FPS_TMP;//默认为25帧每秒
	for (long k = 0; k < seconds.size() - 4; k++)
	{
		if (seconds[k] == second_tmp)
		{
			fps_acc1++;
		}
		else if (seconds[k + 3] == second_tmp)
		{
			fps_acc1 += 3;
		}
		else
		{
			if ((second_tmp == 59 && seconds[k] == 0) || (seconds[k] - second_tmp == 1))//视为正常跳变
			{
				fps_tmp1 = fps_acc1;
				fps_acc1 = 1;
				second_tmp = seconds[k];
				if (fps_tmp1 > 0)
				{
					FPS_TMP.push_back(fps_tmp1);
				}
			}
			else//非正常跳变
			{
				second_tmp = seconds[k];
				fps_acc1 = 1;
			}
		}
	}
	//根据统计的最佳FPS
	std::vector<Int_Long> FPSHC = IntVectorHistCal(FPS_TMP);
	int tmp0 = 0;
	int best_fps = 25;
	for (int k = 0; k < FPSHC.size(); k++)
	{
		if (FPSHC[k].int_count > tmp0)
		{
			tmp0 = FPSHC[k].int_count;
			best_fps = FPSHC[k].int_value;
		}
	}
	if (best_fps > 20)
	{
		return best_fps = 25;
	}
	else
	{
		return best_fps = 8;
	}
}