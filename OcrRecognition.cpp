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
int g_upleft_width = 65535;//������¼ʱ����ٶȵķָ���
int g_downleft_width = 65535;//������¼���źͺ����ַ��ķָ��
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
	rect.x = 30;//ֱ�ӽ�����������ȥ����9.28����
	rect.width = rect.width - 30;
	im_down= imGray(rect).clone();
	Cut_PreProcess(im_down, im_down1);
	if (m_wordHeigh > 25 || im_down1.cols > PICWIDTH)
	{
		resize(im_up1, im_up1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_up1, im_up1, 127, 255, CV_THRESH_BINARY);//����ʱ�ᵼ�±�Ե����ֵС��255��Ӱ��ʶ��Ч��
		resize(im_down1, im_down1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_down1, im_down1, 127, 255, CV_THRESH_BINARY);
	}
	//ImageOcr_up(im_up1, wordInfor.infor_up);

	numWidth[CHARSUM] = { 0 };
	numHeight[CHARSUM] = { 0 };
	bool find = false;
	find = Crop_up(im_up1, im_up2, im_up3);//�ж���û���ҵ��ֶμ�ķֽ���
	if (find)
	{
		ImageOcr_up1(im_up2, wordInfor.infor_up);
		ImageOcr_up2(im_up3, wordInfor.infor_up);
	}
	else
	{
		ImageOcr_up(im_up1, wordInfor.infor_up);
	}
	//9.28����
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
		if (acc == 0)//������ȫ����
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
	rect.x = 30;//ֱ�ӽ�����������ȥ����9.28����
	rect.width = rect.width - 30;
	im_down = imGray(rect).clone();
	Cut_PreProcess(im_down, im_down1);
	if (m_wordHeigh > 25 || im_down1.cols > PICWIDTH)
	{
		resize(im_up1, im_up1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_up1, im_up1, 127, 255, CV_THRESH_BINARY);//����ʱ�ᵼ�±�Ե����ֵС��255��Ӱ��ʶ��Ч��
		resize(im_up, im_up, Size(PICWIDTH, 25), (0, 0), (0, 0));
		resize(im_down1, im_down1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_down1, im_down1, 127, 255, CV_THRESH_BINARY);
		resize(im_down, im_down, Size(PICWIDTH, 25), (0, 0), (0, 0));
	}
	bool find = false;
	find = Crop_up(im_up1, im_up2, im_up3);//�ж���û���ҵ��ֶμ�ķֽ���
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
	find = Crop_up(im_down1, im_down2, im_down3);//�ж���û���ҵ��ֶμ�ķֽ���
	if (find)
	{
		cv::Mat orig_down1, orig_down2;
		//ֻ������ߵ�
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
	rect.x = 30;//ֱ�ӽ�����������ȥ����9.28����
	rect.width = rect.width - 30;
	im_down = imGray(rect).clone();
	Cut_PreProcess(im_down, im_down1);
	if (m_wordHeigh > 25 || im_down1.cols > PICWIDTH)
	{
		resize(im_up1, im_up1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_up1, im_up1, 127, 255, CV_THRESH_BINARY);//����ʱ�ᵼ�±�Ե����ֵС��255��Ӱ��ʶ��Ч��
		resize(im_up, im_up, Size(PICWIDTH, 25), (0, 0), (0, 0));//��ԭͼ��ȡ�ַ�
		resize(im_down1, im_down1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_down1, im_down1, 127, 255, CV_THRESH_BINARY);
		resize(im_down, im_down, Size(PICWIDTH, 25), (0, 0), (0, 0));//��ԭͼ��ȡ�ַ�
	}
	bool find = false;
	find = Crop_up(im_up1, im_up2, im_up3);//�ж���û���ҵ��ֶμ�ķֽ���
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
	find = Crop_up(im_down1, im_down2, im_down3);//�ж���û���ҵ��ֶμ�ķֽ���
	if (find)
	{
		cv::Mat orig_down1, orig_down2;
		//ֻ������ߵ�
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
	rect.x = 30;//ֱ�ӽ�����������ȥ����9.28����
	rect.width = rect.width - 30;
	im_down = imGray(rect).clone();
	Cut_PreProcess(im_down, im_down1);
	if (m_wordHeigh > 25 || im_down1.cols > PICWIDTH)
	{
		resize(im_up1, im_up1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_up1, im_up1, 127, 255, CV_THRESH_BINARY);//����ʱ�ᵼ�±�Ե����ֵС��255��Ӱ��ʶ��Ч��
		resize(im_up, im_up, Size(PICWIDTH, 25), (0, 0), (0, 0));//��ԭͼ��ȡ�ַ�
		resize(im_down1, im_down1, Size(PICWIDTH, 25), (0, 0), (0, 0));
		threshold(im_down1, im_down1, 127, 255, CV_THRESH_BINARY);
		resize(im_down, im_down, Size(PICWIDTH, 25), (0, 0), (0, 0));//��ԭͼ��ȡ�ַ�
	}
	bool find = false;
	find = Crop_up(im_up1, im_up2, im_up3);//�ж���û���ҵ��ֶμ�ķֽ���
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
	find = Crop_down(im_down1, im_down2, im_down3);//�ж���û���ҵ��ֶμ�ķֽ���
	if (find)
	{
		cv::Mat orig_down1, orig_down2;
		//ֻ������ߵ�
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
	RowCharacter(charNums1, colNum); //�����������
	ColCharacter(charNums1, colNum); //������������
	DivideVector(charNums1); //������������


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
		else if (i >= 8 && i < 16) //ʱ��һ��16���ַ�
		{
			str_time += NumCharIdentify(i, colNum);
		}
		else
		{
			if (numHeight[i] > 12 || numWidth[i] > 13) //�ַ��߶ȴ���12���߿�ȴ���13������Ϊ���֣��ݲ�ʶ��
			//if (numWidth[i] > 13) //�ַ���ȴ���13������Ϊ���֣��ݲ�ʶ��
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
	//����
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
	RowCharacter(charNums1, colNum); //�����������
	ColCharacter(charNums1, colNum); //������������
	DivideVector(charNums1); //������������


	string str_date, str_time;
	for (int i = 0; i < charNums1; i++)
	{
		if (i < 8)
		{
			if (i == 7 && str_date[4] == '.')//���ڿ���ȱλ
			{
				str_time += NumCharIdentify(i, colNum);
			}
			else
			{
				str_date += NumCharIdentify(i, colNum);
			}			
		}
		else if (i >= 8 && i < 16) //ʱ��һ��16���ַ�
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
	RowCharacter(charNums1, colNum); //�����������
	ColCharacter(charNums1, colNum); //������������
	DivideVector(charNums1); //������������


	int flag = 0;
	int flag1 = 0;
	string str_km;
	int tag = 5;
	for (int i = 0; i < charNums1; i++)
	{
		if (numHeight[i] > 12 || numWidth[i] > 13) //�ַ��߶ȴ���12���߿�ȴ���13������Ϊ���֣��ݲ�ʶ��
		//if (numWidth[i] > 13) //�ַ���ȴ���13������Ϊ���֣��ݲ�ʶ��
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
	RowCharacter(charNums2, colNum); //�����������
	ColCharacter(charNums2, colNum); //������������
	DivideVector(charNums2); //������������

	for (int i = 0; i < charNums2; i++)
	{
		if (numHeight[i] > 12 || numWidth[i] > 13) //�ַ��߶ȴ���12���߿�ȴ���13������Ϊ���֣��ݲ�ʶ��
		//if (numWidth[i] > 13) //�ַ���ȴ���13������Ϊ���֣��ݲ�ʶ��
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
*	������:	otsu
*	�����������㷨������ĻҶ�ͼ���ж�ֵ��
*			  ��ֱ��ͼ�ֳ��������֣�ʹ��������֮��ľ�����󡣻��ֵ�����õ����?
*	�β���:  picPtr    //ָ��ͼ�������׵�ַ��ָ��
*			   picRows   //ͼ����������
*			   picCols   //ͼ����������
*	����ֵ����ֵ����ֵ
*   ȫ�ֱ���:  ��
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	     Revised by	  Item Description
*   V1.0	2018/4/12    �Ź⻪      ԭʼ�汾
******************************************************************/
void otsu(unsigned char *picPtr, int startRows, int picHeight, int picCols)
{
	/*	unsigned char *np;  // ͼ��ָ��
	int ihist[256] = {0};   // ͼ��ֱ��ͼ��256����
	int i, j, k;  // various counters
	int n, n1, n2, gmin, gmax, sum, csum;
	double m1, m2, fmax, sb;
	int thresholdValue = 0;


	//	memset(ihist, 0, sizeof(ihist));

	gmin = 255; gmax = 0;

	// ����ֱ��ͼ
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
	sum += k * ihist[k];    // x*f(x)������
	n += ihist[k];       // f(x)����
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
		for (int j = 0; j < picCols; j++)  //ǰ��
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
	int oneNum = 0, upTag = 0, leftRightFlag = 0, upDownFlag = 0; //����´�Ҫɨ��ı߽磬0��ʾ��߽磬1��ʾ�ұ߽�
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
			oneNum += picPtr[i*picCols + j]; //��¼ÿ����255���ص�ĸ���
		}

		if (i == startRows + 10)
		{
			areaStart = 1; //����10�л�δ�ҵ���˵�������Ϸ�������
			break;
		}
		if (oneNum <= 255 * 2) //����2���������
		{
			areaStart = i;  //�޳������Ϸ�����
			break;
		}
	}

	for (i = areaStart; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picCols / 2; j++)
		{
			if (picPtr[i*picCols + j] == 255 && picPtr[i*picCols + j + 1] == 255)
			{
				areaUpBord = i; //���������ϱ߽�
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
			oneNum += picPtr[i*picCols + j]; //��¼ÿ����255���ص�ĸ���
		}

		if (oneNum <= 255 * 4) //����4���������
		{
			areaDownBord = i; //���ַ��±߽��ŵ������� 
			break;
		}
		if (i == startRows + picHeight - 1)
		{
			areaDownBord = startRows + picHeight - 1; //����û�ҵ�ȡĩ��
		}
	}

	//	numHeight = areaDownBord - areaUpBord;

	for (j = 0; j < picCols; j++)
	{
		if (leftRightFlag == 0)  //leftRightFlag=0��ʾҪɨ���ַ�����߽�
		{
			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				if (picPtr[i*picCols + j] == 255 && picPtr[(i + 1)*picCols + j] == 255)
				{
					leftBord = j; //������߽�
					leftRightFlag = 1; //��leftRightFlag��1����ʾ�´�Ҫɨ���ұ߽�
					if (j < picCols - 2)
					{
						j += 2; //�ұ߽�Ӻ�3�п�ʼɨ��
					}
					break;
				} //if (picPtr[i*outlinesize + j] == 255)...end			
			} //for (i = 0; i <= picRows; i++)...end
		}  //if (leftRightFlag == 0)...end


		if (leftRightFlag == 1) //leftRightFlag=1��ʾҪɨ���ַ����ұ߽�
		{
			oneNum = 0;

			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				oneNum += picPtr[i*picCols + j]; //��¼ÿ����255���ص�ĸ���
			}

			if (oneNum <= 0) //�ҵ�ȫ�ڵ�һ�У�֮ǰ����1����������ȡ����
			{
				rightBord = j; //���ַ��ұ߽��ŵ������� 
				leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
				numWidth[k] = rightBord - leftBord;
			}
			else if (j == leftBord + 16) //ɨ��16��(����14��)����δ�ҵ��ַ��ұ߽磬����2���ַ�δ�ֿ�
			{
				rightBord = leftBord + 8; //�ַ����һ��С��12�����䶨Ϊ�ұ߽�
				j = leftBord + 8; //�¸��ַ���߽���ʼ��
				leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
				numWidth[k] = rightBord - leftBord;
			}
			else if (j == picCols - 1) //�ַ��ұ߽�պ��������ұ߽�
			{
				rightBord = j; //���ַ��ұ߽��ŵ������� 
				leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
				numWidth[k] = rightBord - leftBord;
			}

			if (leftRightFlag == 0) //�ҵ��ұ߽磬�������·ָ�
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
							oneNum += picPtr[i*picCols + j]; //��¼ÿ����255���ص�ĸ���
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
	int oneNum = 0, upTag = 0, leftRightFlag = 0, upDownFlag = 0; //����´�Ҫɨ��ı߽磬0��ʾ��߽磬1��ʾ�ұ߽�
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
			oneNum += picPtr[i*picCols + j]; //��¼ÿ����255���ص�ĸ���
		}

		if (i == startRows + 10)
		{
			areaStart = 1; //����10�л�δ�ҵ���˵�������Ϸ�������
			break;
		}
		if (oneNum <= 255 * 2) //����2���������
		{
			areaStart = i;  //�޳������Ϸ�����
			break;
		}
	}

	for (i = areaStart; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picCols / 2; j++)
		{
			if (picPtr[i*picCols + j] == 255 && picPtr[i*picCols + j + 1] == 255)
			{
				areaUpBord = i; //���������ϱ߽�
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
			oneNum += picPtr[i*picCols + j]; //��¼ÿ����255���ص�ĸ���
		}

		if (oneNum <= 255 * 4) //����4���������
		{
			areaDownBord = i; //���ַ��±߽��ŵ������� 
			break;
		}
		if (i == startRows + picHeight - 1)
		{
			areaDownBord = startRows + picHeight - 1; //����û�ҵ�ȡĩ��
		}
	}

	//	numHeight = areaDownBord - areaUpBord;

	for (j = 0; j < picCols; j++)
	{
		if (leftRightFlag == 0)  //leftRightFlag=0��ʾҪɨ���ַ�����߽�
		{
			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				if (picPtr[i*picCols + j] == 255 && picPtr[(i + 1)*picCols + j] == 255)
				{
					leftBord = j; //������߽�
					leftRightFlag = 1; //��leftRightFlag��1����ʾ�´�Ҫɨ���ұ߽�
					if (j < picCols - 2)
					{
						j += 2; //�ұ߽�Ӻ�3�п�ʼɨ��
					}
					break;
				} //if (picPtr[i*outlinesize + j] == 255)...end			
			} //for (i = 0; i <= picRows; i++)...end
		}  //if (leftRightFlag == 0)...end


		if (leftRightFlag == 1) //leftRightFlag=1��ʾҪɨ���ַ����ұ߽�
		{
			oneNum = 0;

			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				oneNum += picPtr[i*picCols + j]; //��¼ÿ����255���ص�ĸ���
			}

			if (oneNum <= 0) //�ҵ�ȫ�ڵ�һ�У�֮ǰ����1����������ȡ����
			{
				rightBord = j; //���ַ��ұ߽��ŵ������� 
				leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
				//numWidth[k] = rightBord - leftBord;
			}
			/*else if (j == leftBord + 16) //ɨ��16��(����14��)����δ�ҵ��ַ��ұ߽磬����2���ַ�δ�ֿ�
			{
				rightBord = leftBord + 10; //�ַ����һ��С��12�����䶨Ϊ�ұ߽�
				j = leftBord + 10; //�¸��ַ���߽���ʼ��
				leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
				//numWidth[k] = rightBord - leftBord;
			}*/
			else if (j == picCols - 1) //�ַ��ұ߽�պ��������ұ߽�
			{
				rightBord = j; //���ַ��ұ߽��ŵ������� 
				leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
				//numWidth[k] = rightBord - leftBord;
			}

			if (leftRightFlag == 0) //�ҵ��ұ߽磬�������·ָ�
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
							oneNum += picPtr[i*picCols + j]; //��¼ÿ����255���ص�ĸ���
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

				int strnum = round((double)(rightBord - leftBord) / 8.0);//�趨һ���ַ���Ϊ10
				//CvRect rect;
				if (numHeight[k]<13 && strnum > 1)//�ַ�������ճ����ǿ�Ʒָ�,�������ַ�����
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
	int oneNum = 0, upTag = 0, leftRightFlag = 0, upDownFlag = 0; //����´�Ҫɨ��ı߽磬0��ʾ��߽磬1��ʾ�ұ߽�
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
			oneNum += picPtr[i*picWidth + j]; //��¼ÿ����255���ص�ĸ���
		}

		if (i == startRows + 10)
		{
			areaStart = 1; //����10�л�δ�ҵ���˵�������Ϸ�������
			break;
		}
		if (oneNum <= 255 * 2) //����2���������
		{
			areaStart = i;  //�޳������Ϸ�����
			break;
		}
	}

	for (i = areaStart; i < startRows + picHeight; i++)
	{
		for (j = 0; j < picWidth / 2; j++)
		{
			if (picPtr[i*picWidth + j] == 255 && picPtr[i*picWidth + j + 1] == 255)
			{
				areaUpBord = i; //���������ϱ߽磨�����ϱ߽�������һ�£�
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
			oneNum += picPtr[i*picWidth + j]; //��¼ÿ����255���ص�ĸ���
		}

		if (oneNum <= 255 * 4) //����4���������
		{
			areaDownBord = i; //���ַ��±߽��ŵ������� 
			break;
		}
		if (i == startRows + picHeight - 1)
		{
			areaDownBord = startRows + picHeight - 1; //����û�ҵ�ȡĩ��
		}
	}

	for (j = 0; j < picWidth; j++)
	{
		if (leftRightFlag == 0)  //leftRightFlag=0��ʾҪɨ���ַ�����߽�
		{
			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				if (picPtr[i*picWidth + j] == 255 && picPtr[(i + 1)*picWidth + j] == 255)//ɨ����
				{
					leftBord = j; //������߽�
					leftRightFlag = 1; //��leftRightFlag��1����ʾ�´�Ҫɨ���ұ߽�
					if (j < picWidth - 2)
					{
						j += 2; //�ұ߽�Ӻ�3�п�ʼɨ��
					}
					break;
				} //if (picPtr[i*outlinesize + j] == 255)...end			
			} //for (i = 0; i <= picRows; i++)...end
		}  //if (leftRightFlag == 0)...end


		if (leftRightFlag == 1) //leftRightFlag=1��ʾҪɨ���ַ����ұ߽�
		{
			oneNum = 0;

			for (i = areaUpBord; i <= areaDownBord; i++)
			{
				oneNum += picPtr[i*picWidth + j]; //��¼ÿ����255���ص�ĸ���
			}

			if (oneNum <= 0) //�ҵ�ȫ�ڵ�һ�У�֮ǰ����1����������ȡ����
			{
				rightBord = j; //���ַ��ұ߽��ŵ������� 
				leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
								   //numWidth[k] = rightBord - leftBord;
			}
			/*else if (j == leftBord + 16) //ɨ��16��(����14��)����δ�ҵ��ַ��ұ߽磬����2���ַ�δ�ֿ�
			{
			rightBord = leftBord + 10; //�ַ����һ��С��12�����䶨Ϊ�ұ߽�
			j = leftBord + 10; //�¸��ַ���߽���ʼ��
			leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
			//numWidth[k] = rightBord - leftBord;
			}*/
			else if (j == picWidth - 1) //�ַ��ұ߽�պ��������ұ߽�
			{
				rightBord = j; //���ַ��ұ߽��ŵ������� 
				leftRightFlag = 0; //��leftRightFlag��0����ʾ�´�Ҫɨ����߽�
			}

			if (leftRightFlag == 0) //�ҵ��ұ߽磬�������·ָ�
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
							oneNum += picPtr[i*picWidth + j]; //��¼ÿ����255���ص�ĸ���
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
				//int charnum = round((double)(rightBord - leftBord) / 7.0);//�趨һ���ַ���Ϊ7�������������
				if (rightBord - leftBord <= 16)
				{
					charnum = round((double)(rightBord - leftBord) / 8.6);//1026���ģ����m��ȫ������
				}
				else
				{
					charnum = round((double)(rightBord - leftBord) / 8.0);//���ܳ��ֶ���ַ�ճ��
				}
																		 
				if (numHeight[k]<13 && numHeight[k] > 5 && charnum >= 1)//�ַ�������ճ����ǿ�Ʒָ�,�������ַ�����
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
						//cv::imwrite("F:/workspace/videos/tmp.jpg", char_img);//�����ַ�ͼ��
						leftBord = rightBordTmp;//10.25�Ĳ���1
						k++;
					}
					j = rightBord;
				}
				else if(charnum<1)//�����ǵ�.��:
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
					//cv::imwrite("F:/workspace/videos/tmp.jpg", char_img);//�����ַ�ͼ��
					k++;
				}
			}
		} //if (leftRightFlag == 1)...end
	} //for (j=0;j<picCols; j++)...end	
}

float colLineCharacter[30][2]; //�洢�ַ���������
float rowLineCharacter[30][2]; //�洢�ַ���������
int charColVector[30]; //�洢ǰ4���ַ���������������
int charRowVector[30]; //�洢ǰ4���ַ��ĺ�����������
					   /*****************************************************************
					   *	������:	ColCharacter
					   *	��������: ��2X1�Ŀ��¼��������ֵ
					   *	��ʽ����: ��
					   *	����ֵ����
					   *   ȫ�ֱ�����
					   *   �ļ���̬��������
					   *   ������̬��������
					   *------------------------------------------------------------------
					   *	Revision History
					   *	No.	    Date	     Revised by	  Item Description
					   *   V1.0	2018/04/17    ZhangGH	  	ԭʼ�汾
					   ******************************************************************/
void ColCharacter(int charNums, int picCols)
{
	int k, i;
	int continueZeroNum = 0; //��¼����0���ص�ĸ���
	int maxContinueZeroNum = 0; //��¼��������0���ص�ĸ���

	memset(colLineCharacter[0], 0, sizeof(colLineCharacter));

	for (k = 0; k < charNums; k++)
	{

		//������
		continueZeroNum = 0;
		maxContinueZeroNum = 0;
		for (i = 0; i < numHeight[k]; i++)
		{
			if (numData[k][i*picCols] == 255 || numData[k][i*picCols + 1] == 255)  //2X1�Ĵ���
			{
				continueZeroNum++; //��������2?��ص�continueZeroNum+1
				if (continueZeroNum > maxContinueZeroNum)
				{
					maxContinueZeroNum = continueZeroNum; //��¼��������0���ص����
				}
			}
			else
			{
				continueZeroNum = 0; //���0���ص�����Ǹ����ص㲻Ϊ0��������0���ص�������?
			}
		}
		//¼��������ֵ
		colLineCharacter[k][0] = maxContinueZeroNum / (float)numHeight[k]; //��¼���������ٷֱ�

		continueZeroNum = 0;
		maxContinueZeroNum = 0;
		for (i = 0; i < numHeight[k]; i++)
		{
			if (numData[k][i*picCols + numWidth[k] - 2] == 255 || numData[k][i*picCols + numWidth[k] - 1] == 255)   //2X1�Ĵ���
			{
				continueZeroNum++; //������?��0���ص㣬continueZeroNum+1
				if (continueZeroNum > maxContinueZeroNum)
				{
					maxContinueZeroNum = continueZeroNum; //�Ǽ�������?���ص����
				}
			}
			else
			{
				continueZeroNum = 0; //���0���ص�����Ǹ����ص㲻Ϊ0��������0���ص������
			}
		}
		//��¼��������ֵ
		colLineCharacter[k][1] = maxContinueZeroNum / (float)numHeight[k]; //��¼���������ٷֱ�
	}
}

/*****************************************************************
*	������:	RowCharacter
*	��������: ��2X1�Ŀ��¼��������ֵ
*	��ʽ����: ��
*	����ֵ����
*   ȫ�ֱ�����g_charBreadth[CHAR_NUM]��g_charHeight[CHAR_NUM]
g_charSection[CHAR_NUM][CHAR_ROWSIZE][CHAR_COLSIZE]
g_perRowLineCharacter[CHAR_NUM][CHAR_ROWSIZE]
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	     Revised by	  Item Description
*   V1.0	2018/04/17     ZhangGH	   		ԭʼ�汾
******************************************************************/
void RowCharacter(int charNums, int picCols)
{
	int k, j;
	int continueZeroNum = 0; //��¼����0���ص�ĸ���
	int maxContinueZeroNum = 0; //��¼��������0���ص�ĸ���
								//	Mat rowTest(20, picCols, CV_8U);

	memset(rowLineCharacter[0], 0, sizeof(rowLineCharacter));

	//�Ϻ���
	for (k = 0; k < charNums; k++) //ֻ����ǰ��4���ַ��ĺ���������
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
			if (numData[k][0 * picCols + j] == 255 || numData[k][1 * picCols + j] == 255)  //2X1�Ĵ���
			{
				continueZeroNum++; //��������2��0���ص㣬continueZeroNum��1
				if (continueZeroNum > maxContinueZeroNum)
				{
					maxContinueZeroNum = continueZeroNum; //��¼��������0���ص����
				}
			}
			else
			{
				continueZeroNum = 0; //���0���ص�����Ǹ����ص㲻Ϊ0��������0���ص��������
			}
		}
		//��¼��������ֵ
		rowLineCharacter[k][0] = maxContinueZeroNum / (float)numWidth[k]; //��¼���������ٷֱ�

		continueZeroNum = 0;
		maxContinueZeroNum = 0;
		for (j = 0; j < numWidth[k]; j++)
		{
			if (numData[k][(numHeight[k] - 2) * picCols + j] == 255 || numData[k][(numHeight[k] - 1) * picCols + j] == 255) //2X1�Ĵ���
			{
				continueZeroNum++; //��������2��0���ص㣬continueZeroNum��1

				if (continueZeroNum > maxContinueZeroNum)
				{
					maxContinueZeroNum = continueZeroNum; //��¼��������0���ص����
				}
			}
			else
			{
				continueZeroNum = 0; //���0���ص�����Ǹ����ص㲻Ϊ0��������0���ص��������
			}
		}
		//��¼��������ֵ
		rowLineCharacter[k][1] = maxContinueZeroNum / (float)numWidth[k]; //��¼���������ٷֱ�
	}
}



/*****************************************************************
*	������:	DivideVector
*	��������: �����ַ���
*	��ʽ����: ��
*	����ֵ����
*   ȫ�ֱ�����charRowVector[CHAR_NUM]
charColVector[CHAR_NUM]
g_perRowLineCharacter[CHAR_NUM][CHAR_ROWSIZE]
g_perColLineCharacter[CHAR_NUM][CHAR_ROWSIZE]
g_leftRightBord[BORD_NUM]��g_upDownBord[BORD_NUM]
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	     Revised by	  Item Description
*   V1.0	2018/04/17     ZhangGH	       ԭʼ�汾
******************************************************************/
void  DivideVector(int charNums)

{
	for (int k = 0; k < charNums; k++)
	{
		if (colLineCharacter[k][1] >= 0.65) //���ַ������������ٷֱȴ��ڵ���0.65����ʾ���ַ���������  
		{
			if (colLineCharacter[k][0] >= 0.65) //���ַ������������ٷֱȴ��ڵ���0.65����ʾ���ַ���������  
			{
				charColVector[k] = 4; //���ַ����������ߣ�Ҳ�������ߣ������ַ���������������4
			}
			else
			{
				charColVector[k] = 2; //���ַ��������ߣ��������ߣ������ַ���������������2
			}
		}
		else
		{
			if (colLineCharacter[k][0] >= 0.65)
			{
				charColVector[k] = 1; //���ַ��������ߣ��������ߣ������ַ���������������1
			}
			else
			{
				charColVector[k] = 0; //����ַ�û���κ����������������ַ���������������0
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

	connectDomainNum = ConnectDomain(k, picCols);  //��ȡ��ͨ��
	if (numWidth[k] < 6 && connectDomainNum == 1) //���С��7,������1�͵�
	{
		dataNums = DomainScan(k, 0, numHeight[k], 0, numWidth[k], picCols); //ͳ��ǰ�����ص������ʶ��1�͵�
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
				if (numData[k][i*picCols + 1] == 255 || numData[k][i*picCols + 2] == 255)  //2X1�Ĵ���
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
			if (maxContinueZeroNum >= numHeight[k] - 1) //������������Ϊ1
			{
				flag = PointCoordUD(k, 2, 3, picCols, 4);
				//ÿ�д�������ɨ��,�������ǰ�����ص������������ڵ���4����Ϊk
				if (flag == 1)
				{
					return 'k'; //�����Ƿ���ʶ��km�żӽ��� һ��ֻʶ������
				}
				return '1';  //������Ӱ�죬1���ż������
			}

			//ÿ�д�������ɨ��,�������ǰ�����ص����������С�ڵ���4����Ϊ7
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

			//��2-4����,ÿ�д�������ɨ��,�������ǰ�����ص�����������С�ڵ���numWidth[k]-4����Ϊ5
			flag = PointCoordRL(k, 2, 3, picCols, numWidth[k] - 4);
			if (flag == 1)
			{
				return '5';
			}

			//�ڵ���3-5����,ÿ�д�������ɨ��,�������ǰ�����ص�����������С�ڵ���numWidth[k]-3,��Ϊ2
			flag = PointCoordRL(k, numHeight[k] - 5, numHeight[k] - 3, picCols, numWidth[k] - 3);
			return '2'*(flag == 1) + '3'*(flag == 0);

		}
		if (connectDomainNum == 2)
		{
			int continueZeroNum = 0, maxContinueZeroNum = 0;

			if (g_flagNum[0] > 12)
			{
				return '0';  //�������ص����12��,��϶���0
			}

			points = PointScanMidCol(k, picCols);
			if (points > 1) //�м伸�н���Ϊ3����������4,����Ϊ69
			{
				for (i = 0; i < numHeight[k] + 2; i++)
				{
					for (j = 0; j < numWidth[k] + 2; j++)
					{
						if (g_connectBuf[i][j] == 3)
						{
							if (i <= 4) //�жϻ�������λ��ʶ��6��9
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
			if (numData[k][i*picCols + 1] == 255 || numData[k][i*picCols + 2] == 255)  //2X1�Ĵ���
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
			if (maxContinueZeroNum <= 6) //�ж���������
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
}  //����ʶ����� 


   /*****************************************************************
   *	������:	MixIdentify
   *	��������: ʶ����ĸ������
   *	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
   *	����ֵ��
   *   ȫ�ֱ�����
   *  �ļ���̬��������
   *  ������̬��������
   *------------------------------------------------------------------
   *	Revision History
   *   No.	     Date	      Revised by	  Item Description
   *	V1.0   	2018/04/17	    ZhangGH 		 ԭʼ�汾
   ******************************************************************/
char MixIdentify(int k, int picCols)
{
	int pointNums = 0; // �������
	int oneNum1 = 0, oneNum2 = 0;
	int flag; //���������ʶλ
	int connectDomainNum; //��ͨ������
	Mat numTest8(numHeight[k], picCols, CV_8U);

	for (int i = 0; i < numHeight[k]; i++)
	{
		for (int j = 0; j < numWidth[k]; j++)
		{
			numTest8.data[i*picCols + j] = numData[k][i*picCols + j];
		}
	}
	connectDomainNum = ConnectDomain(k, picCols);
	if (numWidth[k] < 6 && connectDomainNum == 1) //�ַ����С��6,����ʶ��Ϊ1��I
	{
		oneNum1 = DomainScan(k, 1, 4, 0, numWidth[k], picCols); //�����1-4��ǰ�����ص����
		oneNum2 = DomainScan(k, numHeight[k] - 5, numHeight[k] - 2, 0, numWidth[k], picCols); //���㵹��2-5��ǰ�����ص����
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

		if (charColVector[k] == 0 && charRowVector[k] == 0)  //����������Ϊ00,����ʶ��ΪA,V,X,Y,4
		{
			if (connectDomainNum == 1)  //����ͨ�����1,�������V,X,Y
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //ɨ��ĩ3�н�����
				if (pointNums > 1)  //ĩ3���н���Ϊ2����������1����ΪX
				{
					return 'X';
				}

				flag = PointCoordUD(k, numWidth[k] / 2 - 1, numWidth[k] / 2, picCols, 3);
				return  'V'*(flag == 1) + 'Y'*(flag != 1);
			}
			else if (connectDomainNum == 2)
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //ɨ��ĩ3�н�����
				if (pointNums > 1)  //ĩ3���н���Ϊ2����������1����ΪA
				{
					return 'A';
				}
				return '4';
			}
			else
			{
				return '8';  //��������8����������������00�������ǿ��ǵ�����Ӱ�죬��ֻ��8��3����ͨ�򣬷���ʶ��ӽ����� 
			}

		}

		if (charColVector[k] == 0 && charRowVector[k] == 1)  //����������Ϊ01,��ʶ��ΪT,J��7
		{
			if (connectDomainNum == 1)
			{
				//�ڵ�2-3����,ÿ�д�������ɨ��,�������ǰ�����ص�������������ڵ���4,��Ϊ7��T
				flag = PointCoordLR(k, 2, 3, picCols, 4);
				if (flag == 1)
				{
					pointNums = PointScanCol(k, 2, 4, picCols, 2); //ɨ��ǰ3�н�����
					return '7'*(pointNums > 1) + 'T'*(pointNums <= 1);
				}
				pointNums = PointScanMidCol(k, picCols);
				if (pointNums > 1)
				{
					return '5';
				}
				pointNums = PointScanCol(k, 0, 3, picCols, 2); //ɨ��ǰ3�н�����
				if (pointNums > 1)  //ǰ3���н���Ϊ2����������1����ΪJ������ΪT
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

		if (charColVector[k] == 0 && charRowVector[k] == 2)  //����������Ϊ02,��ʶ��ΪA��1��2��8
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

		if (charColVector[k] == 0 && charRowVector[k] == 4)  //����������Ϊ04,�����ʶ��ΪA,S,Z,X,J,2,3,5
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanMidCol(k, picCols); //ɨ���м伸�н���Ϊ3������
				if (pointNums > 1)
				{
					//�ڵ�����3-5����,ÿ�д�������ɨ��,�������ǰ�����ص�����������С�ڵ���numWidth[k] - 3,��Ϊ2Z
					flag = PointCoordRL(k, numHeight[k] - 4, numHeight[k] - 2, picCols, numWidth[k] - 4);
					if (flag == 1)
					{
						return Identify2Z(k, picCols);
					}
					return Identify5S(k, picCols);
				}
				pointNums = PointScanRow(k, 0, 3, picCols, 2); //ɨ��ǰ3�н�����
				if (pointNums > 1)  //ĩ3���н���Ϊ2����������1����ΪX������ΪJ
				{
					return 'X';
				}

				pointNums = PointScanCol(k, 0, 3, picCols, 2); //ɨ��ǰ3�н�����
				if (pointNums > 1)  //ǰ3���н���Ϊ2����������1����ΪJ������ΪT
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

		if (charColVector[k] == 1 && charRowVector[k] == 0)  //����������Ϊ10,��ʶ��ΪK,
		{
			if (connectDomainNum == 1)
			{
				return 'K';
			}
			else
			{
				return Identify8B(k, picCols);//�����쳣������
			}
		}

		if (charColVector[k] == 1 && charRowVector[k] == 1)  //����������Ϊ11,�����ʶ��ΪP,R,F
		{
			if (connectDomainNum == 1)
			{
				return 'F';
			}
			else if (connectDomainNum == 2)
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //ɨ��ĩ3�н�����
				if (pointNums > 1)  //ĩ3���н���Ϊ2����������1����ΪR,����ΪP
				{
					return 'R';
				}
				return 'P';
			}
			else
			{
				return Identify8B(k, picCols);//�����쳣������	
			}

		}

		if (charColVector[k] == 1 && charRowVector[k] == 2)  //����������Ϊ12,����ʶ��ΪL,6
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
				return Identify8B(k, picCols);//�����쳣������	
			}
		}

		if (charColVector[k] == 1 && charRowVector[k] == 4)  //����������Ϊ14,�����ʶ��ΪB,C,E,G,O,Q,6
		{
			if (connectDomainNum == 1)
			{
				//�ں�3����ÿ�д�������ɨ��,�������ǰ�����ص���������С�ڵ���numWidth[k] - 4,�����ΪC��E
				flag = PointCoordRL(k, numHeight[k] - 4, numHeight[k] - 2, picCols, numWidth[k] - 4);
				if (flag == 1)
				{
					pointNums = PointScanCol(k, 2, 4, picCols, 3); //ɨ��2-4�н�����
					if (pointNums > 1)  //3���н���Ϊ3����������1����ΪE������ΪC
					{
						return 'E';
					}
					return 'C';
				}
				pointNums = PointScanCol(k, numWidth[k] - 4, numWidth[k] - 2, picCols, 3); //ɨ���3�н�����
				if (pointNums > 1)  //��3���н���Ϊ3����������1����ΪG������ΪC
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
							if (i > 4) //�жϻ�������λ��ʶ��6
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
				return Identify8B(k, picCols);//��ͨ��Ϊ3����Ϊ8��B
			}
		}

		if (charColVector[k] == 2 && charRowVector[k] == 0)  //����������Ϊ20,��ʶ��Ϊ4
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
				return Identify8B(k, picCols);//�����쳣������	
			}
		}

		if (charColVector[k] == 2 && (charRowVector[k] == 1 || charRowVector[k] == 2 || charRowVector[k] == 4))
		{   //����������Ϊ21��22��24,��ʶ��Ϊ3��9
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
				return Identify8B(k, picCols);//�����쳣������	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 0)  //����������Ϊ40,�����ʶ��ΪH,M,N,W
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanRow(k, 0, 3, picCols, 3); //ɨ��ǰ3�н�����
				if (pointNums > 1)
				{
					return 'W';  //ǰ3���н���Ϊ2����������1����ΪR,����ΪP
				}
				//�ڵ�2-3����,ÿ�д�������ɨ��,�������ǰ�����ص������������ڵ���3,��ΪH
				flag = PointCoordUD(k, 2, 3, picCols, 3);
				if (flag == 1)
				{
					return 'H';

				}
				//�ڵ�����2-3����,ÿ�д�������ɨ��,�������ǰ�����ص������������ڵ���4,��ΪN,����ΪM
				flag = PointCoordUD(k, numWidth[k] - 4, numWidth[k] - 3, picCols, 4);
				return 'N' * (flag == 1) + 'M' * (flag != 1);
			}
			else if (connectDomainNum == 2)
			{
				return 'R'; //��ͨ��Ϊ2����ΪR
			}
			else
			{
				return Identify8B(k, picCols);//�����쳣������	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 1)  //����������Ϊ41,��ʶ��ΪR
		{
			if (connectDomainNum == 2)
			{
				return 'R'; //��ͨ��Ϊ2����ΪR
			}
			else
			{
				return Identify8B(k, picCols); //�����쳣������	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 2)  //����������Ϊ42,��ʶ��ΪU
		{
			if (connectDomainNum == 1)
			{
				return 'U'; //��ͨ��Ϊ2����ΪR
			}
			else
			{
				return Identify8B(k, picCols);//�����쳣������	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 4)  //����������Ϊ44,�����ʶ��ΪB,D,O,Q
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

		return '?';  //��ĸʶ�����,���޷�ʶ�𣬷���'��'   
	}  //"if (numWidth[k] <= SHORTCHAR)...else..." ����
}


/*****************************************************************
*	������:	LetterIdentify
*	��������: ʶ����ĸ
*	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
*	����ֵ��
*   ȫ�ֱ�����
*  �ļ���̬��������
*  ������̬��������
*------------------------------------------------------------------
*	Revision History
*   No.	     Date	      Revised by	  Item Description
*	V1.0   	2018/04/17	    ZhangGH 		 ԭʼ�汾
******************************************************************/

char LetterIdentify(int k, int picCols)
{
	int pointNums = 0; // �������
	int flag; //���������ʶλ
	int connectDomainNum; //��ͨ������

	connectDomainNum = ConnectDomain(k, picCols);
	if (numWidth[k] < 5 && connectDomainNum == 1) //�ַ����С�ڵ���SHORTCHAR��ʾ�Ƕ��ַ�
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

		if (charColVector[k] == 0 && charRowVector[k] == 0)  //����������Ϊ00,����ʶ��ΪA,V,X,Y
		{
			if (connectDomainNum == 1)  //����ͨ�����1,�������V,X,Y
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //ɨ��ĩ3�н�����
				if (pointNums > 1)  //ĩ3���н���Ϊ2����������1����ΪX
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
				return 'B';//�����쳣������	
			}

		}

		if (charColVector[k] == 0 && charRowVector[k] == 1)  //����������Ϊ01,��ʶ��ΪT,J��A
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanCol(k, 0, 3, picCols, 2); //ɨ��ǰ3�н�����
				if (pointNums > 1)  //ǰ3���н���Ϊ2����������1����ΪJ������ΪT
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
				return 'B';//�����쳣������	
			}
		}

		if (charColVector[k] == 0 && charRowVector[k] == 2)  //����������Ϊ02,��ʶ��ΪA
		{
			if (connectDomainNum == 2)
			{
				return 'A';
			}
			else
			{
				return 'B';//�����쳣������	
			}
		}

		if (charColVector[k] == 0 && charRowVector[k] == 4)  //����������Ϊ04,�����ʶ��ΪA,S,Z,X,J,2,3,5
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanMidCol(k, picCols); //ɨ���м伸�н���Ϊ3������
				if (pointNums > 1)
				{
					//�ڵ�����3-5����,ÿ�д�������ɨ��,�������ǰ�����ص�����������С�ڵ���numWidth[k] - 3,��Ϊ2Z
					flag = PointCoordRL(k, numHeight[k] - 4, numHeight[k] - 2, picCols, numWidth[k] - 3);
					return  'Z'*(flag == 1) + 'S'*(flag != 1);
				}
				pointNums = PointScanRow(k, 0, 3, picCols, 2); //ɨ��ǰ3�н�����
				if (pointNums > 1)  //ĩ3���н���Ϊ2����������1����ΪX������ΪJ
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
				return 'B';//�����쳣������	
			}

		}

		if (charColVector[k] == 1 && charRowVector[k] == 0)  //����������Ϊ10,��ʶ��ΪK,
		{
			if (connectDomainNum == 1)
			{
				return 'K';
			}
			else
			{
				return 'B';//�����쳣������	
			}
		}

		if (charColVector[k] == 1 && charRowVector[k] == 1)  //����������Ϊ11,�����ʶ��ΪP,R,F
		{
			if (connectDomainNum == 1)
			{
				return 'F';
			}
			else if (connectDomainNum == 2)
			{
				pointNums = PointScanRow(k, numHeight[k] - 3, numHeight[k], picCols, 2); //ɨ��ĩ3�н�����
				if (pointNums > 1)  //ĩ3���н���Ϊ2����������1����ΪR,����ΪP
				{
					return 'R';
				}
				return 'P';
			}
			else
			{
				return 'B';//�����쳣������	
			}

		}

		if (charColVector[k] == 1 && charRowVector[k] == 2)  //����������Ϊ12,����ʶ��ΪL
		{
			if (connectDomainNum == 1)
			{
				return 'L';
			}
			else
			{
				return 'B';//�����쳣������	
			}
		}

		if (charColVector[k] == 1 && charRowVector[k] == 4)  //����������Ϊ14,�����ʶ��ΪB,C,E,G,O,Q,6
		{
			if (connectDomainNum == 1)
			{
				//�ں�3����ÿ�д�������ɨ��,�������ǰ�����ص���������С�ڵ���numWidth[k] - 4,�����ΪC��E
				flag = PointCoordRL(k, numHeight[k] - 4, numHeight[k] - 2, picCols, numWidth[k] - 4);
				if (flag == 1)
				{
					pointNums = PointScanCol(k, 2, 4, picCols, 3); //ɨ��2-4�н�����
					if (pointNums > 1)  //3���н���Ϊ3����������1����ΪE������ΪC
					{
						return 'E';
					}
					return 'C';
				}
				pointNums = PointScanCol(k, numWidth[k] - 4, numWidth[k] - 2, picCols, 3); //ɨ���3�н�����
				if (pointNums > 1)  //��3���н���Ϊ3����������1����ΪG������ΪC
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

		if (charColVector[k] == 4 && charRowVector[k] == 0)  //����������Ϊ40,�����ʶ��ΪH,M,N,W��R
		{
			if (connectDomainNum == 1)
			{
				pointNums = PointScanRow(k, 0, 3, picCols, 3); //ɨ��ǰ3�н�����
				if (pointNums > 1)
				{
					return 'W';  //ǰ3���н���Ϊ2����������1����ΪR,����ΪP
				}
				//�ڵ�2-3����,ÿ�д�������ɨ��,�������ǰ�����ص������������ڵ���3,��ΪH
				flag = PointCoordUD(k, 2, 3, picCols, 3);
				if (flag == 1)
				{
					return 'H';

				}
				//�ڵ�����2-3����,ÿ�д�������ɨ��,�������ǰ�����ص������������ڵ���4,��ΪN,����ΪM
				flag = PointCoordUD(k, numWidth[k] - 4, numWidth[k] - 3, picCols, 4);
				return 'N' * (flag == 1) + 'M' * (flag != 1);
			}
			else if (connectDomainNum == 2)
			{
				return 'R';
			}
			else
			{
				return 'B';//�����쳣������	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 1)  //����������Ϊ41,��ʶ��ΪR
		{
			if (connectDomainNum == 2)
			{
				return 'R';
			}
			else
			{
				return 'B';//�����쳣������	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 2)  //����������Ϊ42,��ʶ��ΪU
		{
			if (connectDomainNum == 1)
			{
				return 'U';
			}
			else
			{
				return 'B';//�����쳣������	
			}
		}

		if (charColVector[k] == 4 && charRowVector[k] == 4)  //����������Ϊ44,�����ʶ��ΪB,D,O,Q
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

		return '?';  //��ĸʶ�����,���޷�ʶ�𣬷���'��'   
	}  //"if (numWidth[k] <= SHORTCHAR)...else..." ����
}


/*****************************************************************
*	������:	IdentifyDOQ
*	��������: ʶ����ĸD��O��Q
*	��ʽ����: int k //��ʾ�ڼ����ַ�
*	����ֵ��D��O��Q
*   ȫ�ֱ���:
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             ԭʼ�汾
******************************************************************/
char IdentifyDOQ(int k, int picCols)
{
	int leftUpNum = 0, leftDownNum = 0;

	if (colLineCharacter[k][0] == 1)
	{
		return 'D'; //��1��������������,��ΪD  
	}

	leftUpNum = DomainScan(k, 0, 2, 0, 2, picCols);
	leftDownNum = DomainScan(k, numHeight[k] - 2, numHeight[k], 0, 2, picCols);
	if (leftUpNum + leftDownNum >= 6)
	{
		return 'D';  //���ϣ�������ǰ�����ص���ΪD������ΪO��Q
	}

	return IdentifyOQ(k, picCols);
}

/*****************************************************************
*	������:	IdentifyDQ0
*	��������: ʶ����ĸD��Q������0
*	��ʽ����: int k //��ʾ�ڼ����ַ�
*	����ֵ��D��Q��0
*   ȫ�ֱ���:
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             ԭʼ�汾
******************************************************************/
char IdentifyDQ0(int k, int picCols)
{
	int leftUpNum = 0, leftDownNum = 0;

	if (colLineCharacter[k][0] == 1)
	{
		return 'D'; //��1��������������,��ΪD  
	}

	leftUpNum = DomainScan(k, 0, 2, 0, 2, picCols);
	leftDownNum = DomainScan(k, numHeight[k] - 2, numHeight[k], 0, 2, picCols);
	if (leftUpNum + leftDownNum >= 6)
	{
		return 'D';  //���ϣ�������ǰ�����ص���ΪD������ΪO��Q
	}

	if (IdentifyOQ(k, picCols) == 'O')
	{
		return '0'; //��������0����ĸO�޷����֣�����ͳһ��Ϊ����0
	}
	else
	{
		return 'Q';
	}
}

/*****************************************************************
*	������:	IdentifyOQ
*	��������: ʶ����ĸO��Q
*	��ʽ����: int k //��ʾ�ڼ����ַ�
*	����ֵ��O��Q
*   ȫ�ֱ���:
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             ԭʼ�汾
******************************************************************/
char IdentifyOQ(int k, int picCols)
{
	int i, j;
	int yRecord = 0, xRecord = 0;  //��������
	int goTimes = 0; //���ߴ���
	int maxSum = 0;

	//�����½�2X2�Ŀ��һ���
	for (j = numWidth[k] - 1; j >= numWidth[k] - 2; j--)
	{
		for (i = numHeight[k] - 1; i >= numHeight[k] - 2; i--)
		{
			if (numData[k][i*picCols + j] == 255)  //���ǰ׵����������	
			{
				if (i + j > maxSum)
				{
					maxSum = i + j;	 //���㵱ǰ����ײ��ڵ�������
					yRecord = i;  //��¼����
					xRecord = j;
					break;
				}
			}  // "if (g_charSection[k][i][j] == 0)"...end
		}  // "for (j..)"...end	
	}  // "for (i..)"...end	

	if (xRecord == 0)  //������2X2�Ŀ���û�ҵ��ڵ㣬˵��û"β��"����ΪO
	{
		return 'O';
	}

	//�б����½�б�����ߴ���
	while (numData[k][(yRecord - 1)*picCols + xRecord - 1] == 255 && yRecord > 1 && xRecord > 1)
	{
		goTimes++;  //���ߴ���++	

		yRecord--;
		xRecord--;
	}


	if (goTimes > 2)
	{
		return 'Q';  //б�����ߴ�������2��ΪQ������ΪO
	}
	else
	{
		return 'O';
	}
}


/*****************************************************************
*	������:	Identify2Z
*	��������: ʶ����ĸ2��Z
*	��ʽ����: int k //��ʾ�ڼ����ַ�
*	����ֵ��2��Z
*   ȫ�ֱ���:
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             ԭʼ�汾
******************************************************************/
char Identify2Z(int k, int picCols)
{
	if (numData[k][0 * picCols + 0] == 0 && numData[k][0 * picCols + numWidth[k] - 1] == 0)
	{
		return '2'; //��1�е�1�����ص��Լ���1�����1�����ص�Ϊ����,����Ϊ2
	}

	return 'Z';
}

/*****************************************************************
*	������:	Identify5S
*	��������: ʶ����ĸ5��S
*	��ʽ����: int k //��ʾ�ڼ����ַ�
*	����ֵ��5��S
*   ȫ�ֱ���:
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             ԭʼ�汾
******************************************************************/
char Identify5S(int k, int picCols)
{
	int flag = 0;
	//�ڵ�2-3����,ÿ�д�������ɨ��,�������ǰ�����ص�����������С�ڵ���numWidth[k] - 4,��Ϊ5S
	flag = PointCoordRL(k, 2, 3, picCols, numWidth[k] - 4);
	if (flag == 1)
	{
		if (numData[k][0 * picCols + 0] == 0 && numData[k][0 * picCols + numWidth[k] - 1] == 0)
		{
			return 'S'; //��1�е�1�����ص��Լ���1�����1�����ص�Ϊ����,����ΪS������Ϊ5
		}
		return '5';
	}

	return '3';
}


/*****************************************************************
*	������:	Identify8B
*	��������: ʶ����ĸ8��B
*	��ʽ����: int k //��ʾ�ڼ����ַ�
*	����ֵ��8��B
*   ȫ�ֱ���:
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	      Date	      Revised by	    Item Description
*	V1.0    2018/04/18     ZhangGH             ԭʼ�汾
******************************************************************/
char Identify8B(int k, int picCols)
{
	if (numData[k][0 * picCols + 0] == 0 && numData[k][(numHeight[k] - 1)* picCols + 0] == 0)
	{
		return '8'; //��1�е�1�����ص��Լ���1�����1�����ص�Ϊ����,����Ϊ8������ΪB
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
		for (j = 0; j < numWidth[k] + 2; j++)  //���ַ���Χ��һȦ������ص㣬������ͨ��?
		{
			if (i == 0 || j == 0 || i == numHeight[k] + 1 || j == numWidth[k] + 1)
			{
				g_connectBuf[i][j] = 0; //��ÿ���ַ����ܼ�һȦ0
			}

			if (i < numHeight[k] && j < numWidth[k])
			{
				g_connectBuf[i + 1][j + 1] = numData[k][i*picCols + j]; //ԭ���ص��������ƶ�
			}

		}
	}

	/*
	for (i = 0; i < numHeight[k] + 2; i++)
	{
	for (j = 0; j < numWidth[k] + 2; j++)  //���ַ���Χ��һȦ������ص㣬������ͨ��?
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
				g_connectBuf[i][j] = flag; //��flag������ص�,����������
				n++;
				Dp[dom].row = i;
				Dp[dom].col = j;
				dom++;


				while (dom != 0) //�ж�ջ�Ƿ�Ϊ��
				{
					xi = Dp[dom - 1].row;
					xj = Dp[dom - 1].col;
					dom--;
					//�������ص���Χ���ص�,����ǳ����ص�,����������ջ,֮���ټ������,������ջѭ��
					if (xi == 0) //��һ��
					{
						if (xj == 0)  //�ǵ�һ���׸����ص�ֻ���ж��·����ҷ����ص?
						{
							Check(xi + 1, xj, flag, Dp); //�������ص��ҷ����ص�,����Ǳ�����ص㣬�����������?
							Check(xi, xj + 1, flag, Dp); //�������ص��·����ص�,����Ǳ�����ص㣬�����������?
						}
						else
							if (xj == numWidth[k] + 1)  //���ǵ�һ����ĩ�����ص�ֻ���ж��·��������ص?
							{
								Check(xi + 1, xj, flag, Dp);
								Check(xi, xj - 1, flag, Dp);
							}
							else //����ǵ�һ�����������ص����ж��¡�����������ص?
							{
								Check(xi + 1, xj, flag, Dp);
								Check(xi, xj - 1, flag, Dp);
								Check(xi, xj + 1, flag, Dp);
							}
					}

					if (xi == numHeight[k] + 1) //���һ��
					{
						if (xj == 0)  //�ϣ���
						{
							Check(xi - 1, xj, flag, Dp);
							Check(xi, xj + 1, flag, Dp);
						}
						else
							if (xj == numWidth[k] + 1)  //�ϡ���
							{
								Check(xi - 1, xj, flag, Dp);
								Check(xi, xj - 1, flag, Dp);
							}
							else //�Ϣ���?
							{
								Check(xi - 1, xj, flag, Dp);
								Check(xi, xj - 1, flag, Dp);
								Check(xi, xj + 1, flag, Dp);
							}
					}

					if (xj == 0 && xi != 0 && xi != numHeight[k] + 1) //����?�ҡ��ϡ���
					{
						Check(xi, xj + 1, flag, Dp);
						Check(xi - 1, xj, flag, Dp);
						Check(xi + 1, xj, flag, Dp);
					}

					if (xj == numWidth[k] + 1 && xi != 0 && xi != numHeight[k] + 1) //������ ���ϡ��?
					{
						Check(xi, xj - 1, flag, Dp);
						Check(xi - 1, xj, flag, Dp);
						Check(xi + 1, xj, flag, Dp);
					}

					if (xi != 0 && xi != numHeight[k] + 1 && xj != 0 && xj != numWidth[k] + 1) //м?�ϡ��¡�����
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
					if (g_flagNum[0] > 1) //����1�����ص��������,������ͨ��
					{
						flag++;  //ѭ��һ�κ�,flag��1,����ͨ���1
					}
					else   //�������ֻ��1�����ص�,��ʾ������,������ͨ��
					{
						n--;

						numData[k][(i - 1)*picCols + j - 1] = 255; //����ֻ��1�����ص�,���
						g_flagNum[0] = 1;
					}
				}
				if (n == 3)
				{
					if (g_flagNum[1] > 1)
					{
						flag++; //ѭ��һ�κ�,flag��1,���ͨ��?
					}
					else
					{
						n--;

						numData[k][(i - 1)*picCols + j - 1] = 255; //����ֻ��1�����ص�,���
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

	connectDomainNum = flag - 2; //�����ͨ���ֵΪflag -2


								 /*		for (i = 0; i < numHeight[k] + 2; i++)
								 {
								 for (j = 0; j < numWidth[k] + 2; j++)  //���ַ���Χ��һȦ������ص㣬������ͨ��?
								 {
								 numTest.data[i*(picCols + 2) + j] = g_connectBuf[i][j];
								 }
								 }*/

	return connectDomainNum;
} // " ConnectDomain( int k )..end "


  //�жϺ���
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
*	������:	PointScanMidCol
*	��������: ����2-6���н���������3������
*	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
*	����ֵ������������3������
*   ȫ�ֱ����� ��
*   �ļ���̬��������
*   ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/16     ZhangGH            ԭʼ�汾
******************************************************************/
int PointScanMidCol(int k, int picCols)
{
	return PointScanCol(k, 2, numWidth[k] - 2, picCols, 3); //ɨ��2-6�н�����
}

/*****************************************************************
*	������:	DomainScan
*	��������: ����ĳ������ǰ�����ص����
*	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
Uint8 rowStart //�������ʼ��
Uint8 rowEnd//�������ֹ��
Uint8 colStart //�������ʼ��
Uint8 colEnd//������ֹ�?
*	����ֵ��zeroNum //������ǰ�����ص����
*   ȫ�ֱ����� g_charSection[CHAR_NUM][CHAR_ROWSIZE][CHAR_COLSIZE]
*   �ļ���̬����: ��
*   ������̬����: ��
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0   2018/04/16    ZhangGH             ԭʼ�汾
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
				zeroNum++;  //����������ǰ�����ص��ܸ���
			}
		}
	}
	return zeroNum;
}

/*****************************************************************
*	������:	PointScan
*	��������: ����ĳЩ�еĽ������
*	��ʽ����: Uint8 k //��ʾ�ڼ�����?
Uint8 rowStart //�������ʼ��
Uint8 rowEnd   //�������ֹ��
*	����ֵ��g_pointNum[m++]  //�����ڽ������
*   ȫ�ֱ�����numWidth[k]
g_charSection[CHAR_NUM][CHAR_ROWSIZE][CHAR_COLSIZE]
*  �ļ���̬��������
*  �����̬�������?*------------------------------------------------------------------
*	Revision History
*	No.	    Date	      Revised by	  Item	Description
*	V1.0    2018/04/16      ZhangGH            ԭʼ�汾
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
*	������:	PointScanCol
*	��������: ����ĳ�н������
*	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
Uint8 colStart //�������ʼ��
Uint8 colEnd//�������ֹ��
*	����ֵ��g_pointNum[m++] //�����ڽ������
*   ȫ�ֱ����� numHeight[k]
*  �ļ���̬��������
*  ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/16    ZhangGH             ԭʼ�汾
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
*	������:	PointCoordRL
*	��������: ��������ɨ�裬�жϵ�һ������ǰ�����ص�ʱ��λ��
*	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
int rowStart //�������ʼ��
int rowEnd//�������ʼ��
Uint8 divideCol//�߽�ֵ
*	����ֵ��flag
*   ȫ�ֱ�����numWidth[k]
*  �ļ���̬��������
*  ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/16    ZhangGH              ԭʼ�汾
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
*	������:	PointCoordLR
*	��������: ��������ɨ�裬�жϵ�һ������ǰ�����ص�ʱ��λ��
*	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
int rowStart //�������ʼ��
int rowEnd//�������ʼ��
Uint8 divideCol//�߽�ֵ
*	����ֵ��flag
*   ȫ�ֱ�����numWidth[k]
*  �ļ���̬��������
*  ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/18    ZhangGH              ԭʼ�汾
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
*	������:	PointCoordUD
*	��������: ��������ɨ�裬�жϵ�һ������ǰ�����ص�ʱ��λ��
*	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
int rowStart //�������ʼ��
int rowEnd//�������ʼ��
Uint8 divideCol//�߽�ֵ
*	����ֵ��flag
*   ȫ�ֱ�����numWidth[k]
*  �ļ���̬��������
*  ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/18    ZhangGH              ԭʼ�汾
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
*	������:	PointCoordDU
*	��������: ��������ɨ�裬�жϵ�һ������ǰ�����ص�ʱ��λ��
*	��ʽ����: Uint8 k //��ʾ�ڼ����ַ�
int rowStart //�������ʼ��
int rowEnd//�������ʼ��
Uint8 divideCol//�߽�ֵ
*	����ֵ��flag
*   ȫ�ֱ�����numWidth[k]
*  �ļ���̬��������
*  ������̬��������
*------------------------------------------------------------------
*	Revision History
*	No.	    Date	    Revised by	      Item	Description
*	V1.0    2018/04/18    ZhangGH              ԭʼ�汾
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
	//�Һڱ�
	des = src.clone();
	des += 1;
	cv::Mat imedge;
	float k[2] = { -1,10 };
	cv::Mat kernel = cv::Mat(1, 2, CV_32FC1, k);
	cv::filter2D(src, imedge, -1, kernel);
	
	//һ����ɫ�����أ�����������Ҷ�û�г��ְ�ɫ������Ϊ�������ֱ߿�
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

	//��ԭͼ������Ϊ���ڡ��������ɫ��������������Ϊ�������������ֱ߿�
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
	
	//��ͼ���Ե���ظ�ֵ255	
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

	//Ѱ���ϱ߽���±߽�
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

	//���ϱ߽���±߽�֮������ֵΪ0
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

	//ʹ����䷨���ڱ���֮ͨ���������Ϊ0
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

	//��ͼ����X�ᷭת180
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

	//Y�ᷭת
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

	//��ͼ����X�ᷭת180
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


bool Crop_up(Mat src, Mat &left, Mat &right)//�������ֲַ���ͼ��ֳ���������
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
		rect.x = tmp;//��0��ʼ�����ﲻ�ü�1
		rect.width = src.cols - tmp - 1;
		right = src(rect).clone();
		return true;
	}
	else
	{
		return false;
	}
}

bool Crop_down(cv::Mat src, cv::Mat &left, cv::Mat &right)//�������ֲַ���ͼ��ֳ���������
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
		rect.x = tmp;//��0��ʼ�����ﲻ�ü�1
		rect.width = src.cols - tmp - 1;
		right = src(rect).clone();
		return true;
	}
	else
	{
		return false;
	}
}

//�����ַ���������ͳ��ֱ��ͼ
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

//����Int������ͳ��ֱ��ͼ
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
	//ת������
	char tmp[2], tmp0[4];
	string mydate = "18.01.01";//��ʼ��ֵ
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

	//ת��ʱ��
	string mytime = "08:01:01";//��ʼ��ֵ
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

	//ʶ���ٶȺ����
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
			else//����������ǿ������Ϊ0
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
	//ת���ٶ�
	char* s;
	if (V == "")
	{
		ocr_recgInfor.Velocity = 0;//���VΪ�գ����ٶ���Ϊ0
	}
	/*else if (V[0] == '0')//����ٶȵ�һλΪ0�����ٶ�Ϊ0
	{
		ocr_recgInfor.Velocity = 0;//������Щ��Ƶ���ٶ�����λ��
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
		else//Ϊ��
		{
			ocr_recgInfor.Velocity = 0;
		}
	}
	

	//ת�����
	if(S.find_first_of('.')<0 && (!S.empty()))//�����ȱʧ
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
	else//Ϊ��
	{
		ocr_recgInfor.Mileage = 0;
	}
	
	//ʶ�𳵴κͳ���
	int len2 = wordInfor.infor_down.str_result.length();
	flag = 0;
	int acc2 = 0;//��¼�����е����ֳ�
	bool VIN_start = false;
	/*for (int j = 2; j < len2; j++)
	{
		if (flag == 0)
		{
			if ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9')|| wordInfor.infor_down.str_result[j] == 'K' || wordInfor.infor_down.str_result[j] == 'T' || wordInfor.infor_down.str_result[j] == 'Z' || wordInfor.infor_down.str_result[j] == 'L' || wordInfor.infor_down.str_result[j] == 'D' || wordInfor.infor_down.str_result[j] == 'G' || wordInfor.infor_down.str_result[j] == 'X')
			{
				if (ocr_recgInfor.Train_Number.size() < 1)//��һλֻ���������ֻ�KTZGDLX
				{
					ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
				}
				else if ((ocr_recgInfor.Train_Number[0] > '9' || ocr_recgInfor.Train_Number[0] < '0'))//�����һ���������֣��������5λ
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
				else //����һλ������
				{
					if (acc2 < 5)//����һλ�����֣����ֳ������Ϊ5
					{						
						if ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9'))
						{
							acc2++;
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
						}
						else//����֮�����KTZGDLX
						{
							ocr_recgInfor.Train_Number = "";
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
							acc2 = 0;
						}
					}					
				}				
			}
			if (wordInfor.infor_down.str_result[j + 1] == '?'&& j <= 3)//��ǰ��λ������˵��֮ǰ�Ĳ��ǳ���
			{
				ocr_recgInfor.Train_Number = "";
				acc2 = 0;
			}
			if ((j >= 3&&j<len2-3)&&(wordInfor.infor_down.str_result[j] == '?'||(wordInfor.infor_down.str_result[j+1]=='H'&&wordInfor.infor_down.str_result[j+2] == 'X'&&wordInfor.infor_down.str_result[j+3] == 'D') || (wordInfor.infor_down.str_result[j + 1] == 'H'&&wordInfor.infor_down.str_result[j + 2] == 'X') || (wordInfor.infor_down.str_result[j + 1] == 'H'&&wordInfor.infor_down.str_result[j + 3] == 'D') || (wordInfor.infor_down.str_result[j + 2] == 'X'&&wordInfor.infor_down.str_result[j + 3] == 'D')))
			{
				flag = 1;
			}
		}*/
	//9.28���޸�
	for (int j = 0; j < len2; j++)
	{
		if (VIN_start == 0)//VIN��δ��ʼ
		{
			if (flag==0 && ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9') || wordInfor.infor_down.str_result[j] == 'K'|| wordInfor.infor_down.str_result[j] == 'k' || wordInfor.infor_down.str_result[j] == 'T' || wordInfor.infor_down.str_result[j] == 'Z' || wordInfor.infor_down.str_result[j] == 'L' || wordInfor.infor_down.str_result[j] == 'D' || wordInfor.infor_down.str_result[j] == 'G' || wordInfor.infor_down.str_result[j] == 'X'))
			{
				if (ocr_recgInfor.Train_Number.size() < 1)//��һλֻ���������ֻ�KTZGDLX
				{
					ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
				}
				else if ((ocr_recgInfor.Train_Number[0] > '9' || ocr_recgInfor.Train_Number[0] < '0'))//�����һ���������֣��������5λ
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
				else //����һλ������
				{
					if (acc2 < 4)//����һλ�����֣����ֳ������Ϊ5
					{
						if ((wordInfor.infor_down.str_result[j] >= '0'&& wordInfor.infor_down.str_result[j] <= '9'))
						{
							acc2++;
							ocr_recgInfor.Train_Number += wordInfor.infor_down.str_result[j];
						}
						else//����֮�����KTZGDLX
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
		else//����Ϊ9λ
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
	std::vector<string> Train_Num;//����
	std::vector<string> Train_VIN;//����
	std::vector<double> Mileage;//���
	std::vector<int> Velocity;//ʱ��
	std::vector<int> Years;//ʱ��
	std::vector<int> Monthes;//ʱ��
	std::vector<int> Days;//ʱ��
	std::vector<int> Hours;//ʱ��
	std::vector<int> Minutes;//ʱ��
	std::vector<int> Seconds;//ʱ��
	OCR_Processed = OCR_Original;
	for (int k = 0; k < frame_Num; k++)
	{
		frame_result = OCR_Original.MTV.at(k);
		//��ȡ���κͳ���
		Train_Num.push_back(frame_result.Train_Number);
		Train_VIN.push_back(frame_result.VIN);
		//��ȡʱ�ٺ����
		Velocity.push_back(frame_result.Velocity);
		Mileage.push_back(frame_result.Mileage);
		//��ȡ����
		Years.push_back(frame_result.Date_Time.tm_year);
		Monthes.push_back(frame_result.Date_Time.tm_mon);
		Days.push_back(frame_result.Date_Time.tm_mday);
		//��ȡʱ��
		Hours.push_back(frame_result.Date_Time.tm_hour);
		Minutes.push_back(frame_result.Date_Time.tm_min);
		Seconds.push_back(frame_result.Date_Time.tm_sec);
	}
	//������,�������仯һ��	
	std::vector<Str_Long> TNHC = StrVectorHistCal(Train_Num);
	long tmp1 = 0;
	std::string PrimaryTN = "";
	std::string SecondTN = "";
	std::vector<long> P2S;//PrimaryTN���SecondTN
	std::vector<long> S2P;//SecondTN���PrimaryTN
	long PrimaryTN_start = frame_Num-1;
	//long PrimaryTN_end = 0;
	long SecondTN_start = frame_Num-1;
	//long SecondTN_end = 0;
	for (int k = 0; k < TNHC.size(); k++)//������һ����
	{
		if (TNHC[k].str_count > tmp1)
		{
			tmp1 = TNHC[k].str_count;
			PrimaryTN = TNHC[k].str_value;
		}
	}
	long tmp2 = 0;
	for (int k = 0; k < TNHC.size(); k++)//�����ڶ�����
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
			else if (Train_Num[j] == PrimaryTN&&PrimaryTN_start == frame_Num - 1)//���ת���������������������󳵺�
			{
				PrimaryTN_start = j;
			}
			else if (Train_Num[j] == SecondTN&&SecondTN_start == frame_Num - 1)
			{
				SecondTN_start = j;
			}
		}

		if (P2S.size() < 1 && S2P.size() == 1)//����һ��S2P�仯
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
		else if (S2P.size() < 1 && P2S.size() == 1)//����һ��P2S�仯
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

	//�����ţ�ͬһ��Ƶ�г��Ų���
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

	//У������
	//��
	std::vector<Int_Long> MHC = IntVectorHistCal(Monthes);
	long tmp0 = 0;
	long PrimaryFrame = frame_Num;//��¼����ʱ�̵Ĺؼ�֡
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
		CurrentMonth = 0;//ǿ������Ϊ0
	}
	
	int premonth = 0;
	int nextmonth = 0;
	long firstmonth_start = frame_Num;//ֻ���ܳ���2��
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

	//�ж��Ƿ�����·ݱ仯
	if (count(Monthes.begin(), Monthes.end(), premonth) >= count(Monthes.begin(), Monthes.end(), nextmonth))//����ͳ��ȷ�����ֵ��·�
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
		else//��Ϊû�п���
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
			PrimaryFrame = secondmonth_start;//�ڶ�����ʼ֡��Ϊ�ؼ�֡
		}
	}
	else //if (count(Monthes.begin(), Monthes.end(), premonth) < count(Monthes.begin(), Monthes.end(), nextmonth))
	{
		int acc1 = 0, acc2 = 0;
		for (long j = frame_Num-1; j >= 0; j--)//�Ӻ���ǰ����
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
		else//��Ϊû�п���
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
			PrimaryFrame = firstmonth_start + 1;//��һ���β֡��Ϊ�ؼ�֡
		}
	}

	//��
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
		currentyear = 2018 - 1900;//ǿ������Ϊ2018��
	}

	int preyear = currentyear - 1;
	int nextyear = currentyear + 1;
	
	if (IsNextYear == false)//���û�п���
	{
		for (long j = 0; j < frame_Num; j++)
		{
			Years[j] = currentyear;
			OCR_Processed.MTV[j].Date_Time.tm_year = currentyear;
		}
	}
	else//����
	{
		int acc1 = 0;
		long currentyear_start = 0;
		for (long j = 0; j < frame_Num; j++)//ȷ��currentyear�ǵ�һ�껹�ǵڶ���
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

		if (currentyear_start < PrimaryFrame)//currentyearΪ��һ��
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
		else//currentyearΪ�ڶ���
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
	//ǰһ��
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
	//��һ��
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

	/*//��
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
	//ǰһ��
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
	//��һ��
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

	//У��ʱ��
	int step = OCR_Original.Frame_Rate;//֡��
	bool SecondsReliable = false;//��ǵ�ǰʱ���Ƿ����
	bool MinutesReliable = false;//��ǵ�ǰʱ���Ƿ����
	bool HoursReliable = false;//��ǵ�ǰʱ���Ƿ����
	long start_tag = 0;//��¼������ʼλ��
	long end_tag = 0;//��¼������ֹλ��
	bool IsNextSecond = false;
	long FramesPerDay = 0;//�������Ļ�����¼ÿ���֡��
	vector<int> OneDay;//���һ��
	
	/*//������ı仯��������Ƶ֡��
	int fps_acc1 = 0;
	int second_tmp = 0;
	int fps_tmp1 = 0;
	std::vector<int> FPS_TMP;//Ĭ��Ϊ25֡ÿ��
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
			if ((second_tmp == 59 && Seconds[k] == 0)|| (Seconds[k] - second_tmp == 1))//��Ϊ��������
			{
				fps_tmp1 = fps_acc1;
				fps_acc1 = 1;
				second_tmp = Seconds[k];
				if (fps_tmp1 > 0)
				{
					FPS_TMP.push_back(fps_tmp1);
				}
			}
			else//����������
			{
				second_tmp = Seconds[k];
				fps_acc1 = 1;
			}
		}
	}
	//����ͳ�Ƶ����FPS
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

	//��ʼУ��ʱ��
	for (long k = 0; k < frame_Num - 1; k++)
	{
		IsNextMinutes = false;
		IsNextHours = false;
		IsNextDay = false;
		IsNextMonth = false;
		IsNextYear = false;

		//��
		IsNextSecond = false;
		if (Seconds[k] == 59 && Seconds[k + 1] == 0)//��һ����
		{
			IsNextMinutes = true;
			IsNextSecond = true;
		}
		else if ((Seconds[k + 1] - Seconds[k] > 1) || (Seconds[k + 1] - Seconds[k] < 0))//����������
		{
			if (k - start_tag > step - 1 && k - start_tag < step + 1 && SecondsReliable)//��������28-32֮�䣬���䵽��һ��
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
			else if (k - start_tag < step - 1 && SecondsReliable)//������С��28,ʱ�䲻��
			{
				//OCR_Processed.MTV[k + 1].Date_Time.tm_sec = Seconds[k];
				Seconds[k + 1] = Seconds[k];
			}
			OCR_Processed.MTV[k + 1].Confidence_Time = 0;//���ڶ�ʱ�������У�������Ŷȼ�С
		}

		if ((Seconds[k + 1] - Seconds[k] == 1))//��������
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
			Seconds[k] = 0;//��ǿ������Ϊ0
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

		//����
		if (Minutes[k]==59 && Minutes[k + 1] == 0 && IsNextMinutes)//��һСʱ
		{
			IsNextHours = true;
		}
		else if ((Minutes[k+1] - Minutes[k] >1)|| (Minutes[k + 1] - Minutes[k] <0))//����������
		{
			if (IsNextMinutes == false)
			{
				if (MinutesReliable == true)
				{
					//OCR_Processed.MTV[k + 1].Date_Time.tm_min = Minutes[k];
					Minutes[k + 1] = Minutes[k];
				}				
			}
			else if(Minutes[k]==59)//�������ƶ���ת����һ����
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
		else if ((Minutes[k + 1] - Minutes[k] == 1) && IsNextMinutes==false)//���Ӻ������䲻һ��
		{
			OCR_Processed.MTV[k + 1].Confidence_Time = 0;
		}
		if (Minutes[k] > 59 || Minutes[k] < 0)
		{
			OCR_Processed.MTV[k].Date_Time.tm_min = 0;//����ǿ������Ϊ0
		}
		OCR_Processed.MTV[k].Date_Time.tm_min = Minutes[k];

		//Сʱ
		if (Hours[k]==23 && Hours[k + 1] == 0 && IsNextHours)//��һ��
		{
			IsNextDay =true;
		}
		else if ((Hours[k + 1] - Hours[k] > 1) || (Hours[k + 1] - Hours[k] < 0))//����������
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
		else if (Hours[k + 1] - Hours[k] == 1 && IsNextHours==false)//Сʱ�ͷ������䲻һ��
		{
			OCR_Processed.MTV[k + 1].Confidence_Time  = 0;
		}

		if (Hours[k] > 23 || Hours[k] < 0)
		{
			OCR_Processed.MTV[k].Date_Time.tm_hour = 0;//Сʱǿ������Ϊ0
		}

		OCR_Processed.MTV[k].Date_Time.tm_hour = Hours[k];

		//���һ֡ǿ�Ƶ��ڵ����ڶ�֡
		if (k == frame_Num - 2)
		{
			OCR_Processed.MTV[k + 1].Date_Time.tm_sec = Seconds[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_min = Minutes[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_hour = Hours[k];
		}

		//��
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

	//У��ʱ��
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

//����֡��У��ʱ�������
void OcrRecognition::Suf_Process_OCR_BasedFPS(OCR_Results_video &OCR_Original, OCR_Results_video &OCR_Processed)
{
	long frame_Num = OCR_Original.frame_Num;
	OCR_Result frame_result;
	std::vector<string> Train_Num;//����
	std::vector<string> Train_VIN;//����
	std::vector<double> Mileage;//���
	std::vector<int> Velocity;//ʱ��
	std::vector<int> Years;//ʱ��
	std::vector<int> Monthes;//ʱ��
	std::vector<int> Days;//ʱ��
	std::vector<int> Hours;//ʱ��
	std::vector<int> Minutes;//ʱ��
	std::vector<int> Seconds;//ʱ��
	OCR_Processed = OCR_Original;
	for (int k = 0; k < frame_Num; k++)
	{
		frame_result = OCR_Original.MTV.at(k);
		//��ȡ���κͳ���
		Train_Num.push_back(frame_result.Train_Number);
		Train_VIN.push_back(frame_result.VIN);
		//��ȡʱ�ٺ����
		Velocity.push_back(frame_result.Velocity);
		Mileage.push_back(frame_result.Mileage);
		//��ȡ����
		Years.push_back(frame_result.Date_Time.tm_year);
		Monthes.push_back(frame_result.Date_Time.tm_mon);
		Days.push_back(frame_result.Date_Time.tm_mday);
		//��ȡʱ��
		Hours.push_back(frame_result.Date_Time.tm_hour);
		Minutes.push_back(frame_result.Date_Time.tm_min);
		Seconds.push_back(frame_result.Date_Time.tm_sec);
	}

	//������,�������仯һ��	
	std::vector<Str_Long> TNHC = StrVectorHistCal(Train_Num);
	long tmp1 = 0;
	std::string PrimaryTN = "";
	std::string SecondTN = "";
	std::vector<long> P2S;//PrimaryTN���SecondTN
	std::vector<long> S2P;//SecondTN���PrimaryTN
	long PrimaryTN_start = frame_Num - 1;
	//long PrimaryTN_end = 0;
	long SecondTN_start = frame_Num - 1;
	//long SecondTN_end = 0;
	for (int k = 0; k < TNHC.size(); k++)//������һ����
	{
		if (TNHC[k].str_count > tmp1)
		{
			tmp1 = TNHC[k].str_count;
			PrimaryTN = TNHC[k].str_value;
		}
	}
	long tmp2 = 0;
	for (int k = 0; k < TNHC.size(); k++)//�����ڶ�����
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
			else if (Train_Num[j] == PrimaryTN&&PrimaryTN_start == frame_Num - 1)//���ת���������������������󳵺�
			{
				PrimaryTN_start = j;
			}
			else if (Train_Num[j] == SecondTN&&SecondTN_start == frame_Num - 1)
			{
				SecondTN_start = j;
			}
		}

		if (P2S.size() < 1 && S2P.size() == 1)//����һ��S2P�仯
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
		else if (S2P.size() < 1 && P2S.size() == 1)//����һ��P2S�仯
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

	//�����ţ�ͬһ��Ƶ�г��Ų���
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

	//У��ʱ��
	//���ͻ��㣨�ٶȱ仯����5��1014����
	int v0_acc = 0;//�ٶȱ���Ϊ0
	//��ǰ����
	for (long j = 1; j < Velocity.size(); j++)
	{
		//У���ٶ�
		if (Velocity[j] - Velocity[j - 1] > 5)//�ٶȱ仯����5
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
	//�Ӻ���ǰ
	v0_acc = 0;//�ٶȱ���Ϊ0
	for (long j = Velocity.size()- 2; j > 0; j--)
	{
		//У���ٶ�
		if (Velocity[j] - Velocity[j+1] > 5)//�ٶȱ仯����5
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

	//У������
	bool IsNextYear = false;
	bool IsNextMonth = false;
	bool IsNextDay = false;
	bool IsNextHours = false;
	bool IsNextMinutes = false;
	
	//У��ʱ��
	int FPS = OCR_Original.Frame_Rate;//֡��
	bool SecondsReliable = false;//��ǵ�ǰʱ���Ƿ����
	bool MinutesReliable = false;//��ǵ�ǰʱ���Ƿ����
	bool HoursReliable = false;//��ǵ�ǰʱ���Ƿ����
	long start_tag = 0;//��¼������ʼλ��
	long end_tag = 0;//��¼������ֹλ��
	bool IsNextSecond = false;
	long FramesPerDay = 0;//�������Ļ�����¼ÿ���֡��

	//��ʼУ��ʱ��		
	int secondACC1 = 0;//���ڼ�¼����������ת��

	//Ѱ��һ�������ɿ���У���׼��������������������3
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
		if (secondACC1 >= 3)//�ҵ��ɿ��Ļ�׼��
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
			break;//����ѭ��
		}
	}
	
	int secondACC2 = 0;//���ڼ�¼ͬһ����ۻ�֡��
	long minute_start = 0, minute_end, hour_start = 0, hour_end,day_start = 0,day_end;
	for (long k = 0; k < frame_Num - 1; k++)
	{
		IsNextMinutes = false;
		IsNextHours = false;
		IsNextDay = false;
		IsNextMonth = false;
		IsNextYear = false;

		//��
		if (IsNextSecond)//����֡���ж���ת����һ��
		{
			if (Seconds[k] == 59 && Seconds[k + 1] == 0)//��һ����
			{
				IsNextMinutes = true;//��һ���ӿ�ʼ
				IsNextSecond = false;//��һ֡
				secondACC2 = 0;//���¼���
			}
			else if (Seconds[k + 1] == Seconds[k] && secondACC2 <= FPS)//�����ѵ����������һ֡û������,�����ͺ�һ֡����
			{
				secondACC2++;
				IsNextSecond = true;//ǿ����һ֡��ת��һ��
				OCR_Processed.MTV[k + 1].Confidence_Time = 0;//ʱ��У���������
			}
			else//��֡ǿ����ת
			{
				IsNextSecond = false;
				secondACC2 = 0;//���¼���
				if (Seconds[k] == 59)
				{
					Seconds[k + 1] = 0;
					IsNextMinutes = true;//��һ���ӿ�ʼ
				}
				else
				{
					Seconds[k + 1] = Seconds[k] + 1;
				}
			}			
		}
		else//���ݼ���δ��ת��һ��
		{
			secondACC2++;
			//�жϼ����Ƿ���������
			if (secondACC2 >= FPS-1)
			{
				if (((Seconds[k] == 59 && Seconds[k + 1] == 0) || Seconds[k + 1] - Seconds[k] == 1))//�������㣬�����ֳ���ȷ����ļ���������ǰһ֡����
				{
					OCR_Processed.MTV[k + 1].Confidence_Time = 0;//ʱ��У���������
					secondACC2 = 0;//���¼���
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

		if (Seconds[k] > 59 || Seconds[k] < 0)//�쳣����
		{
			Seconds[k] = 0;//��ǿ������Ϊ0
			OCR_Processed.MTV[k].Confidence_Time = 0;
			SecondsReliable = false;
		}
		OCR_Processed.MTV[k].Date_Time.tm_sec = Seconds[k];

		//����
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

			//�޸ķ���
			for (long j = minute_start; j <= k; j++)
			{
				Minutes[j] = CurrentMinute;
				OCR_Processed.MTV[j].Date_Time.tm_min = CurrentMinute;
			}

			//�ж�Сʱ�Ƿ�����
			if (Minutes[k] == 59 && Minutes[k + 1] == 0)//��һСʱ
			{
				IsNextHours = true;
			}

			//�ָ�ȫ�ֲ���
			IsNextMinutes = false;
			minute_start = k + 1;
		}

		//Сʱ
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

			//�޸�Сʱ
			for (long j = hour_start; j <= k; j++)
			{
				Hours[j] = CurrentHour;
				OCR_Processed.MTV[j].Date_Time.tm_hour = CurrentHour;
			}

			//�ж�Сʱ�Ƿ�����
			if (Hours[k] == 23 && Hours[k + 1] == 0)//��һСʱ
			{
				IsNextDay = true;
			}

			//�ָ�ȫ�ֲ���
			IsNextHours = false;
			hour_start = k + 1;
		}

		//��
		if (IsNextDay)
		{
			std::vector<int> OneDay;//���һ��
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

			//�޸���
			for (long j = day_start; j <= k; j++)
			{
				Days[j] = CurrentDay;
				OCR_Processed.MTV[j].Date_Time.tm_mday = CurrentDay;
			}

			//�ָ�ȫ�ֲ���
			IsNextDay = false;
			day_start = k + 1;
		}

		//�����δ������ת�ķ֡�Сʱ�������У��
		if (k == frame_Num - 2)
		{
			//��
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

				//�޸ķ���
				for (long j = minute_start; j <= k; j++)
				{
					Minutes[j] = CurrentMinute;
					OCR_Processed.MTV[j].Date_Time.tm_min = CurrentMinute;
				}
			}			

			//Сʱ
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

				//�޸�Сʱ
				for (long j = hour_start; j <= k; j++)
				{
					Hours[j] = CurrentHour;
					OCR_Processed.MTV[j].Date_Time.tm_hour = CurrentHour;
				}
			}			

			//��
			std::vector<int> OneDay;//���һ��
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
				//�޸���
				for (long j = day_start; j <= k; j++)
				{
					Days[j] = CurrentDay;
					OCR_Processed.MTV[j].Date_Time.tm_mday = CurrentDay;
				}
			}
			

			//������һ֡ǿ�Ƶ��ڵ����ڶ�֡
			OCR_Processed.MTV[k + 1].Date_Time.tm_sec = Seconds[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_min = Minutes[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_hour = Hours[k];
			OCR_Processed.MTV[k + 1].Date_Time.tm_mday = Days[k];
		}
	}

	//��
	std::vector<Int_Long> MonHC = IntVectorHistCal(Monthes);
	long tmp0 = 0;
	long PrimaryFrame = frame_Num;//��¼����ʱ�̵Ĺؼ�֡
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
		CurrentMonth = 0;//ǿ������Ϊ0
	}

	int premonth = 0;
	int nextmonth = 0;
	long firstmonth_start = frame_Num;//ֻ���ܳ���2��
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

	//�ж��Ƿ�����·ݱ仯
	if (count(Monthes.begin(), Monthes.end(), premonth) >= count(Monthes.begin(), Monthes.end(), nextmonth))//����ͳ��ȷ�����ֵ��·�
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
		else//��Ϊû�п���
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
			PrimaryFrame = secondmonth_start;//�ڶ�����ʼ֡��Ϊ�ؼ�֡
		}
	}
	else //if (count(Monthes.begin(), Monthes.end(), premonth) < count(Monthes.begin(), Monthes.end(), nextmonth))
	{
		int acc1 = 0, acc2 = 0;
		for (long j = frame_Num - 1; j >= 0; j--)//�Ӻ���ǰ����
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
		else//��Ϊû�п���
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
			PrimaryFrame = firstmonth_start + 1;//��һ���β֡��Ϊ�ؼ�֡
		}
	}

	//��
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
		currentyear = 2018 - 1900;//ǿ������Ϊ2018��
	}

	int preyear = currentyear - 1;
	int nextyear = currentyear + 1;

	if (IsNextYear == false)//���û�п���
	{
		for (long j = 0; j < frame_Num; j++)
		{
			Years[j] = currentyear;
			OCR_Processed.MTV[j].Date_Time.tm_year = currentyear;
		}
	}
	else//����
	{
		int acc1 = 0;
		long currentyear_start = 0;
		for (long j = 0; j < frame_Num; j++)//ȷ��currentyear�ǵ�һ�껹�ǵڶ���
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

		if (currentyear_start < PrimaryFrame)//currentyearΪ��һ��
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
		else//currentyearΪ�ڶ���
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

//������Ƶ����ı仯��������Ƶ֡��
int OcrRecognition::GetVideoFrame(std::vector<int> seconds)
{	
	int fps_acc1 = 0;
	int second_tmp = 0;
	int fps_tmp1 = 0;
	std::vector<int> FPS_TMP;//Ĭ��Ϊ25֡ÿ��
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
			if ((second_tmp == 59 && seconds[k] == 0) || (seconds[k] - second_tmp == 1))//��Ϊ��������
			{
				fps_tmp1 = fps_acc1;
				fps_acc1 = 1;
				second_tmp = seconds[k];
				if (fps_tmp1 > 0)
				{
					FPS_TMP.push_back(fps_tmp1);
				}
			}
			else//����������
			{
				second_tmp = seconds[k];
				fps_acc1 = 1;
			}
		}
	}
	//����ͳ�Ƶ����FPS
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