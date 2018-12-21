#include<iostream>
#include<opencv.hpp>
#include"hand_model.h"
using cv::Mat;
using std::cout;
using std::endl;
/*
程序说明：左手手摸
*/
HandModel::HandModel()
{
	initialize();

	xd = 0; yd = 0; zd = 0; yaw = 0; pitch = 0; roll = 0;
	mcp_xT = 0; mcp_zT = 0; dipT = 0;
	mcp_xI = 0; mcp_zI = 0; pipI = 0; dipI = 0;
	mcp_xM = 0; mcp_zM = 0; pipM = 0; dipM = 0;
	mcp_xR = 0; mcp_zR = 0; pipR = 0; dipR = 0;
	mcp_xL = 0; mcp_zL = 0; pipL = 0; dipL = 0;

	/*手模自由度取值限制，根据MSRA14数据集Subject2文件夹数据确定*/
	low_range_dof[0] = -70.0;   up_range_dof[0] = 70.0;  //xd、yd、ad的取值范围统计MSRA14数据集中"Subject+cur_subject+1"文件夹下的数据的数据得到
	low_range_dof[1] = -70.0;   up_range_dof[1] = 70.0;
	low_range_dof[2] = -380.0;   up_range_dof[2] = -250.0;

	low_range_dof[3] = -90.0;   up_range_dof[3] = 90.0;   //绕x轴
	low_range_dof[4] = 0.0;   up_range_dof[4] = 180.0;    //绕y轴，左手pitch的取值范围与右手不一致，右手为[-180°，0°]
	low_range_dof[5] = -45.0;   up_range_dof[5] = 180;    //绕z轴

	low_range_dof[6] = 0.0;   up_range_dof[6] = 90.0;    //大拇指
	low_range_dof[7] = -90.0;   up_range_dof[7] = 0.0;    //大拇指指掌关节绕z轴旋转的取值范围与右手不一致，右手为[0°，90°]
	low_range_dof[8] = 0.0;   up_range_dof[8] = 90.0;

	low_range_dof[9] = 0.0;   up_range_dof[9] = 90.0;   //食指
	low_range_dof[10] = -30.0;   up_range_dof[10] = 0; //mcp_zI的取值范围与右手不一致，右手为[0°，30°]
	low_range_dof[11] = 0.0;   up_range_dof[11] = 120.0;
	low_range_dof[12] = 0.0;   up_range_dof[12] = 80.0;

	low_range_dof[13] = 0.0;   up_range_dof[13] = 95.0;   //中指
	low_range_dof[14] = -5.0;   up_range_dof[14] = 5.0;
	low_range_dof[15] = 0.0;   up_range_dof[15] = 120.0;
	low_range_dof[16] = 0.0;   up_range_dof[16] = 80.0;

	low_range_dof[17] = 0.0;   up_range_dof[17] = 95.0;    //无名指
	low_range_dof[18] = 0.0;   up_range_dof[18] = 25.0;     //mcp_zR的取值范围与右手不一致，右手为[-25°，0°]
	low_range_dof[19] = 0.0;   up_range_dof[19] = 120.0;
	low_range_dof[20] = 0.0;   up_range_dof[20] = 80.0;

	low_range_dof[21] = 0.0;   up_range_dof[21] = 95.0;    //小拇指
	low_range_dof[22] = 0.0;   up_range_dof[22] = 30.0;    //mcp_zL的取值范围与右手不一致，右手为[-30°，0°]
	low_range_dof[23] = 0.0;   up_range_dof[23] = 120.0;
	low_range_dof[24] = 0.0;   up_range_dof[24] = 80.0;
}

void HandModel::initialize()
{
	//赋值initCylind[14]数组
	double x, y;  /*圆柱体下表面中心坐标的x、y分量*/
	
	/*大拇指：*/
	x = handStru.fingerRidiusM * 2.0 + handStru.fingerRidiusI * 2.0;
	y = handStru.highMCP0;
	//set_all_params()的3个参数依次是半径、上底面、下底面中心
	initCylind[0].set_all_params(handStru.fingerRidiusT, Position(x, y + handStru.len01, 0), Position(x, y, 0)); 
	y = y + handStru.len01;
	initCylind[1].set_all_params(handStru.fingerRidiusT, Position(x, y + handStru.len02, 0), Position(x, y, 0));

	/*食指：*/
	x = handStru.fingerRidiusM * 2 + handStru.fingerRidiusI;
	y = handStru.highMCP1;
	initCylind[2].set_all_params(handStru.fingerRidiusI, Position(x, y + handStru.len11, 0), Position(x, y, 0));
	y = y + handStru.len11;
	initCylind[3].set_all_params(handStru.fingerRidiusI, Position(x, y + handStru.len12, 0), Position(x, y, 0));
	y = y + handStru.len12;
	initCylind[4].set_all_params(handStru.fingerRidiusI, Position(x, y + handStru.len13, 0), Position(x, y, 0));

	/*中指：*/
	x = handStru.fingerRidiusM;
	y = handStru.highMCP2;
	initCylind[5].set_all_params(handStru.fingerRidiusM, Position(x, y + handStru.len21, 0), Position(x, y, 0));
	y = y + handStru.len21;
	initCylind[6].set_all_params(handStru.fingerRidiusM, Position(x, y + handStru.len22, 0), Position(x, y, 0));
	y = y + handStru.len22;
	initCylind[7].set_all_params(handStru.fingerRidiusM, Position(x, y + handStru.len23, 0), Position(x, y, 0));

	/*无名指：*/
	x = -1.0 * handStru.fingerRidiusR;
	y = handStru.highMCP3;
	initCylind[8].set_all_params(handStru.fingerRidiusR, Position(x, y + handStru.len31, 0), Position(x, y, 0));
	y = y + handStru.len31;
	initCylind[9].set_all_params(handStru.fingerRidiusR, Position(x, y + handStru.len32, 0), Position(x, y, 0));
	y = y + handStru.len32;
	initCylind[10].set_all_params(handStru.fingerRidiusR, Position(x, y + handStru.len33, 0), Position(x, y, 0));

	/*小拇指：*/
	x = -2.0 * handStru.fingerRidiusR - handStru.fingerRidiusL;
	y = handStru.highMCP4;
	initCylind[11].set_all_params(handStru.fingerRidiusL, Position(x, y + handStru.len41, 0), Position(x, y, 0));
	y = y + handStru.len41;
	initCylind[12].set_all_params(handStru.fingerRidiusL, Position(x, y + handStru.len42, 0), Position(x, y, 0));
	y = y + handStru.len42;
	initCylind[13].set_all_params(handStru.fingerRidiusL, Position(x, y + handStru.len43, 0), Position(x, y, 0));

	//赋值initCub1:
	double x_lrm = 2.0 * handStru.fingerRidiusM + 2.0 * handStru.fingerRidiusI;
	initCub1.set_all_params(Position(0, 0, 0), Position(0, handStru.highMCP4, 0), Position(x_lrm, 0, 0), Position(0, 0, 1.0 / 2.0 * handStru.palmThick));  /*1/2.0不能写成1/2，前者的结果为double型0.5，后者为int 0然后被转换为double 0.0*/

	 //赋值initCub0:
	double x_low = -2.0 * handStru.fingerRidiusR + 1.0 / 2.0 * (handStru.fingerRidiusR * 2 + handStru.fingerRidiusM * 2 + handStru.fingerRidiusI * 2.0);
	initCub0.set_all_params(Position(x_low, handStru.highMCP4, 0), Position(x_low, handStru.highMCP1, 0), Position(2 * (handStru.fingerRidiusM + handStru.fingerRidiusI), handStru.highMCP4, 0), Position(x_low, handStru.highMCP4, 1.0 / 2.0 * handStru.palmThick));

	//赋值changedCylind[]的半径
	for (int i = 0; i < 14; i++)
		changedCylind[i].set_radius(initCylind[i].get_radius()); //执行手势前后圆柱体半径不变
}

void HandModel::setDoF(double* DoF, int num)
{
	if (num != 25)
	{
		cout << "getStateChanged()要求的自由度参数个数不符合" << endl;
		exit(EXIT_FAILURE);
	}
	xd = DoF[0]; yd = DoF[1]; zd = DoF[2]; yaw = DoF[3]; pitch = DoF[4]; roll = DoF[5];
	mcp_xT = DoF[6]; mcp_zT = DoF[7]; dipT = DoF[8];
	mcp_xI = DoF[9]; mcp_zI = DoF[10]; pipI = DoF[11]; dipI = DoF[12];
	mcp_xM = DoF[13]; mcp_zM = DoF[14]; pipM = DoF[15]; dipM = DoF[16];
	mcp_xR = DoF[17]; mcp_zR = DoF[18]; pipR = DoF[19]; dipR = DoF[20];
	mcp_xL = DoF[21]; mcp_zL = DoF[22]; pipL = DoF[23]; dipL = DoF[24];
}

void HandModel::calChangedPose()
{
	/*-----------------------------------------------------------变换矩阵--------------------------------------------------------*/
	double mcp_x = 0;
	double mcp_z = 0;   /*指掌关节绕z轴旋转*/
	double pip = 0;     /*四指指间关节绕x轴旋转，或大拇指指端关节*/
	double dip = 0;     /*四指指端关节绕x轴旋转，大拇指对应的变换矩阵不需要该参数*/

	double jx = 0, jy = 0, jz = 0;  /*指掌关节的x、y、z坐标，当几何模型在初始状态下时*/
	double mx = 0, my = 0, mz = 0;  /*四指指间关节的x、y、z坐标，或大拇指指端关节*/
	double qx = 0, qy = 0, qz = 0;  /*四指指端关节的x、y、z坐标，大拇指对应的变换矩阵不需要这3个参数*/

	double Lnear = 0;  /*近指节长度*/
	double Lmid = 0;   /*四指中指节长度，大拇指对应的变换矩阵不需要该参数*/


	double aT5[4][4] = { { cos(pitch / 180 * CV_PI),0,sin(pitch / 180 * CV_PI),0 },{ 0,1,0,0 },{ -sin(pitch / 180 * CV_PI) ,0,cos(pitch / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double aT6[4][4] = { { 1,0,0,0 },{ 0,cos(yaw / 180 * CV_PI),-sin(yaw / 180 * CV_PI) ,0 },{ 0,sin(yaw / 180 * CV_PI) ,cos(yaw / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double aT7[4][4] = { { cos(roll / 180 * CV_PI),-sin(roll / 180 * CV_PI) ,0,0 },{ sin(roll / 180 * CV_PI) ,cos(roll / 180 * CV_PI) ,0,0 },{ 0,0,1,0 },{ 0,0,0,1 } };
	double aT8[4][4] = { { 1,0,0,xd },{ 0,1,0,yd },{ 0,0,1,zd },{ 0,0,0,1 } };
	Mat T5(4, 4, CV_64F, aT5);
	Mat T6(4, 4, CV_64F, aT6);
	Mat T7(4, 4, CV_64F, aT7);
	Mat T8(4, 4, CV_64F, aT8);

	Mat T8567 = T8*T5*T6*T7;

	//matPp
	Mat matPp = T8567;


	//matTn、matTf
	mcp_x = mcp_xT;
	mcp_z = mcp_zT;
	pip = dipT;
	jx = initCylind[0].get_lowSurface().x;
	jy = initCylind[0].get_lowSurface().y;
	jz = initCylind[0].get_lowSurface().z;
	mx = initCylind[1].get_lowSurface().x;   /*大拇指指端关节*/
	my = initCylind[1].get_lowSurface().y;
	mz = initCylind[1].get_lowSurface().z;
	Lnear = initCylind[0].getHeight();

	double atT1[4][4] = { { 1,0,0,-jx },{ 0,1,0,-jy },{ 0,0,1,-jz },{ 0,0,0,1 } };
	double atT2[4][4] = { { 1,0,0,0 },{ 0,cos(mcp_x / 180 * CV_PI),-sin(mcp_x / 180 * CV_PI),0 },{ 0,sin(mcp_x / 180 * CV_PI),cos(mcp_x / 180 * CV_PI),0 },{ 0,0,0,1 } };
	double atT3[4][4] = { { cos(mcp_z / 180 * CV_PI),-sin(mcp_z / 180 * CV_PI),0,0 },{ sin(mcp_z / 180 * CV_PI),cos(mcp_z / 180 * CV_PI),0,0 },{ 0,0,1,0 },{ 0,0,0,1 } };
	double atT4[4][4] = { { 1,0,0,jx },{ 0,1,0,jy },{ 0,0,1,jz },{ 0,0,0,1 } };

	double atP1[4][4] = { { 1,0,0,-mx },{ 0,1,0,-my },{ 0,0,1,-mz },{ 0,0,0,1 } };
	double atP2[4][4] = { { 1,0,0,0 },{ 0,cos(pip / 180 * CV_PI),-sin(pip / 180 * CV_PI) ,0 },{ 0,sin(pip / 180 * CV_PI) ,cos(pip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double atP3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lnear },{ 0,0,1,0 },{ 0,0,0,1 } };

	Mat tT1(4, 4, CV_64F, atT1);
	Mat tT2(4, 4, CV_64F, atT2);
	Mat tT3(4, 4, CV_64F, atT3);
	Mat tT4(4, 4, CV_64F, atT4);

	Mat tP1(4, 4, CV_64F, atP1);
	Mat tP2(4, 4, CV_64F, atP2);
	Mat tP3(4, 4, CV_64F, atP3);

	Mat T8567423_T = T8567*tT4*tT2*tT3;

	Mat matTn = T8567423_T*tT1;
	Mat matTf = T8567423_T*tP3*tP2*tP1;


	//matIn、matIm、matIf
	mcp_x = mcp_xI;
	mcp_z = mcp_zI;
	pip = pipI;
	dip = dipI;
	jx = initCylind[2].get_lowSurface().x;
	jy = initCylind[2].get_lowSurface().y;
	jz = initCylind[2].get_lowSurface().z;
	mx = initCylind[3].get_lowSurface().x;   /*食指指间关节*/
	my = initCylind[3].get_lowSurface().y;
	mz = initCylind[3].get_lowSurface().z;
	qx = initCylind[4].get_lowSurface().x;
	qy = initCylind[4].get_lowSurface().y;
	qz = initCylind[4].get_lowSurface().z;
	Lnear = initCylind[2].getHeight();
	Lmid = initCylind[3].getHeight();

	double aiT1[4][4] = { { 1,0,0,-jx },{ 0,1,0,-jy },{ 0,0,1,-jz },{ 0,0,0,1 } };
	double aiT2[4][4] = { { 1,0,0,0 },{ 0,cos(mcp_x / 180 * CV_PI),-sin(mcp_x / 180 * CV_PI),0 },{ 0,sin(mcp_x / 180 * CV_PI),cos(mcp_x / 180 * CV_PI),0 },{ 0,0,0,1 } };
	double aiT3[4][4] = { { cos(mcp_z / 180 * CV_PI),-sin(mcp_z / 180 * CV_PI),0,0 },{ sin(mcp_z / 180 * CV_PI),cos(mcp_z / 180 * CV_PI),0,0 },{ 0,0,1,0 },{ 0,0,0,1 } };
	double aiT4[4][4] = { { 1,0,0,jx },{ 0,1,0,jy },{ 0,0,1,jz },{ 0,0,0,1 } };

	double aiP1[4][4] = { { 1,0,0,-mx },{ 0,1,0,-my },{ 0,0,1,-mz },{ 0,0,0,1 } };
	double aiP2[4][4] = { { 1,0,0,0 },{ 0,cos(pip / 180 * CV_PI),-sin(pip / 180 * CV_PI) ,0 },{ 0,sin(pip / 180 * CV_PI) ,cos(pip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double aiP3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lnear },{ 0,0,1,0 },{ 0,0,0,1 } };

	double aiN1[4][4] = { { 1,0,0,-qx },{ 0,1,0,-qy },{ 0,0,1,-qz },{ 0,0,0,1 } };
	double aiN2[4][4] = { { 1,0,0,0 },{ 0,cos(dip / 180 * CV_PI),-sin(dip / 180 * CV_PI) ,0 },{ 0,sin(dip / 180 * CV_PI) ,cos(dip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double aiN3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lmid },{ 0,0,1,0 },{ 0,0,0,1 } };

	Mat iT1(4, 4, CV_64F, aiT1);
	Mat iT2(4, 4, CV_64F, aiT2);
	Mat iT3(4, 4, CV_64F, aiT3);
	Mat iT4(4, 4, CV_64F, aiT4);

	Mat iP1(4, 4, CV_64F, aiP1);
	Mat iP2(4, 4, CV_64F, aiP2);
	Mat iP3(4, 4, CV_64F, aiP3);

	Mat iN1(4, 4, CV_64F, aiN1);
	Mat iN2(4, 4, CV_64F, aiN2);
	Mat iN3(4, 4, CV_64F, aiN3);

	Mat iT423 = iT4*iT2*iT3;
	Mat iT8567423 = T8567*iT423;
	Mat iT8567423P32 = iT8567423*iP3*iP2;

	Mat matIn = iT8567423*iT1;
	Mat matIm = iT8567423P32*iP1;
	Mat matIf = iT8567423P32*iN3*iN2*iN1;


	//matMn、matMm、matMf
	mcp_x = mcp_xM;
	mcp_z = mcp_zM;
	pip = pipM;
	dip = dipM;
	jx = initCylind[5].get_lowSurface().x;
	jy = initCylind[5].get_lowSurface().y;
	jz = initCylind[5].get_lowSurface().z;
	mx = initCylind[6].get_lowSurface().x;
	my = initCylind[6].get_lowSurface().y;
	mz = initCylind[6].get_lowSurface().z;
	qx = initCylind[7].get_lowSurface().x;
	qy = initCylind[7].get_lowSurface().y;
	qz = initCylind[7].get_lowSurface().z;
	Lnear = initCylind[5].getHeight();
	Lmid = initCylind[6].getHeight();

	double amT1[4][4] = { { 1,0,0,-jx },{ 0,1,0,-jy },{ 0,0,1,-jz },{ 0,0,0,1 } };
	double amT2[4][4] = { { 1,0,0,0 },{ 0,cos(mcp_x / 180 * CV_PI),-sin(mcp_x / 180 * CV_PI),0 },{ 0,sin(mcp_x / 180 * CV_PI),cos(mcp_x / 180 * CV_PI),0 },{ 0,0,0,1 } };
	double amT3[4][4] = { { cos(mcp_z / 180 * CV_PI),-sin(mcp_z / 180 * CV_PI),0,0 },{ sin(mcp_z / 180 * CV_PI),cos(mcp_z / 180 * CV_PI),0,0 },{ 0,0,1,0 },{ 0,0,0,1 } };
	double amT4[4][4] = { { 1,0,0,jx },{ 0,1,0,jy },{ 0,0,1,jz },{ 0,0,0,1 } };

	double amP1[4][4] = { { 1,0,0,-mx },{ 0,1,0,-my },{ 0,0,1,-mz },{ 0,0,0,1 } };
	double amP2[4][4] = { { 1,0,0,0 },{ 0,cos(pip / 180 * CV_PI),-sin(pip / 180 * CV_PI) ,0 },{ 0,sin(pip / 180 * CV_PI) ,cos(pip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double amP3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lnear },{ 0,0,1,0 },{ 0,0,0,1 } };

	double amN1[4][4] = { { 1,0,0,-qx },{ 0,1,0,-qy },{ 0,0,1,-qz },{ 0,0,0,1 } };
	double amN2[4][4] = { { 1,0,0,0 },{ 0,cos(dip / 180 * CV_PI),-sin(dip / 180 * CV_PI) ,0 },{ 0,sin(dip / 180 * CV_PI) ,cos(dip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double amN3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lmid },{ 0,0,1,0 },{ 0,0,0,1 } };

	Mat mT1(4, 4, CV_64F, amT1);
	Mat mT2(4, 4, CV_64F, amT2);
	Mat mT3(4, 4, CV_64F, amT3);
	Mat mT4(4, 4, CV_64F, amT4);

	Mat mP1(4, 4, CV_64F, amP1);
	Mat mP2(4, 4, CV_64F, amP2);
	Mat mP3(4, 4, CV_64F, amP3);

	Mat mN1(4, 4, CV_64F, amN1);
	Mat mN2(4, 4, CV_64F, amN2);
	Mat mN3(4, 4, CV_64F, amN3);

	Mat mT8567423 = T8567*mT4*mT2*mT3;
	Mat mT8567423P32 = mT8567423*mP3*mP2;

	Mat matMn = mT8567423*mT1;
	Mat matMm = mT8567423P32*mP1;
	Mat matMf = mT8567423P32*mN3*mN2*mN1;


	//matRn、matRm、matRf
	mcp_x = mcp_xR;
	mcp_z = mcp_zR;
	pip = pipR;
	dip = dipR;
	jx = initCylind[8].get_lowSurface().x;
	jy = initCylind[8].get_lowSurface().y;
	jz = initCylind[8].get_lowSurface().z;
	mx = initCylind[9].get_lowSurface().x;
	my = initCylind[9].get_lowSurface().y;
	mz = initCylind[9].get_lowSurface().z;
	qx = initCylind[10].get_lowSurface().x;
	qy = initCylind[10].get_lowSurface().y;
	qz = initCylind[10].get_lowSurface().z;
	Lnear = initCylind[8].getHeight();
	Lmid = initCylind[9].getHeight();

	double arT1[4][4] = { { 1,0,0,-jx },{ 0,1,0,-jy },{ 0,0,1,-jz },{ 0,0,0,1 } };
	double arT2[4][4] = { { 1,0,0,0 },{ 0,cos(mcp_x / 180 * CV_PI),-sin(mcp_x / 180 * CV_PI),0 },{ 0,sin(mcp_x / 180 * CV_PI),cos(mcp_x / 180 * CV_PI),0 },{ 0,0,0,1 } };
	double arT3[4][4] = { { cos(mcp_z / 180 * CV_PI),-sin(mcp_z / 180 * CV_PI),0,0 },{ sin(mcp_z / 180 * CV_PI),cos(mcp_z / 180 * CV_PI),0,0 },{ 0,0,1,0 },{ 0,0,0,1 } };
	double arT4[4][4] = { { 1,0,0,jx },{ 0,1,0,jy },{ 0,0,1,jz },{ 0,0,0,1 } };

	double arP1[4][4] = { { 1,0,0,-mx },{ 0,1,0,-my },{ 0,0,1,-mz },{ 0,0,0,1 } };
	double arP2[4][4] = { { 1,0,0,0 },{ 0,cos(pip / 180 * CV_PI),-sin(pip / 180 * CV_PI) ,0 },{ 0,sin(pip / 180 * CV_PI) ,cos(pip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double arP3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lnear },{ 0,0,1,0 },{ 0,0,0,1 } };

	double arN1[4][4] = { { 1,0,0,-qx },{ 0,1,0,-qy },{ 0,0,1,-qz },{ 0,0,0,1 } };
	double arN2[4][4] = { { 1,0,0,0 },{ 0,cos(dip / 180 * CV_PI),-sin(dip / 180 * CV_PI) ,0 },{ 0,sin(dip / 180 * CV_PI) ,cos(dip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double arN3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lmid },{ 0,0,1,0 },{ 0,0,0,1 } };

	Mat rT1(4, 4, CV_64F, arT1);
	Mat rT2(4, 4, CV_64F, arT2);
	Mat rT3(4, 4, CV_64F, arT3);
	Mat rT4(4, 4, CV_64F, arT4);

	Mat rP1(4, 4, CV_64F, arP1);
	Mat rP2(4, 4, CV_64F, arP2);
	Mat rP3(4, 4, CV_64F, arP3);

	Mat rN1(4, 4, CV_64F, arN1);
	Mat rN2(4, 4, CV_64F, arN2);
	Mat rN3(4, 4, CV_64F, arN3);

	Mat rT8567423 = T8567*rT4*rT2*rT3;
	Mat rT8567423P32 = rT8567423*rP3*rP2;

	Mat matRn = rT8567423*rT1;
	Mat matRm = rT8567423P32*rP1;
	Mat matRf = rT8567423P32*rN3*rN2*rN1;


	//matLn、matLm、matLf
	mcp_x = mcp_xL;
	mcp_z = mcp_zL;
	pip = pipL;
	dip = dipL;
	jx = initCylind[11].get_lowSurface().x;
	jy = initCylind[11].get_lowSurface().y;
	jz = initCylind[11].get_lowSurface().z;
	mx = initCylind[12].get_lowSurface().x;
	my = initCylind[12].get_lowSurface().y;
	mz = initCylind[12].get_lowSurface().z;
	qx = initCylind[13].get_lowSurface().x;
	qy = initCylind[13].get_lowSurface().y;
	qz = initCylind[13].get_lowSurface().z;
	Lnear = initCylind[11].getHeight();
	Lmid = initCylind[12].getHeight();

	double aLT1[4][4] = { { 1,0,0,-jx },{ 0,1,0,-jy },{ 0,0,1,-jz },{ 0,0,0,1 } };
	double aLT2[4][4] = { { 1,0,0,0 },{ 0,cos(mcp_x / 180 * CV_PI),-sin(mcp_x / 180 * CV_PI),0 },{ 0,sin(mcp_x / 180 * CV_PI),cos(mcp_x / 180 * CV_PI),0 },{ 0,0,0,1 } };
	double aLT3[4][4] = { { cos(mcp_z / 180 * CV_PI),-sin(mcp_z / 180 * CV_PI),0,0 },{ sin(mcp_z / 180 * CV_PI),cos(mcp_z / 180 * CV_PI),0,0 },{ 0,0,1,0 },{ 0,0,0,1 } };
	double aLT4[4][4] = { { 1,0,0,jx },{ 0,1,0,jy },{ 0,0,1,jz },{ 0,0,0,1 } };

	double aLP1[4][4] = { { 1,0,0,-mx },{ 0,1,0,-my },{ 0,0,1,-mz },{ 0,0,0,1 } };
	double aLP2[4][4] = { { 1,0,0,0 },{ 0,cos(pip / 180 * CV_PI),-sin(pip / 180 * CV_PI) ,0 },{ 0,sin(pip / 180 * CV_PI) ,cos(pip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double aLP3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lnear },{ 0,0,1,0 },{ 0,0,0,1 } };

	double aLN1[4][4] = { { 1,0,0,-qx },{ 0,1,0,-qy },{ 0,0,1,-qz },{ 0,0,0,1 } };
	double aLN2[4][4] = { { 1,0,0,0 },{ 0,cos(dip / 180 * CV_PI),-sin(dip / 180 * CV_PI) ,0 },{ 0,sin(dip / 180 * CV_PI) ,cos(dip / 180 * CV_PI) ,0 },{ 0,0,0,1 } };
	double aLN3[4][4] = { { 1,0,0,0 },{ 0,1,0,Lmid },{ 0,0,1,0 },{ 0,0,0,1 } };

	Mat LT1(4, 4, CV_64F, aLT1);
	Mat LT2(4, 4, CV_64F, aLT2);
	Mat LT3(4, 4, CV_64F, aLT3);
	Mat LT4(4, 4, CV_64F, aLT4);

	Mat LP1(4, 4, CV_64F, aLP1);
	Mat LP2(4, 4, CV_64F, aLP2);
	Mat LP3(4, 4, CV_64F, aLP3);

	Mat LN1(4, 4, CV_64F, aLN1);
	Mat LN2(4, 4, CV_64F, aLN2);
	Mat LN3(4, 4, CV_64F, aLN3);

	Mat LT8567423 = T8567*LT4*LT2*LT3;
	Mat LT8567423P32 = LT8567423*LP3*LP2;

	Mat matLn = LT8567423*LT1;
	Mat matLm = LT8567423P32*LP1;
	Mat matLf = LT8567423P32*LN3*LN2*LN1;


	/*-----------------------------------------------------------坐标变换--------------------------------------------------------*/
	Position posInit;
	Position posChanged;

	/*大拇指：*/ 
	/*finger指明是哪根手指：T、I、M、R、L, finger = P表示手掌；section指明哪个指节：远指节f、中指节m、近指节n，section = p表示手掌*/
	getPosChanged(initCylind[0].get_lowSurface(), matPp, posChanged);
	changedCylind[0].set_lowSurface(posChanged);
	getPosChanged(initCylind[0].get_upSurface(), matTn, posChanged);
	changedCylind[0].set_upSurface(posChanged);

	changedCylind[1].set_lowSurface(posChanged);    /*changedCylind[1]*/
	getPosChanged(initCylind[1].get_upSurface(), matTf, posChanged);
	changedCylind[1].set_upSurface(posChanged);

	/*食指：*/
	getPosChanged(initCylind[2].get_lowSurface(), matPp, posChanged);
	changedCylind[2].set_lowSurface(posChanged);
	getPosChanged(initCylind[2].get_upSurface(), matIn, posChanged);
	changedCylind[2].set_upSurface(posChanged);

	changedCylind[3].set_lowSurface(posChanged);  /*changedCylind[3]*/
	getPosChanged(initCylind[3].get_upSurface(), matIm, posChanged);
	changedCylind[3].set_upSurface(posChanged);

	changedCylind[4].set_lowSurface(posChanged);   /*changedCylind[4]*/
	getPosChanged(initCylind[4].get_upSurface(), matIf, posChanged);
	changedCylind[4].set_upSurface(posChanged);

	/*中指：*/                        
	getPosChanged(initCylind[5].get_lowSurface(), matPp, posChanged);
	changedCylind[5].set_lowSurface(posChanged);
	getPosChanged(initCylind[5].get_upSurface(), matMn, posChanged);
	changedCylind[5].set_upSurface(posChanged);

	changedCylind[6].set_lowSurface(posChanged);   /*changedCylind[6]*/
	getPosChanged(initCylind[6].get_upSurface(), matMm, posChanged);
	changedCylind[6].set_upSurface(posChanged);

	changedCylind[7].set_lowSurface(posChanged);    /*changedCylind[7]*/
	getPosChanged(initCylind[7].get_upSurface(), matMf, posChanged);
	changedCylind[7].set_upSurface(posChanged);

	/*无名指：*/
	getPosChanged(initCylind[8].get_lowSurface(), matPp, posChanged);
	changedCylind[8].set_lowSurface(posChanged);
	getPosChanged(initCylind[8].get_upSurface(), matRn, posChanged);
	changedCylind[8].set_upSurface(posChanged);

	changedCylind[9].set_lowSurface(posChanged);   /*changedCylind[9]*/
	getPosChanged(initCylind[9].get_upSurface(), matRm, posChanged);
	changedCylind[9].set_upSurface(posChanged);

	changedCylind[10].set_lowSurface(posChanged);   /*changedCylind[10]*/
	getPosChanged(initCylind[10].get_upSurface(), matRf, posChanged);
	changedCylind[10].set_upSurface(posChanged);

	/*小拇指：*/
	getPosChanged(initCylind[11].get_lowSurface(), matPp, posChanged);
	changedCylind[11].set_lowSurface(posChanged);
	getPosChanged(initCylind[11].get_upSurface(), matLn, posChanged);
	changedCylind[11].set_upSurface(posChanged);

	changedCylind[12].set_lowSurface(posChanged);     /*changedCylind[12]*/
	getPosChanged(initCylind[12].get_upSurface(), matLm, posChanged);
	changedCylind[12].set_upSurface(posChanged);

	changedCylind[13].set_lowSurface(posChanged);    /*changedCylind[13]*/
	getPosChanged(initCylind[13].get_upSurface(), matLf, posChanged);
	changedCylind[13].set_upSurface(posChanged);

	/*赋值changedCub0、changedCub1：*/
	getPosChanged(initCub0.lowSurface, matPp, changedCub0.lowSurface);
	getPosChanged(initCub0.upSurface, matPp, changedCub0.upSurface);
	getPosChanged(initCub0.lrMidPoint, matPp, changedCub0.lrMidPoint);
	getPosChanged(initCub0.lbMidPoint, matPp, changedCub0.lbMidPoint);

	getPosChanged(initCub1.lowSurface, matPp, changedCub1.lowSurface);
	getPosChanged(initCub1.upSurface, matPp, changedCub1.upSurface);
	getPosChanged(initCub1.lrMidPoint, matPp, changedCub1.lrMidPoint);
	getPosChanged(initCub1.lbMidPoint, matPp, changedCub1.lbMidPoint);
}

void HandModel::calChangedPose(double* DoF, int num)  /*赋值changedCylind[]、changedCub0、changedCub1*/
{
	setDoF(DoF, num);
	calChangedPose();
}

void HandModel::getPosChanged(const Position& posInit, const Mat& mat, Position& posChanged)const
{
	double posInitA[4][1] = { { posInit.x },{ posInit.y },{ posInit.z },{ 1 } };
	Mat posInitMat(4, 1, CV_64F, posInitA);
	double posChangedA[4][1] = { { 0 },{ 0 },{ 0 },{ 1 } };
	Mat posChangedMat(4, 1, CV_64F, posChangedA);
	posChangedMat = mat*posInitMat;
	posChanged.x = posChangedMat.at<double>(0, 0);
	posChanged.y = posChangedMat.at<double>(1, 0);
	posChanged.z = posChangedMat.at<double>(2, 0);
};

void HandModel::get_joint_xyz_in_msra14_order(double* joint_xyz, int num)
{
	/*
	MSRA数据集定义了21个关节，关节顺序为：
	手腕、                                  0
	mcpI、pipI、dipI、tipI、                1-4
	mcpM、pipM、dipM、tipM、                5-8
	mcpR、pipR、dipR、tipR、                9-12
	mcpL、pipL、dipL、tipL、                13-16
	大拇指掌腕关节、大拇指指掌关节、大拇指dip、大拇指tip    17-20
	*/
	if (num != 21 * 3)
	{
		cout << "关节数与MSRA手模不匹配" << endl;
		exit(EXIT_FAILURE);
	}
	joint_xyz[0 * 3 + 0] = changedCub1.lowSurface.x; joint_xyz[0 * 3 + 1] = changedCub1.lowSurface.y; joint_xyz[0 * 3 + 2] = changedCub1.lowSurface.z;

	joint_xyz[1 * 3 + 0] = changedCylind[2].get_lowSurface().x; joint_xyz[1 * 3 + 1] = changedCylind[2].get_lowSurface().y; joint_xyz[1 * 3 + 2] = changedCylind[2].get_lowSurface().z;
	joint_xyz[2 * 3 + 0] = changedCylind[3].get_lowSurface().x; joint_xyz[2 * 3 + 1] = changedCylind[3].get_lowSurface().y; joint_xyz[2 * 3 + 2] = changedCylind[3].get_lowSurface().z;
	joint_xyz[3 * 3 + 0] = changedCylind[4].get_lowSurface().x; joint_xyz[3 * 3 + 1] = changedCylind[4].get_lowSurface().y; joint_xyz[3 * 3 + 2] = changedCylind[4].get_lowSurface().z;
	joint_xyz[4 * 3 + 0] = changedCylind[4].get_upSurface().x;  joint_xyz[4 * 3 + 1] = changedCylind[4].get_upSurface().y;  joint_xyz[4 * 3 + 2] = changedCylind[4].get_upSurface().z;

	joint_xyz[5 * 3 + 0] = changedCylind[5].get_lowSurface().x; joint_xyz[5 * 3 + 1] = changedCylind[5].get_lowSurface().y; joint_xyz[5 * 3 + 2] = changedCylind[5].get_lowSurface().z;
	joint_xyz[6 * 3 + 0] = changedCylind[6].get_lowSurface().x; joint_xyz[6 * 3 + 1] = changedCylind[6].get_lowSurface().y; joint_xyz[6 * 3 + 2] = changedCylind[6].get_lowSurface().z;
	joint_xyz[7 * 3 + 0] = changedCylind[7].get_lowSurface().x; joint_xyz[7 * 3 + 1] = changedCylind[7].get_lowSurface().y; joint_xyz[7 * 3 + 2] = changedCylind[7].get_lowSurface().z;
	joint_xyz[8 * 3 + 0] = changedCylind[7].get_upSurface().x;  joint_xyz[8 * 3 + 1] = changedCylind[7].get_upSurface().y;  joint_xyz[8 * 3 + 2] = changedCylind[7].get_upSurface().z;

	joint_xyz[9 * 3 + 0] = changedCylind[8].get_lowSurface().x;   joint_xyz[9 * 3 + 1] = changedCylind[8].get_lowSurface().y;   joint_xyz[9 * 3 + 2] = changedCylind[8].get_lowSurface().z;
	joint_xyz[10 * 3 + 0] = changedCylind[9].get_lowSurface().x;  joint_xyz[10 * 3 + 1] = changedCylind[9].get_lowSurface().y;  joint_xyz[10 * 3 + 2] = changedCylind[9].get_lowSurface().z;
	joint_xyz[11 * 3 + 0] = changedCylind[10].get_lowSurface().x; joint_xyz[11 * 3 + 1] = changedCylind[10].get_lowSurface().y; joint_xyz[11 * 3 + 2] = changedCylind[10].get_lowSurface().z;
	joint_xyz[12 * 3 + 0] = changedCylind[10].get_upSurface().x;  joint_xyz[12 * 3 + 1] = changedCylind[10].get_upSurface().y;  joint_xyz[12 * 3 + 2] = changedCylind[10].get_upSurface().z;

	joint_xyz[13 * 3 + 0] = changedCylind[11].get_lowSurface().x; joint_xyz[13 * 3 + 1] = changedCylind[11].get_lowSurface().y; joint_xyz[13 * 3 + 2] = changedCylind[11].get_lowSurface().z;
	joint_xyz[14 * 3 + 0] = changedCylind[12].get_lowSurface().x; joint_xyz[14 * 3 + 1] = changedCylind[12].get_lowSurface().y; joint_xyz[14 * 3 + 2] = changedCylind[12].get_lowSurface().z;
	joint_xyz[15 * 3 + 0] = changedCylind[13].get_lowSurface().x; joint_xyz[15 * 3 + 1] = changedCylind[13].get_lowSurface().y; joint_xyz[15 * 3 + 2] = changedCylind[13].get_lowSurface().z;
	joint_xyz[16 * 3 + 0] = changedCylind[13].get_upSurface().x;  joint_xyz[16 * 3 + 1] = changedCylind[13].get_upSurface().y;  joint_xyz[16 * 3 + 2] = changedCylind[13].get_upSurface().z;

	
	joint_xyz[17 * 3 + 0] = 0;  joint_xyz[17 * 3 + 1] = 0;  joint_xyz[17 * 3 + 2] = 0;//HandModel类定义的手模没有大拇指掌腕关节，因此这里将该关节的3d位置设为(0，0，0)

	joint_xyz[18 * 3 + 0] = changedCylind[0].get_lowSurface().x; joint_xyz[18 * 3 + 1] = changedCylind[0].get_lowSurface().y; joint_xyz[18 * 3 + 2] = changedCylind[0].get_lowSurface().z;
	joint_xyz[19 * 3 + 0] = changedCylind[1].get_lowSurface().x; joint_xyz[19 * 3 + 1] = changedCylind[1].get_lowSurface().y; joint_xyz[19 * 3 + 2] = changedCylind[1].get_lowSurface().z;
	joint_xyz[20 * 3 + 0] = changedCylind[1].get_upSurface().x; joint_xyz[20 * 3 + 1] = changedCylind[1].get_upSurface().y; joint_xyz[20 * 3 + 2] = changedCylind[1].get_upSurface().z;
}

void HandModel::showInitPose()const
{
	cout << "初始化后手模initCylind的半径是：" << endl;
	for (int i = 0; i < 14; i++)
		cout <<initCylind[i].get_radius() << " ";
	cout << endl;

	cout << "0到14号圆柱体的上下表面中心的坐标依次是：" << endl;
	for (int i = 0; i < 14; i++)
	{
		cout << initCylind[i].get_upSurface().x << " " << initCylind[i].get_upSurface().y << " " << initCylind[i].get_upSurface().z << "; " << initCylind[i].get_lowSurface().x << " " << initCylind[i].get_lowSurface().y << " " << initCylind[i].get_lowSurface().z << "   ";
		if (i == 1 || i == 4 || i == 7 || i == 10 || i == 13)
			cout << endl;
	}
	cout << "0、1号长方体的下、上、右、后四个点的坐标依次是：" << endl;
	cout << initCub0.lowSurface.x << " " << initCub0.lowSurface.y << " " << initCub0.lowSurface.z << "; " << initCub0.upSurface.x << " " << initCub0.upSurface.y << " " << initCub0.upSurface.z << "; ";
	cout << initCub0.lrMidPoint.x << " " << initCub0.lrMidPoint.y << " " << initCub0.lrMidPoint.z << "; " << initCub0.lbMidPoint.x << " " << initCub0.lbMidPoint.y << " " << initCub0.lbMidPoint.z << endl;

	cout << initCub1.lowSurface.x << " " << initCub1.lowSurface.y << " " << initCub1.lowSurface.z << "; " << initCub1.upSurface.x << " " << initCub1.upSurface.y << " " << initCub1.upSurface.z << "; ";
	cout << initCub1.lrMidPoint.x << " " << initCub1.lrMidPoint.y << " " << initCub1.lrMidPoint.z << "; " << initCub1.lbMidPoint.x << " " << initCub1.lbMidPoint.y << " " << initCub1.lbMidPoint.z << endl;
}

void HandModel::showChangedPose()const
{
	cout << "运动变换后手模initCylind的半径是：" << endl;
	for (int i = 0; i < 14; i++)
		cout << changedCylind[i].get_radius() << " ";
	cout << endl;

	cout << "0到14号圆柱体的上、下表面中心的坐标依次是：" << endl;
	for (int i = 0; i < 14; i++)
	{
		cout << changedCylind[i].get_upSurface().x << " " << changedCylind[i].get_upSurface().y << " " << changedCylind[i].get_upSurface().z << ";  " << changedCylind[i].get_lowSurface().x << " " << changedCylind[i].get_lowSurface().y << " " << changedCylind[i].get_lowSurface().z << "     ";
		if (i == 1 || i == 4 || i == 7 || i == 10 || i == 13)
			cout << endl;
	}
	cout << "0、1号长方体的下、上、右、后四个点的坐标依次是：" << endl;
	cout << changedCub0.lowSurface.x << " " << changedCub0.lowSurface.y << " " << changedCub0.lowSurface.z << "; " << changedCub0.upSurface.x << " " << changedCub0.upSurface.y << " " << changedCub0.upSurface.z << "; ";
	cout << changedCub0.lrMidPoint.x << " " << changedCub0.lrMidPoint.y << " " << changedCub0.lrMidPoint.z << "; " << changedCub0.lbMidPoint.x << " " << changedCub0.lbMidPoint.y << " " << changedCub0.lbMidPoint.z << endl;

	cout << changedCub1.lowSurface.x << " " << changedCub1.lowSurface.y << " " << changedCub1.lowSurface.z << "; " << changedCub1.upSurface.x << " " << changedCub1.upSurface.y << " " << changedCub1.upSurface.z << "; ";
	cout << changedCub1.lrMidPoint.x << " " << changedCub1.lrMidPoint.y << " " << changedCub1.lrMidPoint.z << "; " << changedCub1.lbMidPoint.x << " " << changedCub1.lbMidPoint.y << " " << changedCub1.lbMidPoint.z << endl;
}

const double* HandModel::getLowRangeDof()const
{
	return low_range_dof;
}
const double* HandModel::getUpRangeDof()const
{
	return up_range_dof;
}
const Cylinder* HandModel::getChangedCylind()const
{
	return changedCylind;
}
const Cuboid HandModel::getChangedCub0()const
{
	return changedCub0;
}
const Cuboid HandModel::getChangedCub1()const
{
	return changedCub1;
}
