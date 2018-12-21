#pragma once
#include<iostream>
#include<opencv.hpp>
#include"data_struct.h"
#include"geometry.h"
using cv::Mat;

class HandSkeletalStruct   /*人手骨骼结构：指节长度、五指指掌关节在手腕局部关节坐标系中的y坐标、手指半径、手掌厚度*/
{
public:
	HandSkeletalStruct() {}
public:
	/*就地初始化，C++ 11新特性*/
	const double len01 = 20.0+10;//大拇指近指节长度,单位mm
	const double len02 = 20.0;  //大拇指远指节长度

	const double len11 = 28.0;  //食指近指节长度
	const double len12 = 21.0;  //食指中指节长度
	const double len13 = 19.0;  //食指远指节长度

	const double len21 = 33.0;  //中指
	const double len22 = 22.0;
	const double len23 = 20.0;

	const double len31 = 29.0;  //无名指
	const double len32 = 19.0;
	const double len33 = 18.0;

	const double len41 = 21.0;  //小拇指
	const double len42 = 13.0;
	const double len43 = 14.0;

	const double highMCP0 = 32;  //大拇指指掌关节在手腕局部关节坐标系中的y坐标 
	const double highMCP1 = 68.3759+10;  //食指指掌关节在手腕局部关节坐标系中的y坐标 
	const double highMCP2 = 68.3759+10;  //中指
	const double highMCP3 = 68.3759+10;  //无名指
	const double highMCP4 = 60.8759+10;  //小拇指

	const double fingerRidiusT = 9.0;  //大拇指半径，单位毫米
	const double fingerRidiusI = 9.0;
	const double fingerRidiusM = 9.0;
	const double fingerRidiusR = 9.0;
	const double fingerRidiusL = 9.0;

	const double palmThick = 18.0;     //手掌厚度
};

class HandModel  /*人手模型，包括几何模型和运动建模、自由度*/
{
private:
	/*
	手模共25个DoF，其中全局全局坐标系和关节局部坐标系都采用左手坐标系，全局坐标系与Kinect坐标系的唯一区别是z轴指向相反
	大拇指仅考虑指端关节（DIP）和指掌关节（MCP），其中指端关节1个自由度——弯曲/伸展，指掌关节2个自由度——弯曲/伸展（绕x轴旋转）和内收/外展（绕z轴旋转）
	四指指端关节（DIP）1个自由度，指间关节（PIP）1个自由度，指掌关节（MCP）2个自由度
	*/
	double xd;            /*全局平移之xd，单位为毫米*/
	double yd;           /*全局平移之yd，单位为毫米*/
	double zd;         /*全局平移之zd，单位为毫米*/

	double yaw;    /*绕x轴旋转，单位角度*/
	double pitch;  /*绕y轴旋转，单位角度*/
	double roll;   /*绕z轴旋转，单位角度*/

	double mcp_xT;    /*大拇指指掌关节绕x轴旋转角度*/
	double mcp_zT;    /*大拇指指掌关节绕z轴旋转角度*/
	double dipT;      /*大拇指指端关节绕x轴旋转角度*/

	double mcp_xI;    /*食指指掌关节绕x轴旋转*/
	double mcp_zI;    /*食指指掌关节绕z轴旋转*/
	double pipI;      /*食指指间关节*/
	double dipI;      /*食指指端关节*/

	double mcp_xM;    /*中指*/
	double mcp_zM;
	double pipM;
	double dipM;

	double mcp_xR;    /*无名指*/
	double mcp_zR;
	double pipR;
	double dipR;

	double mcp_xL;    /*小拇指*/
	double mcp_zL;
	double pipL;
	double dipL;

private:
	double low_range_dof[25];  /*手模自由度的取值范围,单位mm*/
	double up_range_dof[25];
	HandSkeletalStruct handStru;
private:
	/*
	长方体-圆柱体几何模型：
	每个指节一个圆柱体，圆柱体高为指节长度，半径为手指半径
	手掌建模为2个长方体，一个长宽高(60，20，10)，另一个长宽高(80，20，62)
	*/
	/*几何模型的初始状态：*/
	Cylinder initCylind[14];  /*依次是T近指节、T远指节、I近指节、I中指节、I远指节、M近指节、M中指节、……、L远指节*/
	Cuboid initCub0;   /*长宽高为(60，20，10)的长方体*/
	Cuboid initCub1;   /*长宽高为(80，20，62)的长方体*/

	/*几何模型执行手势后的状态：*/
	Cylinder changedCylind[14];
	Cuboid changedCub0;
	Cuboid changedCub1;
public:
	HandModel();//赋值DoF为0，赋值low_range_dof[25]和up_range_dof[25]
	void calChangedPose(double* DoF, int num); /*赋值新的DoF[]，并计算changedCylind[]、changedCub0、changedCub1*/
	void get_joint_xyz_in_msra14_order(double* joint_xyz, int num);
	
	void showInitPose()const;
	void showChangedPose()const;
	const double* getLowRangeDof()const;
	const double* getUpRangeDof()const;
	const Cylinder* getChangedCylind()const;
	const Cuboid getChangedCub0()const;
	const Cuboid getChangedCub1()const;
private:
	void getPosChanged(const Position& posInit, const Mat& mat, Position& posChanged) const;	//人手运动建模
	void initialize(); /*赋值initCylind[]、initCub0、initCub1、changedCylind[]的半径*/
	void setDoF(double* DoF, int num);  //设置DoF
	void calChangedPose(); //根据成员变量DoF当前值计算changedCylind[]、changedCub0、changedCub1
};

