#pragma once
#include<iostream>
#include<opencv.hpp>
#include"data_struct.h"
#include"geometry.h"
using cv::Mat;

class HandSkeletalStruct   /*���ֹ����ṹ��ָ�ڳ��ȡ���ָָ�ƹؽ�������ֲ��ؽ�����ϵ�е�y���ꡢ��ָ�뾶�����ƺ��*/
{
public:
	HandSkeletalStruct() {}
public:
	/*�͵س�ʼ����C++ 11������*/
	const double len01 = 20.0+10;//��Ĵָ��ָ�ڳ���,��λmm
	const double len02 = 20.0;  //��ĴָԶָ�ڳ���

	const double len11 = 28.0;  //ʳָ��ָ�ڳ���
	const double len12 = 21.0;  //ʳָ��ָ�ڳ���
	const double len13 = 19.0;  //ʳָԶָ�ڳ���

	const double len21 = 33.0;  //��ָ
	const double len22 = 22.0;
	const double len23 = 20.0;

	const double len31 = 29.0;  //����ָ
	const double len32 = 19.0;
	const double len33 = 18.0;

	const double len41 = 21.0;  //СĴָ
	const double len42 = 13.0;
	const double len43 = 14.0;

	const double highMCP0 = 32;  //��Ĵָָ�ƹؽ�������ֲ��ؽ�����ϵ�е�y���� 
	const double highMCP1 = 68.3759+10;  //ʳָָ�ƹؽ�������ֲ��ؽ�����ϵ�е�y���� 
	const double highMCP2 = 68.3759+10;  //��ָ
	const double highMCP3 = 68.3759+10;  //����ָ
	const double highMCP4 = 60.8759+10;  //СĴָ

	const double fingerRidiusT = 9.0;  //��Ĵָ�뾶����λ����
	const double fingerRidiusI = 9.0;
	const double fingerRidiusM = 9.0;
	const double fingerRidiusR = 9.0;
	const double fingerRidiusL = 9.0;

	const double palmThick = 18.0;     //���ƺ��
};

class HandModel  /*����ģ�ͣ���������ģ�ͺ��˶���ģ�����ɶ�*/
{
private:
	/*
	��ģ��25��DoF������ȫ��ȫ������ϵ�͹ؽھֲ�����ϵ��������������ϵ��ȫ������ϵ��Kinect����ϵ��Ψһ������z��ָ���෴
	��Ĵָ������ָ�˹ؽڣ�DIP����ָ�ƹؽڣ�MCP��������ָ�˹ؽ�1�����ɶȡ�������/��չ��ָ�ƹؽ�2�����ɶȡ�������/��չ����x����ת��������/��չ����z����ת��
	��ָָ�˹ؽڣ�DIP��1�����ɶȣ�ָ��ؽڣ�PIP��1�����ɶȣ�ָ�ƹؽڣ�MCP��2�����ɶ�
	*/
	double xd;            /*ȫ��ƽ��֮xd����λΪ����*/
	double yd;           /*ȫ��ƽ��֮yd����λΪ����*/
	double zd;         /*ȫ��ƽ��֮zd����λΪ����*/

	double yaw;    /*��x����ת����λ�Ƕ�*/
	double pitch;  /*��y����ת����λ�Ƕ�*/
	double roll;   /*��z����ת����λ�Ƕ�*/

	double mcp_xT;    /*��Ĵָָ�ƹؽ���x����ת�Ƕ�*/
	double mcp_zT;    /*��Ĵָָ�ƹؽ���z����ת�Ƕ�*/
	double dipT;      /*��Ĵָָ�˹ؽ���x����ת�Ƕ�*/

	double mcp_xI;    /*ʳָָ�ƹؽ���x����ת*/
	double mcp_zI;    /*ʳָָ�ƹؽ���z����ת*/
	double pipI;      /*ʳָָ��ؽ�*/
	double dipI;      /*ʳָָ�˹ؽ�*/

	double mcp_xM;    /*��ָ*/
	double mcp_zM;
	double pipM;
	double dipM;

	double mcp_xR;    /*����ָ*/
	double mcp_zR;
	double pipR;
	double dipR;

	double mcp_xL;    /*СĴָ*/
	double mcp_zL;
	double pipL;
	double dipL;

private:
	double low_range_dof[25];  /*��ģ���ɶȵ�ȡֵ��Χ,��λmm*/
	double up_range_dof[25];
	HandSkeletalStruct handStru;
private:
	/*
	������-Բ���弸��ģ�ͣ�
	ÿ��ָ��һ��Բ���壬Բ�����Ϊָ�ڳ��ȣ��뾶Ϊ��ָ�뾶
	���ƽ�ģΪ2�������壬һ�������(60��20��10)����һ�������(80��20��62)
	*/
	/*����ģ�͵ĳ�ʼ״̬��*/
	Cylinder initCylind[14];  /*������T��ָ�ڡ�TԶָ�ڡ�I��ָ�ڡ�I��ָ�ڡ�IԶָ�ڡ�M��ָ�ڡ�M��ָ�ڡ�������LԶָ��*/
	Cuboid initCub0;   /*�����Ϊ(60��20��10)�ĳ�����*/
	Cuboid initCub1;   /*�����Ϊ(80��20��62)�ĳ�����*/

	/*����ģ��ִ�����ƺ��״̬��*/
	Cylinder changedCylind[14];
	Cuboid changedCub0;
	Cuboid changedCub1;
public:
	HandModel();//��ֵDoFΪ0����ֵlow_range_dof[25]��up_range_dof[25]
	void calChangedPose(double* DoF, int num); /*��ֵ�µ�DoF[]��������changedCylind[]��changedCub0��changedCub1*/
	void get_joint_xyz_in_msra14_order(double* joint_xyz, int num);
	
	void showInitPose()const;
	void showChangedPose()const;
	const double* getLowRangeDof()const;
	const double* getUpRangeDof()const;
	const Cylinder* getChangedCylind()const;
	const Cuboid getChangedCub0()const;
	const Cuboid getChangedCub1()const;
private:
	void getPosChanged(const Position& posInit, const Mat& mat, Position& posChanged) const;	//�����˶���ģ
	void initialize(); /*��ֵinitCylind[]��initCub0��initCub1��changedCylind[]�İ뾶*/
	void setDoF(double* DoF, int num);  //����DoF
	void calChangedPose(); //���ݳ�Ա����DoF��ǰֵ����changedCylind[]��changedCub0��changedCub1
};

