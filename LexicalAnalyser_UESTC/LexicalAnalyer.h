#include<stdio.h>
#include<string.h>
#include<Windows.h>
#define MAX_COUNT 1024
#define ILLEGAL_CHAR_ERR 1
#define UNKNOWN_OPERATOR_ERR 2

/*�ӱ�׼��������һ���ǿհ��ַ�(���з�����)*/
char getnbc();
/*�ж�character�Ƿ�Ϊ��ĸ*/
bool letter(char character);
/*�ж�character�Ƿ�Ϊ����*/
bool digit(char character);
/*�����ַ�*/
void retract(char& character);
/*���ر����ֵĶ�Ӧ�ֱ�*/
int reserve(char* token);
/*���ر�ʶ���Ķ�Ӧ�ֱ�*/
int symbol();
/*���س����Ķ�Ӧ�ֱ�*/
int constant();
/*���ո�ʽ������ʷ��ź��ֱ�*/
void output(const char* token, int kindNum);
/*�ʷ�����������ÿ����һ��ʶ��һ������*/
bool LexAnalyze();
/*���·��*/
void getPath(char* in, char* out);
/*����ļ�������������չ*/
void getFilename(char* in, char* out);
/*��ʼ�����������������ļ���ַ���������롢����������ļ�������׼�����ض��������ļ�������׼����ض�������ļ�����׼�����ض��򵽴����ļ�*/
bool init(int argc, char* argv[]);
