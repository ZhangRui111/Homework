#include<stdio.h>
#include<string.h>
#include<Windows.h>
#define MAX_COUNT 1024
#define ILLEGAL_CHAR_ERR 1
#define UNKNOWN_OPERATOR_ERR 2

/*从标准输入读入第一个非空白字符(换行符除外)*/
char getnbc();
/*判断character是否为字母*/
bool letter(char character);
/*判断character是否为数字*/
bool digit(char character);
/*回退字符*/
void retract(char& character);
/*返回保留字的对应种别*/
int reserve(char* token);
/*返回标识符的对应种别*/
int symbol();
/*返回常数的对应种别*/
int constant();
/*按照格式输出单词符号和种别*/
void output(const char* token, int kindNum);
/*词法分析函数，每调用一次识别一个符号*/
bool LexAnalyze();
/*获得路径*/
void getPath(char* in, char* out);
/*获得文件名，不包括扩展*/
void getFilename(char* in, char* out);
/*初始化函数，接收输入文件地址，并打开输入、输出、错误文件、将标准输入重定向到输入文件，将标准输出重定向到输出文件，标准错误重定向到错误文件*/
bool init(int argc, char* argv[]);
