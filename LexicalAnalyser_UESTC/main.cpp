#include<stdio.h>
#include<Windows.h>
#include "LexicalAnalyer.h"

int main()
{
	char* inputFile = "f:\\input.pas";  //包含待分析源程序的文件路径和文件名
    char outputFile[MAX_COUNT] = "f:\\result.dyd";  //词法分析结果--二元式
    char errputFile[MAX_COUNT] = "f:\\errinfo.err";  //错误输出

    if (freopen(inputFile, "r", stdin) == NULL || freopen(outputFile, "w", stdout) == NULL || freopen(errputFile, "w", stderr) == NULL){
        printf("输出重定向出错！\n");
        exit(1);
    } else{
        while (LexAnalyze()){
                //do nothing
		}
		fclose(stdin);
        fclose(stdout);
        fclose(stderr);
        printf("LexicalAnalyser is OK!");
        return 0;
    }
}
