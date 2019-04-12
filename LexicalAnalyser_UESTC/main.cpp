#include<stdio.h>
#include<Windows.h>
#include "LexicalAnalyer.h"

int main()
{
	char* inputFile = "f:\\input.pas";  //����������Դ������ļ�·�����ļ���
    char outputFile[MAX_COUNT] = "f:\\result.dyd";  //�ʷ��������--��Ԫʽ
    char errputFile[MAX_COUNT] = "f:\\errinfo.err";  //�������

    if (freopen(inputFile, "r", stdin) == NULL || freopen(outputFile, "w", stdout) == NULL || freopen(errputFile, "w", stderr) == NULL){
        printf("����ض������\n");
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
