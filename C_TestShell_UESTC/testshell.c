#include "testshellfunc.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    char input_command[MAX_INPUT_COMMAND];  //�����ָ��
    //�������е�PCB����
    all_process = allPCBList();
    //�����������ȼ���Ľ��̶���
    head_process_zero_list = creatPriorityZeroList();
    head_process_one_list = creatPriorityOneList();
    head_process_two_list = creatPriorityTwoList();
    //���������ȴ���Դ�Ľ��̶���
    head_resource_R1_list = creatResourceR1List();
    head_resource_R2_list = creatResourceR2List();
    head_resource_R3_list = creatResourceR3List();
    head_resource_R4_list = creatResourceR4List();
    //init������
    current_process = init();

    //do until get char[] = "exit"
    printf("input command & end with \"exit\"\n");
    while(1){
        gets(input_command);  //�������л���û�����������
        if(strcmp(input_command, "exit") == 0){
            printf("TestShell exit!");
            break;  //�û�����"exit"���˳�����
        }
        analyseAndDoCommand(input_command);
    }
/**
    �������-start
**/
//    creatProcess("x", 1);
//    checkProInfo("x");
//    creatProcess("p", 1);
//    creatProcess("q", 1);
//    creatProcess("r", 1);
//    Time_out();
//    requestResource(current_process, 1, 1);
//    Time_out();
//    requestResource(current_process, 2, 3);
//    Time_out();
//    requestResource(current_process, 3, 3);
//    Time_out();
//    Time_out();
//    requestResource(current_process, 2, 1);
//    requestResource(current_process, 3, 2);
//    listAllPCBList();
//    requestResource(current_process, 1, 2);
//    Time_out();
//    deleteProcess("q");
//    Time_out();
//    Time_out();
/**
    �������-end
**/

    //�ͷŸ���������ռ�Ŀռ�
    destoryAllList();
    return 0;
}
