#include "testshellfunc.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    char input_command[MAX_INPUT_COMMAND];  //输入的指令
    //创建所有的PCB链表
    all_process = allPCBList();
    //创建各个优先级别的进程队列
    head_process_zero_list = creatPriorityZeroList();
    head_process_one_list = creatPriorityOneList();
    head_process_two_list = creatPriorityTwoList();
    //创建各个等待资源的进程队列
    head_resource_R1_list = creatResourceR1List();
    head_resource_R2_list = creatResourceR2List();
    head_resource_R3_list = creatResourceR3List();
    head_resource_R4_list = creatResourceR4List();
    //init根进程
    current_process = init();

    //do until get char[] = "exit"
    printf("input command & end with \"exit\"\n");
    while(1){
        gets(input_command);  //从命令行获得用户的输入命令
        if(strcmp(input_command, "exit") == 0){
            printf("TestShell exit!");
            break;  //用户输入"exit"则退出程序
        }
        analyseAndDoCommand(input_command);
    }
/**
    测试命令集-start
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
    测试命令集-end
**/

    //释放各个链表所占的空间
    destoryAllList();
    return 0;
}
