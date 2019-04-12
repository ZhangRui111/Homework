#include "testshellfunc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

PCBNode * init()
{
    PCBNode * newProcess;
    int i;
    newProcess = (PCBNode *)malloc(sizeof(PCBNode));
    newProcess->Process_ID = "init";
    for(i=0;i<4;i++){
        newProcess->Other_resources[i] = 0;
    }
    for(i=0;i<4;i++){
        resource[i] = (i+1);
    }
    newProcess->status = STATUS_RUNNING;
    newProcess->parent = NULL;
    newProcess->sonNum = 0;
    for(i = 0;i<10;i++){
        newProcess->son[i] = NULL;
    }
    newProcess->Priority = 0;

    insertAllPCBList(newProcess);
    insertProcessList(newProcess->Priority,newProcess);  //保存进程到相应的优先级链表

    printf("init\n");
    return newProcess;
}

PCBNode * allPCBList()
{
    PCBNode * head;
    if((head=(PCBNode *)malloc(sizeof(PCBNode)))==NULL){
        printf("Can't allocate space to allProcess List!!");
        exit(0);
    }
    head -> Process_ID = "head_node";
    head -> status = STATUS_READY;
    head -> next = NULL;
    return head;
}

insertAllPCBList(PCBNode * node)
{
    PCBNode * head,*p;
    head = all_process;
    while (head->next != NULL){
        head = head->next;
    }
    head->next = node;

    node->next = NULL;
}

PCBNode * creatProcess(char * pid, int priority)
{
    if(priority<0||priority>2) {
        printf("The new process priority is valid!\n");
        exit(0);
    }
    //create PCB data structure
    PCBNode * newProcess;
    int i;
    newProcess = (PCBNode *)malloc(sizeof(PCBNode));
    newProcess->Process_ID = pid;
    for(i=0;i<4;i++){
        newProcess->Other_resources[i] = 0;
    }
    newProcess->status = STATUS_READY;
    newProcess->parent = current_process;
    newProcess->sonNum = 0;
    for(i = 0;i<10;i++){
        newProcess->son[i] = NULL;
    }
    newProcess->Priority = priority;
    newProcess->next = NULL;

    PCBNode * parent;
    parent = current_process;
    parent->son[parent->sonNum] = newProcess;  //新建进程连接到父进程的子节点上
    parent->sonNum ++;

    insertAllPCBList(newProcess);  //保存PCB到allPCBList*
    insertProcessList(newProcess->Priority,newProcess);  //保存进程到相应的优先级链表
    Scheduler();
    //travelAllPCBList();
    return newProcess;
}

void deleteProcess(char * pid)
{
    PCBNode *ppp, *pp, * p = getProcessById(pid);  //*p,*pp,*ppp指向要撤销的进程的PCB结构
    pp = p;
    ppp = pp;
    releaseResource(p);
    removeProcessById(ppp->Process_ID);  //从PCB链表中删除相应的PCB

//    if(p->son != NULL){  //撤销子进程
//        while(p->son != NULL){
//            p=p->son;
//            removeFromHead(pp);
//            pp = p;
//        }
//        removeFromHead(pp);
//    } else {  //没有子进程
//        removeFromHead(ppp);
//    }
    removeFromHead(ppp);
    Scheduler();
}

PCBNode * getProcessById(char * Pid)
{
    PCBNode *p;
    p = all_process;
    while(p->next != NULL) {
        p = p->next;
        if(strcmp(Pid,p->Process_ID) == 0){
            return p;
        }
    }
    return NULL;
}

void removeProcessById(char * Pid)
{
    PCBNode *p,*q;
    p = all_process;
    q = p;
    while(p->next != NULL){
        p = p->next;
        if(strcmp(Pid,p->Process_ID) == 0){
            q->next = p->next;
            free(p);
        }
        q = p;
    }
}

analyseAndDoCommand(char input_command[])
{
    char firstChar;  //输入命令的第一个字母
    int length,i;
    char *firstArgs,*secondArgs,*thirdArgs  ;  //第一个，第二个，第三个参数
    char *delim = ' ';  //分隔符
    firstArgs = strtok(input_command, " ");
    secondArgs = strtok(NULL, " ");
    thirdArgs = strtok(NULL, " ");
    //打印分割后的各个参数
//    if(firstArgs)    printf("firstArgs:%s\n", firstArgs);
//    if(secondArgs)    printf("secondArgs:%s\n", secondArgs);
//    if(thirdArgs)    printf("thirdArgs:%s\n", thirdArgs);

    if (strcmp(firstArgs ,"cr") == 0) {
        //"cr"命令
        if(strcmp(secondArgs,"x")==0){
            new_process = creatProcess("x", atoi(thirdArgs));
            //travelAllPCBList();
        }
        if(strcmp(secondArgs,"p")==0){
            new_process = creatProcess("p", atoi(thirdArgs));
            //travelAllPCBList();
        }
        if(strcmp(secondArgs,"q")==0){
            new_process = creatProcess("q", atoi(thirdArgs));
            //travelAllPCBList();
        }
        if(strcmp(secondArgs,"r")==0){
            new_process = creatProcess("r", atoi(thirdArgs));
            //travelAllPCBList();
        }
        //遍历三个优先级链表
        //travelProcessList(head_process_zero_list);
        //travelProcessList(head_process_one_list);
        //travelProcessList(head_process_two_list);
    } else if (strcmp(firstArgs ,"req") == 0){
        //"req"命令
        if(strcmp(secondArgs,"R1")==0){
            requestResource(current_process, 0, atoi(thirdArgs));
        }
        if(strcmp(secondArgs,"R2")==0){
            requestResource(current_process, 1, atoi(thirdArgs));
        }
        if(strcmp(secondArgs,"R3")==0){
            requestResource(current_process, 2, atoi(thirdArgs));
        }
        if(strcmp(secondArgs,"R4")==0){
            requestResource(current_process, 3, atoi(thirdArgs));
        }
    } else if (strcmp(firstArgs ,"de") == 0){
        //"de"命令
        if(strcmp(secondArgs,"x")==0){
            deleteProcess("x");
        }
        if(strcmp(secondArgs,"p")==0){
            deleteProcess("p");
        }
        if(strcmp(secondArgs,"q")==0){
            deleteProcess("q");
        }
        if(strcmp(secondArgs,"r")==0){
            deleteProcess("r");
        }
    } else if (strcmp(firstArgs ,"to") == 0){
       //"to"命令
       Time_out();
    } else if (strcmp(firstArgs ,"rel") == 0){
       //"rel"命令
       releaseResource(current_process);
    } else if (strcmp(firstArgs ,"listPro") == 0){
       //-list all processes and their status
       listAllPCBList();
    } else if (strcmp(firstArgs ,"listRec") == 0){
       //-list all resources and their status
      listAllRec();
    } else if (strcmp(firstArgs ,"checkPro") == 0){
       //-provide information about a given process
       checkProInfo(secondArgs);
    } else {
        //无效命令
        printf("valid command!    Please input again!\n");
    }
}

void Time_out()
{
    removeFromHead(current_process);  //将当前正在运行的进程从相应的链表的表头删除
    insertProcessList(current_process->Priority, current_process);  //将当前的进程插入到相应的优先级链表尾部
    Scheduler();
}

void Scheduler()
{
    char * highestProcessId = getHighestProcess();
    //printf("highestProcessId:%s\n",highestProcessId);
    PCBNode * highestProcess = getProcessById(highestProcessId);
    //printf("PCBid:%s\n",highestProcess->Process_ID);
    if(current_process->Priority < highestProcess->Priority) {
        if(current_process->status == STATUS_RUNNING){
            current_process->status = STATUS_READY;  //修改上一个进程的状态为READY
        }
        current_process = highestProcess;  //进程抢占
        current_process->status = STATUS_RUNNING;
        printf("current process:%s\n",current_process->Process_ID);
    } else if(current_process->status != STATUS_RUNNING ||
        current_process == NULL) {
        current_process = highestProcess;  //进程抢占
        current_process->status = STATUS_RUNNING;
        printf("current process:%s\n",current_process->Process_ID);
    } else if(current_process->Priority >= highestProcess->Priority){
        current_process->status = STATUS_RUNNING;
        printf("current process:%s\n",current_process->Process_ID);
    } else {
        printf("Something error while Scheduler!\n");
    }
}

char * getHighestProcess()
{
    Process_List_Note * priority_two,* priority_one, *priority_zero;
    priority_zero = head_process_zero_list;
    priority_one = head_process_one_list;
    priority_two = head_process_two_list;
    if(priority_two->next != NULL){
        //printf("priority_two");
        return priority_two->next->process_id;
    } else if (priority_one->next != NULL){
        //printf("priority_one");
        return priority_one->next->process_id;
    } else if (priority_zero->next != NULL){
        //printf("priority_zero");
        return priority_zero->next->process_id;
    } else {
        printf("There is not a process ready in the priority list!\n");
        return NULL;
    }
}

void insertProcessList(int priority, PCBNode * node)
{
    switch (priority)
    {
    case 0:
        insertProcessListZero(node);
        //travelProcessList(head_process_zero_list);
        break;
    case 1:
        insertProcessListOne(node);
        //travelProcessList(head_process_one_list);
        break;
    case 2:
        insertProcessListTwo(node);
        //travelProcessList(head_process_two_list);
        break;
    default:
        printf("The new process priority is valid!\n");
        break;
    }
}

void removeFromHead(PCBNode * node)
{
    PCBNode * p;
    p = node;
    p->status = STATUS_READY;
    switch (node->Priority)
    {
    case 0:
        removeFromHeadZero(node->Process_ID);
        break;
    case 1:
        removeFromHeadOne(node->Process_ID);
        break;
    case 2:
        removeFromHeadTwo(node->Process_ID);
        break;
    default:
        printf("The current process priority is valid!\n");
        break;
    }
}

Process_List_Note * creatPriorityZeroList()
{
    Process_List_Note * head;
    if((head=(Process_List_Note *)malloc(sizeof(Process_List_Note)))==NULL){
        printf("Can't allocate space to process_0 List!!");
        exit(0);
    }
    head -> process_id = "head_node_Pr0";    //头结点的进程id是-1(无效进程)
    head -> next = NULL;
    return head;
}

void insertProcessListZero(PCBNode * node)
{
   //printf("insert process in List zero!!");
    Process_List_Note * head,*p;
    head = head_process_zero_list;
    while (head->next != NULL){
        head = head->next;
    }
    if((p=(Process_List_Note *)malloc(sizeof(Process_List_Note)))==NULL){
        printf("Can't allocate new space to process_0 List!!");
        exit(0);
    }
    p->process_id = node -> Process_ID;
    p->next = NULL;
    head->next = p;
}

void removeFromHeadZero(char * ProId)
{
    Process_List_Note *p,*q;
    p = head_process_zero_list;
    q = p;
    while(p->next != NULL){
        p = p->next;
        if(strcmp(p->process_id,ProId) == 0){
            q->next = p->next;
            free(p);
            break;
        }
        q = p;
    }
}

Process_List_Note * creatPriorityOneList()
{
    Process_List_Note * head;
    if((head=(Process_List_Note *)malloc(sizeof(Process_List_Note)))==NULL){
        printf("Can't allocate space to process_1 List!!");
        exit(0);
    }
    head -> process_id = "head_node_Pr1";    //头结点的进程id是-1(无效进程)
    head -> next = NULL;
    return head;
}

void insertProcessListOne(PCBNode * node)
{
    //printf("insert process in List one!!");
    Process_List_Note * head,*p;
    head = head_process_one_list;
    while (head->next != NULL){
        head = head->next;
    }
    if((p=(Process_List_Note *)malloc(sizeof(Process_List_Note)))==NULL){
        printf("Can't allocate new space to process_1 List!!");
        exit(0);
    }
    p->process_id = node -> Process_ID;
    p->next = NULL;
    head->next = p;
}

void removeFromHeadOne(char * ProId)
{
    Process_List_Note *p,*q;
    p = head_process_one_list;
    q = p;
    while(p->next != NULL){
        p = p->next;
        if(strcmp(p->process_id,ProId) == 0){
            q->next = p->next;
            free(p);
            break;
        }
        q = p;
    }
}

Process_List_Note * creatPriorityTwoList()
{
    Process_List_Note * head;
    if((head=(Process_List_Note *)malloc(sizeof(Process_List_Note)))==NULL){
        printf("Can't allocate space to process_2 List!!");
        exit(0);
    }
    head -> process_id = "head_node_Pr2";    //头结点的进程id是-1(无效进程)
    head -> next = NULL;
    return head;
}

void insertProcessListTwo(PCBNode * node)
{
    //printf("insert process in List two!!");
    Process_List_Note * head,*p;
    head = head_process_two_list;
    while (head->next != NULL){
        head = head->next;
    }
    if((p=(Process_List_Note *)malloc(sizeof(Process_List_Note)))==NULL){
        printf("Can't allocate new space to process_2 List!!");
        exit(0);
    }
    p->process_id = node -> Process_ID;
    p->next = NULL;
    head->next = p;
}

void removeFromHeadTwo(char * ProId)
{
    Process_List_Note *p,*q;
    p = head_process_two_list;
    q = p;
    while(p->next != NULL){
        p = p->next;
        if(strcmp(p->process_id,ProId) == 0){
            q->next = p->next;
            free(p);
            break;
        }
        q = p;
    }
}

//RIDNode * creatResourcesList()
//{
//    RIDNode * head,*currentPtr,*newPtr;
//    int i;
//    if((head=(RIDNode *)malloc(sizeof(RIDNode)))==NULL){
//        printf("Can't allocate space to resources List!!");
//        exit(0);
//    }
//    head -> Resource_ID = 0;
//    head -> Status = -1;
//    head -> next = NULL;
//    currentPtr = head;
//    for(i = 1 ; i < 5; i ++){
//        if((newPtr= (RIDNode *) malloc(sizeof(RIDNode)))==NULL){
//                printf("Can't allocate space to resources List!!");
//                exit(0);
//        }
//        newPtr -> Resource_ID = i;
//        newPtr -> Status = i;
//        newPtr -> next = NULL;
//        currentPtr -> next = newPtr;
//        currentPtr = newPtr;
//    }
//    return head;
//}

//ProResList * creatProResList(int res1,int res2,int res3,int res4)
//{
//    ProResList * head,*currentPtr,*newPtr;
//    int i, arr[4];
//    arr[0] = res1;arr[1] = res2;arr[2] = res3;arr[3] = res4;
//    if((head=(ProResList *)malloc(sizeof(ProResList)))==NULL){
//        printf("Can't allocate space to resources List!!");
//        exit(0);
//    }
//    head -> Resource_ID = 0;
//    head -> Resource_Number = -1;
//    head -> next = NULL;
//    currentPtr = head;
//    for(i = 0 ; i < 4; i ++){
//        if((newPtr= (ProResList *) malloc(sizeof(ProResList)))==NULL){
//                printf("Can't allocate space to resources List!!");
//                exit(0);
//        }
//        newPtr -> Resource_ID = (i+1);
//        newPtr -> Resource_Number = arr[i];
//        newPtr -> next = NULL;
//        currentPtr -> next = newPtr;
//        currentPtr = newPtr;
//    }
//    return head;
//}

void releaseResource(PCBNode * node)
{
    int i;
    for(i = 0;i < 4;i++){
        resource[i] += node->Other_resources[i];
        checkIfResourceListOk();
    }
}

void requestResource(PCBNode * pro, int rid,int num)
{
    if(num <= resource[rid]) {
        pro->Other_resources[rid] = num;
        resource[rid] -= num;
        pro->status=STATUS_RUNNING;
        printf("current process:%s\n",pro->Process_ID);
    } else {
        removeFromHead(pro);  //将当前正在运行的进程从相应的链表的表头删除
        insertResourceReList(pro, rid,num );
        pro->status = STATUS_BLOCK;
        Scheduler();
    }
}

void checkIfResourceListOk()
{
    Waiting_List_Node * firstBlockProR1, * firstBlockProR2, * firstBlockProR3, * firstBlockProR4;
    if(head_resource_R1_list->next != NULL){
        Waiting_List_Node * firstBlockProR1 = head_resource_R1_list->next;
        if(firstBlockProR1->requireNum <= resource[0]){
            int require = firstBlockProR1->requireNum;
            //printf("firstBlockR1Process:%s\n",firstBlockProR1->process_id);
            PCBNode * pro = getProcessById(firstBlockProR1->process_id);
            pro->status = STATUS_READY;
            removeResourceR1List(firstBlockProR1->process_id);
            insertProcessList(pro->Priority,pro);
            resource[0] -= require;
        }
    }
    if(head_resource_R2_list->next != NULL){
        Waiting_List_Node * firstBlockProR2 = head_resource_R2_list->next;
        if(firstBlockProR2->requireNum <= resource[1]){
            int require = firstBlockProR2->requireNum;
            //printf("firstBlockR2Process:%s\n",firstBlockProR2->process_id);
            PCBNode * pro = getProcessById(firstBlockProR2->process_id);
            pro->status = STATUS_READY;
            removeResourceR2List(firstBlockProR2->process_id);
            insertProcessList(pro->Priority,pro);
            resource[1] -= require;
        }
    }
    if(head_resource_R3_list->next != NULL){
        Waiting_List_Node * firstBlockProR3 = head_resource_R3_list->next;
        if(firstBlockProR3->requireNum <= resource[2]){
            int require = firstBlockProR3->requireNum;
            PCBNode * pro = getProcessById(firstBlockProR3->process_id);
            pro->status = STATUS_READY;
            removeResourceR3List(firstBlockProR3->process_id);
            insertProcessList(pro->Priority,pro);
            resource[2] -= require;
        }
    }
    if(head_resource_R4_list->next != NULL){
        Waiting_List_Node * firstBlockProR4 = head_resource_R4_list->next;
        if(firstBlockProR4->requireNum <= resource[3]){
            int require = firstBlockProR4->requireNum;
            //printf("firstBlockR4Process:%s\n",firstBlockProR4->process_id);
            PCBNode * pro = getProcessById(firstBlockProR4->process_id);
            pro->status = STATUS_READY;
            removeResourceR4List(firstBlockProR4->process_id);
            insertProcessList(pro->Priority,pro);
            resource[3] -= require;
        }
    }
}

insertResourceReList(PCBNode * node, int rid, int num)
{
    switch (rid)
    {
    case 0:
        insertResourceR1List(node,rid,num);
        break;
    case 1:
        insertResourceR2List(node,rid,num);
        break;
    case 2:
        insertResourceR3List(node,rid,num);
        break;
    case 3:
        insertResourceR4List(node,rid,num);
        break;
    default:
        printf("rid is valid\n");
        break;
    }
}

Waiting_List_Node * creatResourceR1List()
{
    Waiting_List_Node * head;
    if((head=(Waiting_List_Node *)malloc(sizeof(Waiting_List_Node)))==NULL){
        printf("Can't allocate space to Resource1 List!!");
        exit(0);
    }
    head -> process_id = "head_node_Re0";    //头结点
    head -> next = NULL;
    return head;
}
void insertResourceR1List(PCBNode * node,int rid, int num)
{
    Waiting_List_Node * head,*p;
    head = head_resource_R1_list;
    while (head->next != NULL){
        head = head->next;
    }
    if((p=(Waiting_List_Node *)malloc(sizeof(Waiting_List_Node)))==NULL){
        printf("Can't allocate new space to Resource1 List!!");
        exit(0);
    }
    p->process_id = node -> Process_ID;
    p->Rid = rid;
    p->requireNum = num;
    p->next = NULL;
    head->next = p;
}
void removeResourceR1List(char *ProId)
{
    Waiting_List_Node *p,*q;
    p = head_resource_R1_list;
    q = p;
    while(p->next != NULL){
        p = p->next;
        if(strcmp(p->process_id,ProId) == 0){
            q->next = p->next;
            free(p);
            break;
        }
        q = p;
    }
}
Waiting_List_Node * creatResourceR2List()
{
    Waiting_List_Node * head;
    if((head=(Waiting_List_Node *)malloc(sizeof(Waiting_List_Node)))==NULL){
        printf("Can't allocate space to Resource1 List!!");
        exit(0);
    }
    head -> process_id = "head_node_Re0";    //头结点
    head -> next = NULL;
    return head;
}
void insertResourceR2List(PCBNode * node, int rid, int num)
{
    Waiting_List_Node * head,*p;
    head = head_resource_R2_list;
    while (head->next != NULL){
        head = head->next;
    }
    if((p=(Waiting_List_Node *)malloc(sizeof(Waiting_List_Node)))==NULL){
        printf("Can't allocate new space to Resource1 List!!");
        exit(0);
    }
    p->process_id = node -> Process_ID;
    p->Rid = rid;
    p->requireNum = num;
    p->next = NULL;
    head->next = p;
}
void removeResourceR2List(char *ProId)
{
    Waiting_List_Node *p,*q;
    p = head_resource_R2_list;
    q = p;
    while(p->next != NULL){
        p = p->next;
        if(strcmp(p->process_id,ProId) == 0){
            q->next = p->next;
            free(p);
            break;
        }
        q = p;
    }
}
Waiting_List_Node * creatResourceR3List()
{
    Waiting_List_Node * head;
    if((head=(Waiting_List_Node *)malloc(sizeof(Waiting_List_Node)))==NULL){
        printf("Can't allocate space to Resource1 List!!");
        exit(0);
    }
    head -> process_id = "head_node_Re0";    //头结点
    head -> next = NULL;
    return head;
}
void insertResourceR3List(PCBNode * node, int rid, int num)
{
    Waiting_List_Node * head,*p;
    head = head_resource_R3_list;
    while (head->next != NULL){
        head = head->next;
    }
    if((p=(Waiting_List_Node *)malloc(sizeof(Waiting_List_Node)))==NULL){
        printf("Can't allocate new space to Resource1 List!!");
        exit(0);
    }
    p->process_id = node -> Process_ID;
    p->Rid = rid;
    p->requireNum = num;
    p->next = NULL;
    head->next = p;
}
void removeResourceR3List(char *ProId)
{
    Waiting_List_Node *p,*q;
    p = head_resource_R3_list;
    q = p;
    while(p->next != NULL){
        p = p->next;
        if(strcmp(p->process_id,ProId) == 0){
            q->next = p->next;
            free(p);
            break;
        }
        q = p;
    }
}
Waiting_List_Node * creatResourceR4List()
{
    Waiting_List_Node * head;
    if((head=(Waiting_List_Node *)malloc(sizeof(Waiting_List_Node)))==NULL){
        printf("Can't allocate space to Resource1 List!!");
        exit(0);
    }
    head -> process_id = "head_node_Re0";    //头结点
    head -> next = NULL;
    return head;
}
void insertResourceR4List(PCBNode * node, int rid, int num)
{
    Waiting_List_Node * head,*p;
    head = head_resource_R4_list;
    while (head->next != NULL){
        head = head->next;
    }
    if((p=(Waiting_List_Node *)malloc(sizeof(Waiting_List_Node)))==NULL){
        printf("Can't allocate new space to Resource1 List!!");
        exit(0);
    }
    p->process_id = node -> Process_ID;
    p->Rid = rid;
    p->requireNum = num;
    p->next = NULL;
    head->next = p;
}
void removeResourceR4List(char *ProId)
{
    Waiting_List_Node *p,*q;
    p = head_resource_R4_list;
    q = p;
    while(p->next != NULL){
        p = p->next;
        if(strcmp(p->process_id,ProId) == 0){
            q->next = p->next;
            free(p);
            break;
        }
        q = p;
    }
}

void travelAllPCBList()
{
    PCBNode * p;
	p = all_process;
	if(!p){
        printf("AllPCBList is NULL");
		exit(0);
	}
	while(p->next != NULL){
		printf("%s\t", p->Process_ID);
		printf("%d\t\n", p->status);
		p = p->next;
	}
	printf("%s\t", p->Process_ID);
	printf("%d\t\n", p->status);
}

void listAllPCBList()
{
    PCBNode * p;
	p = all_process;
	if(!p){
        printf("AllPCBList is NULL");
		exit(0);
	}
	printf("All processes and their status:\n");
	while(p->next != NULL){
        p = p->next;
		printf("pid:%4s\t", p->Process_ID);
		int status = p->status;
		if (status == STATUS_RUNNING) {
            printf("status:Running\t");
		} else if (status == STATUS_READY){
		    printf("status:Ready\t");
		} else if(status == STATUS_BLOCK){
		    printf("status:Block\t");
		} else {
		    printf("status:NULL\t");
		}
		if (p->parent == NULL){
            printf("parent:NULL\t");
		} else {
		    printf("parent:%4s\t", p->parent->Process_ID);
		}
		if (p->son[0] == NULL){
            printf("son: NULL\t\n");
		} else {
		    int i = 0;
		    printf("son:");
		    while(p->son[i] != NULL){
                printf("%4s", p->son[i]->Process_ID);
                i++;
		    }
		    printf("\n");
		}
	}
}

void listAllRec()
{
    int i;
    printf("All resources and their status:\n");
    for (i = 0;i<4;i++){
        printf("Resource %d remain:%4d\n",i+1,resource[i]);
    }
}

void checkProInfo(char * pid)
{
    PCBNode * node = getProcessById(pid);
    if(node == NULL) {
        printf("The pid is valid!");
        exit(0);
    }
    printf("Process ID:%4s\n",node->Process_ID);
    printf("Process Resources:R1:%2d  R2:%2d  R3:%2d  R4:%2d\n",node->Other_resources[0],node->Other_resources[1],node->Other_resources[2],node->Other_resources[3]);
    if(node->parent != NULL){
        printf("Process parent: %s\n",node->parent->Process_ID);
    }
    if(node->son[0]!=NULL){
            int i = 0;
		    printf("son:");
		    while(node->son[i] != NULL){
                printf("%4s", node->son[i]->Process_ID);
                i++;
		    }

    } else {
        printf("son: NULL");
    }
    printf("\n");
}

void travelProcessList(Process_List_Note * head)
{
    Process_List_Note * p;
	p = head;
	if(!p){
        printf("ProcessList is NULL!\n");
		exit(0);
	}
	while(p->next != NULL){
		printf("%s\t", p->process_id);
		p = p->next;
	}
	printf("%s\t\n", p->process_id);
}

//void travelResourcesList(RIDNode * head)
//{
//    RIDNode * p;
//	p = head;
//	if(!p){
//        printf("ResourceList is NULL");
//		exit(0);
//	}
//	while(p->next != NULL){
//		printf("%s\t", p->Resource_ID);
//		printf("%4d\n", p->Status);
//		p = p->next;
//	}
//	printf("%s\t", p->Resource_ID);
//	printf("%4d\n", p->Status);
//}
//
//void travelPCBResourcesList(ProResList * head)
//{
//    ProResList * p;
//	p = head;
//	if(!p){
//        printf("ResourceList is NULL");
//		exit(0);
//	}
//	while(p->next != NULL){
//		printf("%s\t", p->Resource_ID);
//		printf("%4d\n", p->Resource_Number);
//		p = p->next;
//	}
//	printf("%s\t", p->Resource_ID);
//	printf("%4d\n", p->Resource_Number);
//}

void destoryAllList()
{
    //RIDNode * lastResPtr;
    Process_List_Note *lastProPtr;
    //释放资源链表
//    while (head_resource_list){
//            lastResPtr = head_resource_list -> next;
//            free(head_resource_list);
//            head_resource_list = lastResPtr;
//    }
    //释放进程链表
    while (head_process_zero_list){
            lastProPtr = head_process_zero_list -> next;
            free(head_process_zero_list);
            head_process_zero_list = lastProPtr;
    }
        while (head_process_one_list){
            lastProPtr = head_process_one_list -> next;
            free(head_process_one_list);
            head_process_one_list = lastProPtr;
    }
    while (head_process_two_list){
            lastProPtr = head_process_two_list -> next;
            free(head_process_two_list);
            head_process_two_list = lastProPtr;
    }
    while (head_resource_R1_list){
            lastProPtr = head_resource_R1_list -> next;
            free(head_resource_R1_list);
            head_resource_R1_list = lastProPtr;
    }
    while (head_resource_R2_list){
            lastProPtr = head_resource_R2_list -> next;
            free(head_resource_R2_list);
            head_resource_R2_list = lastProPtr;
    }
    while (head_resource_R3_list){
            lastProPtr = head_resource_R3_list -> next;
            free(head_resource_R3_list);
            head_resource_R3_list = lastProPtr;
    }while (head_resource_R4_list){
            lastProPtr = head_resource_R4_list -> next;
            free(head_resource_R4_list);
            head_resource_R4_list = lastProPtr;
    }
    free(current_process);
}
