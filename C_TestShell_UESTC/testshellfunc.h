#define MAX_INPUT_COMMAND 30
#define STATUS_RUNNING 1
#define STATUS_READY 2
#define STATUS_BLOCK 3

//进程队列的节点
typedef struct processListNote
{
    char * process_id;
    struct processListNote * next;
}Process_List_Note;

//资源等待队列的节点
typedef struct waitingListNode
{
    char * process_id;
    int Rid;
    int requireNum;
    struct waitingListNode * next;
}Waiting_List_Node;

//PCB 结构
typedef struct PCBNode
{
	int CPU_state ;  //not used--CPU status
	int Memory;  //not used--Memory state
	int open_files;  //not used

	char * Process_ID;  //process ID
	int Other_resources[4];  //资源使用情况
	int status;  //Type & List  取值STATUS_BLOCK or STATUS_RUNNING
	struct PCBNode * parent;  //父子关系树
	struct PCBNode * son[10] ;  //父子关系树
	int sonNum;  //子进程的个数
	int Priority;  //优先级0,1,2

	struct PCBNode * next ;
}PCBNode;

//RID结构
//typedef struct RID
//{
//    int Resource_ID;  //资源的ID
//    int Status;  //空闲单元的数量
//    struct RID * next;
//}RIDNode;

//进程占有资源的队列
//typedef struct ProResList
//{
//    int Resource_ID;  //资源的ID
//    int Resource_Number;  //占有资源单元的数量
//    struct RID * next;
//}ProResList;

int resource[4];
//RIDNode * head_resource_list;  //资源链表指针
PCBNode * current_process;  //正在运行进程
PCBNode * all_process;  //所有的PCB
Process_List_Note * head_process_zero_list;  //优先级0的进程队列指针
Process_List_Note * head_process_one_list;  //优先级1的进程队列指针
Process_List_Note * head_process_two_list;  //优先级2的进程队列指针
Waiting_List_Node * head_resource_R1_list;  //等待资源R1的进程队列指针
Waiting_List_Node * head_resource_R2_list;  //等待资源R2的进程队列指针
Waiting_List_Node * head_resource_R3_list;  //等待资源R3的进程队列指针
Waiting_List_Node * head_resource_R4_list;  //等待资源R4的进程队列指针

//PCB管理
PCBNode * init();  //init进程，根进程
PCBNode * allPCBList();  //所有的PCB结构链表
PCBNode * new_process;
insertAllPCBList(PCBNode * node);  //插入PCB到链表
PCBNode * creatProcess(char * pid, int priority);  //创建进程，pid为进程id，priority为优先级
void deleteProcess(char * pid);  //撤销相应的进程
PCBNode * getProcessById(char * Pid);  //根据ID从PCB链表得到相应的PCB
void removeProcessById(char * Pid);  //根据ID从PCB链表撤销相应的PCB

//功能性函数
analyseAndDoCommand(char input_command[]);  //分析并执行输入的指令
void Time_out();  //时钟中断Time out:模拟时间片到或者外部硬件中断
void Scheduler();  //重新调度
char * getHighestProcess();  //得到当前优先级最高的进程

//优先级队列管理
void insertProcessList(int priority, PCBNode * node);  //将进程插入到相应的优先级队列
void removeFromHead(PCBNode * node);  //将正在运行的进程从相应的优先级别去除
Process_List_Note * creatPriorityZeroList();  //创建优先级0的进程队列
void insertProcessListZero(PCBNode * node);  //插入进程到队列1
void removeFromHeadZero(char *ProId);  //从head删除正在运行的进程
Process_List_Note * creatPriorityOneList();  //创建优先级1的进程队列
void insertProcessListOne(PCBNode * node);  //插入进程到队列1
void removeFromHeadOne(char *ProId);  //从head删除正在运行的进程
Process_List_Note * creatPriorityTwoList();  //创建优先级2的进程队列
void insertProcessListTwo(PCBNode * node);  //插入进程到队列1
void removeFromHeadTwo(char *ProId);  //从head删除正在运行的进程

//资源链表
//RIDNode * creatResourcesList();  //创建资源链表
//ProResList * creatProResList(int res1,int res2,int res3,int res4);  //创建PCB资源链表
void releaseResource(PCBNode * node);  //释放进程占用的资源
void requestResource(PCBNode * pro, int rid,int num);  //进程请求资源rid，数目为num个
void checkIfResourceListOk();  //检查阻塞队列是否有可以得到充足资源从而运行的进程

insertResourceReList(PCBNode * node, int rid, int num);  //根据rid区分插入到哪个链表
Waiting_List_Node * creatResourceR1List();  //创建资源R1的等待队列
void insertResourceR1List(PCBNode * node, int rid, int num);  //插入进程到资源R1的等待队列
void removeResourceR1List(char *ProId);  //从资源R1的等待队列删除正在等待的进程
Waiting_List_Node * creatResourceR2List();  //创建资源R2的等待队列
void insertResourceR2List(PCBNode * node, int rid, int num);  //插入进程到资源R2的等待队列
void removeResourceR2List(char *ProId);  //从资源R2的等待队列删除正在等待的进程
Waiting_List_Node * creatResourceR3List();  //创建资源R3的等待队列
void insertResourceR3List(PCBNode * node, int rid, int num);  //插入进程到资源R3的等待队列
void removeResourceR3List(char *ProId);  //从资源R3的等待队列删除正在等待的进程
Waiting_List_Node * creatResourceR4List();  //创建资源R4的等待队列
void insertResourceR4List(PCBNode * node, int rid, int num);  //插入进程到资源R4的等待队列
void removeResourceR4List(char *ProId);  //从资源R4的等待队列删除正在等待的进程


//遍历和销毁链表
void travelAllPCBList();  //遍历PCB链表
void listAllPCBList();  //-list all processes and their status
void listAllRec();//-list all resources and their status
void checkProInfo(char * pid); //-provide information about a given process
void travelProcessList(Process_List_Note * head);  //遍历进程链表
//void travelResourcesList(RIDNode * head);  //遍历资源链表
//void travelPCBResourcesList(ProResList * head);  //遍历PCB资源链表
void destoryAllList();  //释放所有链表
