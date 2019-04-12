#define MAX_INPUT_COMMAND 30
#define STATUS_RUNNING 1
#define STATUS_READY 2
#define STATUS_BLOCK 3

//���̶��еĽڵ�
typedef struct processListNote
{
    char * process_id;
    struct processListNote * next;
}Process_List_Note;

//��Դ�ȴ����еĽڵ�
typedef struct waitingListNode
{
    char * process_id;
    int Rid;
    int requireNum;
    struct waitingListNode * next;
}Waiting_List_Node;

//PCB �ṹ
typedef struct PCBNode
{
	int CPU_state ;  //not used--CPU status
	int Memory;  //not used--Memory state
	int open_files;  //not used

	char * Process_ID;  //process ID
	int Other_resources[4];  //��Դʹ�����
	int status;  //Type & List  ȡֵSTATUS_BLOCK or STATUS_RUNNING
	struct PCBNode * parent;  //���ӹ�ϵ��
	struct PCBNode * son[10] ;  //���ӹ�ϵ��
	int sonNum;  //�ӽ��̵ĸ���
	int Priority;  //���ȼ�0,1,2

	struct PCBNode * next ;
}PCBNode;

//RID�ṹ
//typedef struct RID
//{
//    int Resource_ID;  //��Դ��ID
//    int Status;  //���е�Ԫ������
//    struct RID * next;
//}RIDNode;

//����ռ����Դ�Ķ���
//typedef struct ProResList
//{
//    int Resource_ID;  //��Դ��ID
//    int Resource_Number;  //ռ����Դ��Ԫ������
//    struct RID * next;
//}ProResList;

int resource[4];
//RIDNode * head_resource_list;  //��Դ����ָ��
PCBNode * current_process;  //�������н���
PCBNode * all_process;  //���е�PCB
Process_List_Note * head_process_zero_list;  //���ȼ�0�Ľ��̶���ָ��
Process_List_Note * head_process_one_list;  //���ȼ�1�Ľ��̶���ָ��
Process_List_Note * head_process_two_list;  //���ȼ�2�Ľ��̶���ָ��
Waiting_List_Node * head_resource_R1_list;  //�ȴ���ԴR1�Ľ��̶���ָ��
Waiting_List_Node * head_resource_R2_list;  //�ȴ���ԴR2�Ľ��̶���ָ��
Waiting_List_Node * head_resource_R3_list;  //�ȴ���ԴR3�Ľ��̶���ָ��
Waiting_List_Node * head_resource_R4_list;  //�ȴ���ԴR4�Ľ��̶���ָ��

//PCB����
PCBNode * init();  //init���̣�������
PCBNode * allPCBList();  //���е�PCB�ṹ����
PCBNode * new_process;
insertAllPCBList(PCBNode * node);  //����PCB������
PCBNode * creatProcess(char * pid, int priority);  //�������̣�pidΪ����id��priorityΪ���ȼ�
void deleteProcess(char * pid);  //������Ӧ�Ľ���
PCBNode * getProcessById(char * Pid);  //����ID��PCB����õ���Ӧ��PCB
void removeProcessById(char * Pid);  //����ID��PCB��������Ӧ��PCB

//�����Ժ���
analyseAndDoCommand(char input_command[]);  //������ִ�������ָ��
void Time_out();  //ʱ���ж�Time out:ģ��ʱ��Ƭ�������ⲿӲ���ж�
void Scheduler();  //���µ���
char * getHighestProcess();  //�õ���ǰ���ȼ���ߵĽ���

//���ȼ����й���
void insertProcessList(int priority, PCBNode * node);  //�����̲��뵽��Ӧ�����ȼ�����
void removeFromHead(PCBNode * node);  //���������еĽ��̴���Ӧ�����ȼ���ȥ��
Process_List_Note * creatPriorityZeroList();  //�������ȼ�0�Ľ��̶���
void insertProcessListZero(PCBNode * node);  //������̵�����1
void removeFromHeadZero(char *ProId);  //��headɾ���������еĽ���
Process_List_Note * creatPriorityOneList();  //�������ȼ�1�Ľ��̶���
void insertProcessListOne(PCBNode * node);  //������̵�����1
void removeFromHeadOne(char *ProId);  //��headɾ���������еĽ���
Process_List_Note * creatPriorityTwoList();  //�������ȼ�2�Ľ��̶���
void insertProcessListTwo(PCBNode * node);  //������̵�����1
void removeFromHeadTwo(char *ProId);  //��headɾ���������еĽ���

//��Դ����
//RIDNode * creatResourcesList();  //������Դ����
//ProResList * creatProResList(int res1,int res2,int res3,int res4);  //����PCB��Դ����
void releaseResource(PCBNode * node);  //�ͷŽ���ռ�õ���Դ
void requestResource(PCBNode * pro, int rid,int num);  //����������Դrid����ĿΪnum��
void checkIfResourceListOk();  //������������Ƿ��п��Եõ�������Դ�Ӷ����еĽ���

insertResourceReList(PCBNode * node, int rid, int num);  //����rid���ֲ��뵽�ĸ�����
Waiting_List_Node * creatResourceR1List();  //������ԴR1�ĵȴ�����
void insertResourceR1List(PCBNode * node, int rid, int num);  //������̵���ԴR1�ĵȴ�����
void removeResourceR1List(char *ProId);  //����ԴR1�ĵȴ�����ɾ�����ڵȴ��Ľ���
Waiting_List_Node * creatResourceR2List();  //������ԴR2�ĵȴ�����
void insertResourceR2List(PCBNode * node, int rid, int num);  //������̵���ԴR2�ĵȴ�����
void removeResourceR2List(char *ProId);  //����ԴR2�ĵȴ�����ɾ�����ڵȴ��Ľ���
Waiting_List_Node * creatResourceR3List();  //������ԴR3�ĵȴ�����
void insertResourceR3List(PCBNode * node, int rid, int num);  //������̵���ԴR3�ĵȴ�����
void removeResourceR3List(char *ProId);  //����ԴR3�ĵȴ�����ɾ�����ڵȴ��Ľ���
Waiting_List_Node * creatResourceR4List();  //������ԴR4�ĵȴ�����
void insertResourceR4List(PCBNode * node, int rid, int num);  //������̵���ԴR4�ĵȴ�����
void removeResourceR4List(char *ProId);  //����ԴR4�ĵȴ�����ɾ�����ڵȴ��Ľ���


//��������������
void travelAllPCBList();  //����PCB����
void listAllPCBList();  //-list all processes and their status
void listAllRec();//-list all resources and their status
void checkProInfo(char * pid); //-provide information about a given process
void travelProcessList(Process_List_Note * head);  //������������
//void travelResourcesList(RIDNode * head);  //������Դ����
//void travelPCBResourcesList(ProResList * head);  //����PCB��Դ����
void destoryAllList();  //�ͷ���������
