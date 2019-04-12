#include "LexicalAnalyer.h"

char getnbc()
{
	char ch;
	ch = getchar();
	while (1)
	{
		if (ch == '\r' || ch == '\t' || ch == ' ')
		{
			ch = getchar();
		}
		else
		{
			break;
		}
	}
	return ch;
}

bool letter(char character)
{
	if ((character >= 'a'&&character <= 'z') || (character >= 'A'&&character <= 'Z'))
		return true;
	else
		return false;
}

bool digit(char character)
{
	if (character >= '0'&&character <= '9')
		return true;
	else
		return false;
}

void retract(char& character)
{
	ungetc(character, stdin);
	character = NULL;
}

int reserve(char* token)
{
	if (strcmp(token, "begin") == 0)
		return 1;
	else if (strcmp(token, "end") == 0)
		return 2;
	else if (strcmp(token, "integer") == 0)
		return 3;
	else if (strcmp(token, "if") == 0)
		return 4;
	else if (strcmp(token, "then") == 0)
		return 5;
	else if (strcmp(token, "else") == 0)
		return 6;
	else if (strcmp(token, "function") == 0)
		return 7;
	else if (strcmp(token, "read") == 0)
		return 8;
	else if (strcmp(token, "write") == 0)
		return 9;
	else
		return 0;
}

int symbol()
{
	return 10;
}

int constant()
{
	return 11;
}

void output(const char* token, int kindNum)
{
	printf("%16s %2d\n", token, kindNum);
}

bool error(int lineNum, int errNum)
{
	char* errInfo;
	switch (errNum)
	{
	case ILLEGAL_CHAR_ERR:
		errInfo = "������ĸ������ķǷ��ַ�";
		break;
	case UNKNOWN_OPERATOR_ERR:
		errInfo = "����δ֪�����";
		break;
	default:
		errInfo = "δ֪����";
	}
	if (fprintf(stderr, "***LINE:%d  %s\n", lineNum, errInfo) >= 0)
		return true;
	else
		return false;
}

bool LexAnalyze()
{
	static int lineNum = 1;
	char character;
	char token[17] = "";
	character = getnbc();

	switch (character)
	{
	case '\n':
		output("EOLN", 24);
		lineNum++;
		break;
	case EOF:
		output("EOF", 25);
		return false;//false��ʾ�Ѷ����ļ�ĩβ
	case 'a':
	case 'b':
	case 'c':
	case 'd':
	case 'e':
	case 'f':
	case 'g':
	case 'h':
	case 'i':
	case 'j':
	case 'k':
	case 'l':
	case 'm':
	case 'n':
	case 'o':
	case 'p':
	case 'q':
	case 'r':
	case 's':
	case 't':
	case 'u':
	case 'v':
	case 'w':
	case 'x':
	case 'y':
	case 'z':
	case 'A':
	case 'B':
	case 'C':
	case 'D':
	case 'E':
	case 'F':
	case 'G':
	case 'H':
	case 'I':
	case 'J':
	case 'K':
	case 'L':
	case 'M':
	case 'N':
	case 'O':
	case 'P':
	case 'Q':
	case 'R':
	case 'S':
	case 'T':
	case 'U':
	case 'V':
	case 'W':
	case 'X':
	case 'Y':
	case 'Z':

		while (letter(character) || digit(character))
		{
			char s[2] = { character };
			strcat(token, s);
			character = getchar();
		}
		retract(character);
		int num;
		num = reserve(token);
		if (num != 0)
			output(token, num);
		else
		{
			int val;
			val = symbol();
			output(token, val);
		}
		break;

	case '0':
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
	case '8':
	case '9':
		while (digit(character))
		{
			char s[2] = { character };
			strcat(token, s);
			character = getchar();
		}
		retract(character);
		int val;
		val = constant();
		output(token, val);
		break;
	case '=':
		output("=", 12);
		break;
	case '<':
		character = getchar();
		if (character == '>')
			output("<>", 13);
		else if (character == '=')
			output("<=", 14);
		else
		{
			retract(character);
			output("<", 15);
		}
		break;
	case '>':
		character = getchar();
		if (character == '=')
			output(">=", 16);
		else
		{
			retract(character);
			output(">", 17);
		}
		break;
	case '-':
		output("-", 18);
		break;
	case '*':
		output("*", 19);
		break;
	case ':':
		character = getchar();
		if (character == '=')
			output(":=", 20);
		else
			error(lineNum, 2);//�����δ֪�����������
		break;
	case '(':
		output("(", 21);
		break;
	case ')':
		output(")", 22);
		break;
	case ';':
		output(";", 23);
		break;
	default:
		error(lineNum, 1);//���"������ĸ������ķǷ��ַ�"����
	}
	return true;
}

void getPath(char* in, char* out)
{
	char* name;
	name = strrchr(in, '\\');
	if (name != NULL)
		strncpy(out, in, strlen(in) - strlen(name) + 1);
	else
		strcpy(out, "");
}

void getFilename(char* in, char* out)
{
	char* fullName;
	char* extension;
	fullName = strrchr(in, '\\');
	extension = strrchr(in, '.');
	if (fullName != NULL)
		strncpy(out, fullName + 1, strlen(fullName) - 1 - strlen(extension));
	else
		strncpy(out, in, strlen(in) - strlen(extension));
}

bool init(int argc, char* argv[])
{
	if (argc != 2)
	{
		return false;
	}
	else
	{
		char* inFilename = argv[1];//argv[1];
		char outFilename[MAX_COUNT] = "";
		char errFilename[MAX_COUNT] = "";
		char filename[MAX_COUNT] = "";
		char path[MAX_COUNT] = "";
		//����ļ�������������չ������·��
		getFilename(inFilename, filename);
		getPath(inFilename, path);
		//��������ļ�ȫ��·��

		strcat(outFilename, path);
		//strcat(outFilename, "\\");
		strcat(outFilename, filename);
		strcat(outFilename, ".dyd");
		//���ɴ����ļ�ȫ��·��

		strcat(errFilename, path);
		//strcat(errFilename, "\\");
		strcat(errFilename, filename);
		strcat(errFilename, ".err");
		if (freopen(inFilename, "r", stdin) != NULL&&freopen(outFilename, "w", stdout) != NULL&&freopen(errFilename, "w", stderr) != NULL)
			return true;
		else
			return false;
	}
}
