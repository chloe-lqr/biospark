#include "sfilerecord.h"

const char SFileRecord::RECORD_SEPARATOR[] = {'S','F','R','X',1,65,243,72,36,217,55,18,134,11,234,83};

SFileRecord::SFileRecord()
    :name(""),type(""),dataSize(0)
{
}

SFileRecord::SFileRecord(string name, string type, int64_t dataSize)
    :name(name),type(type),dataSize(dataSize)
{

}
