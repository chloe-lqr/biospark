#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "file.h"
#include "runtimeexception.h"
#include "sfilerecord.h"

using std::string;

File::File()
{
}

File::~File()
{
}

void File::readFully(void* vbuffer, size_t length)
{
    unsigned char* buffer = (unsigned char*)vbuffer;
    size_t bytesRead;
    do
    {
        bytesRead=read(buffer, length);
        buffer += bytesRead;
        length -= bytesRead;
    }
    while (length > 0 && bytesRead != 0);
}

bool File::isSFile()
{
    bool ret=false;
    unsigned char header[sizeof(SFileRecord::RECORD_SEPARATOR)];
    openReadOnly();
    size_t bytesRead=read(header, sizeof(SFileRecord::RECORD_SEPARATOR));
    if (bytesRead == sizeof(SFileRecord::RECORD_SEPARATOR) && memcmp(header, SFileRecord::RECORD_SEPARATOR, sizeof(SFileRecord::RECORD_SEPARATOR)) == 0)
        ret=true;
    close();
    return ret;
}

SFileRecord File::readNextSFileRecord()
{
    // Make sure that a separator is next.
    unsigned char separator[sizeof(SFileRecord::RECORD_SEPARATOR)];
    readFully(separator, sizeof(SFileRecord::RECORD_SEPARATOR));
    if (memcmp(separator, SFileRecord::RECORD_SEPARATOR, sizeof(SFileRecord::RECORD_SEPARATOR)) != 0)
    {
        for (int i=0; i<sizeof(SFileRecord::RECORD_SEPARATOR); i++)
        {
            printf("%u ",(int)separator[i]);
        }
        printf("\n");
        throw RuntimeException(string("Error reading next sfile separator."));
    }

    // Read the name.
    int32_t nameSize;
    readFully(&nameSize, sizeof(nameSize));
    char* name = new char[nameSize+1];
    readFully(name, nameSize);
    name[nameSize] = 0;

    // Read the type.
    int32_t typeSize;
    readFully(&typeSize, sizeof(typeSize));
    char* type = new char[typeSize+1];
    readFully(type, typeSize);
    type[typeSize] = 0;

    // Read the data size
    int64_t dataSize;
    readFully(&dataSize, sizeof(dataSize));

    return SFileRecord(string(name), string(type), dataSize);
}
