#ifndef FILE_H
#define FILE_H

#include <string>

#include <stdint.h>

#include "sfilerecord.h"

using std::string;

class File
{
public:
    File();
    virtual ~File();
    virtual bool isFile()=0;
    virtual bool isSFile();
    virtual bool isDir()=0;
    virtual bool exists()=0;
    virtual string getFilename()=0;
    virtual int64_t getSize()=0;
    virtual void create()=0;
    virtual void openAppendOnly()=0;
    virtual void openReadOnly()=0;
    virtual size_t read(void* buffer, size_t length)=0;
    virtual void readFully(void* buffer, size_t length);
    virtual void skip(uint64_t length)=0;
    virtual size_t write(void* buffer, size_t length)=0;
    virtual bool isEof()=0;
    virtual void close()=0;
    virtual SFileRecord readNextSFileRecord();
};

#endif // FILE_H
