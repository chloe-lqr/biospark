#ifndef LOCALFILE_H
#define LOCALFILE_H

#include <cstdio>
#include <string>

#include <stdint.h>
#include <sys/stat.h>

#include "file.h"

using std::string;

class LocalFile : public File
{
public:
    LocalFile(string filename);
    virtual ~LocalFile();
    virtual bool isFile();
    virtual bool isDir();
    virtual bool exists();
    virtual string getFilename();
    virtual int64_t getSize();
    virtual void create();
    virtual void openAppendOnly();
    virtual void openReadOnly();
    virtual size_t read(void* buffer, size_t length);
    virtual void skip(uint64_t length);
    virtual bool isEof();
    virtual size_t write(void* buffer, size_t length);
    virtual void close();

protected:
    string filename;
    struct stat filestats;
    int filestatsRet;
    FILE* fp;
};

#endif // LOCALFILE_H
