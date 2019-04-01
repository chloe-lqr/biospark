#ifdef OPT_HDFS

#ifndef HDFSFile_H
#define HDFSFile_H

#include <cstdio>
#include <string>

#include <stdint.h>
#include <sys/stat.h>

#include <hdfs.h>

#include "file.h"
using std::string;

class HDFSFile : public File
{
public:
    HDFSFile(string url, string path);
    virtual ~HDFSFile();
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
    string url;
    string path;
    hdfsFS fs;
    hdfsFileInfo* fileInfo;
    hdfsFile fp;
    tOffset currentPosition;
};

#endif // HDFSFile_H

#endif
