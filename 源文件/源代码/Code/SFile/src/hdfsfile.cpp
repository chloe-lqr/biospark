#ifdef OPT_HDFS

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <sys/stat.h>

#include <hdfs.h>

#include "hdfsfile.h"
#include "runtimeexception.h"

using std::string;

HDFSFile::HDFSFile(string url, string path)
    :url(url),path(path),fs(NULL),fileInfo(NULL),fp(NULL),currentPosition(0)
{
    // Create a connection builder.
    hdfsBuilder* builder = hdfsNewBuilder();
    if (builder ==  NULL) throw RuntimeException("Could not create new hdfs builder.");
    hdfsBuilderSetNameNode(builder, url.c_str());
    if (getenv("HADOOP_USER_NAME") != NULL && strlen(getenv("HADOOP_USER_NAME")) > 0)
        hdfsBuilderSetUserName(builder, getenv("HADOOP_USER_NAME"));

    // Connect to hdfs.
    fs = hdfsBuilderConnect(builder);
    if (fs == NULL) throw RuntimeException(string("Could not connect to hdfs filesystem: ")+url);
}

HDFSFile::~HDFSFile()
{
    close();
    if (fileInfo != NULL) hdfsFreeFileInfo(fileInfo, 1); fileInfo = NULL;
    if (fs != NULL) hdfsDisconnect(fs); fs = NULL;
}

bool HDFSFile::isFile()
{
    // Get some file info, if necessary.
    if (fileInfo == NULL)
        fileInfo = hdfsGetPathInfo(fs, path.c_str());
    if (fileInfo != NULL)
        return fileInfo->mKind == kObjectKindFile;
    throw RuntimeException(string("Could not retrieve hdfs file info: ")+path);
}

bool HDFSFile::isDir()
{
    // Get some file info, if necessary.
    if (fileInfo == NULL)
        fileInfo = hdfsGetPathInfo(fs, path.c_str());
    if (fileInfo != NULL)
        return fileInfo->mKind == kObjectKindDirectory;
    throw RuntimeException(string("Could not retrieve hdfs file info: ")+path);
}

bool HDFSFile::exists()
{
    return (hdfsExists(fs, path.c_str())==0);
}

string HDFSFile::getFilename()
{
    return url+path;
}

int64_t HDFSFile::getSize()
{
    if (fileInfo == NULL)
        fileInfo = hdfsGetPathInfo(fs, path.c_str());
    if (fileInfo != NULL)
        return fileInfo->mSize;
    throw RuntimeException(string("Could not retrieve hdfs file info: ")+path);
}

void HDFSFile::create()
{
    if (fs != NULL && fp == NULL)
    {
        fp = hdfsOpenFile(fs, path.c_str(), O_WRONLY, 0, 0, 0);
        if (fp == NULL) throw RuntimeException(string("Could not create file: ")+url+path);
        currentPosition = 0;
    }
}

void HDFSFile::openAppendOnly()
{
    if (fs != NULL && fp == NULL)
    {
        fp = hdfsOpenFile(fs, path.c_str(), O_WRONLY|O_APPEND, 0, 0, 0);
        if (fp == NULL) throw RuntimeException(string("Could not open file for appending: ")+url+path);
        currentPosition = 0;
    }
}

void HDFSFile::openReadOnly()
{
    if (fs != NULL && fp == NULL)
    {
        fp = hdfsOpenFile(fs, path.c_str(), O_RDONLY, 0, 0, 0);
        if (fp == NULL) throw RuntimeException(string("Could not open file for reading: ")+url+path);
        currentPosition = 0;
    }
}

size_t HDFSFile::read(void* buffer, size_t count)
{
    if (fs != NULL  && fp != NULL)
    {
        tSize ret=hdfsRead(fs, fp, buffer, count);
        if (ret == -1) throw RuntimeException(string("Error reading from file: ")+url+path);
        currentPosition += ret;
        return ret;
    }
    return 0;
}

void HDFSFile::skip(uint64_t length)
{
    if (fs != NULL  && fp != NULL)
    {
        const int bufferSize = 100*1024;
        unsigned char* buffer[bufferSize];
        while (length > 0)
        {
            tSize ret=hdfsRead(fs, fp, buffer, (length<bufferSize)?length:bufferSize);
            if (ret <= 0) throw RuntimeException(string("Error skipping in file: ")+url+path);
            length -= ret;
            currentPosition += ret;
        }
    }
}

bool HDFSFile::isEof()
{
    if (fs != NULL  && fp != NULL)
    {
        return (currentPosition >= fileInfo->mSize);
    }
    return true;
}

size_t HDFSFile::write(void* buffer, size_t count)
{
    if (fs != NULL  && fp != NULL)
    {
        tSize ret=hdfsWrite(fs, fp, buffer, count);
        if (ret == -1 || ((tSize)count) != ret) throw RuntimeException(string("Error writing to file: ")+url+path);
        currentPosition += ret;
        return ret;
    }
    return 0;
}

void HDFSFile::close()
{
    if (fs != NULL  && fp != NULL)
    {
        int ret = hdfsCloseFile(fs, fp);
        fp = NULL;
        if (ret != 0) throw RuntimeException(string("Could not close file: ")+url+path);
    }
}

#endif
