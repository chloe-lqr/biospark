#include <cstdio>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

#include "localfile.h"
#include "runtimeexception.h"

using std::string;

LocalFile::LocalFile(string filename)
    :filename(filename),fp(NULL)
{
    filestatsRet=stat(filename.c_str(), &filestats);
}

LocalFile::~LocalFile()
{
    close();
}

bool LocalFile::isFile()
{
    return S_ISREG(filestats.st_mode);

}

bool LocalFile::isDir()
{
    return S_ISDIR(filestats.st_mode);
}

bool LocalFile::exists()
{
    return (filestatsRet==0);
}

string LocalFile::getFilename()
{
    return filename;
}

int64_t LocalFile::getSize()
{
    return (int64_t)filestats.st_size;
}

void LocalFile::create()
{
    if (fp == NULL)
    {
        fp = fopen(filename.c_str(), "w");
        if (fp == NULL) throw RuntimeException(string("Could not create file: ")+filename);
    }
}

void LocalFile::openAppendOnly()
{
    if (fp == NULL)
    {
        fp = fopen(filename.c_str(), "a");
        if (fp == NULL) throw RuntimeException(string("Could not open file for appending: ")+filename);
    }
}

void LocalFile::openReadOnly()
{
    if (fp == NULL)
    {
        fp = fopen(filename.c_str(), "r");
        if (fp == NULL) throw RuntimeException(string("Could not open file for reading: ")+filename);
    }
}

size_t LocalFile::read(void* buffer, size_t count)
{
    if (fp != NULL)
    {
        size_t ret=fread(buffer, sizeof(unsigned char), count, fp);
        if (count != ret && ferror(fp) != 0) throw RuntimeException(string("Error reading from file: ")+filename);
        return ret;
    }
    return 0;
}

void LocalFile::skip(uint64_t length)
{
    if (fp != NULL)
    {
        if (fseek(fp, length, SEEK_CUR) != 0) throw RuntimeException(string("Error seeking in file: ")+filename);
    }
}

bool LocalFile::isEof()
{
    if (fp != NULL)
    {
        return (ftell(fp) == getSize());
    }
    return true;
}

size_t LocalFile::write(void* buffer, size_t count)
{
    if (fp != NULL)
    {
        size_t ret=fwrite(buffer, sizeof(unsigned char), count, fp);
        if (count != ret) throw RuntimeException(string("Error writing to file: ")+filename);
        return ret;
    }
    return 0;
}

void LocalFile::close()
{
    if (fp != NULL)
    {
        int ret=fclose(fp);
        fp = NULL;
        if (ret != 0) throw RuntimeException(string("Could not close file: ")+filename);
    }
}
