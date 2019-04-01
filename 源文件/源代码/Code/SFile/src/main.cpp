#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <stdint.h>
#include <glob.h>

#include "hrtime.h"
#include "optionparser.h"
#include "runtimeexception.h"
#include "sfilerecord.h"

using std::string;
using std::vector;

void setupEnvironment();
void functionCopy();
void copyToSFile(vector<File*> sources, File* targetSFile);
void writeRecord(File* dest, SFileRecord& record);
void copyData(File* source, File* dest, int64_t size);
void extractToDir(vector<File*> sources, File* targetDir);
void functionList();

int main(int argc, char** argv)
{
    try
    {
        setupEnvironment();

        printCopyright(argc, argv);
        if (!parseArguments(argc, argv))
        {
            printf("Invalid command line.\n");
            printUsage(argc, argv);
            return -1;
        }

        if (function == "help")
        {
            printUsage(argc, argv);
        }
        else if (function == "version")
        {
        }
        else if (function == "cp")
        {
            functionCopy();
        }
        else if (function == "ls")
        {
            functionList();
        }
        return 0;
    }
    catch (RuntimeException e)
    {
        printf("Exception: %s\n", e.what());
    }
    catch (std::exception e)
    {
        printf("STD Exception: %s\n", e.what());
    }
}

void setupEnvironment()
{
#ifdef OPT_HDFS
    char* hadoopRoot = getenv("HADOOP_INSTALL");
    if (hadoopRoot == NULL || strlen(hadoopRoot) == 0) hadoopRoot = getenv("HADOOP_PREFIX");
    if (hadoopRoot == NULL || strlen(hadoopRoot) == 0) throw RuntimeException("Could not determine hadoop installation path.");

    printf("Searching for HADOOP in %s...",hadoopRoot);

    //Declare glob_t for storing the results of globbing
    glob_t globbuf;

    // Find the jar files.
    glob((string(hadoopRoot)+"/share/hadoop/*.jar").c_str(), 0, NULL, &globbuf);
    glob((string(hadoopRoot)+"/share/hadoop/*/*.jar").c_str(), GLOB_APPEND, NULL, &globbuf);
    glob((string(hadoopRoot)+"/share/hadoop/*/*/*.jar").c_str(), GLOB_APPEND, NULL, &globbuf);
    glob((string(hadoopRoot)+"/share/hadoop/*/*/*/*.jar").c_str(), GLOB_APPEND, NULL, &globbuf);
    glob((string(hadoopRoot)+"/share/hadoop/*/*/*/*/*.jar").c_str(), GLOB_APPEND, NULL, &globbuf);
    glob((string(hadoopRoot)+"/share/hadoop/*/*/*/*/*/*.jar").c_str(), GLOB_APPEND, NULL, &globbuf);
    glob((string(hadoopRoot)+"/share/hadoop/*/*/*/*/*/*/*.jar").c_str(), GLOB_APPEND, NULL, &globbuf);

    if (globbuf.gl_pathc > 0)
    {
        printf("%lu jar files.\n",globbuf.gl_pathc);
        string classpath="";
        if (getenv("CLASSPATH") != NULL && strlen(getenv("CLASSPATH")) > 0)
            classpath += string(getenv("CLASSPATH"))+string(":");
        classpath += string(globbuf.gl_pathv[0]);
        for (int i=1; i<globbuf.gl_pathc; i++)
            classpath += string(":")+string(globbuf.gl_pathv[i]);
        if (setenv("CLASSPATH", classpath.c_str(), 1) != 0)
            throw RuntimeException("Could not set classpath.");
    }
    else
    {
        printf("NOT FOUND\n");
    }

    //Free the globbuf structure
    if( globbuf.gl_pathc > 0 )
      globfree(&globbuf);
#endif
}

void functionCopy()
{
    // Figure out if we are copying to or from an sfile.

    // If the target doesn't exist, make a new sfile.
    if (!target->exists())
    {
        printf("Copying to new sfile: %s\n", target->getFilename().c_str());
        target->create();
        copyToSFile(sources, target);
        target->close();
    }

    // If the target is an sfile, add everything to into it.
    else if(target->isFile() && target->isSFile())
    {
        printf("Appending to existing sfile: %s\n", target->getFilename().c_str());
        target->openAppendOnly();
        copyToSFile(sources, target);
        target->close();
    }

    // If the target is a local directory, extract everything into it.
    else if (target->isDir())
    {
        printf("Extracting into directory: %s\n", target->getFilename().c_str());
        extractToDir(sources, target);
    }

    // Otherwise, it is improper usage.
    else
    {
        throw RuntimeException("Invalid destination for copy: "+target->getFilename());
    }
}

void copyToSFile(vector<File*> sources, File* targetSFile)
{
    for (unsigned int i=0; i<sources.size(); i++)
    {
        File* source = sources[i];
        if (source->isFile() && source->getSize() > 0)
        {
            if (source->isSFile())
            {
                source->openReadOnly();
                int count=0;
                while (!source->isEof())
                {
                    SFileRecord record = source->readNextSFileRecord();
                    printf("\t%s%s->%s (%lld bytes)\n", source->getFilename().c_str(), record.name.c_str(), record.name.c_str(), record.dataSize);
                    writeRecord(targetSFile, record);
                    copyData(source, targetSFile, record.dataSize);
                    count++;
                }
                source->close();
            }
            else
            {
                SFileRecord record;
                record.name = string("/")+source->getFilename();
                record.type = recordType;
                record.dataSize = source->getSize();
                printf("\t%s->%s (%lld bytes)\n", source->getFilename().c_str(), record.name.c_str(), record.dataSize);
                writeRecord(targetSFile, record);
                source->openReadOnly();
                copyData(source, targetSFile, record.dataSize);
                source->close();
            }
        }
    }
}

void extractToDir(vector<File*> sources, File* targetDir)
{
    for (unsigned int i=0; i<sources.size(); i++)
    {
        File* source = sources[i];
        if (source->isFile() && source->getSize() > 0)
        {
            if (source->isSFile())
            {
                source->openReadOnly();
                int count=0;
                while (!source->isEof())
                {
                    SFileRecord record = source->readNextSFileRecord();
                    printf("\t%s%s->%s (%lld bytes)\n", source->getFilename().c_str(), record.name.c_str(), record.name.c_str(), record.dataSize);
                    LocalFile target(targetDir->getFilename()+"/"+record.name);
                    if (target.exists())
                        throw RuntimeException("Quit while trying to extract over an existing file",target.getFilename());
                    target.create();
                    copyData(source, &target, record.dataSize);
                    count++;
                }
                source->close();
            }
            else
            {
                throw RuntimeException("Quit while trying to extract from a regular file",source->getFilename());
            }
        }
    }
}

void writeRecord(File* dest, SFileRecord& record)
{
    dest->write((void*)SFileRecord::RECORD_SEPARATOR, sizeof(SFileRecord::RECORD_SEPARATOR));
    uint32_t nameLength = record.name.size();
    dest->write(&nameLength, sizeof(nameLength));
    dest->write((void*)record.name.c_str(), nameLength);
    uint32_t typeLength = record.type.size();
    dest->write(&typeLength, sizeof(typeLength));
    dest->write((void*)record.type.c_str(), typeLength);
    dest->write(&record.dataSize, sizeof(record.dataSize));
}

void copyData(File* source, File* dest, int64_t size)
{
    const unsigned int bufferSize = 1024*128;
    unsigned char buffer[bufferSize];
    hrtime readTime = 0;
    hrtime writeTime = 0;
    hrtime startTime = getHrTime();
    double lastSecs=0.0;
    int64_t sizeCopied=0;
    while (sizeCopied < size)
    {
        hrtime readStartTime = getHrTime();
        size_t read = source->read(buffer, (size-sizeCopied>bufferSize)?(bufferSize):(size-sizeCopied));
        readTime += getHrTime()-readStartTime;
        if (read == 0) break;

        hrtime writeStartTime = getHrTime();
        size_t written = dest->write(buffer, read);
        writeTime += getHrTime()-writeStartTime;
        if (written != read) break;

        double secs = convertHrToSeconds(getHrTime()-startTime);
        if (secs-lastSecs > 10.0)
        {
            printf("\t\t%3.2f%% r=%0.2f MB/s (%0.2e s) w=%0.2f MB/s (%0.2e s)\n", 100.0*((double)sizeCopied)/((double)size), ((double)sizeCopied)/(1024.0*1024.0*convertHrToSeconds(readTime)), convertHrToSeconds(readTime), ((double)sizeCopied)/(1024.0*1024.0*convertHrToSeconds(writeTime)), convertHrToSeconds(writeTime));
            lastSecs = secs;
        }

        // Increase the size copied.
        sizeCopied += read;
    }
    printf("\t\t%3.0f%% r=%0.2f MB/s (%0.2e s) w=%0.2f MB/s (%0.2e s)\n", 100.0*((double)sizeCopied)/((double)size), ((double)sizeCopied)/(1024.0*1024.0*convertHrToSeconds(readTime)), convertHrToSeconds(readTime), ((double)sizeCopied)/(1024.0*1024.0*convertHrToSeconds(writeTime)), convertHrToSeconds(writeTime));
}

void functionList()
{
    // Make sure the target exists and is an sfile.
    if (!target->exists())
        throw RuntimeException(string("File does not exist: "+target->getFilename()));
    if (!target->isFile())
        throw RuntimeException(string("Not a valid file: "+target->getFilename()));
    if (!target->isSFile())
        throw RuntimeException(string("Not a valid sfile: "+target->getFilename()));

    target->openReadOnly();

    int count=0;
    while (!target->isEof())
    {
        SFileRecord record = target->readNextSFileRecord();
        target->skip(record.dataSize);
        printf("%s\t%lld\t%s\n", record.name.c_str(), record.dataSize, record.type.c_str());
        count++;
    }
    printf("Total records: %d\n",count);

    target->close();
}
