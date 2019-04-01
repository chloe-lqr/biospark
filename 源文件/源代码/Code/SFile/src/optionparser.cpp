
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "file.h"
#include "hdfsfile.h"
#include "localfile.h"
#include "optionparser.h"
#include "runtimeexception.h"

using std::string;
using std::vector;

string function;
vector<File*> sources;
File* target=NULL;
string recordType("mime:application/octet-stream");


/**
 * Parses the command line arguments.
 */
bool parseArguments(int argc, char** argv)
{
    // Parse any arguments.
    for (int i=1; i<argc; i++)
    {
        char *option = argv[i];
        while (*option == ' ') option++;
        
        //See if the user is trying to get help.
        if (strcmp(option, "-h") == 0 || strcmp(option, "--help") == 0) {
            function = "help";
        	break;
        }
        
        //See if the user is trying to get the version info.
        else if (strcmp(option, "-v") == 0 || strcmp(option, "--version") == 0) {
            function = "version";
            break;
        }

        //See if the user is trying to set the type value.
        else if ((strcmp(option, "-t") == 0 || strcmp(option, "--record-type") == 0) && i < (argc-1))
        {
            recordType=string(argv[++i]);
        }
        else if (strncmp(option, "--record-type=", strlen("--record-type=")) == 0)
        {
            recordType=string(option+strlen("--record-type="));
        }

        // See if the user is trying to perform cp.
        else if (function == "" && strcmp(option, "-cp") == 0 && (argc-i-1) >= 2)
        {
            function = "cp";
        }
        else if (function == "cp" && i < argc-1)
        {
            sources.push_back(parseFile(string(option)));
        }
        else if (function == "cp" && i == argc-1)
        {
            target = parseFile(string(option));
        }

        // See if the user is trying to perform ls.
        else if (function == "" && strcmp(option, "-ls") == 0 && (argc-i-1) == 1)
        {
            function = "ls";
        }
        else if (function == "ls" && i == argc-1)
        {
            target = parseFile(string(option));
        }

        /*
        // See if the output aligned images flag has been set.
        else if (strcmp(option, "-oa") == 0 || strcmp(option, "--output-aligned") == 0) {
            outputAlignedImages=true;
        }

        // Get the output prefix.
        else if (argc >= 5 && i == argc-4) {
            outputPrefix = argv[i];
        }
            
        // Get the output prefix.
        else if (argc >= 5 && i == argc-3) {
            imageFilename = argv[i];
        }

        // Get the output prefix.
        else if (argc >= 5 && i == argc-2) {
            backgroundFilename = argv[i];
        }

        // Get the processing area.
        else if (argc >= 5 && i == argc-1) {

            // Parse the coordinates.
            vector<int> coords;
            parseIntListArg(coords, argv[i]);

            // Figure out the width and height.
            int width=coords[2]-coords[0]+1;
            int height=coords[3]-coords[1]+1;

            // Make sure the width and height are divisible by two to work around a Qt image library bug.
            width-=width%2;
            height-=height%2;

            // Create the processing area.
            processingArea=QRect(coords[0],coords[1],width,height);
            functionOption = "processImage";
        }
        */

        //This must be an invalid option.
        else {
            return false;
        }
    }

    return true;
}

void parseIntListArg(vector<int> & list, char * arg)
{
    list.clear();
    char * argbuf = new char[strlen(arg)+1];
    strcpy(argbuf,arg);
    char * pch = strtok(argbuf," ,;:\"");
    while (pch != NULL)
    {
        char * rangeDelimiter;
        if ((rangeDelimiter=strstr(pch,"-")) != NULL)
        {
            *rangeDelimiter='\0';
            int begin=atoi(pch);
            int end=atoi(rangeDelimiter+1);
            for (int i=begin; i<=end; i++)
                list.push_back(i);
        }
        else
        {
            if (strlen(pch) > 0) list.push_back(atoi(pch));
        }
        pch = strtok(NULL," ,;:");
    }
    delete[] argbuf;
}

File* parseFile(string pathname)
{
    // See if this is an hdfs url
    if (pathname.find("hdfs://",0) == 0)
    {
#ifdef OPT_HDFS
        std::size_t t;
        size_t index = pathname.find_first_of('/',strlen("hdfs://"));
        if (index != string::npos)
        {
            string url = pathname.substr(0, index);
            string path = pathname.substr(index);
            return new HDFSFile(url, path);
        }
        throw RuntimeException(string("Could not parse HDFS URL: ")+pathname);
#else
        throw RuntimeException("HDFS support not available.");
#endif
    }
    else
    {
        return new LocalFile(pathname);
    }
}

/**
 * Prints the usage for the program.
 */
void printUsage(int argc, char** argv)
{
    printf("Usage: %s (-h|--help)\n", argv[0]);
    printf("Usage: %s (-v|--version)\n", argv[0]);
    printf("Usage: %s -cp [OPTIONS] [hdfs://hostname/]source_file+ [hdfs://hostname/]target_sfile\n", argv[0]);
    printf("Usage: %s -cp [OPTIONS] [hdfs://hostname/]source_file+ target_dir\n", argv[0]);
    printf("Usage: %s -ls [OPTIONS] [hdfs://hostname/]sfile\n", argv[0]);
    printf(" WHERE\n");
    printf(" CP_OPTIONS\n");
    printf("       -t value            --record-type=value          The value to use when adding record types (default mime:application/octet-stream).\n");
}

/**
 * Prints the copyright notice.
 */
void printCopyright(int argc, char** argv)
{
    printf("SFile v2016.02.19 %lu-bit mode with options:", (sizeof(void*)*8));
#ifdef OPT_HDFS
    printf(" HDFS");
#endif
    printf(".\n");
    printf("Copyright (C) 2014-2016 Roberts Group, Johns Hopkins University.\n");
}

