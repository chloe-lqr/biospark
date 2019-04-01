#ifndef OPTIONPARSER_H
#define OPTIONPARSER_H

#include <cstring>
#include <string>
#include <vector>

#include <stdint.h>

#include "file.h"
#include "localfile.h"

using std::string;
using std::vector;

extern string function;
extern vector<File*> sources;
extern File* target;
extern string recordType;

bool parseArguments(int argc, char** argv);
void parseIntListArg(vector<int> & list, char * arg);
File* parseFile(string pathname);
void printUsage(int argc, char** argv);
void printCopyright(int argc, char** argv);

#endif
