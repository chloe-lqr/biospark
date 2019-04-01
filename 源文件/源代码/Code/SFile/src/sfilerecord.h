#ifndef SFILERECORD_H
#define SFILERECORD_H

#include <string>
#include <stdint.h>

using std::string;

class SFileRecord
{
public:
    static const char RECORD_SEPARATOR[16];

public:
    SFileRecord();
    SFileRecord(string name, string type, int64_t dataSize);
    string name;
    string type;
    int64_t dataSize;
};

#endif // SFILERECORD_H

