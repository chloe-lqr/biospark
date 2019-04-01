#include <exception>
#include <string>
#include "runtimeexception.h"

using std::string;

RuntimeException::RuntimeException(string exception)
:exception(exception),message("")
{
}

RuntimeException::RuntimeException(string exception, string message)
:exception(exception),message(message)
{
}

RuntimeException::~RuntimeException() throw()
{

}

const char* RuntimeException::what() const throw()
{
    if (message != "")
        return (exception+": "+message).c_str();
    else
        return exception.c_str();
}
