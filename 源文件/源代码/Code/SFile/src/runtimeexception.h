#ifndef RUNTIMEEXCEPTION_H
#define RUNTIMEEXCEPTION_H

#include <exception>
#include <string>

using std::string;

class RuntimeException : public std::exception
{
public:
    RuntimeException(string exception);
    RuntimeException(string exception, string message);
    virtual ~RuntimeException() throw();
    virtual const char* what() const throw();
protected:
    string exception;
    string message;
};

#endif // RUNTIMEEXCEPTION_H
