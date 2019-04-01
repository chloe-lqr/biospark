#ifndef HRTIME_H
#define HRTIME_H

#include <stdint.h>

#if defined(WINDOWS)
    #define NOMINMAX
    #include <Windows.h>
    typedef long long hrtime;
#elif defined(MACOSX)
    #include <mach/mach_time.h>
    typedef uint64_t hrtime;
#elif defined(LINUX)
    #include <time.h>
    typedef uint64_t hrtime;
#endif


inline hrtime getHrTime()
{
#if defined(WINDOWS)
    LARGE_INTEGER ticks;
    QueryPerformanceCounter(&ticks);
    return ticks.QuadPart;
#elif defined(MACOSX)
    return mach_absolute_time();
#elif defined(LINUX)
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t ret=now.tv_sec;
    ret *= 1000000000ULL;
    ret += now.tv_nsec;
    return ret;
#endif
}

inline double convertHrToSeconds(hrtime time)
{
#if defined(WINDOWS)
    static bool _time_mult_init = false;
    static double _time_mult = 0.0;
    if (!_time_mult_init)
    {
        LARGE_INTEGER tickFreqOrig;
        QueryPerformanceFrequency(&tickFreqOrig);
        _time_mult = 1.0/(double)tickFreqOrig.QuadPart;
        _time_mult_init = true;
    }
    return ((double)time)*_time_mult;

#elif defined(MACOSX)
    static bool _time_mult_init = false;
    static double _time_mult = 0.0;
    if (!_time_mult_init)
    {
        mach_timebase_info_data_t _time_info;
        mach_timebase_info(&_time_info);
        _time_mult=1e-9*((double)_time_info.numer)/((double)_time_info.denom);
        _time_mult_init = true;
    }
    return ((double)time)*_time_mult;
#elif defined(LINUX)
    return ((double)time)*1e-9;
#endif
}

#endif // HRTIME_H
