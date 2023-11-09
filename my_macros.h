#pragma once

//#ifdef MY_LIBRARY_EXPORTS
//#define MY_LIBRARY_API __declspec(dllexport)
//#else
//#define MY_LIBRARY_API __declspec(dllimport)
//#endif

#ifdef _WIN32
#ifdef MY_LIBRARY_EXPORTS
#define MY_LIBRARY_API __declspec(dllexport)
#else
#define MY_LIBRARY_API __declspec(dllimport)
#endif
#else
#define MY_LIBRARY_API
#endif