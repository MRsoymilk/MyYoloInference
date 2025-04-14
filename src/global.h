#ifndef GLOBAL_H
#define GLOBAL_H

#if defined(_MSC_VER) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(WIN32) || \
    defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define MYYOLOINFERENCE_EXPORT __declspec(dllexport)
#define MYYOLOINFERENCE_IMPORT __declspec(dllimport)
#else
#define MYYOLOINFERENCE_EXPORT __attribute__((visibility("default")))
#define MYYOLOINFERENCE_IMPORT __attribute__((visibility("default")))
#endif

#if defined(MYYOLOINFERENCE_LIBRARY)
#define MYYOLOINFERENCE_API MYYOLOINFERENCE_EXPORT
#else
#define MYYOLOINFERENCE_API MYYOLOINFERENCE_IMPORT
#endif

#endif  // GLOBAL_H
