#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <functional>

float random_float() {
    static std::uniform_real_distribution<> dist(0.0, 1.0);
    static std::default_random_engine e(time(0));
    return dist(e);
}

#endif
