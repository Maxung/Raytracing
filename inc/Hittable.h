#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"

class Material;

struct hit_record {
	float t;
	Vec3 p;
	Vec3 normal;
    Material *mat_ptr;
};

class Hittable {
public:
	__device__ virtual bool hit (const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif
