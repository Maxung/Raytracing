#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"

class Camera {
public:
	__device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect) {
        Vec3 u, v, w;
        float theta = vfov * M_PI / 180;
        float half_height = tan(theta/2);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
		lower_left_corner = origin - half_width * u - half_height * v - w;
    	horizontal = 2 * half_width * u;
    	vertical = 2 * half_height * v;
	}
	__device__ Ray get_ray(float u, float v) {
		return Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	}
	
	Vec3 origin;
	Vec3 lower_left_corner;
	Vec3 horizontal;
	Vec3 vertical;
};

#endif
