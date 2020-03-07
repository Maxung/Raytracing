#include "Sphere.h"
#include "HittableList.h"
#include "Vec3.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cfloat>
#include "Camera.h"
#include "Material.h"

Vec3 color(const Ray& r, Hittable *world, int depth) {
    hit_record rec;
    if (world->hit(r, 0.001, FLT_MAX, rec)) { // color hit Sphere
        Ray scattered;
        Vec3 attenuation;
        if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * color(scattered, world, depth + 1);
        else
            return Vec3(0, 0, 0);
    }
    else { // background
        Vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5 * (unit_direction.y() + 1.0); // scaling to 0.0 <-> 1.0
        return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0); // blend
    }
}

Hittable *random_scene() {
    int n = 500;
    Hittable **list = new Hittable*[n+1];
    list[0] =  new Sphere(Vec3(0,-1000,0), 1000, new Lambertian(Vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = random_float();
            Vec3 center(a+0.9*random_float(),0.2,b+0.9*random_float());
            if ((center-Vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new Sphere(center, 0.2,
                        new Lambertian(Vec3(random_float()*random_float(),
                                            random_float()*random_float(),
                                            random_float()*random_float())
                        )
                    );
                }
                else if (choose_mat < 0.95) { // Metal
                    list[i++] = new Sphere(center, 0.2,
                            new Metal(Vec3(0.5*(1 + random_float()),
                                           0.5*(1 + random_float()),
                                           0.5*(1 + random_float())),
                                           0.5*random_float()));
                }
            }
        }
    }

    list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));
    list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)));

    return new HittableList(list,i);
}

int main() {
    int nx = 1000;
    int ny = 500;
    int ns = 10;
    unsigned char image[nx * ny * 3]; // RGB image
    int index = 0;
    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);

    // setup scene
    Hittable *list[4];
    list[0] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.8, 0.3, 0.3)));
    list[1] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
    list[2] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0.5));
    list[3] = new Sphere(Vec3(-1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.8, 0.8), 0.5));
    Hittable *world = new HittableList(list, 4);
    world = random_scene();

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(nx) / float(ny));

    // going from left top to right bottom (row by row)
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + random_float()) / float(nx);
                float v = float(j + random_float()) / float(ny);
                Ray r = cam.get_ray(u, v);
                col += color(r, world, 0);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);
            image[index++] = ir;
            image[index++] = ig;
            image[index++] = ib;
        }
    }
    stbi_write_jpg("render.jpg", nx, ny, 3, image, 100);
}
