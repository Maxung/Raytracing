#include "Sphere.h"
#include "HittableList.h"
#include "Vec3.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cfloat>
#include "Camera.h"
#include "Material.h"
#include <mpi.h>
#include <ctime>
#include <fstream>

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

Hittable* share_scene(HittableList *world, int &rank) {
    int list_size;
    if (rank == 0)
        list_size = world->list_size;
    MPI_Bcast(&list_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    Hittable **list = new Hittable*[list_size]; 
    for (int i = 0; i < list_size; i++) {
        float data[8];
        if (rank == 0) {
            Sphere *sph = static_cast<Sphere*>(world->list[i]);
            data[0] = sph->center.x();
            data[1] = sph->center.y();
            data[2] = sph->center.z();
            data[3] = sph->radius;
            data[4] = 0;
            data[5] = 0;
            data[6] = 0;
            data[7] = -1;
            if (sph->mat_type == 0) {
                Lambertian *lamb = static_cast<Lambertian*>(sph->mat_ptr);
                data[4] = lamb->albedo.r();
                data[5] = lamb->albedo.g();
                data[6] = lamb->albedo.b();
            } else if (sph->mat_type == 1) {
                Metal *met = static_cast<Metal*>(sph->mat_ptr);
                data[4] = met->albedo.r();
                data[5] = met->albedo.g();
                data[6] = met->albedo.b();
                data[7] = met->fuzz;
            }
        }

        MPI_Bcast(&data, 8, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            Vec3 center(data[0], data[1], data[2]);
            float radius = data[3];
            Vec3 albedo(data[4], data[5], data[6]);
            if (data[7] >= 0) {
                list[i] = new Sphere(center, radius, new Metal(albedo, data[7]), 1);
            } else {
                list[i] = new Sphere(center, radius, new Lambertian(albedo), 0);
            } 
        }
    }
    if (rank == 0)
        return world;
    else
        return new HittableList(list, list_size);
}

HittableList* random_scene() {
    int n = 500;
    Hittable **list = new Hittable*[n+1];
    list[0] =  new Sphere(Vec3(0,-1000,0), 1000, new Lambertian(Vec3(0.5, 0.5, 0.5)), 0);
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
                                            random_float()*random_float())),0);
                }
                else if (choose_mat < 0.95) { // Metal
                    list[i++] = new Sphere(center, 0.2,
                            new Metal(Vec3(0.5*(1 + random_float()),
                                           0.5*(1 + random_float()),
                                           0.5*(1 + random_float())),
                                           0.5*random_float()), 1);
                }
            }
        }
    }

    list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0), 1);
    list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)), 0);

    return new HittableList(list, i);
}

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::clock_t start_time;

    if (rank == 0)
        start_time = std::clock();

    int nx = 2000;
    int ny = 1000;
    int ns = 10;
    std::vector<unsigned char> image;

    if (rank != 0)
        image.reserve(nx * ny * 3 / size); // RGB image for each process
    else
        image.reserve(nx * ny * 3);

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);

    HittableList *world_list;

    if (rank == 0) {
        // setup scene
        world_list = random_scene();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    Hittable* world = share_scene(world_list, rank);

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(nx) / float(ny));

    // calculate start and stop point for each rank
    int start = (ny) - ((int((ny) / size)) * rank) - 1;
    int stop = (ny) - ((int((ny) / size)) * (rank+1));

    // to compensate for the int division, the last rank might has to do some more
    if (rank == size - 1)
        stop = 0;

    // going from left top to right bottom (row by row)
    for (int j = start; j >= stop; j--) {
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
            image.push_back(ir);
            image.push_back(ig);
            image.push_back(ib);
        }
    }

    if (rank != 0) {
        int image_size = image.size();
        MPI_Send(&image_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(image.data(), image_size, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    else if (rank == 0) {
        std::vector<int> processed_pixels;
        for (int i = 1; i < size; i++) {
            int image_size;
            MPI_Recv(&image_size, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, nullptr);
            processed_pixels.push_back(image_size);
            unsigned char buf[image_size];
            MPI_Recv(&buf, image_size, MPI_UNSIGNED_CHAR, i, MPI_ANY_TAG, MPI_COMM_WORLD, nullptr);
            image.insert(image.end(), buf, buf + image_size);
        }

        // make sure we didn't miss a pixel
        assert(image.size() == nx * ny * 3);

        int avg_pixels;
        for (const int &i : processed_pixels) {
           avg_pixels += i/3; 
        }
        avg_pixels /= size;
        std::cout << "Each process processed ~" << avg_pixels << " pixels" <<  std::endl;

        std::clock_t end = std::clock();
        std::cout << "CPU time: " << 1000.0 * (end - start) / CLOCKS_PER_SEC << "ms" << std::endl;
        stbi_write_jpg("render.jpg", nx, ny, 3, image.data(), 100);
    }

    MPI_Finalize();
    return 0;
}
