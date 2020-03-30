#include "Sphere.h"
#include "HittableList.h"
#include "Vec3.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cfloat>
#include <ctime>
#include <iostream>
#include "Camera.h"
#include "Material.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Vec3 color(const Ray& r, Hittable **world, int depth, curandState *lrs) {
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1, 1, 1);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, lrs)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return Vec3(0, 0, 0);
            }
        }
        else { // background
            Vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f); // scaling to 0.0 <-> 1.0
            Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0); // blend
            return cur_attenuation * c;
        }
    }
    return Vec3(0, 0, 0);
}

__global__ void random_scene(Hittable **list,  Hittable **world, Camera **camera, int nx, int ny, curandState
        *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState lrs = *rand_state;
        list[0] =  new Sphere(Vec3(0,-1000,0), 1000, new Lambertian(Vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = curand_uniform(&lrs);
                Vec3 center(a+0.9*curand_uniform(&lrs),0.2,b+0.9*curand_uniform(&lrs));
                if ((center-Vec3(4,0.2,0)).length() > 0.9) {
                    if (choose_mat < 0.8) {  // diffuse
                        list[i++] = new Sphere(center, 0.2,
                            new Lambertian(Vec3(curand_uniform(&lrs)*curand_uniform(&lrs),
                                            curand_uniform(&lrs)*curand_uniform(&lrs),
                                            curand_uniform(&lrs)*curand_uniform(&lrs))
                            )
                        );
                    }
                    else if (choose_mat < 0.95) { // Metal
                        list[i++] = new Sphere(center, 0.2,
                                new Metal(Vec3(0.5*(1 + curand_uniform(&lrs)),
                                           0.5*(1 + curand_uniform(&lrs)),
                                           0.5*(1 + curand_uniform(&lrs))),
                                           0.5*curand_uniform(&lrs)));
                    }
                }
            }
        }

        list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));
        list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)));

        *world = new HittableList(list,i);
        *camera = new Camera(Vec3(13, 2, 3), Vec3(0, 0, 0), Vec3(0, 1, 0), 20, float(nx) / float(ny));
    }
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        curand_init(1984, 0, 0, rand_state);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int max_x, int max_y,  curandState *rand_state, Camera **cam, int ns,
        Hittable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState lrs = rand_state[pixel_index];
    Vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&lrs)) / float(max_x);
        float v = float(j + curand_uniform(&lrs)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v);
        col += color(r, world, 0, &lrs);
    }
    col /= float(ns);
    col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

    fb[pixel_index] = col;
}

__global__ void cleanup(Hittable **list, Hittable **world,  Camera **camera) {
    for(int i = 0; i < 500; i++) {
        delete (reinterpret_cast<Sphere*>(list[i])->mat_ptr);
        delete list[i]; 
    }
    delete *world;
    delete *camera;
}

int main() {
    const int nx = 2000;
    const int ny = 1000;
    int ns = 10;
    unsigned char image[nx * ny * 3]; // RGB image
    Vec3 *fb;
    Hittable **list;
    Hittable **world;
    Camera **camera;
    std::clock_t start = std::clock();
	
    int tx = 8;
    int ty = 8;
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    curandState *rand_state;
    curandState *rand_state2;

    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&fb), nx * ny * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&rand_state), nx * ny * sizeof(curandState)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&rand_state2), 1 * sizeof(curandState)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&list), 500 * sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&world), sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&camera), sizeof(Camera *)));

    rand_init<<<1, 1>>>(rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    random_scene<<<1, 1>>>(list, world, camera, nx, ny, rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render_init<<<blocks, threads>>>(nx, ny, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, nx, ny, rand_state, camera, ns, world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int index = 0;
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99f * fb[pixel_index].r());
            int ig = int(255.99f * fb[pixel_index].g());
            int ib = int(255.99f * fb[pixel_index].b());
            image[index++] = ir;
            image[index++] = ig;
            image[index++] = ib;
        }
    }
	
    std::clock_t end = std::clock();
    std::cout << "CPU time: " << 1000.0 * (end - start) / CLOCKS_PER_SEC << "ms" << std::endl;    

    stbi_write_jpg("render.jpg", nx, ny, 3, image, 100);

    checkCudaErrors(cudaDeviceSynchronize());
    cleanup<<<1, 1>>>(list, world, camera);
    checkCudaErrors(cudaGetLastError());
    //checkCudaErrors(cudaFree(camera));
    //checkCudaErrors(cudaFree(world));
    //checkCudaErrors(cudaFree(list));
    //checkCudaErrors(cudaFree(rand_state));
    
    cudaDeviceReset();
}
