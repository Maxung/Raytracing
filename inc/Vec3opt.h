#ifndef VEC3_OPT_H
#define VEC3_OPT_H

#include <immintrin.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

struct Vec3 {
    __m128 xmm;
    Vec3() {};
    Vec3(__m128 const &x) { xmm = x; };
    Vec3(float a, float b, float c) { xmm = _mm_set_ps(0, c, b, a); };
    Vec3(float t) { xmm = _mm_set_ps(0, t, t, t); };
    Vec3& operator=(__m128 const &x) { xmm = x; return *this; };
    void store(float* p) const { _mm_storeu_ps(p, xmm); };
    operator __m128() const { return xmm; };
    inline float x() { return ((float*)&xmm)[0]; }; 
    inline float y() { return ((float*)&xmm)[1]; }; 
    inline float z() { return ((float*)&xmm)[2]; }; 
    inline float r() { return ((float*)&xmm)[0]; };
    inline float g() { return ((float*)&xmm)[1]; };
    inline float b() { return ((float*)&xmm)[2]; }; 
    inline Vec3& operator+=(Vec3 const &v);
    inline Vec3& operator-=(Vec3 const &v);
    inline Vec3& operator*=(Vec3 const &v);
    inline Vec3& operator/=(Vec3 const &v);
    inline float length() {
        __m128 r1     = _mm_mul_ps(xmm, xmm);
        /*__m128 shuf   = _mm_movehdup_ps(r1);
        __m128 sums   = _mm_add_ps(r1, shuf);
        shuf          = _mm_movehl_ps(shuf, sums);
        sums          = _mm_add_ss(sums, shuf);
        return sqrt(_mm_cvtss_f32(sums));*/
		__m128 v = _mm_hadd_ps(r1, r1);
		v = _mm_hadd_ps(v, v);
		return sqrt(_mm_cvtss_f32(v));
    }
    inline float squared_length() {
		__m128 r1 = _mm_mul_ps(xmm, xmm);
        /*__m128 shuf   = _mm_movehdup_ps(r1);
        __m128 sums   = _mm_add_ps(r1, shuf);
        shuf          = _mm_movehl_ps(shuf, sums);
        sums          = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);*/
		__m128 v = _mm_hadd_ps(r1, r1);
		v = _mm_hadd_ps(v, v);
		return _mm_cvtss_f32(v);
    }
    inline void make_unit_vector() {
        float k = 1.0 / length();
        __m128 kvec = _mm_set_ps(0, k, k, k);
        xmm = _mm_mul_ps(xmm, kvec);
    }
	inline Vec3 insert(int index, float value) {
		switch (index & 3) {
		case 0:
			xmm = _mm_insert_ps(xmm, _mm_set_ss(value), 0 << 4);  break;
		case 1:
			xmm = _mm_insert_ps(xmm, _mm_set_ss(value), 1 << 4);  break;
		case 2:
			xmm = _mm_insert_ps(xmm, _mm_set_ss(value), 2 << 4);  break;
		default:
			xmm = _mm_insert_ps(xmm, _mm_set_ss(value), 3 << 4);  break;
		}
		return *this;
	}
};

inline Vec3 operator+(Vec3 const &a, Vec3 const &b) {
    return _mm_add_ps(a, b);
}
inline Vec3& Vec3::operator+=(Vec3 const &v) {
   xmm = _mm_add_ps(xmm, v); 
   return *this;
}

inline Vec3 operator-(Vec3 const &a, Vec3 const &b) {
    return _mm_sub_ps(a, b);
}
inline Vec3& Vec3::operator-=(Vec3 const &v) {
    xmm = _mm_sub_ps(xmm, v);
    return *this;
}

inline Vec3 operator*(Vec3 const &a, Vec3 const &b) {
    return _mm_mul_ps(a, b);
}
inline Vec3& Vec3::operator*=(Vec3 const &v) {
    xmm = _mm_mul_ps(xmm, v);
    return *this;
}
inline Vec3 operator*(Vec3 const &a, float t) {
    return _mm_mul_ps(a, Vec3(t));
}

inline Vec3 operator/(Vec3 const &a, Vec3 const &b) {
    __m128 r = _mm_div_ps(a, b);
	return _mm_insert_ps(r, _mm_set_ss(0.0), 3 << 4);
}
inline Vec3& Vec3::operator/=(Vec3 const &v) {
    xmm = _mm_div_ps(xmm, v);
    return *this;
}
inline Vec3 operator/(Vec3 const &a, float t) {
	__m128 r = _mm_div_ps(a, Vec3(t));
	return _mm_insert_ps(r, _mm_set_ss(0.0), 3 << 4);
}
inline Vec3 operator/(float t, Vec3 const &a) {
	return _mm_div_ps(Vec3(t), a);
}

inline float dot(Vec3 const &a, Vec3 const &b) {
    __m128 r1     = _mm_mul_ps(a, b);
    /*__m128 shuf   = _mm_movehdup_ps(r1);
    __m128 sums   = _mm_add_ps(r1, shuf);
    shuf          = _mm_movehl_ps(shuf, sums);
    sums          = _mm_add_ss(sums, shuf);*/
	__m128 v = _mm_hadd_ps(r1, r1);
	v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

inline Vec3 cross(Vec3 const &a, Vec3 const &b) {
    return _mm_sub_ps(
                _mm_mul_ps(
                    _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)),
                    _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))
                ),
                _mm_mul_ps(
                    _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)),
                    _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1))
                )
            );
}

inline Vec3 unit_vector(Vec3 v) {
    return (v / v.length()).insert(3, 0.0);
}

inline std::ostream& operator<<(std::ostream &os, Vec3 &t) {
    float p[4];
    t.store(p);
    os << p[0] << " " << p[1] << " " << p[2];
    return os;
}

#endif
