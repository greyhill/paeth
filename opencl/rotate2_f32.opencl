// vim: filetype=opencl

inline float Trap_integrate(const float tau0, const float tau1,
                            const float tau2, const float tau3,
                            const float li, const float ri) {
    float accum = 0.f;

    float l = fmin(fmax(li, tau0), tau1);
    float r = fmin(fmax(ri, tau0), tau1);
    accum += ((r - tau0)*(r - tau0) - (l - tau0)*(l - tau0))/(2.f*(tau1 - tau0));

    l = fmin(fmax(li, tau1), tau2);
    r = fmin(fmax(ri, tau1), tau2);
    accum += r - l;

    l = fmin(fmax(li, tau2), tau3);
    r = fmin(fmax(ri, tau2), tau3);
    accum += ((l - tau3)*(l - tau3) - (r - tau3)*(r - tau3))/(2.f*(tau3 - tau2));

    return accum;
}

kernel void shear_x(
        float cx, float cy, float h,
        float tau0, float tau1, float tau2, float tau3,
        int nx, int ny,
        float wx, float wy,
        global float* input,
        global float* tmp) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);

    local float local_val[32*8];
    local int local_idx[32*8];

    int local_id = get_local_id(0) + 32*get_local_id(1);
    int local_di = get_local_id(1) +  8*get_local_id(0);

    if(ix >= nx || iy >= ny) {
        local_val[local_id] = NAN;
        local_idx[local_id] = -1;
    } else {
        float x = ix - wx;
        float y = iy - wy;

        float t0 = x*cx + cy*y + tau0;
        float t1 = x*cx + cy*y + tau1;
        float t2 = x*cx + cy*y + tau2;
        float t3 = x*cx + cy*y + tau3;

        int ix0 = max(0, min((int)(t0 + wx + .5f), nx));
        int ix1 = max(0, min((int)ceil(t3 + wx + .5f), nx));

        float accum = 0.f;
        for(int ix_in=ix0; ix_in<ix1; ++ix_in) {
            float x0_in = ix_in - wx - .5f;
            float x1_in = x0_in + 1.f;
            float w = Trap_integrate(t0, t1, t2, t3, x0_in, x1_in);
            accum += w * input[ix_in + iy*nx];
        }

        local_val[local_id] = h*accum;
        local_idx[local_id] = iy + ny*ix;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float val = local_val[local_di];
    int idx = local_idx[local_di];
    if(idx >= 0) {
        tmp[idx] = val;
    }
}

kernel void shear_y(
        float cy, float cx, float h,
        float tau0, float tau1, float tau2, float tau3,
        int nx, int ny,
        float wx, float wy,
        global float* tmp,
        global float* output) {
    int iy = get_global_id(0);
    int ix = get_global_id(1);

    local float local_val[32*8];
    local int local_idx[32*8];

    int local_id = get_local_id(0) + 32*get_local_id(1);
    int local_di = get_local_id(1) +  8*get_local_id(0);

    if(ix >= nx || iy >= ny) {
        local_val[local_id] = NAN;
        local_idx[local_id] = -1;
    } else {
        float x = ix - wx;
        float y = iy - wy;

        float t0 = x*cx + cy*y + tau0;
        float t1 = x*cx + cy*y + tau1;
        float t2 = x*cx + cy*y + tau2;
        float t3 = x*cx + cy*y + tau3;

        int iy0 = max(0, min((int)(t0 + wy + .5f), ny));
        int iy1 = max(0, min((int)ceil(t3 + wy + .5f), ny));

        float accum = 0.f;
        for(int iy_in=iy0; iy_in<iy1; ++iy_in) {
            float y0_in = iy_in - wy - .5f;
            float y1_in = y0_in + 1.f;
            float w = Trap_integrate(t0, t1, t2, t3, y0_in, y1_in);
            accum += w * tmp[iy_in + ix*ny];
        }

        local_val[local_id] = h*accum;
        local_idx[local_id] = ix + nx*iy;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float val = local_val[local_di];
    int idx = local_idx[local_di];
    if(idx >= 0) {
        output[idx] = val;
    }
}

