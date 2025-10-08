//#define JUMP

struct Particle
{
    float3 pos;
    float3 vel;
    float life;
    float lifetime;
    float size;
    float rot;
    uint alive;
    float4 color;
    uint pad;
};

StructuredBuffer<Particle> gIn : register(t0);
RWStructuredBuffer<Particle> gOut : register(u0);

cbuffer SimCB : register(b0)
{
    float dt;
    float3 gravity; // 16
    float3 sphereC;
    float sphereR; // 16
};

static const float kShell = 0.10;
static const float kSlideFriction = 0.05; // trenie

void DodgeSphere(in float3 C, in float R, inout float3 pos, inout float3 vel)
{
    float3 v = pos - C;
    float d2 = dot(v, v);
    float Rinfl = R + kShell;

    if (d2 < Rinfl * Rinfl)
    {
        float d = sqrt(max(d2, 1e-8));
        float3 n = (d > 1e-6) ? (v / d) : float3(0, 1, 0);

        // push out 
        pos = C + n * Rinfl;

        // kasatelnaya
        float vn = dot(vel, n);
        if (vn < 0.0)
            vel -= vn * n;

        // trenie po kasatelnoy
        float3 tang = vel - n * dot(vel, n);
        vel = tang * (1.0 - kSlideFriction) + n * max(0.0, dot(vel, n));
    }
}

[numthreads(256, 1, 1)]
void CS(uint3 id : SV_DispatchThreadID)
{
    
    uint i = id.x;
    Particle p = gIn[i];
    if (p.alive != 0)
    {
        p.vel += gravity * dt;
        p.pos += p.vel * dt;
        DodgeSphere(sphereC, sphereR, p.pos, p.vel);
        p.life -= dt;
        #ifdef JUMP
        if (p.pos.y < 0.0)
        {
            p.pos.y = 0.0;
            p.vel.y = abs(p.vel.y) * 0.5;
        } 
        #endif // jump
        if (p.life <= 0)
        {
            p.alive = 0;
            p.life = p.lifetime;
            p.vel = float3(0,-1.5,0);
            p.pos.y = 35.0;
            p.alive = 1;
        }
    }
    gOut[i] = p;
}
