struct VSOut
{
    float4 posH : SV_Position;
    float2 uv : TEXCOORD0;
    float4 col : COLOR0;
};

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

StructuredBuffer<Particle> gParticles : register(t0);
Texture2D spriteTex : register(t1);
SamplerState samLinear : register(s2); // linearWrap from static samplers

cbuffer PassCB : register(b0)
{
    float4x4 View;
    float4x4 InvView;
    float4x4 Proj;
    float4x4 InvProj;
    float4x4 ViewProj;
    float4x4 InvViewProj;
    float3 EyePosW;
    float _pad0;
    float2 RTSize;
    float2 InvRTSize;
    float NearZ;
    float FarZ;
    float TotalTime;
    float DeltaTime;
    float4 Ambient;
    float4 _padL[16]; // padding
};

VSOut VS(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    VSOut o;
    Particle p = gParticles[iid];
    if (p.alive == 0)
    {
        o.posH = float4(0, 0, 0, 1);
        o.uv = 0;
        o.col = 0;
        return o;
    }

    float2 c = float2((vid & 1) ? +0.5 : -0.5, (vid & 2) ? +0.5 : -0.5);
    float4 v = mul(float4(p.pos, 1), View);
    float s = p.size;
    v.xyz += float3(c.x * s, c.y * s, 0);
    o.posH = mul(v, Proj);
    o.uv = c * 0.5 + 0.5;
    o.col = p.color * saturate(p.life / max(1e-3, p.lifetime));
    return o;
}

float4 PS(VSOut i) : SV_Target
{
    float4 t = spriteTex.Sample(samLinear, i.uv);
    return float4(0,0,1,1) * i.col;
}
