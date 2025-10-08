//deferred lighting
//#include "PBR_IBL.hlsl"
Texture2D gAlbedo : register(t0);
Texture2D gNormalGloss : register(t1);
Texture2D gSpecMisc : register(t2);
Texture2D gDepth : register(t3);
SamplerState gsamPoint : register(s0);

TextureCube gEnvCube : register(t4);
TextureCube gIrradiance : register(t5);
TextureCube gPrefilter : register(t6);
Texture2D gBRDF_LUT : register(t7);

SamplerState sClamp : register(s3); 
SamplerState sTriLinear : register(s2);
struct Light
{
    float3 Strength;
    float FalloffStart;
    float3 Direction;
    float FalloffEnd;
    float3 Position;
    float SpotPower;
};

cbuffer PassCB : register(b0) // light pass
{
    float4x4 gView;
    float4x4 gInvView;
    float4x4 gProj;
    float4x4 gInvProj;
    float4x4 gViewProj;
    float4x4 gInvViewProj;
    float3 gEyePosW;
    float _pad0;
    float2 gRTSize;
    float2 gInvRTSize;
    float gNearZ;
    float gFarZ;
    float gTotalTime;
    float gDeltaTime;
    float4 gAmbientLight;
    
    Light gLights[16];

    
    int gNumDirLights = 0;
    int gNumPointLights = 0;
    int gNumSpotLights = 0;
    float _PadLights = 0.0f; 
};

float3 ReconstructWorldPos(float2 uv, float depth01)
{
    // NDC from uv
    float2 ndc = uv * float2(2, -2) + float2(-1, 1);
    float4 clip = float4(ndc, depth01, 1);

    // in view space
    float4 view = mul(clip, gInvProj);
    view.xyz /= view.w;

    // in world
    float4 world = mul(float4(view.xyz, 1), gInvView);
    return world.xyz;
}

struct VSOut
{
    float4 PosH : SV_Position;
    float2 TexC : TEXCOORD;
};

VSOut FullscreenVS(uint vid : SV_VertexID)
{
    VSOut o;
    float2 uv = float2((vid << 1) & 2, vid & 2);
    o.PosH = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    o.TexC = uv;
    return o;
}

float LinearizeDepth(float d01)
{
    // from PassCB (NearZ/FarZ)
    float z = d01; // [0..1] â D3D
    float n = gNearZ, f = gFarZ;
    // D3D (z in [0..1]):
    return (n * f) / (f - z * (f - n)); // z_view > 0
}


float Attenuation_Point(float d, float s, float e)
{
    float a = saturate((e - d) / max(e - s, 1e-3));
    return a * a;
}

float3 ShadeDirectional(float3 albedo, float3 N, Light L)
{
    float3 Lw = -normalize(L.Direction);
    return albedo * L.Strength * saturate(dot(N, Lw));
}

float3 ShadePoint(float3 albedo, float3 N, float3 P, Light L)
{
    float3 toL = L.Position - P;
    float d = length(toL);
    float3 Lw = toL / max(d, 1e-3);
    float att = Attenuation_Point(d, L.FalloffStart, L.FalloffEnd);
    return albedo * L.Strength * (saturate(dot(N, Lw)) * att);
}

float3 ShadeSpot(float3 albedo, float3 N, float3 P, Light L)
{
    float3 toL = L.Position - P;
    float d = length(toL);
    float3 Lw = toL / max(d, 1e-3);
    float att = Attenuation_Point(d, L.FalloffStart, L.FalloffEnd);
    float cone = pow(saturate(dot(-normalize(L.Direction), Lw)), L.SpotPower);
    return albedo * L.Strength * (saturate(dot(N, Lw)) * att * cone);
}


// 0=Albedo, 1=Normal, 2=Depth, 3=Spec
static const int DEBUG_MODE = 1;

float4 DeferredPS(VSOut i) : SV_Target
{
    
    float3 albedo = gAlbedo.Sample(gsamPoint, i.TexC).rgb;
    float3 N = normalize(gNormalGloss.Sample(gsamPoint, i.TexC).xyz * 2 - 1);
    float depth = gDepth.Sample(gsamPoint, i.TexC).r;
    float3 P = ReconstructWorldPos(i.TexC, depth);
    //return float4(abs(N), 1);
    float3 color = albedo * gAmbientLight.rgb;

    // dir
    [loop]
    for (int iL = 0; iL < gNumDirLights; ++iL)
        color += ShadeDirectional(albedo, N, gLights[iL]);

    // point
    int basePoint = gNumDirLights;
    [loop]
    for (int iL = 0; iL < gNumPointLights; ++iL)
        color += ShadePoint(albedo, N, P, gLights[basePoint + iL]);

    // spot
    int baseSpot = gNumDirLights + gNumPointLights;
    [loop]
    for (int iL = 0; iL < gNumSpotLights; ++iL)
        color += ShadeSpot(albedo, N, P, gLights[baseSpot + iL]);

    return float4(color, 1);
}