// DeferredPBR
Texture2D gAlbedo : register(t0); // RGB=albedo (linear)
Texture2D gNormalGloss : register(t1); // RGB=normal (0..1), A=roughness
Texture2D gSpecMisc : register(t2); // RGB=(optional F0), A=metallic
Texture2D gDepth : register(t3); // depth 0..1

//TextureCube gEnvCube : register(t4); // debug sky
TextureCube gIrradiance : register(t5); // diffuse IBL
TextureCube gPrefilter : register(t6); // spec IBL (mipped)
Texture2D gBRDF_LUT : register(t7); // 2D LUT (RG16F)

SamplerState gsamPoint : register(s0);
SamplerState sTriLinear : register(s2);
SamplerState sClamp : register(s3);

// PassCB
struct Light
{
    float3 Strength;
    float FalloffStart;
    float3 Direction;
    float FalloffEnd;
    float3 Position;
    float SpotPower;
};

cbuffer PassCB : register(b0)
{
    float4x4 gView, gInvView, gProj, gInvProj, gViewProj, gInvViewProj;
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
    int gNumDirLights;
    int gNumPointLights;
    int gNumSpotLights;
    float _PadLights;
};

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

float3 ReconstructWorldPos(float2 uv, float depth01)
{
    // uv->[0..1], D3D NDC z=[0..1]
    float2 ndc = uv * float2(2, -2) + float2(-1, 1);
    float4 clip = float4(ndc, depth01, 1);
    float4 view = mul(clip, gInvProj);
    view.xyz /= max(view.w, 1e-6);
    float4 world = mul(float4(view.xyz, 1), gInvView);
    return world.xyz;
}

float3 SafeNormalize(float3 v)
{
    float len2 = dot(v, v);
    return (len2 > 1e-10) ? v * rsqrt(len2) : float3(0, 0, 1);
}

float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    // UE4: lower F with more rough
    float m = saturate(1.0 - roughness);
    return F0 + (max(float3(m, m, m), F0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float4 DeferredPS(VSOut i) : SV_Target
{
    // GBuffer fetch
    float3 albedo = gAlbedo.Sample(gsamPoint, i.TexC).rgb;

    float4 ng = gNormalGloss.Sample(gsamPoint, i.TexC);
    float3 N = SafeNormalize(ng.xyz * 2.0 - 1.0);
    float roughness = saturate(ng.w);

    float4 sm = gSpecMisc.Sample(gsamPoint, i.TexC);
    float metallic = saturate(sm.a);

    // F0: clear metallic workflow
    // (if hardcoded F0 from SpecMisc.rgb — blend max(0.04, sm.rgb) here
    const float3 F0_dielectric = 0.04.xxx;
    float3 F0 = lerp(F0_dielectric, albedo, metallic);

    // position/view 
    float depth01 = gDepth.Sample(gsamPoint, i.TexC).r;
    float3 P = ReconstructWorldPos(i.TexC, depth01);
    float3 V = SafeNormalize(gEyePosW - P);
    float NdotV = max(dot(N, V), 1e-4);

    // IBL diffuse
    float3 irradiance = gIrradiance.Sample(sClamp, N).rgb;
    float3 diffuseIBL = irradiance * albedo;

    // --- IBL specular ---
    uint w, h, mipCount;
    gPrefilter.GetDimensions(0, w, h, mipCount);
    float lod = roughness * (mipCount - 1);

    float3 R = reflect(-V, N);
    float3 prefiltered = gPrefilter.SampleLevel(sTriLinear, R, lod).rgb;
    float2 brdf = gBRDF_LUT.Sample(sClamp, float2(NdotV, roughness)).rg;

    float3 F = FresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic); // kD = 0 (no diffuse) if metal

    float3 specIBL = prefiltered * (F * brdf.x + brdf.y);

    float ao = 1.0;
    
    float specMul = max(gAmbientLight.a, 0.0);
    float3 diffuseMul = gAmbientLight.rgb;

    float3 color = (kD * diffuseIBL + specIBL) * ao * gAmbientLight.rgb;
    color = kD * diffuseIBL * diffuseMul * ao + specIBL * specMul * ao;

    return float4(color, 1);
}
