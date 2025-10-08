#ifndef PBR_IBL_HLSL
#define PBR_IBL_HLSL

static const float PI = 3.14159265359;

// Schlick Fresnel
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0.xxx - F0) * pow(1.0 - cosTheta, 5.0);
}

// Schlick Fresnel with roughness tweak for ambient
float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max(1.0.xxx - roughness, F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

// Семпл префильтрованного куба с LOD=roughness*(mipCount-1)
float3 SamplePrefiltered(TextureCube pref, SamplerState sTri, float3 R, float roughness)
{
    uint w, h, levels; // доступно в SM5+
    pref.GetDimensions(0, w, h, levels);
    float lod = roughness * max(0, int(levels) - 1);
    return pref.SampleLevel(sTri, R, lod).rgb;
}

// Полная IBL-ambient часть: diffuse(irr) + specular(prefilter+LUT)
float3 ComputeIBL(
    float3 N, float3 V,
    float3 albedo, float roughness, float metallic, float3 F0,
    TextureCube irr, TextureCube pref, Texture2D brdfLUT,
    SamplerState sClamp, SamplerState sTri)
{
    float NdotV = saturate(dot(N, V));
    float3 F_amb = FresnelSchlickRoughness(NdotV, F0, roughness);

    // diffuse (Lambert): irradiance * albedo, затенённый kD=(1-F)*nonMetal
    float3 irradiance = irr.Sample(sClamp, N).rgb;
    float3 kD = (1.0.xxx - F_amb) * (1.0 - metallic);
    float3 diffuseIBL = irradiance * albedo;

    // specular: prefiltered env * BRDF LUT(A,B)
    float3 R = reflect(-V, N);
    float3 prefiltered = SamplePrefiltered(pref, sTri, R, roughness);

    float2 ab = brdfLUT.Sample(sClamp, float2(NdotV, roughness)).rg; // A,B
    float3 specIBL = prefiltered * (F_amb * ab.x + ab.y);

    return kD * diffuseIBL + specIBL;
}

#endif // PBR_IBL_HLSL
