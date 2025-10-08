// ===================== GBuffer =====================
cbuffer ObjectCB : register(b0)
{
    float4x4 gWorld;
    float4x4 gTexTransform;
    float4x4 gWorldInvTranspose;

    // per-object overrides; <0 => использовать материал/дефолт
    float Roughness;
    float Metallic;
    float2 padRM;
};

cbuffer MaterialCB : register(b1)
{
    float4 gDiffuseAlbedo;
    float3 gFresnelR0;
    float gRoughness;
    float4x4 gMatTransform;
};

cbuffer PassCB : register(b2)
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
}

Texture2D gDiffuseMap : register(t0);
SamplerState gsamAnisoWrap : register(s0);

struct VSIn
{
    float3 PosL : POSITION;
    float3 NormalL : NORMAL;
    float2 TexC : TEXCOORD;
};

struct VSOut
{
    float4 PosH : SV_Position;
    float3 PosW : POSITIONWS;
    float3 NormalW : NORMALWS;
    float2 TexC : TEXCOORD;
};

VSOut VS(VSIn v)
{
    VSOut o;

    float4 posW = mul(float4(v.PosL, 1.0f), gWorld);
    o.PosW = posW.xyz;

    float3x3 Nmat = (float3x3) gWorldInvTranspose;
    o.NormalW = normalize(mul(Nmat, v.NormalL));

    o.PosH = mul(posW, gViewProj);
    // при желании можно применить gTexTransform к UV
    o.TexC = v.TexC;
    return o;
}

struct PSOut
{
    float4 Albedo : SV_Target0; // RGB: albedo, A: зарезервировано
    float4 NormalGloss : SV_Target1; // RGB: normal(0..1), A: roughness
    float4 SpecMisc : SV_Target2; // RGB: F0 (Fresnel R0), A: metallic
};

PSOut PS(VSOut i)
{
    PSOut o;

    // текстура * цвет материала
    float3 albedoTex = gDiffuseMap.Sample(gsamAnisoWrap, i.TexC).rgb;
    float3 albedo = albedoTex * gDiffuseAlbedo.rgb;

    // нормаль -> [0..1]
    float3 n01 = normalize(i.NormalW) * 0.5f + 0.5f;

    // --------- Roughness/Metallic c override ---------
    // Если в ObjectCB значение >= 0 — используем его; иначе берём из материала/дефолт.
    float rough = (Roughness >= 0.0f) ? Roughness : gRoughness;
    float metal = (Metallic >= 0.0f) ? Metallic : 0.0f;
    
    float3 F0 = lerp(gFresnelR0, albedo, saturate(metal));

    rough = saturate(rough);
    metal = saturate(metal);

    // --------- Запись в MRT ---------
    o.Albedo = float4(albedo, 1.0f);
    o.NormalGloss = float4(n01, rough); // A = roughness (несмотря на имя *Gloss*)
    o.SpecMisc = float4(F0, metal); // A = metallic, RGB = F0

    return o;
}
