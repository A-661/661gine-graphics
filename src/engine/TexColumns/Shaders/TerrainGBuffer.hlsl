// b0 : PassCB
// b1 : TerrainGlobalsCB
// t0 : Texture2DArray gDiffuseArr
// t1 : Texture2DArray gNormalArr 
// t2 : Texture2DArray gHeightArr
// t3 : StructuredBuffer<TerrainInstanceGPU> gInst

cbuffer PassCB : register(b0)
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
};

cbuffer TerrainGlobalsCB : register(b1)
{
    uint gGridRes; // 257
    float gSkirtHeight; //0
    float gRoughness;
    float gMetallic;
};

struct TerrainInstanceGPU
{
    float2 originWS; // XZ
    float tileSize;
    float heightScale;

    uint diffSlice;
    uint normSlice;
    uint heightSlice;
    float morph;

    float2 uvScale;
    float2 _pad_;
};

Texture2DArray gDiffuseArr : register(t0);
Texture2DArray gNormalArr : register(t1);
Texture2DArray gHeightArr : register(t2);
StructuredBuffer<TerrainInstanceGPU> gInst : register(t3);

SamplerState gsamLinearClamp : register(s3);

struct VSIn
{
    float2 Grid : TEXCOORD0; // (x,y) = [0..gGridRes-1]
};

struct VSOut
{
    float4 PosH : SV_Position;
    float3 PosW : POSITIONWS;
    float3 NormalW : NORMALWS;
    float2 UV : TEXCOORD0;
    uint InstId : SV_InstanceID;
};

// [0..1]
float SampleHeight(uint instId, float2 uv01)
{
    TerrainInstanceGPU inst = gInst[instId];
    float3 uvw = float3(uv01, inst.heightSlice);
    float h = gHeightArr.SampleLevel(gsamLinearClamp, uvw, 0).r;
    return h;
}

VSOut VS(VSIn v, uint iid : SV_InstanceID)
{
    VSOut o = (VSOut) 0;
    TerrainInstanceGPU inst = gInst[iid];

    float resM1 = max(1.0, (float) (gGridRes - 1));
    float2 f = v.Grid / resM1;

    // XZ
    float2 xz = inst.originWS + f * inst.tileSize;

    // h
    float h = SampleHeight(iid, f);
    float y = h * inst.heightScale;

    float3 Pw = float3(xz.x, y, xz.y);

    float2 dfu = float2(1.0 / resM1, 0.0);
    float2 dfv = float2(0.0, 1.0 / resM1);

    float hU = SampleHeight(iid, saturate(f + dfu)) * inst.heightScale;
    float hV = SampleHeight(iid, saturate(f + dfv)) * inst.heightScale;

    float dx = inst.tileSize / resM1;
    float dz = inst.tileSize / resM1;

    float3 Pu = float3(xz.x + dx, hU, xz.y);
    float3 Pv = float3(xz.x, hV, xz.y + dz);

    float3 du = Pu - Pw;
    float3 dv = Pv - Pw;

    float3 n = normalize(cross(dv, du));

    o.PosW = Pw;
    o.NormalW = n;
    o.PosH = mul(float4(Pw, 1.0), gViewProj);
    o.UV = f * inst.uvScale;
    o.InstId = iid;
    return o;
}

struct PSOut
{
    float4 Albedo : SV_Target0; // RGB=albedo
    float4 NormalGloss : SV_Target1; // RGB=normal(0..1), A=roughness
    float4 SpecMisc : SV_Target2; // RGB=F0, A=metallic
};

PSOut PS(VSOut i)
{
    TerrainInstanceGPU inst = gInst[i.InstId];
    float3 albedo = gDiffuseArr.Sample(gsamLinearClamp, float3(i.UV, inst.diffSlice)).rgb;
    float3 n01 = normalize(i.NormalW) * 0.5f + 0.5f;

    float rough = saturate(gRoughness);
    float metal = saturate(gMetallic);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metal);

    PSOut o;
    o.Albedo = float4(albedo, 1.0);
    o.NormalGloss = float4(n01, rough);
    o.SpecMisc = float4(F0, metal);
    return o;
}

VSOut VS_Skirt(VSIn v, uint iid : SV_InstanceID, uint vid : SV_VertexID)
{
    VSOut o = (VSOut) 0;

    TerrainInstanceGPU inst = gInst[iid];

    float resM1 = max(1.0, (float) (gGridRes - 1));
    float2 f = v.Grid / resM1;

    float2 xz = inst.originWS + f * inst.tileSize;

    float h = SampleHeight(iid, f);
    float y = h * inst.heightScale;

    float3 Pw = float3(xz.x, y, xz.y);

    float2 dfu = float2(1.0 / resM1, 0.0);
    float2 dfv = float2(0.0, 1.0 / resM1);

    float hU = SampleHeight(iid, saturate(f + dfu)) * inst.heightScale;
    float hV = SampleHeight(iid, saturate(f + dfv)) * inst.heightScale;

    float dx = inst.tileSize / resM1;
    float dz = inst.tileSize / resM1;

    float3 Pu = float3(xz.x + dx, hU, xz.y);
    float3 Pv = float3(xz.x, hV, xz.y + dz);

    float3 du = Pu - Pw;
    float3 dv = Pv - Pw;
    float3 n = normalize(cross(dv, du));

    bool bottom = (vid & 1) == 1;
    if (bottom)
    {
        Pw.y -= gSkirtHeight;
    }

    o.PosW = Pw;
    o.NormalW = n;
    o.PosH = mul(float4(Pw, 1.0), gViewProj);
    o.UV = f * inst.uvScale;
    o.InstId = iid;
    return o;
}