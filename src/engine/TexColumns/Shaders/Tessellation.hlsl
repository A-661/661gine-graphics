//***************************************************************************************
// Tessellation.hlsl by Frank Luna (C) 2015 All Rights Reserved.
//***************************************************************************************


// Include structures and functions for lighting.
#include "LightingUtil.hlsl"
#define WORLD_DISPLACEMENT
Texture2D    gDiffuseMap : register(t0);
Texture2D	 gNormalMap : register(t1); // global


SamplerState gsamPointWrap        : register(s0);
SamplerState gsamPointClamp       : register(s1);
SamplerState gsamLinearWrap       : register(s2);
SamplerState gsamLinearClamp      : register(s3);
SamplerState gsamAnisotropicWrap  : register(s4);
SamplerState gsamAnisotropicClamp : register(s5);

// Constant data that varies per frame.
cbuffer cbPerObject : register(b0)
{
	float4x4 gWorld;
	float4x4 gTexTransform;
};

// Constant data that varies per material.
cbuffer cbPass : register(b1)
{
	float4x4 gView;
	float4x4 gInvView;
	float4x4 gProj;
	float4x4 gInvProj;
	float4x4 gViewProj;
	float4x4 gInvViewProj;
	float3 gEyePosW;
	float cbPerObjectPad1;
	float2 gRenderTargetSize;
	float2 gInvRenderTargetSize;
	float gNearZ;
	float gFarZ;
	float gTotalTime;
	float gDeltaTime;
	float4 gAmbientLight;

	float4 gFogColor;
	float gFogStart;
	float gFogRange;
	float2 cbPerObjectPad2;

	// Indices [0, NUM_DIR_LIGHTS) are directional lights;
	// indices [NUM_DIR_LIGHTS, NUM_DIR_LIGHTS+NUM_POINT_LIGHTS) are point lights;
	// indices [NUM_DIR_LIGHTS+NUM_POINT_LIGHTS, NUM_DIR_LIGHTS+NUM_POINT_LIGHT+NUM_SPOT_LIGHTS)
	// are spot lights for a maximum of MaxLights per object.
	Light gLights[MaxLights];
};

cbuffer cbMaterial : register(b2)
{
	float4   gDiffuseAlbedo;
	float3   gFresnelR0;
	float    gRoughness;
	float4x4 gMatTransform;
};

struct VertexIn
{
    float3 PosL : POSITION;
    float3 NormalL : NORMAL;
    float2 TexC : TEXCOORD;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float3 PosW : W_POSITION;
    float3 PosL : POSITION;
    float3 NormalW : NORMAL;
    float2 TexC : TEXCOORD;
};

VertexOut VS(VertexIn vin)
{
    VertexOut vout = (VertexOut) 0.0f;
    // Transform to world space.
    float4 posW = mul(float4(vin.PosL, 1.0f), gWorld);
    vout.PosW = posW.xyz;

    // Assumes nonuniform scaling; otherwise, need to use inverse-transpose of world matrix.
    vout.NormalW = mul(vin.NormalL, (float3x3) gWorld);

    // Transform to homogeneous clip space.
    vout.PosH = mul(posW, gViewProj);
	
	// Output vertex attributes for interpolation across triangle.
    float4 texC = mul(float4(vin.TexC, 0.0f, 1.0f), gTexTransform);
    vout.TexC = mul(texC, gMatTransform).xy;
	
    vout.PosL = vin.PosL;
    return vout;
}
 
/*struct PatchTess
{
	float EdgeTess[4]   : SV_TessFactor;
	float InsideTess[2] : SV_InsideTessFactor;
};*/
struct PatchTess
{
    float EdgeTess[3] : SV_TessFactor;
    float InsideTess[1] : SV_InsideTessFactor;
};


/*PatchTess ConstantHS(InputPatch<VertexOut, 4> patch, uint patchID : SV_PrimitiveID)
{
	PatchTess pt;
	
	float3 centerL = 0.25f*(patch[0].PosL + patch[1].PosL + patch[2].PosL + patch[3].PosL);
	float3 centerW = mul(float4(centerL, 1.0f), gWorld).xyz;
	
	float d = distance(centerW, gEyePosW);

	// Tessellate the patch based on distance from the eye such that
	// the tessellation is 0 if d >= d1 and 64 if d <= d0.  The interval
	// [d0, d1] defines the range we tessellate in.
	
	const float d0 = 20.0f;
	const float d1 = 100.0f;
	float tess = 64.0f*saturate( (d1-d)/(d1-d0) );

	// Uniformly tessellate the patch.

	pt.EdgeTess[0] = tess;
	pt.EdgeTess[1] = tess;
	pt.EdgeTess[2] = tess;
	pt.EdgeTess[3] = tess;
	
	pt.InsideTess[0] = tess;
	pt.InsideTess[1] = tess;
	
	return pt;
}*/

PatchTess ConstantHS(InputPatch<VertexOut, 3> patch, uint patchID : SV_PrimitiveID)
{
    PatchTess pt;

    float3 centerL = (patch[0].PosH + patch[1].PosH + patch[2].PosH).xyz / 3.0f;
    float3 centerW = mul(float4(centerL, 1.0f), gWorld).xyz;

    float d = distance(centerW, gEyePosW) * 0.4;

    // Tessellate the patch based on distance from the eye such that
	// the tessellation is 0 if d >= d1 and 64 if d <= d0.  The interval
	// [d0, d1] defines the range we tessellate in.
    const float d0 = 20.0f;
    const float d1 = 80.0f;
    if (d < d0)
        d = d0;
    
    float tess = 4 * saturate((d1 - d) / (d1 - d0));
    if (tess <= 0)
        tess = 1;
    // Uniformly tessellate the patch.
    
    //
    float ss = 64.f;
    float2 p0 = (patch[0].PosH.xy / patch[0].PosH.w + 1) / 2 * gRenderTargetSize;
    float2 p1 = (patch[1].PosH.xy / patch[1].PosH.w + 1) / 2 * gRenderTargetSize;
    float2 p2 = (patch[2].PosH.xy / patch[2].PosH.w + 1) / 2 * gRenderTargetSize;
    float3 edge_tess_factors = float3(
    length(p2 - p1) / ss,
    length(p2 - p0) / ss,
    length(p1 - p0) / ss);
    tess = max(edge_tess_factors.x, max(edge_tess_factors.y, edge_tess_factors.z));
    pt.EdgeTess[0] = tess;
    pt.EdgeTess[1] = tess;
    pt.EdgeTess[2] = tess;

    pt.InsideTess[0] = tess;

    return pt;
}


struct HullOut
{
	float3 PosL : POSITION;
    float3 PosW : W_POSITION;
    float3 NormalW : NORMAL;
    float2 TexC : TEXCOORD;
    float4 PosH : H_POSITION;
};

[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("ConstantHS")]
[maxtessfactor(64.0f)]
HullOut HS(InputPatch<VertexOut, 3> p, 
           uint i : SV_OutputControlPointID,
           uint patchId : SV_PrimitiveID)
{
	HullOut hout;
	
	hout.PosL = p[i].PosL;
    hout.TexC = p[i].TexC;
    hout.NormalW = p[i].NormalW;
    hout.PosW = p[i].PosW;
    hout.PosH = p[i].PosH;
	return hout;
}

struct DomainOut
{
	float4 PosH : SV_POSITION;
    float3 NormalW : NORMAL;
    float2 TexC : TEXCOORD;
    float3 PosW : W_POSITION;
};

// The domain shader is called for every vertex created by the tessellator.  
// It is like the vertex shader after tessellation.
[domain("tri")]
DomainOut DS(PatchTess patchTess,
             float2 uv : SV_DomainLocation,
             const OutputPatch<HullOut, 3> tri)
{
    DomainOut dout;
    
    // Barycentric interpolation for triangle
    float3 barycentric = float3(1.0f - uv.x - uv.y, uv.x, uv.y);
    
    // Interpolate positions
#ifdef WORLD_DISPLACEMENT
    float3 p = barycentric.x * tri[0].PosW + barycentric.y * tri[1].PosW + barycentric.z * tri[2].PosW;
#else
    float3 p = barycentric.x * tri[0].PosL + barycentric.y * tri[1].PosL + barycentric.z * tri[2].PosL;
#endif // WORLD_DISPLACEMENT

    // UV LERP
    float2 vt = barycentric.x * tri[0].TexC + barycentric.y * tri[1].TexC + barycentric.z * tri[2].TexC;
    dout.TexC = vt;

    // Displacement mapping
    float height = 0.01f;
    //p.y += height * (p.z * sin(p.x + gTotalTime) + p.x * cos(p.z + gTotalTime));
#ifdef WORLD_DISPLACEMENT
    float4 posW = float4(p, 1.0f); //mul(float4(p, 1.0f), gWorld);
#else
    float4 posW = mul(float4(p, 1.0f), gWorld);
#endif // WORLD_DISPLACEMENT
    dout.PosH = mul(posW, gViewProj);
    dout.PosW = posW;
    
    return dout;
}

/*[domain("quad")]
DomainOut DS(PatchTess patchTess, 
             float2 uv : SV_DomainLocation,
             const OutputPatch<HullOut, 4> quad)
{
	DomainOut dout;
	
	// Bilinear interpolation.
	float3 v1 = lerp(quad[0].PosL, quad[1].PosL, uv.x); 
	float3 v2 = lerp(quad[2].PosL, quad[3].PosL, uv.x); 
	float3 p  = lerp(v1, v2, uv.y); 
	
	// UV LERP
    float2 vt1 = lerp(quad[0].TexC, quad[1].TexC, uv.x);
    float2 vt2 = lerp(quad[2].TexC, quad[3].TexC, uv.x);
    float2 vt = lerp(vt1, vt2, uv.y);
    dout.TexC = vt;
	
	// Displacement mapping
    float height = 0.3f;
    //p.y += height * (p.z * sin(p.x + gTotalTime) + p.x * cos(p.z + gTotalTime));
	
	float4 posW = mul(float4(p, 1.0f), gWorld);
	dout.PosH = mul(posW, gViewProj);
	
	return dout;
}*/

float4 PS(DomainOut pin) : SV_Target
{
    float cosAngle = cos(gTotalTime);
    float sinAngle = sin(gTotalTime);
    
    float2x2 rotationMatrix =
    {
        { cosAngle, -sinAngle },
        { sinAngle, cosAngle }
    };
    float tiling = 10.f;
    float speed = 100.f;

    float2 rotatedTexC = mul(rotationMatrix, frac(pin.TexC * 10 + float2(gTotalTime * 2, 0)) + frac(gTotalTime / speed) - 0.5) + 0.5;
    
    float pixelSize = 0.1;
    //pin.TexC = floor(rotatedTexC / pixelSize) * pixelSize;
    
    float4 diffuseAlbedo = gDiffuseMap.Sample(gsamAnisotropicWrap, pin.TexC) * gDiffuseAlbedo;
    float4 diffuseAlbedo1 = gDiffuseMap.Sample(gsamAnisotropicWrap, pin.TexC + float2(0.01, 0)) * gDiffuseAlbedo;
    float4 diffuseAlbedo2 = gDiffuseMap.Sample(gsamAnisotropicWrap, pin.TexC - float2(0.01, 0)) * gDiffuseAlbedo;


	
    // Interpolating normal can unnormalize it, so renormalize it.
    pin.NormalW = normalize(pin.NormalW);

    // Vector from point being lit to eye. 
    float3 toEyeW = normalize(gEyePosW - pin.PosW);
     
    // Light terms.
    float4 ambient = gAmbientLight * diffuseAlbedo;

    const float shininess = 1.0f - gRoughness;
    Material mat = { diffuseAlbedo, gFresnelR0, shininess };
    float3 shadowFactor = 1.0f;
    float4 directLight = ComputeLighting(gLights, mat, pin.PosW,
        pin.NormalW, toEyeW, shadowFactor);

    float4 litColor = ambient + directLight;

    // Common convention to take alpha from diffuse albedo.
    litColor.a = diffuseAlbedo.a;

    return litColor;
}
