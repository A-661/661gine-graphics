// Sky.hlsl
Texture2D gDepth : register(t0); // depth SRV
TextureCube gEnvCube : register(t1); // environment cubemap

// совпадает со статическими самплерами из GetStaticSamplers():
SamplerState gsamPointClamp : register(s1); // дл€ depth
SamplerState gsamLinearClamp : register(s3); // дл€ cubemap

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
    // (можешь добавить счЄтчики и Lights[], если они есть в общем буфере Ч
    //  лишние пол€ не мешают, важен пор€док до используемых)
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

float3 RayDirWorld(float2 uv)
{
    float2 ndc = uv * float2(2, -2) + float2(-1, 1);
    float4 clip = float4(ndc, 1, 1); // луч к far plane
    float4 view = mul(clip, gInvProj);
    float3 dirV = normalize(view.xyz / view.w);
    float3 dirW = mul(float4(dirV, 0), gInvView).xyz;
    return normalize(dirW);
}

float4 SkyPS(VSOut i) : SV_Target
{
    // рисуем только там, где нет геометрии
    float d = gDepth.Sample(gsamPointClamp, i.TexC).r;
    if (d < 1.f)
        clip(-1);

    float3 dirW = RayDirWorld(i.TexC);
    float3 col = gEnvCube.Sample(gsamLinearClamp, dirW).rgb;
    return float4(col, 1);
}
