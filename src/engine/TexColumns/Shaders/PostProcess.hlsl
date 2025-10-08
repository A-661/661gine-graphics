struct VertexOut
{
    float4 PosH : SV_POSITION;
    float2 TexC : TEXCOORD;
};

//#define PHASMOFOBIA1
//#define HEXAGON_EFFECT
//#define BLUR

#ifdef PHASMOFOBIA1
    #define LINE_THICKNESS 2.
    #define LINE_GAP 2
    #define LINE_OPACITY 0.5
    #define COLOR float3(0.0,0.0,0.0)
#endif

VertexOut VS(uint vid : SV_VertexID)
{
    VertexOut vout;
    
    // Fullscreen triangle
    float2 texcoord = float2((vid << 1) & 2, vid & 2);
    vout.PosH = float4(texcoord * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    vout.TexC = texcoord;
    
    return vout;
}

Texture2D gInputImage : register(t0);
SamplerState gSampler : register(s0);

float3 ApplyChromaticAberration(float2 texcoord)
{
    float2 gChromaticAberrationCenter = float2(0.5f, 0.5f);
    float gChromaticAberrationEdgePower = 3.f;
    float gChromaticAberrationIntensity = 0.9f;
    
    float2 dir = texcoord - gChromaticAberrationCenter;
    
    
    float dist = length(dir);
    
    
    float edgeFactor = pow(dist, gChromaticAberrationEdgePower);
    
    
    float2 offsetR = dir * gChromaticAberrationIntensity * edgeFactor * float2(1., 0.0);
    float2 offsetG = dir * gChromaticAberrationIntensity * edgeFactor * float2(0.0, 0.0);
    float2 offsetB = dir * gChromaticAberrationIntensity * edgeFactor * float2(-1., 0.0);
    
    
    float r = gInputImage.Sample(gSampler, texcoord + offsetR).r;
    float g = gInputImage.Sample(gSampler, texcoord + offsetG).g;
    float b = gInputImage.Sample(gSampler, texcoord + offsetB).b;
    
    return float3(r, g, b);
}
float rand(float2 co) {
    return frac(sin(dot(co.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float2 hexDistortion(float2 uv, float size, float distortion, float time)
{
    
    float hexSize = size * 0.01; 
    
    
    float2 p = uv / hexSize * 11;
    
    // checkboard
    float offset = floor(p.y) * 0.5;
    p.x += offset;
    
    
    float2 cell = floor(p);
    float2 center = cell + 0.5;
    
    // rand for every hex
    float2 randomOffset = float2(
        rand(cell + float2(0.1, 0.1)) - 0.5,
        rand(cell + float2(0.2, 0.2)) - 0.5
    );
    
    // anim
    float timeFactor = sin(time * 2.0 + cell.x * 0.5 + cell.y * 0.3);
    randomOffset *= timeFactor * 0.2;
    
    
    float2 toCenter = (center - p) + randomOffset;
    float dist = length(toCenter);
    
    
    float edgeFactor = 1.0 - smoothstep(0.0, 0.5, dist * 2.0);
    float2 displacement = normalize(toCenter) * distortion * edgeFactor * 0.01;
    
    return uv + displacement;
}
float4 PS(VertexOut pin) : SV_Target
{
    
    float gHexagonSize = 15; 
    float gHexagonDistortion = 2.0f; 
    
    float2 uv = pin.TexC;
    // Default - just pass through
    // float4 color = gInputImage.Sample(gSampler, pin.TexC);
   
#ifdef HEXAGON_EFFECT
    uv = hexDistortion(uv, gHexagonSize, gHexagonDistortion, 0.0f);
#endif
#ifdef PHASMOFOBIA1
    float4 color = float4(ApplyChromaticAberration(uv), 1.0f); // TODO DELTA TIME
    
    if (int(floor(pin.TexC.y * 1600 / LINE_THICKNESS)) % LINE_GAP == 0)
    {
        color.xyz = lerp(color.xyz, COLOR, 0.5f);
    }
    
    
    
#else
    float4 color = gInputImage.Sample(gSampler, uv);
#endif
    
#ifdef BLUR
    float2 texelSize = 5.0 / float2(2560, 1600);
    float4 blurColor = float4(0, 0, 0, 0);
    int size = 2;
    int samples = 0;
    for (int x = -size; x <= size; x++)
    {
        for(int y = -2; y <= 2; y++)
        {
            blurColor += gInputImage.Sample(gSampler, uv + float2(x, y) * texelSize);
            samples++;
        }
    }
    //color = blurColor / 25.0;
    color = lerp(color, blurColor / samples, 0.7f);
#endif
    return color;
}