#define PI 3.14159265f

cbuffer PassCB : register(b0)
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
};

cbuffer AtmosphereCB : register(b1)
{
    float3 gFogColor;
    float gGlobalDensity;
    float gHeightFalloff;
    float gBaseHeight;
    float gFogAnisotropy;
    float gSunIntensity;
    float3 gSunDirection;
    float pad0;
};

Texture2D gSceneColor : register(t0);
Texture2D gDepth : register(t1);

SamplerState gsamLinearClamp : register(s3);

struct VSOut
{
    float4 PosH : SV_POSITION;
    float2 TexC : TEXCOORD;
};

// full-screen triangle 
VSOut FullscreenVS(uint vid : SV_VertexID)
{
    VSOut v;
    float2 pos[3] =
    {
        float2(-1.0f, -1.0f),
        float2(-1.0f, 3.0f),
        float2(3.0f, -1.0f)
    };

    float2 p = pos[vid];
    v.PosH = float4(p, 0.0f, 1.0f);
    v.TexC = float2(0.5f * (p.x + 1.0f), 0.5f * (1.0f - p.y));

    return v;
}

// approximation
float PhaseHG(float mu, float g)
{
    float g2 = g * g;
    float denom = pow(1.0f + g2 - 2.0f * g * mu, 1.5f);
    return (1.0f - g2) / max(denom, 1e-3f);
}

float PhaseRayleigh(float mu)
{
    return (1.0f + mu * mu);
}

float4 HeightFogPS(VSOut pin) : SV_TARGET
{
    float3 sceneColor = gSceneColor.Sample(gsamLinearClamp, pin.TexC).rgb;
    float depth = gDepth.Sample(gsamLinearClamp, pin.TexC).r;

    float2 ndc;
    ndc.x = pin.TexC.x * 2.0f - 1.0f;
    ndc.y = (1.0f - pin.TexC.y) * 2.0f - 1.0f;

    // ray to farplane
    float4 clipFar = float4(ndc.x, ndc.y, 1.0f, 1.0f);
    float4 worldFarH = mul(clipFar, gInvViewProj);
    worldFarH /= worldFarH.w;
    float3 rayDir = normalize(worldFarH.xyz - gEyePosW);

    // calc distance if not pussy
    bool hasGeometry = (depth < 1.0f - 1e-5f);
    float maxDist;

    if (hasGeometry)
    {
        float4 clipPos = float4(ndc.x, ndc.y, depth, 1.0f);
        float4 worldPosH = mul(clipPos, gInvViewProj);
        worldPosH /= worldPosH.w;
        float3 worldPos = worldPosH.xyz;
        maxDist = length(worldPos - gEyePosW);
    }
    else
    {
        maxDist = gFarZ; // no more limits blyat
    }

    maxDist = max(maxDist, 1e-3f);

    // ?(y) = k * exp(-? * (y - BaseHeight)) // gg
    float lambda = gHeightFalloff;
    float k = gGlobalDensity;

    float y0 = gEyePosW.y - gBaseHeight;
    float y1 = y0 + rayDir.y * maxDist;

    float tau;
    if (abs(rayDir.y) < 1e-3f)
    {
        float density = k * exp(-lambda * y0);
        tau = density * maxDist;
    }
    else
    {
        float exp0 = exp(-lambda * y0);
        float exp1 = exp(-lambda * y1);
        tau = (k / (lambda * rayDir.y)) * (exp0 - exp1);
        tau = abs(tau);
    }

    tau = clamp(tau, 0.0f, 50.0f);
    float T = exp(-tau); // transmittance

    // in-scattering

    float3 sunDir = normalize(gSunDirection);
    float mu = dot(-rayDir, sunDir); // angle(cam, sun)

    float rayleighPhase = PhaseRayleigh(mu);
    float miePhase = PhaseHG(mu, gFogAnisotropy);

    // -1-1 to 0-1
    float sunHeight = saturate(sunDir.y * 0.5f + 0.5f);

    // Rayleigh
    float3 rayleighHorizon = float3(1.0f, 0.55f, 0.2f);
    float3 rayleighZenith = gFogColor; // base color
    float3 rayleighColor = lerp(rayleighHorizon, rayleighZenith, sunHeight);

    // Mie
    float3 mieSunset = float3(1.0f, 0.7f, 0.3f);
    float3 mieDay = float3(1.0f, 0.9f, 0.7f);
    float3 mieColor = lerp(mieSunset, mieDay, sunHeight);

    float rayleighWeight = 1.0f;
    float mieWeight = 0.4f;

    float3 scatterColor =
        rayleighColor * rayleighPhase * rayleighWeight +
        mieColor * miePhase * mieWeight;

    float3 fogLight = scatterColor * gSunIntensity;

    // blend
    float3 fogContribution = fogLight * (1.0f - T);
    float3 color = sceneColor * T + fogContribution;

    // tonemap gamma other shit
    color = color / (1.0f + color);
    color = pow(color, 1.0f / 2.2f);

    return float4(color, 1.0f);
}

float4 AtmospherePS(VSOut pin) : SV_TARGET
{
    float3 sceneColor = gSceneColor.Sample(gsamLinearClamp, pin.TexC).rgb;
    float depth = gDepth.Sample(gsamLinearClamp, pin.TexC).r;
    
    float2 ndc;
    ndc.x = pin.TexC.x * 2.0f - 1.0f;
    ndc.y = (1.0f - pin.TexC.y) * 2.0f - 1.0f;

    float4 clipFar = float4(ndc.x, ndc.y, 1.0f, 1.0f);
    float4 worldFarH = mul(clipFar, gInvViewProj);
    worldFarH /= worldFarH.w;
    float3 rayDir = normalize(worldFarH.xyz - gEyePosW);

    bool hasGeometry = (depth < 1.0f - 1e-5f);
    float maxDist;

    if (hasGeometry)
    {
        float4 clipPos = float4(ndc.x, ndc.y, depth, 1.0f);
        float4 worldPosH = mul(clipPos, gInvViewProj);
        worldPosH /= worldPosH.w;
        float3 worldPos = worldPosH.xyz;
        maxDist = length(worldPos - gEyePosW);
    }
    else
    {
        maxDist = gFarZ;
    }

    maxDist = max(maxDist, 1e-3f);

    // exp height fog (tau, T)

    // ?(y) = k * exp(-? * (y - BaseHeight)) (wtf is that)
    float lambda = gHeightFalloff;
    float k = gGlobalDensity;

    float y0 = gEyePosW.y - gBaseHeight;
    float y1 = y0 + rayDir.y * maxDist;

    float tau;
    if (abs(rayDir.y) < 1e-3f)
    {
        float density = k * exp(-lambda * y0);
        tau = density * maxDist;
    }
    else
    {
        float exp0 = exp(-lambda * y0);
        float exp1 = exp(-lambda * y1);
        tau = (k / (lambda * rayDir.y)) * (exp0 - exp1);
        tau = abs(tau);
    }

    tau = clamp(tau, 0.0f, 50.0f);
    float T = exp(-tau); // transmittance

    //nRayleigh + HG-Mie
    float3 sunDir = normalize(gSunDirection);
    float3 lightDir = -sunDir;

    float mu = dot(-rayDir, lightDir);

    float rayleighPhase = PhaseRayleigh(mu);
    float miePhase = PhaseHG(mu, gFogAnisotropy);

    float sunAlt = saturate(sunDir.y);

    float3 rayleighHorizon = float3(1.0f, 0.4f, 0.15f); // orange
    float3 rayleighZenith = gFogColor; // blue
    float3 rayleighBase = lerp(rayleighHorizon, rayleighZenith, sunAlt);

    // halo
    float3 mieSunset = float3(1.0f, 0.7f, 0.3f);
    float3 mieDay = float3(1.0f, 0.9f, 0.7f);
    float3 mieBase = lerp(mieSunset, mieDay, sunAlt);

    float rayleighWeight = 1.0f;
    float mieWeight = 0.4f;

    // what
    float3 rayleighColor = rayleighBase * rayleighPhase * rayleighWeight;
    float3 mieColor = mieBase * miePhase * mieWeight;

    float3 fogLight = (rayleighColor + mieColor) * gSunIntensity;

    // blend
    float3 color;

    if (hasGeometry) {
        float3 fogContribution = fogLight * (1.0f - T);
        color = sceneColor * T + fogContribution;
    }
    else { // just die already
        float up = saturate(rayDir.y * 0.5f + 0.5f);

        float3 zenithDay = rayleighZenith;
        float3 zenithSunset = float3(0.15f, 0.2f, 0.5f);
        float3 zenithColor = lerp(zenithSunset, zenithDay, sunAlt);

        float3 horizonDay = lerp(rayleighZenith, float3(1.0f, 1.0f, 1.0f), 0.1f);
        float3 horizonSunset = rayleighHorizon;
        float3 horizonColor = lerp(horizonSunset, horizonDay, sunAlt);

        float3 baseSky = lerp(horizonColor, zenithColor, up);

        // halo
        float muSun = dot(rayDir, sunDir);
        float sunDisk = pow(saturate(muSun), 1024.0f);
        float sunGlow = pow(saturate(muSun), 64.0f);

        float3 sunDiskColor = float3(1.0f, 0.97f, 0.9f);
        float3 sunGlowColor = float3(1.0f, 0.9f, 0.7f);

        float3 sunContribution =
            sunGlowColor * sunGlow * 2.0f +
            sunDiskColor * sunDisk * 30.0f;

        color = baseSky + sunContribution * gSunIntensity;
    }

    // tonemap gamma other shit
    color = color / (1.0f + color);
    color = pow(color, 1.0f / 2.2f);

    return float4(color, 1.0f);
}
