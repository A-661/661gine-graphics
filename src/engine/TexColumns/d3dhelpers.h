#pragma once

#include <cstdint>
#include <stdexcept>

// COM smart pointers
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

// DX12 core
#include "../../Common/d3dx12.h"

#ifndef ThrowIfFailed
inline void ThrowIfFailedHR(HRESULT hr)
{
    if (FAILED(hr)) {
        throw std::runtime_error("D3D12 call failed.");
    }
}
#define ThrowIfFailed(x) ThrowIfFailedHR((x))
#endif

inline Microsoft::WRL::ComPtr<ID3D12Resource> CreateStructuredBuffer(const Microsoft::WRL::ComPtr<ID3D12Device>& dev, UINT elementCount, UINT stride, D3D12_RESOURCE_STATES initState = D3D12_RESOURCE_STATE_COMMON)
{
    return CreateStructuredBuffer(dev.Get(), elementCount, stride, initState);
}

inline Microsoft::WRL::ComPtr<ID3D12Resource> CreateUploadBuffer(const Microsoft::WRL::ComPtr<ID3D12Device>& dev, UINT64 byteSize)
{
    return CreateUploadBuffer(dev.Get(), byteSize);
}

inline ComPtr<ID3D12Resource> CreateStructuredBuffer(ID3D12Device* dev, UINT elementCount, UINT stride, D3D12_RESOURCE_STATES initState = D3D12_RESOURCE_STATE_COMMON)
{
    const UINT64 byteSize = UINT64(elementCount) * stride;
    auto desc = CD3DX12_RESOURCE_DESC::Buffer(
        byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    ComPtr<ID3D12Resource> res;
    ThrowIfFailed(dev->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initState,
        nullptr,
        IID_PPV_ARGS(&res)));
    return res;
}

inline ComPtr<ID3D12Resource> CreateUploadBuffer(ID3D12Device* dev, UINT64 byteSize)
{
    auto desc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);

    ComPtr<ID3D12Resource> res;
    ThrowIfFailed(dev->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&res)));
    return res;
}