//***************************************************************************************
// TexColumnsApp.cpp by Frank Luna (C) 2015 All Rights Reserved.
//***************************************************************************************


#define NOMINMAX
#include "../../Common/Camera.h"
#include "../../Common/d3dApp.h"
#include "../../Common/MathHelper.h"
#include "../../Common/UploadBuffer.h"
#include "../../Common/GeometryGenerator.h"
#include "FrameResource.h"
#include <DirectXCollision.h>

#include "d3dhelpers.h"
#include <iostream>



//#define		QUADPATCH

//#define DEBUG
#define SKYPASS
#define PBR
//#define OLDMAP
//#define DEFERREDQUADS

Camera cam;
using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace DirectX::PackedVector;

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "D3D12.lib")

const int gNumFrameResources = 3;

// Lightweight structure stores parameters to draw a shape.  This will
// vary from app-to-app.
struct RenderItem
{
	RenderItem() = default;
    RenderItem(const RenderItem& rhs) = delete;

	// FRUSTRUM CULLING
	BoundingSphere BoundSphereLocal = { XMFLOAT3(0,0,0), 1.0f };
	BoundingSphere BoundSphereWS = { XMFLOAT3(0,0,0), 1.0f };

	// LODs
	int lodGroupId = -1;  // -1 = no group
	int lodLevel = 0;   // (0,1,2)

    // World matrix of the shape that describes the object's local space
    // relative to the world space, which defines the position, orientation,
    // and scale of the object in the world.
    XMFLOAT4X4 World = MathHelper::Identity4x4();

	XMFLOAT4X4 TexTransform = MathHelper::Identity4x4();

	// Dirty flag indicating the object data has changed and we need to update the constant buffer.
	// Because we have an object cbuffer for each FrameResource, we have to apply the
	// update to each FrameResource.  Thus, when we modify obect data we should set 
	// NumFramesDirty = gNumFrameResources so that each frame resource gets the update.
	int NumFramesDirty = gNumFrameResources;

	// Index into GPU constant buffer corresponding to the ObjectCB for this render item.
	UINT ObjCBIndex = -1;

	Material* Mat = nullptr;
	MeshGeometry* Geo = nullptr;

    // Primitive topology.
	D3D12_PRIMITIVE_TOPOLOGY PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    // DrawIndexedInstanced parameters.
    UINT IndexCount = 0;
    UINT StartIndexLocation = 0;
    int BaseVertexLocation = 0;
	std::string Name;

	int  sublevelId = 0;   // 0 = const, >0 = sublevel
	bool visible = true;

	float RoughnessOverride = 0.1f;
	float MetallicOverride = 1.0f;
};

static inline void SetItemRM(RenderItem* ri, float rough, float metal) {
	ri->RoughnessOverride = rough;
	ri->MetallicOverride = metal;
	ri->NumFramesDirty = gNumFrameResources;
}


struct Particle
{
	DirectX::XMFLOAT3 pos;
	DirectX::XMFLOAT3 vel;
	float life;
	float lifetime;
	float size; float rot;
	uint32_t alive;
	DirectX::XMFLOAT4 color;
	uint32_t pad;
};
static_assert(sizeof(Particle) % 16 == 0, "aligned");

struct AtmosphereConstants
{
	DirectX::XMFLOAT3 fogColor;
	float globalDensity;
	float heightFalloff;
	float baseHeight;
	float fogAnisotropy;
	float sunIntensity;
	DirectX::XMFLOAT3 sunDirection;
	float pad;
};
static_assert(sizeof(AtmosphereConstants) % 16 == 0, "aligned");


class TexColumnsApp : public D3DApp
{
public:
    TexColumnsApp(HINSTANCE hInstance);
    TexColumnsApp(const TexColumnsApp& rhs) = delete;
    TexColumnsApp& operator=(const TexColumnsApp& rhs) = delete;
    ~TexColumnsApp();

    virtual bool Initialize()override;

private:
    virtual void OnResize()override;
    virtual void Update(const GameTimer& gt)override;
    virtual void Draw(const GameTimer& gt)override;

    virtual void OnMouseDown(WPARAM btnState, int x, int y)override;
    virtual void OnMouseUp(WPARAM btnState, int x, int y)override;
    virtual void OnMouseMove(WPARAM btnState, int x, int y)override;
	virtual void MoveBackFwd(float step)override;
	virtual void MoveLeftRight(float step)override;
	virtual void MoveUpDown(float step)override;
	void OnKeyPressed(const GameTimer& gt, WPARAM key) override;
	void OnKeyReleased(const GameTimer& gt, WPARAM key) override;
	std::wstring GetCamSpeed() override;
	void UpdateCamera(const GameTimer& gt);
	void AnimateMaterials(const GameTimer& gt);
	void UpdateObjectCBs(const GameTimer& gt);
	void UpdateMaterialCBs(const GameTimer& gt);
	void UpdateMainPassCB(const GameTimer& gt);
	XMFLOAT3 AnimateLightOrbitY(const XMFLOAT3& center, float radius, float AngSpeed, const GameTimer& gt, float phase);
	
	void LoadAllTextures();
	void LoadTexture(const std::string& name);
    void BuildRootSignature();
	void BuildDescriptorHeaps();
    void BuildShadersAndInputLayout();
    void BuildShapeGeometry();
	void BuildQuadPatchGeometry();
    void BuildPSOs();
    void BuildFrameResources();
	void TexColumnsApp::CreateMaterial(std::string _name, int _CBIndex, int _SRVHeapIndex, int _normalSRVHeapIndex, XMFLOAT4 _DiffuseAlbedo, XMFLOAT3 _FresnelR0, float _Roughness);
	void BuildMaterials();
	RenderItem* RenderObject(std::string unique_name, std::string meshname, std::string materialName, XMMATRIX Scale, XMMATRIX Rotation, XMMATRIX Translation);
	RenderItem* RenderObject(std::string unique_name, std::string meshname,
		std::string materialName, float roughness, float metallic,
		XMMATRIX Scale, XMMATRIX Rotation, XMMATRIX Translation);
	void BuildCustomMeshGeometry(std::string name, UINT& meshVertexOffset, UINT& meshIndexOffset, UINT& prevVertSize, UINT& prevIndSize, std::vector<Vertex>& vertices, std::vector<std::uint16_t>& indices, MeshGeometry* Geo);
	bool RemoveSublevel(const std::string& sceneKey);
	bool AddSublevelToScene(std::string filename, XMFLOAT3 WorldOffset);
    void RenderWorld();
    void DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems);
	void ReassignObjectCBIndices();

	void BuildGBufferResources();
	void BuildDeferredPSOs();
	void BuildDeferredRootSignature();

	void BuildIBLSRVs();
	void LoadIBLTextures();

	void LoadEnvironmentCube(const std::wstring& file);
	void BuildEnvironmentSRV();
	void BuildEnvPSO();
	void BuildEnvRootSignature();

	void BuildParticleResources();
	void BuildParticleDescriptors();
	void BuildParticleCS_RS();
	void BuildParticleGfx_RS();
	void BuildParticlePSO();
	XMFLOAT3 sphereC = DirectX::XMFLOAT3(0, 18, -15);;
	float sphereR = 10.f;

	void BuildPostProcessResources();
	void BuildPostProcessPSO();
	void BuildPostProcessRootSignature();

	void BuildAtmosphereRootSignature();
	void BuildAtmospherePSO();

	BoundingSphere ComputeLocalSphere(MeshGeometry* geo, const SubmeshGeometry& sm);
	void UpdateWorldSphere(RenderItem* ri);
	void BuildVisibleList();

	struct OctreeNode {
		DirectX::BoundingBox bounds;                 // AABB world
		std::vector<RenderItem*> items;
		std::unique_ptr<OctreeNode> child[8];        // nullptr if leaf
		bool IsLeaf() const {
			for (int i = 0; i < 8; ++i) if (child[i]) return false;
			return true;
		}
	};

	static BoundingBox ComputeSceneAABB(const std::vector<std::unique_ptr<RenderItem>>& items);
	std::unique_ptr<OctreeNode> BuildOctreeRecursive(const BoundingBox& nodeAABB, const std::vector<RenderItem*>& in, int depth);
	static void MakeChildrenAABBs(const BoundingBox& parent, BoundingBox out[8], bool loose, float looseFactor);
	static void GatherAll(const OctreeNode* n, std::vector<RenderItem*>& out, const std::unordered_map<int, int>* lodSel);
	void TraverseOctree(const BoundingFrustum& frWorld, const OctreeNode* n, std::vector<RenderItem*>& out, const std::unordered_map<int, int>* lodSel);
	std::unordered_map<int, int> ComputeSelectedLods() const;

	void TexColumnsApp::BuildOctree(); // MAIN

	void SpawnLODObject(
		const std::string& unique_name_base,
		const std::vector<std::string>& meshNames,
		const std::string& materialName,
		DirectX::XMMATRIX Scale,
		DirectX::XMMATRIX Rotation,
		DirectX::XMMATRIX Translation,
		const std::vector<float>& switchDist);

	void SetupImGuiStyle();
	void BuildUI();

	std::array<const CD3DX12_STATIC_SAMPLER_DESC, 6> GetStaticSamplers();

private:
	std::unordered_map<std::string, unsigned int>ObjectsMeshCount;
    std::vector<std::unique_ptr<FrameResource>> mFrameResources;
    FrameResource* mCurrFrameResource = nullptr;
    int mCurrFrameResourceIndex = 0;
	int mFrameIndex = 0;
    UINT mCbvSrvDescriptorSize = 0;

    ComPtr<ID3D12RootSignature> mRootSignature = nullptr;
	ComPtr<ID3D12RootSignature> mDeferredLightingRootSignature = nullptr;

	ComPtr<ID3D12DescriptorHeap> mSrvDescriptorHeap = nullptr;

	std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> mGeometries;
	std::unordered_map<std::string, std::unique_ptr<Material>> mMaterials;
	std::unordered_map<std::string, std::unique_ptr<Texture>> mTextures;
	std::unordered_map<std::string, ComPtr<ID3DBlob>> mShaders;
	std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> mPSOs;

	//=================================================================PARTICLES
	ComPtr<ID3D12Resource> mParticlesA;      // default SRV/UAV
	ComPtr<ID3D12Resource> mParticlesB;      // default SRV/UAV
	std::unique_ptr<UploadBuffer<Particle>> mParticlesInit; // UPLOAD
	struct SimCB {
		float dt;
		DirectX::XMFLOAT3 gravity;
		DirectX::XMFLOAT3 sphereC;
		float sphereR;
	};
	std::unique_ptr<UploadBuffer<SimCB>> mSimCB;            // UPLOAD
	// shader-visible heap for SRV/UAV particles + SRV billboards
	ComPtr<ID3D12DescriptorHeap> mParticleHeap;
	enum { kSRV_A = 0, kUAV_A = 1, kSRV_B = 2, kUAV_B = 3, kSRV_Sprite = 4 };

	ComPtr<ID3D12RootSignature> mParticleCS_RS;  // CS RS
	ComPtr<ID3D12RootSignature> mParticleGfx_RS; // PS RS
	ComPtr<ID3D12PipelineState> mParticleCS_PSO; // CS PSO
	ComPtr<ID3D12PipelineState> mParticleGfx_PSO;// PS PSO

	UINT mParticleCount = 5000;
	//==========================================================================

	// CUBEMAP
	ComPtr<ID3D12Resource> mEnvCube;
	ComPtr<ID3D12Resource> mEnvUpload;
	int kSrvIdx_Albedo = 0, kSrvIdx_Normal = 1, kSrvIdx_Spec = 2, kSrvIdx_Depth = 3, kSrvIdx_Env = 4, kSrvIdx_Irr = 5, kSrvIdx_Pref = 6, kSrvIdx_BRDF = 7;
	ComPtr<ID3D12RootSignature> mSkyRootSignature;
	DXGI_FORMAT mLightingFormat = DXGI_FORMAT_R11G11B10_FLOAT; // mb DXGI_FORMAT_R16G16B16A16_FLOAT
	ComPtr<ID3D12Resource>       mLightingTarget;
	ComPtr<ID3D12DescriptorHeap> mLightingRTVHeap;
	ComPtr<ID3D12DescriptorHeap> mLightingSRVHeap;
	bool bSkypass = true;

	// PBR
	ComPtr<ID3D12Resource> mIrradianceCube;   // DDS cube (low-res)
	ComPtr<ID3D12Resource> mPrefilteredCube;  // DDS cube (mipped)
	ComPtr<ID3D12Resource> mBRDF_LUT;         // 2D LUT R16G16_FLOAT
	ComPtr<ID3D12Resource> mIrradianceUpload, mPrefilteredUpload, mBRDFUpload;

	// POST
	ComPtr<ID3D12Resource> mPostProcessRenderTarget;
	ComPtr<ID3D12Resource> mPostProcessRenderTargetUpload;
	ComPtr<ID3D12DescriptorHeap> mPostProcessRTVHeap;
	ComPtr<ID3D12DescriptorHeap> mPostProcessSRVHeap;
	ComPtr<ID3D12PipelineState> mPostProcessPSO;
	ComPtr<ID3D12RootSignature> mPostProcessRootSignature;
	std::unordered_map<std::string, ComPtr<ID3DBlob>> mPostProcessShaders;

	// ATMOSPHERE
	std::unique_ptr<UploadBuffer<AtmosphereConstants>> mAtmosphereCB;
	AtmosphereConstants mAtmosphereData{};
	ComPtr<ID3D12RootSignature> mAtmosphereRootSignature;
	ComPtr<ID3D12PipelineState> mAtmospherePSO;
	ComPtr<ID3D12PipelineState> mHeightFogPSO;
	bool mHeightFogOnly = false;

	// LEVEL STREAMING
	int mNextSublevelId = 1;
	int mCurrentSpawningSublevelId = 0;
	std::unordered_map<std::string, int> mSublevelNameToId;         // "testscene2" -> id
	std::unordered_map<int, std::vector<RenderItem*>> mSublevelIdx; // id -> all items

	// FRUSTRUM CULLING
	bool mEnableCulling = 1;
	std::vector<RenderItem*> mVisibleOpaqueRitems;

	

	std::unique_ptr<OctreeNode> mOctreeRoot;
	int  mOctreeMaxDepth = 8;
	int  mOctreeLeafMax = 16;    // max leaves
	bool mUseLoose = true;  // "loose" tree /w scaling
	float mLooseFactor = 1.25f; // scaling

	// LODs
	struct LODGroup {
		int numLevels = 0; // 1-3
		float switchDist[3] = { FLT_MAX,FLT_MAX,FLT_MAX }; // switchDist[0] -> LOD0, [1] -> LOD1, else -> LOD2
		std::vector<RenderItem*> levels[3];
		int lastSelected = 0;
	};
	std::unordered_map<int, LODGroup> mLODGroups;
	int mNextLODGroupId = 1;

	// DEFERRED
	static constexpr int kGBufferCount = 3;

	DXGI_FORMAT mGBufferFormats[kGBufferCount] = {
		DXGI_FORMAT_R8G8B8A8_UNORM,       // Albedo
		DXGI_FORMAT_R16G16B16A16_FLOAT,   // Normal.xyz (в 0..1) + Roughness/gloss in .w
		DXGI_FORMAT_R8G8B8A8_UNORM        // Spec/Metallic/etc
	};

	ComPtr<ID3D12Resource>      mGBuffer[kGBufferCount];
	ComPtr<ID3D12DescriptorHeap> mGBufferRTVHeap; // RTV for MRT
	ComPtr<ID3D12DescriptorHeap> mGBufferSRVHeap; // SRV for lighting pass (t0..t2 + t3=depth)

	// UI
	bool mWireframe = false;



    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;
 
	// List of all the render items.
	std::vector<std::unique_ptr<RenderItem>> mAllRitems;

	// Render items divided by PSO.
	std::vector<RenderItem*> mOpaqueRitems;

    PassConstants mMainPassCB;

	XMFLOAT3 mEyePos = { 0.0f, 0.0f, 0.0f };
	XMFLOAT4X4 mView = MathHelper::Identity4x4();
	XMFLOAT4X4 mProj = MathHelper::Identity4x4();

    float mTheta = 1.5f*XM_PI;
    float mPhi = 0.2f*XM_PI;
    float mRadius = 15.0f;

    POINT mLastMousePos;
};

void TexColumnsApp::SpawnLODObject(
	const std::string& unique_name_base,
	const std::vector<std::string>& meshNames,
	const std::string& materialName,
	XMMATRIX Scale, XMMATRIX Rotation, XMMATRIX Translation,
	const std::vector<float>& switchDist)
{
	const int gid = mNextLODGroupId++;
	LODGroup grp;

	grp.numLevels = (int)meshNames.size();
	// writing steps
	for (int i = 0; i < grp.numLevels - 1; ++i)
		grp.switchDist[i] = switchDist[i];
	for (int i = grp.numLevels - 1; i < 3; ++i)
		grp.switchDist[i] = FLT_MAX;

	// spawn 1 by 1
	for (int level = 0; level < grp.numLevels; ++level)
	{
		const std::string& mesh = meshNames[level];

		size_t before = mAllRitems.size();

		// create submeshes with old func
		(void)RenderObject(
			unique_name_base + "_L" + std::to_string(level),
			mesh, materialName, Scale, Rotation, Translation);

		// ri->lodGroupId, ri->lodLevel, push_back(ri)
		for (size_t idx = before; idx < mAllRitems.size(); ++idx) {
			RenderItem* ri = mAllRitems[idx].get();
			ri->lodGroupId = gid;
			ri->lodLevel = level;
			grp.levels[level].push_back(ri);
		}
	}

	mLODGroups[gid] = std::move(grp);
}

void TexColumnsApp::SetupImGuiStyle()
{
	ImGuiStyle& style = ImGui::GetStyle();
	ImVec4* colors = style.Colors;

	colors[ImGuiCol_WindowBg] = ImVec4(0.09f, 0.39f, 0.30f, 1.0f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.49f, 0.39f, 0.30f, 1.0f);

	colors[ImGuiCol_Button] = ImVec4(0.18f, 0.22f, 0.30f, 1.0f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.09f, 0.09f, 0.10f, 1.0f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.34f, 0.48f, 1.0f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.20f, 0.45f, 0.65f, 1.0f);

	style.WindowRounding = 1.0f;
	style.FrameRounding = 3.0f;
	style.GrabRounding = 3.0f;
}

BoundingSphere TexColumnsApp::ComputeLocalSphere(MeshGeometry* geo, const SubmeshGeometry& sm) {
	const Vertex* vb = reinterpret_cast<const Vertex*>(geo->VertexBufferCPU->GetBufferPointer());
	const uint16_t* ib = reinterpret_cast<const uint16_t*>(geo->IndexBufferCPU->GetBufferPointer());

	std::vector<XMFLOAT3> pts;
	pts.reserve(sm.IndexCount);

	// submesh = local mesh index; abs = BaseVertexLocation + idx
	for (UINT k = 0; k < sm.IndexCount; ++k)
	{
		uint32_t idx = ib[sm.StartIndexLocation + k];
		uint32_t vi = sm.BaseVertexLocation + idx;
		pts.push_back(vb[vi].Pos);
	}

	BoundingSphere s;
	BoundingSphere::CreateFromPoints(s, (UINT)pts.size(), pts.data(), sizeof(XMFLOAT3));
	return s;

}

void TexColumnsApp::UpdateWorldSphere(RenderItem* ri)
{
	XMMATRIX W = XMLoadFloat4x4(&ri->World);

	// local sphere center -> world
	XMVECTOR cL = XMLoadFloat3(&ri->BoundSphereLocal.Center);
	XMVECTOR cW = XMVector3Transform(cL, W);
	XMStoreFloat3(&ri->BoundSphereWS.Center, cW);

	// r * maxScale
	XMVECTOR S, R, T;
	XMMatrixDecompose(&S, &R, &T, W);
	XMFLOAT3 s; XMStoreFloat3(&s, S);
	float maxScale = std::max({ fabsf(s.x), fabsf(s.y), fabsf(s.z) });
	ri->BoundSphereWS.Radius = ri->BoundSphereLocal.Radius * maxScale;
}

void TexColumnsApp::BuildVisibleList()
{
	mVisibleOpaqueRitems.clear();

	BoundingFrustum frView;
	BoundingFrustum::CreateFromMatrix(frView, XMLoadFloat4x4(&mProj));

	BoundingFrustum frWorld;
	XMMATRIX invView = XMMatrixInverse(nullptr, XMLoadFloat4x4(&mView));
	frView.Transform(frWorld, invView);

	// which LOD levels to ON
	auto lodSel = ComputeSelectedLods();

	if (mOctreeRoot)
	{
		// add visible
		TraverseOctree(frWorld, mOctreeRoot.get(), mVisibleOpaqueRitems, &lodSel);
	}
	else
	{
		// if no tree
		for (auto& up : mAllRitems)
		{
			RenderItem* ri = up.get();
			if (!ri->visible) continue;

			// LOD group filter
			if (ri->lodGroupId >= 0)
			{
				auto it = lodSel.find(ri->lodGroupId);
				if (it != lodSel.end() && ri->lodLevel != it->second) continue;
			}

			if (frWorld.Contains(ri->BoundSphereWS) != DirectX::DISJOINT)
				mVisibleOpaqueRitems.push_back(ri);
		}
	}
}

BoundingBox TexColumnsApp::ComputeSceneAABB(const std::vector<std::unique_ptr<RenderItem>>& items)
{
	XMFLOAT3 minP(FLT_MAX, FLT_MAX, FLT_MAX);
	XMFLOAT3 maxP(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for (auto& up : items) {
		const auto& s = up->BoundSphereWS;
		XMFLOAT3 r(s.Radius, s.Radius, s.Radius);
		XMFLOAT3 mn(s.Center.x - r.x, s.Center.y - r.y, s.Center.z - r.z);
		XMFLOAT3 mx(s.Center.x + r.x, s.Center.y + r.y, s.Center.z + r.z);
		minP.x = std::min(minP.x, mn.x); minP.y = std::min(minP.y, mn.y); minP.z = std::min(minP.z, mn.z);
		maxP.x = std::max(maxP.x, mx.x); maxP.y = std::max(maxP.y, mx.y); maxP.z = std::max(maxP.z, mx.z);
	}

	XMFLOAT3 center((minP.x + maxP.x) * 0.5f, (minP.y + maxP.y) * 0.5f, (minP.z + maxP.z) * 0.5f);
	XMFLOAT3 half((maxP.x - minP.x) * 0.5f, (maxP.y - minP.y) * 0.5f, (maxP.z - minP.z) * 0.5f);
	float h = std::max(half.x, std::max(half.y, half.z)) + 1e-2f;

	BoundingBox out;
	out.Center = center;
	out.Extents = XMFLOAT3(h, h, h);
	return out;
}

void TexColumnsApp::MakeChildrenAABBs(const BoundingBox& parent, BoundingBox out[8], bool loose, float looseFactor)
{
	XMFLOAT3 c = parent.Center;
	XMFLOAT3 e = parent.Extents; // half-sizes
	XMFLOAT3 childE(e.x * 0.5f, e.y * 0.5f, e.z * 0.5f);

	// compute centers
	const float dx[2] = { -childE.x, +childE.x };
	const float dy[2] = { -childE.y, +childE.y };
	const float dz[2] = { -childE.z, +childE.z };

	int idx = 0;
	for (int iy = 0; iy < 2; ++iy)
		for (int iz = 0; iz < 2; ++iz)
			for (int ix = 0; ix < 2; ++ix, ++idx)
			{
				out[idx].Center = XMFLOAT3(c.x + dx[ix], c.y + dy[iy], c.z + dz[iz]);
				out[idx].Extents = childE;
				if (loose) {
					out[idx].Extents.x *= looseFactor;
					out[idx].Extents.y *= looseFactor;
					out[idx].Extents.z *= looseFactor;
				}
			}
}

std::unique_ptr<TexColumnsApp::OctreeNode> TexColumnsApp::BuildOctreeRecursive(const BoundingBox& nodeAABB, const std::vector<RenderItem*>& in, int depth)
{
	auto node = std::make_unique<OctreeNode>();
	node->bounds = nodeAABB;

	if (in.size() <= (size_t)mOctreeLeafMax || depth >= mOctreeMaxDepth) {
		node->items = in;
		return node;
	}

	BoundingBox childAABB[8];
	MakeChildrenAABBs(nodeAABB, childAABB, mUseLoose, mLooseFactor);

	std::vector<RenderItem*> childItems[8];
	std::vector<RenderItem*> stayHere;

	for (RenderItem* ri : in)
	{
		// bbox out of sphere
		BoundingBox objBB; BoundingBox::CreateFromSphere(objBB, ri->BoundSphereWS);

		int onlyChild = -1;

		// try to find kids
		for (int c = 0; c < 8; ++c) {
			auto r = childAABB[c].Contains(objBB); // Contains/Intersects/Disjoint
			if (r == DirectX::CONTAINS) {
				if (onlyChild == -1) onlyChild = c;
				else { onlyChild = -2; break; } // >=2 = on top
			}
			else if (r == DirectX::INTERSECTS) {
				onlyChild = -2; break;
			}
		}

		if (onlyChild >= 0) childItems[onlyChild].push_back(ri);
		else stayHere.push_back(ri);
	}

	// if no kids then leaf
	size_t totalKids = 0;
	for (int c = 0; c < 8; ++c) totalKids += childItems[c].size();
	if (totalKids == 0) {
		node->items = std::move(stayHere);
		return node;
	}

	// else - down
	node->items = std::move(stayHere);
	for (int c = 0; c < 8; ++c) {
		if (!childItems[c].empty()) {
			node->child[c] = BuildOctreeRecursive(childAABB[c], childItems[c], depth + 1);
		}
	}
	return node;
}

void TexColumnsApp::GatherAll(const TexColumnsApp::OctreeNode* n,
	std::vector<RenderItem*>& out,
	const std::unordered_map<int, int>* lodSel)
{
	for (auto* ri : n->items) {
		if (lodSel && ri->lodGroupId >= 0) {
			auto it = lodSel->find(ri->lodGroupId);
			if (it != lodSel->end() && ri->lodLevel != it->second) continue;
		}
		out.push_back(ri);
	}
	for (int c = 0; c < 8; ++c) if (n->child[c])
		GatherAll(n->child[c].get(), out, lodSel);
}

void TexColumnsApp::TraverseOctree(const BoundingFrustum& frWorld, const OctreeNode* n, std::vector<RenderItem*>& out, const std::unordered_map<int, int>* lodSel)
{
	if (!n) return;

	auto rel = frWorld.Contains(n->bounds);
	if (rel == DirectX::DISJOINT) return;

	if (rel == DirectX::CONTAINS) {
		GatherAll(n, out, lodSel);
		return;
	}

	// INTERSECTS: check children
	for (auto* ri : n->items) {
		if (!ri->visible) continue;
		if (lodSel && ri->lodGroupId >= 0) {
			auto it = lodSel->find(ri->lodGroupId);
			if (it != lodSel->end() && ri->lodLevel != it->second) continue;
		}
		if (frWorld.Contains(ri->BoundSphereWS) != DirectX::DISJOINT)
			out.push_back(ri);
	}
	for (int c = 0; c < 8; ++c) if (n->child[c])
		TraverseOctree(frWorld, n->child[c].get(), out, lodSel);
}

void TexColumnsApp::BuildOctree()
{
	// 1) candidate array
	std::vector<RenderItem*> items;
	items.reserve(mAllRitems.size());
	for (auto& up : mAllRitems) {
		if (!up->visible) continue;
		items.push_back(up.get());
	}

	// 2) scene AABB (cube)
	BoundingBox sceneBB = ComputeSceneAABB(mAllRitems);

	
	mOctreeRoot = BuildOctreeRecursive(sceneBB, items, /*depth=*/0);
}

std::unordered_map<int, int> TexColumnsApp::ComputeSelectedLods() const
{
	std::unordered_map<int, int> out;
	XMFLOAT3 eye = cam.GetPosition3f();
	XMVECTOR eyeV = XMLoadFloat3(&eye);

	for (auto& kv : mLODGroups)
	{
		const auto& grp = kv.second;
		RenderItem* rep = nullptr;
		for (int l = 0; l < grp.numLevels && !rep; ++l)
			if (!grp.levels[l].empty()) rep = grp.levels[l][0];
		if (!rep) continue;

		XMVECTOR c = XMLoadFloat3(&rep->BoundSphereWS.Center);
		float dist = XMVectorGetX(XMVector3Length(c - eyeV));

		int want = 0;
		if (grp.numLevels >= 2 && dist >= grp.switchDist[0]) want = 1;
		if (grp.numLevels >= 3 && dist >= grp.switchDist[1]) want = 2;
		if (want >= grp.numLevels) want = grp.numLevels - 1;

		out[kv.first] = want;
	}
	return out;
}

void TexColumnsApp::LoadEnvironmentCube(const std::wstring& file) {
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(
		md3dDevice.Get(), mCommandList.Get(), file.c_str(),
		mEnvCube, mEnvUpload));
}

void TexColumnsApp::BuildEnvRootSignature()
{
	CD3DX12_DESCRIPTOR_RANGE range;                // t0..t1: depth, envCube
	range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0);

	CD3DX12_ROOT_PARAMETER params[2];
	params[0].InitAsConstantBufferView(0);         // b0 = PassCB
	params[1].InitAsDescriptorTable(1, &range);    // t0..t1

	auto samplers = GetStaticSamplers();
	CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params,
		(UINT)samplers.size(), samplers.data(),
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	ComPtr<ID3DBlob> sig, err;
	ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
	ThrowIfFailed(md3dDevice->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
		IID_PPV_ARGS(&mSkyRootSignature)));
}

void TexColumnsApp::BuildEnvPSO() {

	D3D12_GRAPHICS_PIPELINE_STATE_DESC sky = {};
	sky.pRootSignature = mSkyRootSignature.Get();
	sky.VS = { mShaders["SkyVS"]->GetBufferPointer(), mShaders["SkyVS"]->GetBufferSize() };
	sky.PS = { mShaders["SkyPS"]->GetBufferPointer(), mShaders["SkyPS"]->GetBufferSize() };
	sky.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	sky.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	sky.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	sky.DepthStencilState.DepthEnable = FALSE;          // read depth as texture, not depth-test
	sky.SampleMask = UINT_MAX;
	sky.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	sky.NumRenderTargets = 1;
	sky.RTVFormats[0] = mBackBufferFormat;               // HDR lighting target
	sky.SampleDesc = { 1,0 };
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&sky, IID_PPV_ARGS(&mPSOs["sky"])));

}

void TexColumnsApp::BuildParticleResources()
{
	const UINT stride = sizeof(Particle);
	const UINT64 bytes = UINT64(mParticleCount) * stride;

	// GPU-buffers A/B -> DEFAULT
	mParticlesA = CreateStructuredBuffer(md3dDevice.Get(), mParticleCount, stride, D3D12_RESOURCE_STATE_COMMON);
	mParticlesB = CreateStructuredBuffer(md3dDevice.Get(), mParticleCount, stride, D3D12_RESOURCE_STATE_COMMON);

	// Upload with random attributes
	mParticlesInit = std::make_unique<UploadBuffer<Particle>>(md3dDevice.Get(), mParticleCount, false);
	for (UINT i = 0; i < mParticleCount; i++) {
		Particle p{};
		p.pos = XMFLOAT3(MathHelper::RandF(-5, 5), MathHelper::RandF(30, 40), MathHelper::RandF(-10, -20));
		p.vel = XMFLOAT3(MathHelper::RandF(-0.5f, 0.5f), MathHelper::RandF(5.5f, 5.5f), MathHelper::RandF(-0.5f, 0.5f));
		p.life = p.lifetime = MathHelper::RandF(2.0f, 5.0f);
		p.size = MathHelper::RandF(0.15f, 0.55f);
		p.rot = 0.0f;
		p.alive = 1;
		p.color = XMFLOAT4(1, 1, 1, 1);
		mParticlesInit->CopyData(i, p);
	}

	// --- copy upload -> A ---
	// A: COMMON -> COPY_DEST
	auto toCopyDest = CD3DX12_RESOURCE_BARRIER::Transition(
		mParticlesA.Get(),
		D3D12_RESOURCE_STATE_COMMON,
		D3D12_RESOURCE_STATE_COPY_DEST);
	mCommandList->ResourceBarrier(1, &toCopyDest);

	// UploadBuffer -> A
	mCommandList->CopyBufferRegion(
		mParticlesA.Get(), 0,
		mParticlesInit->Resource(), 0,
		bytes);

	// A: COPY_DEST -> NON_PIXEL_SHADER_RESOURCE (for CS in first frame)
	auto toSRV = CD3DX12_RESOURCE_BARRIER::Transition(
		mParticlesA.Get(),
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	mCommandList->ResourceBarrier(1, &toSRV);

	// simCB
	mSimCB = std::make_unique<UploadBuffer<SimCB>>(md3dDevice.Get(), 1, true);
}


void TexColumnsApp::BuildParticleDescriptors()
{
	// 5 descriptors: SRV_A, UAV_A, SRV_B, UAV_B, SRV_Sprite
	D3D12_DESCRIPTOR_HEAP_DESC desc{};
	desc.NumDescriptors = 5;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&mParticleHeap)));

	const UINT inc = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	auto cpu = mParticleHeap->GetCPUDescriptorHandleForHeapStart();

	D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
	srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Format = DXGI_FORMAT_UNKNOWN;
	srv.Buffer.FirstElement = 0;
	srv.Buffer.NumElements = mParticleCount;
	srv.Buffer.StructureByteStride = sizeof(Particle);
	srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
	uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uav.Format = DXGI_FORMAT_UNKNOWN;
	uav.Buffer.FirstElement = 0;
	uav.Buffer.NumElements = mParticleCount;
	uav.Buffer.StructureByteStride = sizeof(Particle);
	uav.Buffer.CounterOffsetInBytes = 0;
	uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

	// SRV/UAV for A/B
	md3dDevice->CreateShaderResourceView(mParticlesA.Get(), &srv, cpu); // kSRV_A
	cpu.ptr += inc;
	md3dDevice->CreateUnorderedAccessView(mParticlesA.Get(), nullptr, &uav, cpu); // kUAV_A
	cpu.ptr += inc;
	md3dDevice->CreateShaderResourceView(mParticlesB.Get(), &srv, cpu); // kSRV_B
	cpu.ptr += inc;
	md3dDevice->CreateUnorderedAccessView(mParticlesB.Get(), nullptr, &uav, cpu); // kUAV_B
	cpu.ptr += inc;

	// sprite SRV
	auto* tex = mTextures["checkboard"]->Resource.Get(); // texture name here
	D3D12_SHADER_RESOURCE_VIEW_DESC sprite{};
	sprite.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	sprite.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	sprite.Format = tex->GetDesc().Format;
	sprite.Texture2D.MipLevels = tex->GetDesc().MipLevels;
	md3dDevice->CreateShaderResourceView(tex, &sprite, cpu); // kSRV_Sprite
}

void TexColumnsApp::BuildParticleCS_RS()
{
	CD3DX12_DESCRIPTOR_RANGE t0(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
	CD3DX12_DESCRIPTOR_RANGE u0(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
	CD3DX12_ROOT_PARAMETER p[3];
	p[0].InitAsConstantBufferView(0);          // b0
	p[1].InitAsDescriptorTable(1, &t0);        // t0
	p[2].InitAsDescriptorTable(1, &u0);        // u0
	CD3DX12_ROOT_SIGNATURE_DESC rsDesc(3, p, 0, nullptr);
	ComPtr<ID3DBlob> sig, err;
	ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
	ThrowIfFailed(md3dDevice->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
		IID_PPV_ARGS(&mParticleCS_RS)));
}

void TexColumnsApp::BuildParticleGfx_RS()
{
	CD3DX12_DESCRIPTOR_RANGE t0(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // particles
	CD3DX12_DESCRIPTOR_RANGE t1(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1); // sprite
	CD3DX12_ROOT_PARAMETER p[3];
	p[0].InitAsConstantBufferView(0);                           // b0 (PassCB)
	p[1].InitAsDescriptorTable(1, &t0, D3D12_SHADER_VISIBILITY_ALL);
	p[2].InitAsDescriptorTable(1, &t1, D3D12_SHADER_VISIBILITY_PIXEL);

	auto samplers = GetStaticSamplers();
	CD3DX12_ROOT_SIGNATURE_DESC rsDesc(3, p, (UINT)samplers.size(), samplers.data(),
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	ComPtr<ID3DBlob> sig, err;
	ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
	ThrowIfFailed(md3dDevice->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
		IID_PPV_ARGS(&mParticleGfx_RS)));
}

void TexColumnsApp::BuildParticlePSO()
{
	// Compile
	auto cs = d3dUtil::CompileShader(L"Shaders\\ParticlesCS.hlsl", nullptr, "CS", "cs_5_1");
	auto vs = d3dUtil::CompileShader(L"Shaders\\ParticlesBillboard.hlsl", nullptr, "VS", "vs_5_1");
	auto ps = d3dUtil::CompileShader(L"Shaders\\ParticlesBillboard.hlsl", nullptr, "PS", "ps_5_1");

	// CS PSO
	D3D12_COMPUTE_PIPELINE_STATE_DESC c{}; c.pRootSignature = mParticleCS_RS.Get();
	c.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
	ThrowIfFailed(md3dDevice->CreateComputePipelineState(&c, IID_PPV_ARGS(&mParticleCS_PSO)));

	// Gfx PSO
	D3D12_GRAPHICS_PIPELINE_STATE_DESC g{};
	g.pRootSignature = mParticleGfx_RS.Get();
	g.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
	g.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
	g.InputLayout = { nullptr, 0 };
	g.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	g.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT); g.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
	g.SampleMask = UINT_MAX;
	g.NumRenderTargets = 1; g.RTVFormats[0] = mBackBufferFormat;
	g.DSVFormat = mDepthStencilFormat;
	g.SampleDesc.Count = 1;

	g.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	g.DepthStencilState.DepthEnable = TRUE;
	g.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
	g.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;

	g.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	auto& rt = g.BlendState.RenderTarget[0];
	rt.BlendEnable = TRUE;
	rt.SrcBlend = D3D12_BLEND_ONE; rt.DestBlend = D3D12_BLEND_ONE; rt.BlendOp = D3D12_BLEND_OP_ADD;
	rt.SrcBlendAlpha = D3D12_BLEND_ONE; rt.DestBlendAlpha = D3D12_BLEND_ONE; rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
	rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&g, IID_PPV_ARGS(&mParticleGfx_PSO)));
}

void TexColumnsApp::BuildGBufferResources()
{
	// RTV heap: 3 RTV под G-Buffer
	D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
	rtvDesc.NumDescriptors = kGBufferCount;
	rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&mGBufferRTVHeap)));

	// SRV heap: 4 SRV (3 GBuffer + 1 depthSRV + 1 sky + 3 for pbr)
	D3D12_DESCRIPTOR_HEAP_DESC srvDesc = {};
	srvDesc.NumDescriptors = kGBufferCount + 5;
	srvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&srvDesc, IID_PPV_ARGS(&mGBufferSRVHeap)));

	UINT rtvInc = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	UINT srvInc = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvH(mGBufferRTVHeap->GetCPUDescriptorHandleForHeapStart());
	CD3DX12_CPU_DESCRIPTOR_HANDLE srvH_CPU(mGBufferSRVHeap->GetCPUDescriptorHandleForHeapStart());
	CD3DX12_GPU_DESCRIPTOR_HANDLE srvH_GPU(mGBufferSRVHeap->GetGPUDescriptorHandleForHeapStart());

	for (int i = 0; i < kGBufferCount; ++i)
	{
		auto texDesc = CD3DX12_RESOURCE_DESC::Tex2D(
			mGBufferFormats[i], mClientWidth, mClientHeight, 1, 1, 1, 0,
			D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

		D3D12_CLEAR_VALUE clear = {};
		clear.Format = mGBufferFormats[i];
		clear.Color[0] = clear.Color[1] = clear.Color[2] = 0.0f;
		clear.Color[3] = 1.0f;

		ThrowIfFailed(md3dDevice->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
			&clear,
			IID_PPV_ARGS(&mGBuffer[i])));

		// RTV
		D3D12_RENDER_TARGET_VIEW_DESC rtvView = {};
		rtvView.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
		rtvView.Format = mGBufferFormats[i];
		md3dDevice->CreateRenderTargetView(mGBuffer[i].Get(), &rtvView, rtvH);
		rtvH.Offset(1, rtvInc);

		// SRV
		D3D12_SHADER_RESOURCE_VIEW_DESC srvView = {};
		srvView.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvView.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvView.Format = mGBufferFormats[i];
		srvView.Texture2D.MipLevels = 1;
		md3dDevice->CreateShaderResourceView(mGBuffer[i].Get(), &srvView, srvH_CPU);
		srvH_CPU.Offset(1, srvInc);
	}

	
	D3D12_SHADER_RESOURCE_VIEW_DESC depthSrv = {};
	depthSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	depthSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	depthSrv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; // para SRV
	depthSrv.Texture2D.MipLevels = 1;
	md3dDevice->CreateShaderResourceView(mDepthStencilBuffer.Get(), &depthSrv, srvH_CPU);
	srvH_CPU.Offset(1, srvInc);
}

void TexColumnsApp::BuildEnvironmentSRV() {
	if (!mGBufferSRVHeap || !mEnvCube) return;

	UINT inc = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	CD3DX12_CPU_DESCRIPTOR_HANDLE h(mGBufferSRVHeap->GetCPUDescriptorHandleForHeapStart());
	h.Offset(kSrvIdx_Env, inc); // слот t4

	D3D12_SHADER_RESOURCE_VIEW_DESC envSrv = {};
	envSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	envSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE; // cube (for equirect = TEXTURE2D)
	envSrv.Format = mEnvCube->GetDesc().Format;
	envSrv.TextureCube.MipLevels = mEnvCube->GetDesc().MipLevels;

	md3dDevice->CreateShaderResourceView(mEnvCube.Get(), &envSrv, h);
}

void TexColumnsApp::BuildDeferredPSOs()
{
	// GEOMETRY
	D3D12_GRAPHICS_PIPELINE_STATE_DESC g = {};
	g.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
	g.pRootSignature = mRootSignature.Get();

	// gbuffer
	g.VS = { mShaders["GBufferVS"]->GetBufferPointer(), mShaders["GBufferVS"]->GetBufferSize() };
	g.PS = { mShaders["GBufferPS"]->GetBufferPointer(), mShaders["GBufferPS"]->GetBufferSize() };
	// no tess
	g.HS = { nullptr, 0 };
	g.DS = { nullptr, 0 };

	g.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	g.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	g.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	g.SampleMask = UINT_MAX;
	g.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	//g.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;

	// 3 MRT for GBuffer
	g.NumRenderTargets = kGBufferCount;
	g.RTVFormats[0] = mGBufferFormats[0]; // DXGI_FORMAT_R8G8B8A8_UNORM (Albedo)
	g.RTVFormats[1] = mGBufferFormats[1]; // DXGI_FORMAT_R16G16B16A16_FLOAT (Normal+Rough)
	g.RTVFormats[2] = mGBufferFormats[2]; // DXGI_FORMAT_R8G8B8A8_UNORM (Spec/etc)
	g.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;   // or mDepthStencilFormat

	g.SampleDesc.Count = 1;
	g.SampleDesc.Quality = 0;

	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&g, IID_PPV_ARGS(&mPSOs["gbuf_geom"])));

	// LIGHTING: read G-Buffer, write RT
	D3D12_GRAPHICS_PIPELINE_STATE_DESC l = {};
	l.pRootSignature = mDeferredLightingRootSignature.Get();

	l.VS = { mShaders["DeferredVS"]->GetBufferPointer(), mShaders["DeferredVS"]->GetBufferSize() };
	l.PS = { mShaders["DeferredPS"]->GetBufferPointer(), mShaders["DeferredPS"]->GetBufferSize() };

	l.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	l.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	l.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	l.DepthStencilState.DepthEnable = FALSE; // fullscreen lighting wout depth
	l.SampleMask = UINT_MAX;
	l.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

	l.NumRenderTargets = 1;
	l.RTVFormats[0] = mBackBufferFormat; // gg proebali
	l.SampleDesc.Count = 1;
	l.SampleDesc.Quality = 0;

	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&l, IID_PPV_ARGS(&mPSOs["deferred_light"])));

	D3D12_GRAPHICS_PIPELINE_STATE_DESC dbg = {};
	dbg.pRootSignature = mDeferredLightingRootSignature.Get();
	dbg.VS = { mShaders["GbufDbgVS"]->GetBufferPointer(), mShaders["GbufDbgVS"]->GetBufferSize() };
	dbg.PS = { mShaders["GbufDbgPS"]->GetBufferPointer(), mShaders["GbufDbgPS"]->GetBufferSize() };
	dbg.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	dbg.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	dbg.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	dbg.DepthStencilState.DepthEnable = FALSE;            // это оверлей
	dbg.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	dbg.NumRenderTargets = 1;
	dbg.RTVFormats[0] = mBackBufferFormat;
	dbg.SampleDesc = { 1,0 };
	dbg.SampleMask = UINT_MAX;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&dbg, IID_PPV_ARGS(&mPSOs["gbuf_debug_overlay"])));

	D3D12_GRAPHICS_PIPELINE_STATE_DESC gWireframe = g;
	gWireframe.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(
		&gWireframe, IID_PPV_ARGS(&mPSOs["gbuf_geom_wireframe"])));
}

void TexColumnsApp::BuildDeferredRootSignature()
{
    // Table: t0..t3 = SRV (Albedo, NormalGloss, SpecMisc, Depth, EnvCube, Irradiance, Prefiltered, BRDF_LUT)
    CD3DX12_DESCRIPTOR_RANGE srvTable;
    srvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 8, 0); // count=8, t0 base

    // [0] b0 = PassCB 
    // [1] SRV
    CD3DX12_ROOT_PARAMETER params[2];
    params[0].InitAsConstantBufferView(0);            // b0
    params[1].InitAsDescriptorTable(1, &srvTable);    // t0..t7

    auto samplers = GetStaticSamplers();

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(
        _countof(params), params,
        (UINT)samplers.size(), samplers.data(),
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    );

    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&mDeferredLightingRootSignature)));
}

void TexColumnsApp::BuildIBLSRVs()
{
	if (!mGBufferSRVHeap) return;
	UINT inc = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	CD3DX12_CPU_DESCRIPTOR_HANDLE h(mGBufferSRVHeap->GetCPUDescriptorHandleForHeapStart());
	h.Offset(kSrvIdx_Irr, inc);

	// Irradiance (cube)
	if (mIrradianceCube) {
		D3D12_SHADER_RESOURCE_VIEW_DESC s{};
		s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
		s.Format = mIrradianceCube->GetDesc().Format;
		s.TextureCube.MipLevels = mIrradianceCube->GetDesc().MipLevels;
		md3dDevice->CreateShaderResourceView(mIrradianceCube.Get(), &s, h);
	}
	// Prefiltered (cube)
	h.Offset(1, inc);
	if (mPrefilteredCube) {
		D3D12_SHADER_RESOURCE_VIEW_DESC s{};
		s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
		s.Format = mPrefilteredCube->GetDesc().Format;
		s.TextureCube.MipLevels = mPrefilteredCube->GetDesc().MipLevels;
		md3dDevice->CreateShaderResourceView(mPrefilteredCube.Get(), &s, h);
	}
	// BRDF LUT (2D)
	h.Offset(1, inc);
	if (mBRDF_LUT) {
		D3D12_SHADER_RESOURCE_VIEW_DESC s{};
		s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		s.Format = mBRDF_LUT->GetDesc().Format;
		s.Texture2D.MipLevels = mBRDF_LUT->GetDesc().MipLevels;
		md3dDevice->CreateShaderResourceView(mBRDF_LUT.Get(), &s, h);
	}
}

void TexColumnsApp::LoadIBLTextures()
{
#ifdef OLDMAP
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(), mCommandList.Get(),
		L"../../Textures/skyIrradiance.dds", mIrradianceCube, mIrradianceUpload));
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(), mCommandList.Get(),
		L"../../Textures/skyPrefilter2.dds", mPrefilteredCube, mPrefilteredUpload));
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(), mCommandList.Get(),
		L"../../Textures/skyBrdf.dds", mBRDF_LUT, mBRDFUpload));
#else
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(), mCommandList.Get(),
		L"../../Textures/irradiance3.dds", mIrradianceCube, mIrradianceUpload));
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(), mCommandList.Get(),
		L"../../Textures/skybox.dds", mPrefilteredCube, mPrefilteredUpload));
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(), mCommandList.Get(),
		L"../../Textures/skyBrdf.dds", mBRDF_LUT, mBRDFUpload));
#endif // OLDMAP
}




void TexColumnsApp::BuildPostProcessResources()
{
	// Create render target for post-processing
	D3D12_RESOURCE_DESC renderTargetDesc;
	ZeroMemory(&renderTargetDesc, sizeof(D3D12_RESOURCE_DESC));
	renderTargetDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	renderTargetDesc.Alignment = 0;
	renderTargetDesc.Width = mClientWidth;
	renderTargetDesc.Height = mClientHeight;
	renderTargetDesc.DepthOrArraySize = 1;
	renderTargetDesc.MipLevels = 1;
	renderTargetDesc.Format = mBackBufferFormat;
	renderTargetDesc.SampleDesc.Count = 1;
	renderTargetDesc.SampleDesc.Quality = 0;
	renderTargetDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	renderTargetDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

	D3D12_CLEAR_VALUE clearValue;
	clearValue.Format = mBackBufferFormat;
	memcpy(clearValue.Color, Colors::LightSteelBlue, sizeof(float) * 4);

	ThrowIfFailed(md3dDevice->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&renderTargetDesc,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		&clearValue,
		IID_PPV_ARGS(&mPostProcessRenderTarget)));

	// Create RTV heap
	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
	rtvHeapDesc.NumDescriptors = 1;
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	rtvHeapDesc.NodeMask = 0;
	ThrowIfFailed(md3dDevice->CreateDescriptorHeap(
		&rtvHeapDesc, IID_PPV_ARGS(&mPostProcessRTVHeap)));

	// Create RTV
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
		mPostProcessRTVHeap->GetCPUDescriptorHandleForHeapStart());
	md3dDevice->CreateRenderTargetView(
		mPostProcessRenderTarget.Get(), nullptr, rtvHandle);

	// Create SRV heap // [0] scene color, [1] depth
	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.NumDescriptors = 2;
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	ThrowIfFailed(md3dDevice->CreateDescriptorHeap(
		&srvHeapDesc, IID_PPV_ARGS(&mPostProcessSRVHeap)));

	// Create SRV
	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(
		mPostProcessSRVHeap->GetCPUDescriptorHandleForHeapStart());
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = mBackBufferFormat;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;

	// SceneColor = mPostProcessRenderTarget
	md3dDevice->CreateShaderResourceView(
		mPostProcessRenderTarget.Get(), &srvDesc, srvHandle);

	// Depth = mDepthStencilBuffer
	srvHandle.Offset(1, mCbvSrvDescriptorSize);

	D3D12_SHADER_RESOURCE_VIEW_DESC depthSrv = {};
	depthSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	depthSrv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
	depthSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	depthSrv.Texture2D.MostDetailedMip = 0;
	depthSrv.Texture2D.MipLevels = 1;

	md3dDevice->CreateShaderResourceView(
		mDepthStencilBuffer.Get(), &depthSrv, srvHandle);
}

void TexColumnsApp::BuildPostProcessPSO()
{
	// Compile shaders
	OutputDebugStringA("Compiling Shaders\\GBuffer.hlsl VS\n");
	mPostProcessShaders["postVS"] = d3dUtil::CompileShader(L"Shaders\\PostProcess.hlsl", nullptr, "VS", "vs_5_0");
	OutputDebugStringA("Compiling Shaders\\GBuffer.hlsl VS\n");
	mPostProcessShaders["postPS"] = d3dUtil::CompileShader(L"Shaders\\PostProcess.hlsl", nullptr, "PS", "ps_5_0");

	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
	ZeroMemory(&psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	psoDesc.InputLayout = { nullptr, 0 };
	psoDesc.pRootSignature = mPostProcessRootSignature.Get();
	psoDesc.VS =
	{
		reinterpret_cast<BYTE*>(mPostProcessShaders["postVS"]->GetBufferPointer()),
		mPostProcessShaders["postVS"]->GetBufferSize()
	};
	psoDesc.PS =
	{
		reinterpret_cast<BYTE*>(mPostProcessShaders["postPS"]->GetBufferPointer()),
		mPostProcessShaders["postPS"]->GetBufferSize()
	};
	psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	psoDesc.DepthStencilState.DepthEnable = false;
	psoDesc.DepthStencilState.StencilEnable = false;
	psoDesc.SampleMask = UINT_MAX;
	psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	psoDesc.NumRenderTargets = 1;
	psoDesc.RTVFormats[0] = mBackBufferFormat;
	psoDesc.SampleDesc.Count = 1;
	psoDesc.SampleDesc.Quality = 0;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPostProcessPSO)));
}

void TexColumnsApp::BuildPostProcessRootSignature()	
{
	CD3DX12_DESCRIPTOR_RANGE srvTable;
	srvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

	CD3DX12_ROOT_PARAMETER slotRootParameter[1];
	slotRootParameter[0].InitAsDescriptorTable(1, &srvTable, D3D12_SHADER_VISIBILITY_PIXEL);

	CD3DX12_STATIC_SAMPLER_DESC sampler(
		0, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(1, slotRootParameter,
		1, &sampler,
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
		serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

	if (errorBlob != nullptr)
	{
		OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);

	ThrowIfFailed(md3dDevice->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(&mPostProcessRootSignature)));
}

void TexColumnsApp::BuildAtmosphereRootSignature()
{
	// SceneColor, Depth
	CD3DX12_DESCRIPTOR_RANGE srvTable;
	srvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0);

	// [0] b0 = PassCB
	// [1] b1 = AtmosphereCB
	// [2] t0-t1 = SRV table
	CD3DX12_ROOT_PARAMETER params[3];
	params[0].InitAsConstantBufferView(0);
	params[1].InitAsConstantBufferView(1);
	params[2].InitAsDescriptorTable(1, &srvTable, D3D12_SHADER_VISIBILITY_PIXEL);

	auto samplers = GetStaticSamplers();
	CD3DX12_ROOT_SIGNATURE_DESC rsDesc(
		_countof(params), params,
		(UINT)samplers.size(), samplers.data(),
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	ComPtr<ID3DBlob> sig, err;
	ThrowIfFailed(D3D12SerializeRootSignature(
		&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
	ThrowIfFailed(md3dDevice->CreateRootSignature(
		0, sig->GetBufferPointer(), sig->GetBufferSize(),
		IID_PPV_ARGS(&mAtmosphereRootSignature)));
}

void TexColumnsApp::BuildAtmospherePSO()
{
	mPostProcessShaders["atmoVS"] = d3dUtil::CompileShader(L"Shaders\\AtmospherePost.hlsl", nullptr, "FullscreenVS", "vs_5_1");
	mPostProcessShaders["atmoPS"] = d3dUtil::CompileShader(L"Shaders\\AtmospherePost.hlsl", nullptr, "AtmospherePS", "ps_5_1");
	mPostProcessShaders["heightFogPS"] = d3dUtil::CompileShader(L"Shaders\\AtmospherePost.hlsl", nullptr, "HeightFogPS", "ps_5_1");

	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.InputLayout = { nullptr, 0 }; // fullscreen
	psoDesc.pRootSignature = mAtmosphereRootSignature.Get();
	psoDesc.VS = {
		mPostProcessShaders["atmoVS"]->GetBufferPointer(),
		mPostProcessShaders["atmoVS"]->GetBufferSize()
	};
	psoDesc.PS = {
		mPostProcessShaders["atmoPS"]->GetBufferPointer(),
		mPostProcessShaders["atmoPS"]->GetBufferSize()
	};
	psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	psoDesc.DepthStencilState.DepthEnable = FALSE;
	psoDesc.DepthStencilState.StencilEnable = FALSE;
	psoDesc.SampleMask = UINT_MAX;
	psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	psoDesc.NumRenderTargets = 1;
	psoDesc.RTVFormats[0] = mBackBufferFormat;
	psoDesc.SampleDesc.Count = 1;
	psoDesc.SampleDesc.Quality = 0;

	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(
		&psoDesc, IID_PPV_ARGS(&mAtmospherePSO)));

	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDescHeightFog = psoDesc;
	psoDescHeightFog.PS = {
		mPostProcessShaders["heightFogPS"]->GetBufferPointer(),
		mPostProcessShaders["heightFogPS"]->GetBufferSize()
	};

	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(
		&psoDescHeightFog, IID_PPV_ARGS(&mHeightFogPSO)));
}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
    PSTR cmdLine, int showCmd)
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    try
    {
        TexColumnsApp theApp(hInstance);
        if(!theApp.Initialize())
            return 0;

        return theApp.Run();
    }
    catch(DxException& e)
    {
        MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
        return 0;
    }
}

TexColumnsApp::TexColumnsApp(HINSTANCE hInstance)
    : D3DApp(hInstance)
{
}

TexColumnsApp::~TexColumnsApp()
{
    if(md3dDevice != nullptr)
        FlushCommandQueue();
}
void TexColumnsApp::MoveBackFwd(float step) {
	XMFLOAT3 newPos;
	XMVECTOR fwd = cam.GetLook();
	XMStoreFloat3(&newPos, cam.GetPosition() + fwd * step);
	cam.SetPosition(newPos);
	cam.UpdateViewMatrix();
}
void TexColumnsApp::MoveLeftRight(float step) {
	XMFLOAT3 newPos;
	XMVECTOR right = cam.GetRight();
	XMStoreFloat3(&newPos, cam.GetPosition() + right * step);
	cam.SetPosition(newPos);
	cam.UpdateViewMatrix();
}
void TexColumnsApp::MoveUpDown(float step) {
	XMFLOAT3 newPos;
	XMVECTOR up = cam.GetUp();
	XMStoreFloat3(&newPos, cam.GetPosition() + up * step);
	cam.SetPosition(newPos);
	cam.UpdateViewMatrix();
}

bool TexColumnsApp::Initialize()
{
	AllocConsole();

	// std streams
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

	cam.SetPosition(0, 3, 10);
	cam.RotateY(MathHelper::Pi);
    if(!D3DApp::Initialize())
        return false;

    // Reset the command list to prep for initialization commands.
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

    // Get the increment size of a descriptor in this heap type.  This is hardware specific, 
	// so we have to query this information.
    mCbvSrvDescriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	LoadAllTextures();

	

	
	BuildRootSignature();                 //  RS geometry-pass
	BuildDeferredRootSignature();         //  RS lighting-pass
	BuildEnvRootSignature();

	BuildShadersAndInputLayout();         //  GBufferVS/PS, DeferredVS/PS

	BuildDeferredPSOs();                  //  PSO
	BuildEnvPSO();

#ifdef OLDMAP
	LoadEnvironmentCube(L"../../Textures/skyPrefilter.dds"); // or skybox.dds
#else
	LoadEnvironmentCube(L"../../Textures/skybox2.dds"); // or skybox.dds
#endif // OLDMAP

	
	BuildEnvironmentSRV();

	LoadIBLTextures();
	BuildIBLSRVs();

	BuildParticleResources();          // Copy in A + A->SRV
	BuildParticleDescriptors();        // heap
	BuildParticleCS_RS();
	BuildParticleGfx_RS();
	BuildParticlePSO();

	BuildDescriptorHeaps();               // default luna shit
	BuildPostProcessResources();
	BuildPostProcessRootSignature();
	BuildPostProcessPSO();

	// TODO MAKE IT BEAUTIFUL LATER
	mAtmosphereCB = std::make_unique<UploadBuffer<AtmosphereConstants>>(md3dDevice.Get(), 1, true);

	mAtmosphereData.fogColor = XMFLOAT3(0.6f, 0.7f, 0.9f);
	mAtmosphereData.globalDensity = 0.02f;
	mAtmosphereData.heightFalloff = 0.25f;
	mAtmosphereData.baseHeight = 0.0f;
	mAtmosphereData.fogAnisotropy = 0.0f;
	mAtmosphereData.sunDirection = XMFLOAT3(0.0f, 1.0f, 0.0f);
	mAtmosphereData.sunIntensity = 1.0f;

	mAtmosphereCB->CopyData(0, mAtmosphereData);

	BuildAtmosphereRootSignature();
	BuildAtmospherePSO();

	BuildShapeGeometry();

	BuildMaterials();
    //BuildPSOs();
    RenderWorld();
    BuildFrameResources();

	SetupImGuiStyle();

    // Execute the initialization commands.
    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Wait until initialization is complete.
    FlushCommandQueue();
    return true;
}
 
void TexColumnsApp::OnResize()
{
    D3DApp::OnResize();
	BuildGBufferResources();

	BuildEnvironmentSRV();
	BuildIBLSRVs();
	BuildPostProcessResources();

    // The window resized, so update the aspect ratio and recompute the projection matrix.
    XMMATRIX P = XMMatrixPerspectiveFovLH(0.4f*MathHelper::Pi, AspectRatio(), 1.0f, 1000.0f);
    XMStoreFloat4x4(&mProj, P);
}

void TexColumnsApp::BuildUI() {
	BeginImGuiFrame();
	ImGui::Begin("Debug");
	ImGui::Text("tam syam");
	ImGui::Checkbox("Wireframe", &mWireframe);
	ImGui::Checkbox("Skypass", &bSkypass);
	ImGui::Checkbox("Heightfog only", &mHeightFogOnly);
	ImGui::Text("bems dems");
	ImGui::End();
	ImGui::Begin("Broooo");
	ImGui::Text("tam syam");
	ImGui::End();
	ImGui::Begin("Render Item Tree");
	for (auto& rItem : mAllRitems) {
		ImGui::Text(rItem->Name.c_str());
	}
	ImGui::End();

	ImGui::Begin("Atmosphere");
	ImGui::ColorEdit3("Fog Color", &mAtmosphereData.fogColor.x);
	ImGui::SliderFloat("Global Density", &mAtmosphereData.globalDensity, 0.0f, 0.1f);
	ImGui::SliderFloat("Height Falloff", &mAtmosphereData.heightFalloff, 0.0f, 5.0f);
	ImGui::SliderFloat("Base Height", &mAtmosphereData.baseHeight, -100.0f, 100.0f);
	ImGui::SliderFloat3("Sun Dir", &mAtmosphereData.sunDirection.x, -1.0f, 1.0f);
	ImGui::SliderFloat("Sun Intensity", &mAtmosphereData.sunIntensity, 0.0f, 20.0f);
	ImGui::SliderFloat("Fog Anisotropy", &mAtmosphereData.fogAnisotropy, -0.9f, 0.9f);
	ImGui::End();
}

void TexColumnsApp::Update(const GameTimer& gt)
{
	__m128 leftpos;
	leftpos.m128_f32[0] = 0.73;
	leftpos.m128_f32[1] = 3.9;
	leftpos.m128_f32[2] = 1.1;
	__m128 rightpos;
	rightpos.m128_f32[0] = -0.73;
	rightpos.m128_f32[1] = 3.9;
	rightpos.m128_f32[2] = 1.1;
	XMVECTOR leftDir = XMVector3Normalize(cam.GetPosition() - leftpos);
	XMVECTOR rightDir = XMVector3Normalize(cam.GetPosition() - rightpos);

	// basic forwardVector (along Z)
	XMVECTOR defaultForward = XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f);

	// left eye
	XMVECTOR leftAxis = XMVector3Normalize(XMVector3Cross(defaultForward, leftDir));
	float leftDot = XMVectorGetX(XMVector3Dot(defaultForward, leftDir));
	float leftAngle = acosf(leftDot);
	XMVECTOR leftQuat = XMQuaternionRotationAxis(leftAxis, leftAngle);
	leftQuat = XMQuaternionNormalize(leftQuat);
	XMMATRIX leftRotation = XMMatrixRotationQuaternion(leftQuat);

	// right eye:
	XMVECTOR rightAxis = XMVector3Normalize(XMVector3Cross(defaultForward, rightDir));
	float rightDot = XMVectorGetX(XMVector3Dot(defaultForward, rightDir));
	float rightAngle = acosf(rightDot);
	XMVECTOR rightQuat = XMQuaternionRotationAxis(rightAxis, rightAngle);
	rightQuat = XMQuaternionNormalize(rightQuat);
	XMMATRIX rightRotation = XMMatrixRotationQuaternion(rightQuat);





	UpdateCamera(gt);
	for (auto& rItem : mAllRitems)
	{
		if (rItem->Name == "eyeL")
		{

			XMStoreFloat4x4(&rItem->World,XMMatrixScaling(3, 3, 3)* leftRotation * XMMatrixTranslation(0.63,3.9,1.1));
			rItem->NumFramesDirty = gNumFrameResources;
		}
		if (rItem->Name == "eyeR")
		{
			XMStoreFloat4x4(&rItem->World, XMMatrixScaling(3, 3, 3) * rightRotation * XMMatrixTranslation(-0.63, 3.9, 1.1));
			rItem->NumFramesDirty = gNumFrameResources;
		}
	}
    // Cycle through the circular frame resource array.
    mCurrFrameResourceIndex = (mCurrFrameResourceIndex + 1) % gNumFrameResources;
    mCurrFrameResource = mFrameResources[mCurrFrameResourceIndex].get();

    // Has the GPU finished processing the commands of the current frame resource?
    // If not, wait until the GPU has completed commands up to this fence point.
    if(mCurrFrameResource->Fence != 0 && mFence->GetCompletedValue() < mCurrFrameResource->Fence)
    {
        HANDLE eventHandle = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
        ThrowIfFailed(mFence->SetEventOnCompletion(mCurrFrameResource->Fence, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }

	AnimateMaterials(gt);
	UpdateObjectCBs(gt);
	UpdateMaterialCBs(gt);
	UpdateMainPassCB(gt);
	BuildVisibleList();

	BuildUI();
	{
		XMVECTOR s = XMLoadFloat3(&mAtmosphereData.sunDirection);
		if (XMVector3LengthSq(s).m128_f32[0] > 0.0001f) // на всякий
		{
			s = XMVector3Normalize(s);
			XMStoreFloat3(&mAtmosphereData.sunDirection, s);
		}
	}

	if (mAtmosphereCB) mAtmosphereCB->CopyData(0, mAtmosphereData);
}

void TexColumnsApp::Draw(const GameTimer& gt)
{
	auto alloc = mCurrFrameResource->CmdListAlloc;
	ThrowIfFailed(alloc->Reset());
	auto* gbufPso = mWireframe ? mPSOs["gbuf_geom_wireframe"].Get() : mPSOs["gbuf_geom"].Get(); // TODO: MAKE IN UPDATE/KEY PROCESSING
	ThrowIfFailed(mCommandList->Reset(alloc.Get(), gbufPso));

	mCommandList->RSSetViewports(1, &mScreenViewport);
	mCommandList->RSSetScissorRects(1, &mScissorRect);

	//Geometry pass: transition GBuffer -> RTV, clear, bind
	D3D12_CPU_DESCRIPTOR_HANDLE gRTVs[kGBufferCount];
	{
		UINT rtvInc = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		auto base = mGBufferRTVHeap->GetCPUDescriptorHandleForHeapStart();
		for (int i = 0; i < kGBufferCount; ++i)
		{
			gRTVs[i] = { base.ptr + i * rtvInc };
			mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
				mGBuffer[i].Get(),
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_RENDER_TARGET));
		}
	}

	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		mDepthStencilBuffer.Get(),
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		D3D12_RESOURCE_STATE_DEPTH_WRITE));

	float clear[4] = { 0,0,0,1 };
	for (int i = 0; i < kGBufferCount; ++i) mCommandList->ClearRenderTargetView(gRTVs[i], clear, 0, nullptr);
	mCommandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.f, 0, 0, nullptr);

	mCommandList->OMSetRenderTargets(kGBufferCount, gRTVs, FALSE, &DepthStencilView());

	// ordinary heaps/RS/CBV like ones for geo:
	ID3D12DescriptorHeap* heaps[] = { mSrvDescriptorHeap.Get() };
	mCommandList->SetDescriptorHeaps(_countof(heaps), heaps);
	mCommandList->SetGraphicsRootSignature(mRootSignature.Get());
	auto passCB = mCurrFrameResource->PassCB->Resource();
	mCommandList->SetGraphicsRootConstantBufferView(3, passCB->GetGPUVirtualAddress());


	auto& listToDraw = mEnableCulling ? mVisibleOpaqueRitems : mOpaqueRitems;
	DrawRenderItems(mCommandList.Get(), listToDraw);

	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		mDepthStencilBuffer.Get(),
		D3D12_RESOURCE_STATE_DEPTH_WRITE,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

	//Lighting pass: GBuffer RTV->SRV, PostProcessRT -> RTV
	for (int i = 0; i < kGBufferCount; ++i)
		mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			mGBuffer[i].Get(),
			D3D12_RESOURCE_STATE_RENDER_TARGET,
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		mPostProcessRenderTarget.Get(),
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		D3D12_RESOURCE_STATE_RENDER_TARGET));

	// Lighting PSO + RS
	mCommandList->SetPipelineState(mPSOs["deferred_light"].Get());
	mCommandList->SetGraphicsRootSignature(mDeferredLightingRootSignature.Get());

	// SRV G-Buffer (t0..t2 + t3 depth)
	ID3D12DescriptorHeap* gbufHeaps[] = { mGBufferSRVHeap.Get() };
	mCommandList->SetDescriptorHeaps(_countof(gbufHeaps), gbufHeaps);

	// b0 = PassCB
	mCommandList->SetGraphicsRootConstantBufferView(0, passCB->GetGPUVirtualAddress());
	// t0..t3 = GBuffer SRV
	mCommandList->SetGraphicsRootDescriptorTable(1, mGBufferSRVHeap->GetGPUDescriptorHandleForHeapStart());

	// render to mPostProcessRenderTarget
	auto ppRTV = mPostProcessRTVHeap->GetCPUDescriptorHandleForHeapStart();
	mCommandList->OMSetRenderTargets(1, &ppRTV, TRUE, nullptr);

	mCommandList->IASetVertexBuffers(0, 0, nullptr);
	mCommandList->IASetIndexBuffer(nullptr);
	mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mCommandList->DrawInstanced(3, 1, 0, 0);


	if (bSkypass) {
		// cubemap pass
		mCommandList->SetPipelineState(mPSOs["sky"].Get());
		mCommandList->SetGraphicsRootSignature(mSkyRootSignature.Get());
		ID3D12DescriptorHeap* skyHeaps[] = { mGBufferSRVHeap.Get() };
		mCommandList->SetDescriptorHeaps(_countof(skyHeaps), skyHeaps);
		mCommandList->SetGraphicsRootConstantBufferView(0, passCB->GetGPUVirtualAddress());
		CD3DX12_GPU_DESCRIPTOR_HANDLE skySrv(mGBufferSRVHeap->GetGPUDescriptorHandleForHeapStart());
		skySrv.Offset(kSrvIdx_Depth, mCbvSrvDescriptorSize);
		mCommandList->SetGraphicsRootDescriptorTable(1, skySrv);
		mCommandList->IASetVertexBuffers(0, 0, nullptr);
		mCommandList->IASetIndexBuffer(nullptr);
		mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		mCommandList->DrawInstanced(3, 1, 0, 0);
	}


	// particles
	const bool useA = (mFrameIndex % 2) == 0; // ping-pong here
	ID3D12Resource* inRes = useA ? mParticlesA.Get() : mParticlesB.Get();
	ID3D12Resource* outRes = useA ? mParticlesB.Get() : mParticlesA.Get();

	ID3D12DescriptorHeap* ph[] = { mParticleHeap.Get() };
	mCommandList->SetDescriptorHeaps(1, ph);
	mCommandList->SetPipelineState(mParticleCS_PSO.Get());
	mCommandList->SetComputeRootSignature(mParticleCS_RS.Get());

	// SimCB
	SimCB scb{ gt.DeltaTime(), XMFLOAT3(0,-9.8f,0), sphereC, sphereR };
	mSimCB->CopyData(0, scb);
	mCommandList->SetComputeRootConstantBufferView(0, mSimCB->Resource()->GetGPUVirtualAddress());

	
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(outRes,
		D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

	// bind t0/u0
	auto gpuStart = mParticleHeap->GetGPUDescriptorHandleForHeapStart();
	UINT inc = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	auto gpuAt = [&](UINT idx) { auto h = gpuStart; h.ptr += idx * inc; return h; };

	mCommandList->SetComputeRootDescriptorTable(1, gpuAt(useA ? kSRV_A : kSRV_B));
	mCommandList->SetComputeRootDescriptorTable(2, gpuAt(useA ? kUAV_B : kUAV_A));

	UINT groups = (mParticleCount + 255) / 256;
	mCommandList->Dispatch(groups, 1, 1);

	// UAV barrier + out -> SRV
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(outRes));
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(outRes,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

	//  buildboard render on mPostProcessRenderTarget /w depth-test
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		mDepthStencilBuffer.Get(),
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		D3D12_RESOURCE_STATE_DEPTH_READ));

	// same RTV
	mCommandList->SetPipelineState(mParticleGfx_PSO.Get());
	mCommandList->SetGraphicsRootSignature(mParticleGfx_RS.Get());
	mCommandList->IASetVertexBuffers(0, 0, nullptr);
	mCommandList->IASetIndexBuffer(nullptr);
	mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	mCommandList->SetGraphicsRootConstantBufferView(0, mCurrFrameResource->PassCB->Resource()->GetGPUVirtualAddress());


	// t0 = particles (outRes), t1 = sprite
	mCommandList->SetGraphicsRootDescriptorTable(1, gpuAt(useA ? kSRV_B : kSRV_A)); // t0
	mCommandList->SetGraphicsRootDescriptorTable(2, gpuAt(kSRV_Sprite));           // t1

	mCommandList->OMSetRenderTargets(1, &mPostProcessRTVHeap->GetCPUDescriptorHandleForHeapStart(), TRUE, &DepthStencilView());

	mCommandList->DrawInstanced(4, mParticleCount, 0, 0);

	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		mDepthStencilBuffer.Get(),
		D3D12_RESOURCE_STATE_DEPTH_READ,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

// END OF PARTICLES
	// mPostProcessRenderTarget -> SRV
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mPostProcessRenderTarget.Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

	// post -> backbuffer
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_RENDER_TARGET));

	ID3D12PipelineState* atmoPSO = mHeightFogOnly ? mHeightFogPSO.Get() : mAtmospherePSO.Get();

	mCommandList->SetPipelineState(atmoPSO);
	mCommandList->SetGraphicsRootSignature(mAtmosphereRootSignature.Get());

	ID3D12DescriptorHeap* atmoHeaps[] = { mPostProcessSRVHeap.Get() };
	mCommandList->SetDescriptorHeaps(_countof(atmoHeaps), atmoHeaps);

	mCommandList->SetGraphicsRootConstantBufferView(0, passCB->GetGPUVirtualAddress());

	mCommandList->SetGraphicsRootConstantBufferView(
		1, mAtmosphereCB->Resource()->GetGPUVirtualAddress());

	mCommandList->SetGraphicsRootDescriptorTable(
		2, mPostProcessSRVHeap->GetGPUDescriptorHandleForHeapStart());

	CD3DX12_CPU_DESCRIPTOR_HANDLE backBufferRtv(
		mRtvHeap->GetCPUDescriptorHandleForHeapStart(),
		mCurrBackBuffer,
		mCbvSrvDescriptorSize);
	mCommandList->OMSetRenderTargets(1, &backBufferRtv, TRUE, nullptr);

	mCommandList->IASetVertexBuffers(0, 0, nullptr);
	mCommandList->IASetIndexBuffer(nullptr);
	mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mCommandList->DrawInstanced(3, 1, 0, 0);
	/*
	mCommandList->SetPipelineState(mPostProcessPSO.Get());
	mCommandList->SetGraphicsRootSignature(mPostProcessRootSignature.Get());

	ID3D12DescriptorHeap* ppHeaps[] = { mPostProcessSRVHeap.Get() };
	mCommandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
	mCommandList->SetGraphicsRootDescriptorTable(0, mPostProcessSRVHeap->GetGPUDescriptorHandleForHeapStart());

	CD3DX12_CPU_DESCRIPTOR_HANDLE backBufferRtv(
		mRtvHeap->GetCPUDescriptorHandleForHeapStart(), mCurrBackBuffer, mCbvSrvDescriptorSize);
	mCommandList->OMSetRenderTargets(1, &backBufferRtv, true, nullptr);

	mCommandList->IASetVertexBuffers(0, 0, nullptr);
	mCommandList->IASetIndexBuffer(nullptr);
	mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mCommandList->DrawInstanced(3, 1, 0, 0);
	*/

	// ============================================ DEBUG QUADS
#ifdef DEBUG
	D3D12_VIEWPORT prevVP = mScreenViewport;
	D3D12_RECT     prevSc = mScissorRect;

	// RB quad
	D3D12_VIEWPORT vp{};
	vp.Width = mClientWidth * 0.5f;
	vp.Height = mClientHeight * 0.5f;
	vp.TopLeftX = mClientWidth - vp.Width;
	vp.TopLeftY = mClientHeight - vp.Height;
	vp.MinDepth = 0.0f; vp.MaxDepth = 1.0f;

	D3D12_RECT sc{};
	sc.left = (LONG)vp.TopLeftX;
	sc.top = (LONG)vp.TopLeftY;
	sc.right = (LONG)(vp.TopLeftX + vp.Width);
	sc.bottom = (LONG)(vp.TopLeftY + vp.Height);

	mCommandList->RSSetViewports(1, &vp);
	mCommandList->RSSetScissorRects(1, &sc);

	// PSO + RS as in lighting-pass
	mCommandList->SetPipelineState(mPSOs["gbuf_debug_overlay"].Get());
	mCommandList->SetGraphicsRootSignature(mDeferredLightingRootSignature.Get());

	// SRV: GBuffer heap (t0..t3), b0 = PassCB
	ID3D12DescriptorHeap* dbgHeaps[] = { mGBufferSRVHeap.Get() };
	mCommandList->SetDescriptorHeaps(_countof(dbgHeaps), dbgHeaps);
	mCommandList->SetGraphicsRootConstantBufferView(0, passCB->GetGPUVirtualAddress());
	mCommandList->SetGraphicsRootDescriptorTable(1, mGBufferSRVHeap->GetGPUDescriptorHandleForHeapStart());

	// fullscreen tris
	mCommandList->IASetVertexBuffers(0, 0, nullptr);
	mCommandList->IASetIndexBuffer(nullptr);
	mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mCommandList->DrawInstanced(3, 1, 0, 0);

	// viewport/scissor
	mCommandList->RSSetViewports(1, &prevVP);
	mCommandList->RSSetScissorRects(1, &prevSc);
#endif


	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

	RenderImGui();

	ThrowIfFailed(mCommandList->Close());
	ID3D12CommandList* lists[] = { mCommandList.Get() };
	mCommandQueue->ExecuteCommandLists(_countof(lists), lists);
	ThrowIfFailed(mSwapChain->Present(1, 0));
	mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

	mCurrFrameResource->Fence = ++mCurrentFence;
	mCommandQueue->Signal(mFence.Get(), mCurrentFence);

	++mFrameIndex;
}

void TexColumnsApp::OnMouseDown(WPARAM btnState, int x, int y)
{
    mLastMousePos.x = x;
    mLastMousePos.y = y;

    SetCapture(mhMainWnd);
}

void TexColumnsApp::OnMouseUp(WPARAM btnState, int x, int y)
{
    ReleaseCapture();
}

void TexColumnsApp::OnMouseMove(WPARAM btnState, int x, int y)
{
	if ((btnState & MK_LBUTTON) != 0)
	{
		// Make each pixel correspond to a quarter of a degree.
		float dx = XMConvertToRadians(0.25f * static_cast<float>(x - mLastMousePos.x));
		float dy = XMConvertToRadians(0.25f * static_cast<float>(y - mLastMousePos.y));

		// Update angles based on input to orbit camera around box.

		cam.YawPitch(dx, -dy);

	}
	mLastMousePos.x = x;
	mLastMousePos.y = y;
}

 
void TexColumnsApp::OnKeyPressed(const GameTimer& gt, WPARAM key)
{
	if (GET_WHEEL_DELTA_WPARAM(key) > 0)
	{
		cam.IncreaseSpeed(0.05);
	}
	else if (GET_WHEEL_DELTA_WPARAM(key) < 0)
	{
		cam.IncreaseSpeed(-0.05);
	}
	switch (key)
	{
	case 'A':
		MoveLeftRight(-cam.GetSpeed());
		return;
	case 'W':
		MoveBackFwd(cam.GetSpeed());
		return;
	case 'S':
		MoveBackFwd(-cam.GetSpeed());
		return;
	case 'D':
		MoveLeftRight(cam.GetSpeed());
		return;
	case 'Q':
		MoveUpDown(-cam.GetSpeed());
		return;
	case 'E':
		MoveUpDown(cam.GetSpeed());
		return;
	case VK_SHIFT:
		cam.SpeedUp();
		return;
	}
}

void TexColumnsApp::OnKeyReleased(const GameTimer& gt, WPARAM key)
{
	
	switch (key)
	{
	case VK_SHIFT:
		cam.SpeedDown();
		return;
	}
}

std::wstring TexColumnsApp::GetCamSpeed()
{
	return std::to_wstring(cam.GetSpeed());
}
 
void TexColumnsApp::UpdateCamera(const GameTimer& gt)
{
	// Convert Spherical to Cartesian coordinates.
	float x = mRadius * sinf(mPhi) * cosf(mTheta);
	float z = mRadius * sinf(mPhi) * sinf(mTheta);
	float y = mRadius * cosf(mPhi);

	// Build the view matrix.
	XMVECTOR pos = XMVectorSet(x, y, z, 1.0f);
	XMVECTOR target = XMVectorZero();
	XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

	XMVECTOR campos = cam.GetPosition();
	pos = XMVectorSet(campos.m128_f32[0], campos.m128_f32[1], campos.m128_f32[2], 0.0f);
	target = cam.GetLook();
	up = cam.GetUp();
	
	XMMATRIX view = XMMatrixLookToLH(pos, target, up);
	XMStoreFloat4x4(&mView, view);
}

void TexColumnsApp::AnimateMaterials(const GameTimer& gt)
{
	
}

void TexColumnsApp::UpdateObjectCBs(const GameTimer& gt)
{
	auto currObjectCB = mCurrFrameResource->ObjectCB.get();
	for(auto& e : mAllRitems)
	{

		// Only update the cbuffer data if the constants have changed.  
		// This needs to be tracked per frame resource.
		if(e->NumFramesDirty > 0)
		{
			XMMATRIX world = XMLoadFloat4x4(&e->World);
			XMMATRIX texTransform = XMLoadFloat4x4(&e->TexTransform);

			ObjectConstants objConstants;
			XMStoreFloat4x4(&objConstants.World, XMMatrixTranspose(world));
			XMStoreFloat4x4(&objConstants.TexTransform, XMMatrixTranspose(texTransform));

			XMMATRIX wit = MathHelper::InverseTranspose(world);
			XMStoreFloat4x4(&objConstants.WorldInvTranspose, wit);

			const float roughMat = e->Mat ? e->Mat->Roughness : 0.5f;
			const float rough = e->RoughnessOverride;
			const float metal = e->MetallicOverride;
			objConstants.Roughness = rough;
			objConstants.Metallic = metal;

			UpdateWorldSphere(e.get());

			currObjectCB->CopyData(e->ObjCBIndex, objConstants);

			// Next FrameResource need to be updated too.
			e->NumFramesDirty--;
		}
	}
}

void TexColumnsApp::UpdateMaterialCBs(const GameTimer& gt)
{
	auto currMaterialCB = mCurrFrameResource->MaterialCB.get();
	for(auto& e : mMaterials)
	{
		
		// Only update the cbuffer data if the constants have changed.  If the cbuffer
		// data changes, it needs to be updated for each FrameResource.
		Material* mat = e.second.get();
		if(mat->NumFramesDirty > 0)
		{
			XMMATRIX matTransform = XMLoadFloat4x4(&mat->MatTransform);

			MaterialConstants matConstants;
			matConstants.DiffuseAlbedo = mat->DiffuseAlbedo;
			matConstants.FresnelR0 = mat->FresnelR0;
			matConstants.Roughness = mat->Roughness;
			XMStoreFloat4x4(&matConstants.MatTransform, XMMatrixTranspose(matTransform));

			currMaterialCB->CopyData(mat->MatCBIndex, matConstants);

			// Next FrameResource need to be updated too.
			mat->NumFramesDirty--;
		}
	}
}

XMFLOAT3 TexColumnsApp::AnimateLightOrbitY(const XMFLOAT3& center, float radius, float AngSpeed, const GameTimer& gt, float phase) {
	
	const float angle = XM_2PI * AngSpeed * gt.TotalTime() + phase;

	float s, c;
	XMScalarSinCos(&s, &c, angle);

	return XMFLOAT3(center.x + radius * c,
		center.y,
		center.z + radius * s);
}

void TexColumnsApp::UpdateMainPassCB(const GameTimer& gt)
{
    using namespace DirectX;
    XMMATRIX view = XMLoadFloat4x4(&mView);
    XMMATRIX proj = XMLoadFloat4x4(&mProj);
    XMMATRIX viewProj = XMMatrixMultiply(view, proj);

    XMVECTOR detV;
    XMMATRIX invView     = XMMatrixInverse(&detV, view);
    XMMATRIX invProj     = XMMatrixInverse(&detV, proj);
    XMMATRIX invViewProj = XMMatrixInverse(&detV, viewProj);

    XMStoreFloat4x4(&mMainPassCB.View,        XMMatrixTranspose(view));
    XMStoreFloat4x4(&mMainPassCB.InvView,     XMMatrixTranspose(invView));
    XMStoreFloat4x4(&mMainPassCB.Proj,        XMMatrixTranspose(proj));
    XMStoreFloat4x4(&mMainPassCB.InvProj,     XMMatrixTranspose(invProj));
    XMStoreFloat4x4(&mMainPassCB.ViewProj,    XMMatrixTranspose(viewProj));
    XMStoreFloat4x4(&mMainPassCB.InvViewProj, XMMatrixTranspose(invViewProj));

    mMainPassCB.EyePosW = cam.GetPosition3f();
    mMainPassCB.RenderTargetSize     = XMFLOAT2((float)mClientWidth, (float)mClientHeight);
    mMainPassCB.InvRenderTargetSize  = XMFLOAT2(1.0f/mClientWidth, 1.0f/mClientHeight);
	
    // near/far 
    mMainPassCB.NearZ = 1.0f;
    mMainPassCB.FarZ  = 10000.0f;

    mMainPassCB.TotalTime = gt.TotalTime();
    mMainPassCB.DeltaTime = gt.DeltaTime();
#ifdef OLDMAP
	mMainPassCB.AmbientLight = { 0.4,.4f,.4f,.4f };
#else
	mMainPassCB.AmbientLight = { 0.7,.7f,.7f,.7f };
#endif // OLDMAP

    
    mMainPassCB.Lights[0].Direction = { 0.57735f, -0.57735f, 0.57735f };
	mMainPassCB.Lights[0].Strength = { 0.0f, 0.0f, 0.0f };
    mMainPassCB.Lights[1].Direction = { -0.57735f, -0.57735f, 0.57735f };
    mMainPassCB.Lights[1].Strength  = { 0.0f, 0.0f, 0.0f };
    mMainPassCB.Lights[2].Direction = { 0.0f, -0.707f, -0.707f };
    mMainPassCB.Lights[2].Strength  = { 0.0f, 0.0f, 0.0f };

	mMainPassCB.NumDirLights = 3;
	mMainPassCB.NumPointLights = 1;
	mMainPassCB.NumSpotLights = 0;

	mMainPassCB.Lights[3].Position = { 40, 5, - 20};
	mMainPassCB.Lights[3].Strength = { 3.0f, 3.0f, 3.0f };
	mMainPassCB.Lights[3].FalloffStart = 1.0f;
	mMainPassCB.Lights[3].FalloffEnd = 120.0f;

	mMainPassCB.Lights[4].Position = { 80, 5, 80 };
	mMainPassCB.Lights[4].Strength = { 1.0f, 1.0f, 3.0f };
	mMainPassCB.Lights[4].FalloffStart = 1.0f;
	mMainPassCB.Lights[4].FalloffEnd = 120.0f;

	mMainPassCB.Lights[5].Position = { 10, 5, 80 };
	mMainPassCB.Lights[5].Strength = { 8.0f, 3.0f, 1.0f };
	mMainPassCB.Lights[5].FalloffStart = 1.0f;
	mMainPassCB.Lights[5].FalloffEnd = 30.0f;

	mMainPassCB.Lights[6].Position = AnimateLightOrbitY({ 30, 5, 80 }, 10, 1, gt, 0);
	mMainPassCB.Lights[6].Strength = { 1.0f, 50.0f, 1.0f };
	mMainPassCB.Lights[6].FalloffStart = 1.0f;
	mMainPassCB.Lights[6].FalloffEnd = 20.0f;
	mMainPassCB.NumPointLights = 4;

    mCurrFrameResource->PassCB->CopyData(0, mMainPassCB);
}


void TexColumnsApp::LoadTexture(const std::string& name)                  //============================================================
{
	auto tex = std::make_unique<Texture>();
	tex->Name = name;
	tex->Filename = L"../../Textures/" + std::wstring(name.begin(), name.end()) + L".dds";
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(),
		mCommandList.Get(), tex->Filename.c_str(),
		tex->Resource, tex->UploadHeap));
	mTextures[name] = std::move(tex);
}

void TexColumnsApp::LoadAllTextures()                                     //============================================================
{
	LoadTexture("bricks");
	LoadTexture("stone");
	LoadTexture("tile");
	LoadTexture("texture");
	LoadTexture("checkboard");
	LoadTexture("white");
	LoadTexture("gray");

}

void TexColumnsApp::BuildRootSignature()
{
	CD3DX12_DESCRIPTOR_RANGE texTable;
	texTable.Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 
        1,  // number of descriptors
        0); // register t0

	CD3DX12_DESCRIPTOR_RANGE texTable1;
	texTable1.Init(
		D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
		1,  // number of descriptors
		1); // register t1

    // Root parameter can be a table, root descriptor or root constants.
    CD3DX12_ROOT_PARAMETER slotRootParameter[5];

	// Perfomance TIP: Order from most frequent to least frequent.
	slotRootParameter[0].InitAsDescriptorTable(1, &texTable, D3D12_SHADER_VISIBILITY_PIXEL);
	
    slotRootParameter[1].InitAsConstantBufferView(0); // register b0
    slotRootParameter[2].InitAsConstantBufferView(1); // register b1
    slotRootParameter[3].InitAsConstantBufferView(2); // register b2

	slotRootParameter[4].InitAsDescriptorTable(1, &texTable1, D3D12_SHADER_VISIBILITY_PIXEL);

	auto staticSamplers = GetStaticSamplers();

    // A root signature is an array of root parameters.
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(5, slotRootParameter,
		(UINT)staticSamplers.size(), staticSamplers.data(),
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    // create a root signature with a single slot which points to a descriptor range consisting of a single constant buffer
    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

    if(errorBlob != nullptr)
    {
        ::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
    }
    ThrowIfFailed(hr);

    ThrowIfFailed(md3dDevice->CreateRootSignature(
		0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(mRootSignature.GetAddressOf())));
}
void TexColumnsApp::CreateMaterial(std::string _name, int _CBIndex, int _SRVHeapIndex, int _normalSRVHeapIndex, XMFLOAT4 _DiffuseAlbedo, XMFLOAT3 _FresnelR0, float _Roughness)
{
	auto material = std::make_unique<Material>();
	material->Name = _name;
	material->MatCBIndex = _CBIndex;
	material->DiffuseSrvHeapIndex = _SRVHeapIndex;
	material->NormalSrvHeapIndex = _normalSRVHeapIndex;
	material->DiffuseAlbedo = _DiffuseAlbedo;
	material->FresnelR0 = _FresnelR0;
	material->Roughness = _Roughness;
	mMaterials[_name] = std::move(material);
}
void TexColumnsApp::BuildDescriptorHeaps()
{
	//
	// Create the SRV heap.
	//
	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.NumDescriptors = mTextures.size();
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&mSrvDescriptorHeap)));

	//
	// Fill out the heap with actual descriptors.
	//
	CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(mSrvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	int offset = 0;
	for (const auto& tex : mTextures) {
		auto text = tex.second->Resource;
		srvDesc.Format = text->GetDesc().Format;
		srvDesc.Texture2D.MipLevels = text->GetDesc().MipLevels;
		md3dDevice->CreateShaderResourceView(text.Get(), &srvDesc, hDescriptor);
		hDescriptor.Offset(1, mCbvSrvDescriptorSize);
		CreateMaterial(tex.second->Name, offset, offset, offset, XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f), XMFLOAT3(0.05f, 0.05f, 0.05f), 0.3f);
		offset++;
	}
}

void TexColumnsApp::BuildShadersAndInputLayout()
{
	/*const D3D_SHADER_MACRO alphaTestDefines[] =
	{
		"ALPHA_TEST", "1",
		NULL, NULL
	};*/
	OutputDebugStringA("Compiling Shaders\\GBuffer.hlsl VS\n");
	mShaders["GBufferVS"] = d3dUtil::CompileShader(L"Shaders\\GBuffer.hlsl", nullptr, "VS", "vs_5_1");
	OutputDebugStringA("Compiling Shaders\\GBuffer.hlsl VS\n");
	mShaders["GBufferPS"] = d3dUtil::CompileShader(L"Shaders\\GBuffer.hlsl", nullptr, "PS", "ps_5_1");
	OutputDebugStringA("Compiling Shaders\\GBuffer.hlsl VS\n");
#ifdef DEFERREDQUADS
	mShaders["DeferredVS"] = d3dUtil::CompileShader(L"Shaders\\DeferredPasses.hlsl", nullptr, "FullscreenVS", "vs_5_1"); // Shaders\\DeferredLighting.hlsl
	OutputDebugStringA("Compiling Shaders\\GBuffer.hlsl VS\n");
	mShaders["DeferredPS"] = d3dUtil::CompileShader(L"Shaders\\DeferredPasses.hlsl", nullptr, "DeferredPS", "ps_5_1"); // Shaders\\DeferredLighting.hlsl
#else
	// new debug layer
	mShaders["GbufDbgVS"] = d3dUtil::CompileShader(L"Shaders\\DeferredPasses.hlsl", nullptr, "FullscreenVS", "vs_5_1");
	mShaders["GbufDbgPS"] = d3dUtil::CompileShader(L"Shaders\\DeferredPasses.hlsl", nullptr, "DeferredPS", "ps_5_1");

	mShaders["DeferredVS"] = d3dUtil::CompileShader(L"Shaders\\DeferredLighting.hlsl", nullptr, "FullscreenVS", "vs_5_1");
	OutputDebugStringA("Compiling Shaders\\GBuffer.hlsl VS\n");
#ifdef PBR
	mShaders["DeferredPS"] = d3dUtil::CompileShader(L"Shaders\\DeferredPBR.hlsl", nullptr, "DeferredPS", "ps_5_1");
#else
	mShaders["DeferredPS"] = d3dUtil::CompileShader(L"Shaders\\DeferredLighting.hlsl", nullptr, "DeferredPS", "ps_5_1");
#endif // PBR
#endif // DEFERREDQUADS

	mShaders["SkyVS"] = d3dUtil::CompileShader(L"Shaders\\Sky.hlsl", nullptr, "FullscreenVS", "vs_5_1");
	mShaders["SkyPS"] = d3dUtil::CompileShader(L"Shaders\\Sky.hlsl", nullptr, "SkyPS", "ps_5_1");



	////mShaders["tessVS"] = d3dUtil::CompileShader(L"Shaders\\Tessellation.hlsl", nullptr, "VS", "ps_5_0");
	//mShaders["tessVS"] = d3dUtil::CompileShader(L"Shaders\\Tessellation.hlsl", nullptr, "PS", "ps_5_0");

    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
}

void TexColumnsApp::BuildCustomMeshGeometry(std::string name, UINT& meshVertexOffset, UINT& meshIndexOffset, UINT& prevVertSize, UINT& prevIndSize, std::vector<Vertex>& vertices, std::vector<std::uint16_t>& indices, MeshGeometry* Geo)
{
	GeometryGenerator geoGen;
	std::vector<GeometryGenerator::MeshData> meshes = geoGen.LoadCustomMesh("../../Common/" + name + ".obj", ObjectsMeshCount[name]);
	UINT totalMeshSize = 0;
	UINT k = vertices.size();
	std::vector<SubmeshGeometry>meshSubmeshes;
	for (auto mesh : meshes)
	{
		meshVertexOffset = meshVertexOffset + prevVertSize;
		prevVertSize = mesh.Vertices.size();
		totalMeshSize += mesh.Vertices.size();

		meshIndexOffset = meshIndexOffset + prevIndSize;
		prevIndSize = mesh.Indices32.size();
		SubmeshGeometry meshSubmesh;
		meshSubmesh.IndexCount = (UINT)mesh.Indices32.size();
		meshSubmesh.StartIndexLocation = meshIndexOffset;
		meshSubmesh.BaseVertexLocation = meshVertexOffset;
		meshSubmeshes.push_back(meshSubmesh);
	}
	/////////
	/////
	for (auto mesh : meshes)
	{
		for (size_t i = 0; i < mesh.Vertices.size(); ++i, ++k)
		{
			vertices.push_back(Vertex(mesh.Vertices[i].Position, mesh.Vertices[i].Normal, mesh.Vertices[i].TexC));
		}
	}
	////////

	///////
	for (auto mesh : meshes)
	{
		indices.insert(indices.end(), std::begin(mesh.GetIndices16()), std::end(mesh.GetIndices16()));
	}
	///////
	Geo->MultiDrawArgs[name] = meshSubmeshes;
}



void TexColumnsApp::BuildShapeGeometry()
{
    GeometryGenerator geoGen;
	GeometryGenerator::MeshData box = geoGen.CreateBox(1.0f, 1.0f, 1.0f, 3);
	GeometryGenerator::MeshData grid = geoGen.CreateGrid(20.0f, 30.0f, 60, 40);
	GeometryGenerator::MeshData sphere = geoGen.CreateSphere(0.5f, 20, 20);
	GeometryGenerator::MeshData cylinder = geoGen.CreateCylinder(0.5f, 0.3f, 3.0f, 20, 20);
	GeometryGenerator::MeshData patch = geoGen.CreatePatch();

	//
	// We are concatenating all the geometry into one big vertex/index buffer.  So
	// define the regions in the buffer each submesh covers.
	//

	// Cache the vertex offsets to each object in the concatenated vertex buffer.
	UINT boxVertexOffset = 0;
	UINT gridVertexOffset = (UINT)box.Vertices.size();
	UINT sphereVertexOffset = gridVertexOffset + (UINT)grid.Vertices.size();
	UINT cylinderVertexOffset = sphereVertexOffset + (UINT)sphere.Vertices.size();
	UINT patchVertexOffset = cylinderVertexOffset + (UINT)cylinder.Vertices.size();

	// Cache the starting index for each object in the concatenated index buffer.
	UINT boxIndexOffset = 0;
	UINT gridIndexOffset = (UINT)box.Indices32.size();
	UINT sphereIndexOffset = gridIndexOffset + (UINT)grid.Indices32.size();
	UINT cylinderIndexOffset = sphereIndexOffset + (UINT)sphere.Indices32.size();
	UINT patchIndexOffset = cylinderIndexOffset + (UINT)cylinder.Indices32.size();

	SubmeshGeometry boxSubmesh;
	boxSubmesh.IndexCount = (UINT)box.Indices32.size();
	boxSubmesh.StartIndexLocation = boxIndexOffset;
	boxSubmesh.BaseVertexLocation = boxVertexOffset;

	SubmeshGeometry gridSubmesh;
	gridSubmesh.IndexCount = (UINT)grid.Indices32.size();
	gridSubmesh.StartIndexLocation = gridIndexOffset;
	gridSubmesh.BaseVertexLocation = gridVertexOffset;

	SubmeshGeometry sphereSubmesh;
	sphereSubmesh.IndexCount = (UINT)sphere.Indices32.size();
	sphereSubmesh.StartIndexLocation = sphereIndexOffset;
	sphereSubmesh.BaseVertexLocation = sphereVertexOffset;

	SubmeshGeometry cylinderSubmesh;
	cylinderSubmesh.IndexCount = (UINT)cylinder.Indices32.size();
	cylinderSubmesh.StartIndexLocation = cylinderIndexOffset;
	cylinderSubmesh.BaseVertexLocation = cylinderVertexOffset;

	SubmeshGeometry patchSubmesh;
	patchSubmesh.IndexCount = (UINT)patch.Indices32.size();;
	patchSubmesh.StartIndexLocation = patchIndexOffset;
	patchSubmesh.BaseVertexLocation = patchVertexOffset;

	//
	// Extract the vertex elements we are interested in and pack the
	// vertices of all the meshes into one vertex buffer.
	//
	
	auto totalVertexCount =
		box.Vertices.size() +
		grid.Vertices.size() +
		sphere.Vertices.size() +
		cylinder.Vertices.size() +
		patch.Vertices.size()
		;

	
	std::vector<Vertex> vertices(totalVertexCount);

	UINT k = 0;
	for(size_t i = 0; i < box.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = box.Vertices[i].Position;
		vertices[k].Normal = box.Vertices[i].Normal;
		vertices[k].TexC = box.Vertices[i].TexC;
	}

	for(size_t i = 0; i < grid.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = grid.Vertices[i].Position;
		vertices[k].Normal = grid.Vertices[i].Normal;
		vertices[k].TexC = grid.Vertices[i].TexC;
	}

	for(size_t i = 0; i < sphere.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = sphere.Vertices[i].Position;
		vertices[k].Normal = sphere.Vertices[i].Normal;
		vertices[k].TexC = sphere.Vertices[i].TexC;
	}

	for(size_t i = 0; i < cylinder.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = cylinder.Vertices[i].Position;
		vertices[k].Normal = cylinder.Vertices[i].Normal;
		vertices[k].TexC = cylinder.Vertices[i].TexC;
	}

	for (size_t i = 0; i < patch.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = patch.Vertices[i].Position;
		vertices[k].Normal = patch.Vertices[i].Normal;
		vertices[k].TexC = patch.Vertices[i].TexC;
	}
	
	std::vector<std::uint16_t> indices;
	indices.insert(indices.end(), std::begin(box.GetIndices16()), std::end(box.GetIndices16()));
	indices.insert(indices.end(), std::begin(grid.GetIndices16()), std::end(grid.GetIndices16()));
	indices.insert(indices.end(), std::begin(sphere.GetIndices16()), std::end(sphere.GetIndices16()));
	indices.insert(indices.end(), std::begin(cylinder.GetIndices16()), std::end(cylinder.GetIndices16()));
	indices.insert(indices.end(), std::begin(patch.GetIndices16()), std::end(patch.GetIndices16()));
	
	
	
	UINT meshVertexOffset = patchVertexOffset;
	UINT meshIndexOffset = patchIndexOffset;
	UINT prevIndSize = (UINT)patch.Indices32.size();
	UINT prevVertSize = (UINT)patch.Vertices.size();

	auto geo = std::make_unique<MeshGeometry>();

	geo->Name = "shapeGeo";

	
	BuildCustomMeshGeometry("left", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("right", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("cubee", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("suzanne", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("suzanne2", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("patchModel", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("customgrid", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("will", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("road", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	//BuildCustomMeshGeometry("overload", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("spheree", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("text", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("cubesphere", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	BuildCustomMeshGeometry("cubesphere2", meshVertexOffset, meshIndexOffset, prevVertSize, prevIndSize, vertices, indices, geo.get());
	





	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);




	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R16_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	geo->DrawArgs["box"] = boxSubmesh;
	geo->DrawArgs["grid"] = gridSubmesh;
	geo->DrawArgs["sphere"] = sphereSubmesh;
	geo->DrawArgs["cylinder"] = cylinderSubmesh;
	geo->DrawArgs["patch"] = patchSubmesh;

	mGeometries[geo->Name] = std::move(geo);
}

void TexColumnsApp::BuildQuadPatchGeometry()
{
	std::array<XMFLOAT3, 4> vertices =
	{
		XMFLOAT3(-10.0f, 0.0f, +10.0f),
		XMFLOAT3(+10.0f, 0.0f, +10.0f),
		XMFLOAT3(-10.0f, 0.0f, -10.0f),
		XMFLOAT3(+10.0f, 0.0f, -10.0f)
	};

	std::array<std::int16_t, 4> indices = { 0, 1, 2, 3 };

	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "quadpatchGeo";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(XMFLOAT3);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R16_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	SubmeshGeometry quadSubmesh;
	quadSubmesh.IndexCount = 4;
	quadSubmesh.StartIndexLocation = 0;
	quadSubmesh.BaseVertexLocation = 0;

	geo->DrawArgs["quadpatch"] = quadSubmesh;

	mGeometries[geo->Name] = std::move(geo);
}

void TexColumnsApp::BuildPSOs()
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc;

	//
	// PSO for opaque objects.
	//
    ZeroMemory(&opaquePsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	opaquePsoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
	opaquePsoDesc.pRootSignature = mRootSignature.Get();
	opaquePsoDesc.VS = 
	{ 
		reinterpret_cast<BYTE*>(mShaders["tessVS"]->GetBufferPointer()), 
		mShaders["tessVS"]->GetBufferSize()
	};
	opaquePsoDesc.HS =
	{
		reinterpret_cast<BYTE*>(mShaders["tessHS"]->GetBufferPointer()),
		mShaders["tessHS"]->GetBufferSize()
	};
	opaquePsoDesc.DS =
	{
		reinterpret_cast<BYTE*>(mShaders["tessDS"]->GetBufferPointer()),
		mShaders["tessDS"]->GetBufferSize()
	};
	opaquePsoDesc.PS = 
	{ 
		reinterpret_cast<BYTE*>(mShaders["tessPS"]->GetBufferPointer()),
		mShaders["tessPS"]->GetBufferSize()
	};
	
	opaquePsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	opaquePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
	opaquePsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	opaquePsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	opaquePsoDesc.SampleMask = UINT_MAX;
	opaquePsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH; // from triangle to patch
	opaquePsoDesc.NumRenderTargets = 1;
	opaquePsoDesc.RTVFormats[0] = mBackBufferFormat;
	opaquePsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
	opaquePsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
	opaquePsoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaquePsoDesc, IID_PPV_ARGS(&mPSOs["opaque"])));

	D3D12_GRAPHICS_PIPELINE_STATE_DESC tessPsoDesc;

	ZeroMemory(&tessPsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	tessPsoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
	tessPsoDesc.pRootSignature = mRootSignature.Get();
	tessPsoDesc.VS =
	{
		reinterpret_cast<BYTE*>(mShaders["tessVS"]->GetBufferPointer()),
		mShaders["tessVS"]->GetBufferSize()
	};
	tessPsoDesc.HS =
	{
		reinterpret_cast<BYTE*>(mShaders["tessHS"]->GetBufferPointer()),
		mShaders["tessHS"]->GetBufferSize()
	};
	tessPsoDesc.DS =
	{
		reinterpret_cast<BYTE*>(mShaders["tessDS"]->GetBufferPointer()),
		mShaders["tessDS"]->GetBufferSize()
	};
	tessPsoDesc.PS =
	{
		reinterpret_cast<BYTE*>(mShaders["tessPS"]->GetBufferPointer()),
		mShaders["tessPS"]->GetBufferSize()
	};

	tessPsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	tessPsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	tessPsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	tessPsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
	tessPsoDesc.SampleMask = UINT_MAX;
	tessPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH; // from triangle to patch
	tessPsoDesc.NumRenderTargets = 1;
	tessPsoDesc.RTVFormats[0] = mBackBufferFormat;
	tessPsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
	tessPsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
	tessPsoDesc.DSVFormat = mDepthStencilFormat;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&tessPsoDesc, IID_PPV_ARGS(&mPSOs["tess"])));

}

void TexColumnsApp::BuildFrameResources()
{
	FlushCommandQueue();
	mFrameResources.clear();
    for(int i = 0; i < gNumFrameResources; ++i)
    {
        mFrameResources.push_back(std::make_unique<FrameResource>(md3dDevice.Get(),
            1, (UINT)mAllRitems.size(), (UINT)mMaterials.size()));
    }
	mCurrFrameResourceIndex = 0;
	mCurrFrameResource = mFrameResources[mCurrFrameResourceIndex].get();
	for (auto& ri : mAllRitems)
	{
		ri->NumFramesDirty = gNumFrameResources;
	}
	for (auto& kv : mMaterials)
	{
		kv.second->NumFramesDirty = gNumFrameResources;
	}
}

void TexColumnsApp::BuildMaterials()
{
	
}
RenderItem* TexColumnsApp::RenderObject(std::string unique_name, std::string meshname,
	std::string materialName,
	XMMATRIX Scale, XMMATRIX Rotation, XMMATRIX Translation)
{
	RenderItem* first = nullptr;

	for (int i = 0; i < ObjectsMeshCount[meshname]; ++i)
	{
		auto rItem = std::make_unique<RenderItem>();
		rItem->Name = unique_name;

		XMStoreFloat4x4(&rItem->TexTransform, XMMatrixScaling(1.f, 1.f, 1.f));
		XMStoreFloat4x4(&rItem->World, Scale * Rotation * Translation);

		rItem->Mat = mMaterials[materialName].get();
		rItem->Geo = mGeometries["shapeGeo"].get();
		rItem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

		const auto& draw = rItem->Geo->MultiDrawArgs[meshname][i];
		rItem->IndexCount = draw.IndexCount;
		rItem->StartIndexLocation = draw.StartIndexLocation;
		rItem->BaseVertexLocation = draw.BaseVertexLocation;

		// FOR CULLING
		rItem->BoundSphereLocal = ComputeLocalSphere(rItem->Geo, draw);
		UpdateWorldSphere(rItem.get()); // rItem -> World

		rItem->ObjCBIndex = static_cast<UINT>(mAllRitems.size());
		rItem->NumFramesDirty = gNumFrameResources;

		// SUBLEVEL TAG
		rItem->sublevelId = mCurrentSpawningSublevelId;

		RenderItem* raw = rItem.get();
		if (!first) first = raw;

		mAllRitems.push_back(std::move(rItem));
		
	}

	
	return first;
}

RenderItem* TexColumnsApp::RenderObject(std::string unique_name, std::string meshname,
	std::string materialName, float roughness, float metallic,
	XMMATRIX Scale, XMMATRIX Rotation, XMMATRIX Translation)
{
	RenderItem* first = nullptr;

	for (int i = 0; i < ObjectsMeshCount[meshname]; ++i)
	{
		auto rItem = std::make_unique<RenderItem>();
		rItem->Name = unique_name;

		XMStoreFloat4x4(&rItem->TexTransform, XMMatrixScaling(1.f, 1.f, 1.f));
		XMStoreFloat4x4(&rItem->World, Scale * Rotation * Translation);

		rItem->Mat = mMaterials[materialName].get();
		rItem->Geo = mGeometries["shapeGeo"].get();
		rItem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

		const auto& draw = rItem->Geo->MultiDrawArgs[meshname][i];
		rItem->IndexCount = draw.IndexCount;
		rItem->StartIndexLocation = draw.StartIndexLocation;
		rItem->BaseVertexLocation = draw.BaseVertexLocation;

		// FOR CULLING
		rItem->BoundSphereLocal = ComputeLocalSphere(rItem->Geo, draw);
		UpdateWorldSphere(rItem.get()); // rItem -> World

		rItem->ObjCBIndex = static_cast<UINT>(mAllRitems.size());
		rItem->NumFramesDirty = gNumFrameResources;

		// SUBLEVEL TAG
		rItem->sublevelId = mCurrentSpawningSublevelId;

		rItem->RoughnessOverride = roughness;
		rItem->MetallicOverride = metallic;

		RenderItem* raw = rItem.get();
		if (!first) first = raw;

		mAllRitems.push_back(std::move(rItem));

	}


	return first;
}



bool TexColumnsApp::AddSublevelToScene(std::string sceneKey, XMFLOAT3 worldOffset)
{
	// 1) path
	std::string filename = sceneKey;
	if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".txt")
		filename += ".txt";
	filename = "../../Scenes/" + filename;

	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return false;
	}

	// 2) new id
	const int id = mNextSublevelId++;
	mSublevelNameToId[sceneKey] = id;

	// 3) spawn mode
	mCurrentSpawningSublevelId = id;

	// 4) parsing
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string prefix; iss >> prefix;
		if (prefix.empty() || prefix == "#") continue;

		if (prefix == "obj") {
			std::string name, geoName, matName;
			XMFLOAT3 S, R, T;
			if (iss >> name >> geoName >> matName
				>> S.x >> S.y >> S.z
				>> R.x >> R.y >> R.z
				>> T.x >> T.y >> T.z)
			{
				// WorldOffset
				T.x += worldOffset.x; T.y += worldOffset.y; T.z += worldOffset.z;

				(void)RenderObject(
					name, geoName, matName,
					XMMatrixScaling(S.x, S.y, S.z),
					XMMatrixRotationRollPitchYaw(R.x, R.y, R.z),
					XMMatrixTranslation(T.x, T.y, T.z));
			}
		}
		// L / particle_system
	}

	file.close();

	// 5) not spawn mode blyat
	mCurrentSpawningSublevelId = 0;

	ReassignObjectCBIndices(); // gg
	//RebuildRenderQueues();

	return true;
}

bool TexColumnsApp::RemoveSublevel(const std::string& sublevelName)
{
	auto it = mSublevelNameToId.find(sublevelName);
	if (it == mSublevelNameToId.end()) return false;
	int id = it->second;

	size_t before = mAllRitems.size();
	mAllRitems.erase(std::remove_if(mAllRitems.begin(), mAllRitems.end(), [id](const std::unique_ptr<RenderItem>& p) { return p && p->sublevelId == id; }), mAllRitems.end());

	mSublevelNameToId.erase(it);

	//ReassignObjectCBIndices();
	//RebuildRenderQueues();

	return mAllRitems.size() != before;
}

void TexColumnsApp::ReassignObjectCBIndices()
{
	UINT idx = 0;
	for (auto& ri : mAllRitems) {
		ri->ObjCBIndex = idx++;
		ri->NumFramesDirty = gNumFrameResources;
	}
}




void TexColumnsApp::RenderWorld()
{	
	// RenderObject(objectName, geoName, textureName, XMMatrixScaling(1, 1, 1), XMMatrixRotationRollPitchYaw(0, 3.14, 0), XMMatrixTranslation(0, 5, 0) | XMMatrixIdentity());
	// or
	// AddSublevelToScene(scenename, worldOffset);
	// or
	/*SpawnLODObject(
		"name",
		{ "obj_lod0" },
		"material",
		XMMatrixScaling(1, 1, 1),
		XMMatrixRotationY(1.0f),
		XMMatrixTranslation(0, 0, 0),
		{} no steps
	);*/

	AddSublevelToScene("testscene", XMFLOAT3(0, 0, 0));
	AddSublevelToScene("testscene2", XMFLOAT3(0, 0, 0));
	RemoveSublevel("testscene2");
	RemoveSublevel("testscene");
	//RenderObject("road", "road", "stone", XMMatrixScaling(7, 7, 7), XMMatrixRotationRollPitchYaw(0, 3.14, 0), XMMatrixTranslation(20, -5, -110));

	RenderObject("road", "cubee", "stone", XMMatrixScaling(7, 7, 7), XMMatrixRotationRollPitchYaw(0, MathHelper::Pi/2, 0), XMMatrixTranslation(20, 15, -110));
	RenderObject("road", "cubee", "stone", XMMatrixScaling(7, 7, 7), XMMatrixRotationRollPitchYaw(0, MathHelper::Pi, 0), XMMatrixTranslation(20, -0, -110));
	RenderObject("collider", "spheree","stone", 0.1, 0.1, XMMatrixScaling(sphereR, sphereR, sphereR), XMMatrixRotationRollPitchYaw(0, MathHelper::Pi, 0), XMMatrixTranslation(sphereC.x, sphereC.y, sphereC.z));
	//RenderObject("overload", "overloadsphere", "stone", XMMatrixScaling(10, 10, 10), XMMatrixRotationRollPitchYaw(0, 3.14, 0), XMMatrixTranslation(20, -5, -110));
	/*for (int i = 0; i < 100; i++) {
		std::string baseName = "lodobj" + std::to_string(i);
		float angle = XMConvertToRadians(45.0f * i);
		XMMATRIX rot = XMMatrixRotationY(angle);
		SpawnLODObject(
			baseName,
			{ "suzanne", "suzanne", "cubee" },
			"stone",
			XMMatrixScaling(1, 1, 1),
			rot,
			XMMatrixTranslation(20 + i * 5, 0, 50),
			{ 30.f, 80.f }                   // [0..30) -> L0, [30..80) -> L1, [80..inf) -> L2
		);
	}*/

	for (int i = 1; i < 11; ++i) {
		for (int j = 1; j < 11; ++j) {
			RenderObject("collider", "cubesphere", "white", i*0.1,j*0.1, XMMatrixScaling(2, 2, 2), XMMatrixRotationRollPitchYaw(0, 0, 0), XMMatrixTranslation(50 - i * 5, 5, 50 - j * 5));
			//if (i == 10 && j == 10) RenderObject("collider", "cubee", "white", i * 0.1, j * 0.1, XMMatrixScaling(1, 1, 1), XMMatrixRotationRollPitchYaw(0, 0, 0), XMMatrixTranslation(50 - i * 5, 0, 50 - j * 5));;
		}
	}
	RenderObject("text", "text", "stone", XMMatrixScaling(8, 8, 8), XMMatrixRotationRollPitchYaw(0, MathHelper::Pi, 0), XMMatrixTranslation(25, 0, 28));
	RenderObject("text", "cubee", "white", 0.1, 0.99, XMMatrixScaling(10, 10, 10), XMMatrixRotationRollPitchYaw(0, MathHelper::Pi, 0), XMMatrixTranslation(15, 0, -60));

	for (auto& up : mAllRitems) UpdateWorldSphere(up.get());

	BuildOctree();

	
	

	//for(auto& e : mAllRitems)
		//mOpaqueRitems.push_back(e.get());	
}

void TexColumnsApp::DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems)
{
    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
    UINT matCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(MaterialConstants));
 
	auto objectCB = mCurrFrameResource->ObjectCB->Resource();
	auto matCB = mCurrFrameResource->MaterialCB->Resource();

    // For each render item...
    for(size_t i = 0; i < ritems.size(); ++i)
    {
        auto ri = ritems[i];

        cmdList->IASetVertexBuffers(0, 1, &ri->Geo->VertexBufferView());
        cmdList->IASetIndexBuffer(&ri->Geo->IndexBufferView());
		cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		CD3DX12_GPU_DESCRIPTOR_HANDLE tex(mSrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
		tex.Offset(ri->Mat->DiffuseSrvHeapIndex, mCbvSrvDescriptorSize);
		CD3DX12_GPU_DESCRIPTOR_HANDLE tex1(mSrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
		tex1.Offset(ri->Mat->DiffuseSrvHeapIndex, mCbvSrvDescriptorSize);

        D3D12_GPU_VIRTUAL_ADDRESS objCBAddress = objectCB->GetGPUVirtualAddress() + ri->ObjCBIndex*objCBByteSize;
		D3D12_GPU_VIRTUAL_ADDRESS matCBAddress = matCB->GetGPUVirtualAddress() + ri->Mat->MatCBIndex*matCBByteSize;

		cmdList->SetGraphicsRootDescriptorTable(0, tex);
		cmdList->SetGraphicsRootConstantBufferView(1, objCBAddress); // b0
		cmdList->SetGraphicsRootConstantBufferView(2, matCBAddress); // b1

        cmdList->DrawIndexedInstanced(ri->IndexCount, 1, ri->StartIndexLocation, ri->BaseVertexLocation, 0);
    }
}

std::array<const CD3DX12_STATIC_SAMPLER_DESC, 6> TexColumnsApp::GetStaticSamplers()
{
	// Applications usually only need a handful of samplers.  So just define them all up front
	// and keep them available as part of the root signature.  

	const CD3DX12_STATIC_SAMPLER_DESC pointWrap(
		0, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC pointClamp(
		1, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC linearWrap(
		2, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC linearClamp(
		3, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC anisotropicWrap(
		4, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressW
		0.0f,                             // mipLODBias
		8);                               // maxAnisotropy

	const CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(
		5, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressW
		0.0f,                              // mipLODBias
		8);                                // maxAnisotropy

	return { 
		pointWrap, pointClamp,
		linearWrap, linearClamp, 
		anisotropicWrap, anisotropicClamp };
}

