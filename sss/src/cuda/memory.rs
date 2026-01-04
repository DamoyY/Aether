extern crate alloc;

use alloc::sync::Arc;
use core::ptr;

use anyhow::Result;
use cudarc::driver::{
    sys::{
        CUaddress_mode, CUarray, CUarray_format, CUfilter_mode, CUresourcetype,
        CUtexObject, CUDA_ARRAY3D_DESCRIPTOR, CUDA_RESOURCE_DESC, CUDA_TEXTURE_DESC,
        cuArray3DCreate_v2, cuArrayDestroy, cuMemcpy3D_v2, cuTexObjectCreate,
        cuTexObjectDestroy, CUDA_MEMCPY3D,
    },
    CudaSlice, CudaStream,
};

use scene::Voxel;

use crate::ffi::{GpuRenderParams, GpuVoxelGridParams, Rgba};

pub(crate) struct VoxelTexture {
    array: CUarray,
    texture: CUtexObject,
}

impl VoxelTexture {
    fn new(
        voxel_data: &[Voxel],
        dim_x: u32,
        dim_y: u32,
        dim_z: u32,
    ) -> Result<Self> {
        let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: usize::try_from(dim_x)?,
            Height: usize::try_from(dim_y)?,
            Depth: usize::try_from(dim_z)?,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 1,
            Flags: 0,
        };

        let mut array: CUarray = ptr::null_mut();
        // SAFETY: cuArray3DCreate_v2 is a CUDA Driver API call that creates a 3D array.
        let create_result = unsafe { cuArray3DCreate_v2(&raw mut array, &raw const array_desc) };
        if create_result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow::anyhow!("cuArray3DCreate_v2 failed: {create_result:?}"));
        }

        let copy_params = CUDA_MEMCPY3D {
            srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: voxel_data.as_ptr().cast(),
            srcPitch: usize::try_from(dim_x)?.saturating_mul(size_of::<f32>()),
            srcHeight: usize::try_from(dim_y)?,
            srcXInBytes: 0,
            srcY: 0,
            srcZ: 0,
            srcLOD: 0,
            srcDevice: 0,
            srcArray: ptr::null_mut(),
            dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
            dstArray: array,
            dstXInBytes: 0,
            dstY: 0,
            dstZ: 0,
            dstLOD: 0,
            dstDevice: 0,
            dstHost: ptr::null_mut(),
            dstPitch: 0,
            dstHeight: 0,
            WidthInBytes: usize::try_from(dim_x)?.saturating_mul(size_of::<f32>()),
            Height: usize::try_from(dim_y)?,
            Depth: usize::try_from(dim_z)?,
            reserved0: ptr::null_mut(),
            reserved1: ptr::null_mut(),
        };

        // SAFETY: cuMemcpy3D_v2 copies data to the 3D array.
        let copy_result = unsafe { cuMemcpy3D_v2(&raw const copy_params) };
        if copy_result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            // SAFETY: We created the array, so we must destroy it on failure.
            unsafe { cuArrayDestroy(array); }
            return Err(anyhow::anyhow!("cuMemcpy3D_v2 failed: {copy_result:?}"));
        }

        let res_desc = CUDA_RESOURCE_DESC {
            resType: CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
            res: cudarc::driver::sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                array: cudarc::driver::sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                    hArray: array,
                },
            },
            flags: 0,
        };

        let tex_desc = CUDA_TEXTURE_DESC {
            addressMode: [
                CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER,
                CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER,
                CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER,
            ],
            filterMode: CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
            flags: 0,
            maxAnisotropy: 1,
            mipmapFilterMode: CUfilter_mode::CU_TR_FILTER_MODE_POINT,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0; 4],
            reserved: [0_i32; 12],
        };

        let mut texture: CUtexObject = 0;
        // SAFETY: cuTexObjectCreate creates a texture object from the 3D array.
        let tex_result = unsafe {
            cuTexObjectCreate(&raw mut texture, &raw const res_desc, &raw const tex_desc, ptr::null())
        };
        if tex_result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            // SAFETY: We created the array, so we must destroy it on failure.
            unsafe { cuArrayDestroy(array); }
            return Err(anyhow::anyhow!("cuTexObjectCreate failed: {tex_result:?}"));
        }

        Ok(Self { array, texture })
    }

    pub(crate) const fn texture(&self) -> CUtexObject {
        self.texture
    }
}

impl Drop for VoxelTexture {
    fn drop(&mut self) {
        // SAFETY: We own the texture object.
        unsafe { cuTexObjectDestroy(self.texture); }
        // SAFETY: We own the array.
        unsafe { cuArrayDestroy(self.array); }
    }
}

pub(crate) struct GpuResources {
    pub voxel_texture: VoxelTexture,
    pub framebuffer: CudaSlice<Rgba>,
    pub accumulator: CudaSlice<Rgba>,
    pub voxel_params: CudaSlice<GpuVoxelGridParams>,
    pub render_params: CudaSlice<GpuRenderParams>,
    pub width: u32,
    pub height: u32,
}

impl GpuResources {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        voxel_data: &[Voxel],
        voxel_params: &GpuVoxelGridParams,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let width_usize =
            usize::try_from(width).map_err(|err| anyhow::anyhow!("width too large: {err}"))?;
        let height_usize =
            usize::try_from(height).map_err(|err| anyhow::anyhow!("height too large: {err}"))?;
        let pixel_count = width_usize
            .checked_mul(height_usize)
            .ok_or_else(|| anyhow::anyhow!("pixel count overflow"))?;

        let voxel_texture = VoxelTexture::new(
            voxel_data,
            voxel_params.dim_x,
            voxel_params.dim_y,
            voxel_params.dim_z,
        )?;
        let framebuffer = stream.alloc_zeros::<Rgba>(pixel_count)?;
        let accumulator = stream.alloc_zeros::<Rgba>(pixel_count)?;
        let voxel_params_gpu = stream.clone_htod(&[*voxel_params])?;
        let render_params = stream.alloc_zeros::<GpuRenderParams>(1)?;

        Ok(Self {
            voxel_texture,
            framebuffer,
            accumulator,
            voxel_params: voxel_params_gpu,
            render_params,
            width,
            height,
        })
    }

    pub(crate) fn update_render_params(
        &mut self,
        stream: &Arc<CudaStream>,
        params: &GpuRenderParams,
    ) -> Result<()> {
        self.render_params = stream.clone_htod(&[*params])?;
        Ok(())
    }

    pub(crate) fn read_framebuffer(&self, stream: &Arc<CudaStream>) -> Result<Vec<Rgba>> {
        let data = stream.clone_dtoh(&self.framebuffer)?;
        Ok(data)
    }
}
