extern crate alloc;
use alloc::sync::Arc;
use core::mem::{size_of, MaybeUninit};
use core::ptr;
use anyhow::Result;
use cudarc::driver::sys::{
    cuArray3DCreate_v2, cuArrayDestroy, cuMemcpy3D_v2, cuTexObjectCreate, cuTexObjectDestroy,
    CUaddress_mode, CUarray, CUarray_format, CUfilter_mode, CUmemorytype, CUresourcetype,
    CUresult, CUtexObject, CUDA_ARRAY3D_DESCRIPTOR, CUDA_MEMCPY3D, CUDA_RESOURCE_DESC,
    CUDA_TEXTURE_DESC,
};
use cudarc::driver::{CudaSlice, CudaStream};
use scene_box::Voxel;
use crate::ffi::{GpuRenderParams, GpuVoxelGridParams, GpuVoxelMaterial, Rgba};
pub(crate) struct DensityTexture {
    array: CUarray,
    pub(crate) texture: CUtexObject,
}
impl DensityTexture {
    fn new(voxel_data: &[Voxel], dimensions: [u32; 3]) -> Result<Self> {
        let [dim_x, dim_y, dim_z] = dimensions;
        let width = usize::try_from(dim_x)?;
        let height = usize::try_from(dim_y)?;
        let depth = usize::try_from(dim_z)?;
        let total_voxels = width
            .checked_mul(height)
            .and_then(|count| count.checked_mul(depth))
            .ok_or_else(|| anyhow::anyhow!("voxel count overflow"))?;
        let densities: Vec<f32> = voxel_data
            .iter()
            .take(total_voxels)
            .map(|voxel| voxel.intensity)
            .collect();
        let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: width,
            Height: height,
            Depth: depth,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 1,
            Flags: 0,
        };
        let mut array: CUarray = ptr::null_mut();
        // SAFETY: cuArray3DCreate_v2 initializes the array pointer with valid CUDA array.
        let create_result =
            unsafe { cuArray3DCreate_v2(ptr::from_mut(&mut array), ptr::from_ref(&array_desc)) };
        if create_result != CUresult::CUDA_SUCCESS {
            return Err(anyhow::anyhow!(
                "cuArray3DCreate_v2 failed with error code: {create_result:?}"
            ));
        }
        let mut copy_params_uninit = MaybeUninit::<CUDA_MEMCPY3D>::uninit();
        let copy_params_ptr = copy_params_uninit.as_mut_ptr();
        // SAFETY: Zeroing raw bytes is safe; we set all necessary fields before reading.
        // assume_init_mut is safe because all bytes are zeroed and we set required fields.
        unsafe {
            ptr::write_bytes(copy_params_ptr, 0, 1);
        }
        // SAFETY: All bytes are zeroed and we set required fields before use.
        let copy_params = unsafe { copy_params_uninit.assume_init_mut() };
        copy_params.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
        copy_params.srcHost = densities.as_ptr().cast::<core::ffi::c_void>();
        copy_params.srcPitch = width.saturating_mul(size_of::<f32>());
        copy_params.srcHeight = height;
        copy_params.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
        copy_params.dstArray = array;
        copy_params.WidthInBytes = width.saturating_mul(size_of::<f32>());
        copy_params.Height = height;
        copy_params.Depth = depth;
        // SAFETY: cuMemcpy3D_v2 copies data from host to device array.
        let copy_result = unsafe { cuMemcpy3D_v2(ptr::from_ref(copy_params)) };
        if copy_result != CUresult::CUDA_SUCCESS {
            // SAFETY: array was successfully created above.
            unsafe { cuArrayDestroy(array); }
            return Err(anyhow::anyhow!(
                "cuMemcpy3D_v2 failed with error code: {copy_result:?}"
            ));
        }
        let mut res_desc_uninit = MaybeUninit::<CUDA_RESOURCE_DESC>::uninit();
        // SAFETY: Zeroing raw bytes is safe; we set all necessary fields before reading.
        unsafe {
            ptr::write_bytes(res_desc_uninit.as_mut_ptr(), 0, 1);
        }
        // SAFETY: All bytes are zeroed and we set required fields before use.
        let res_desc = unsafe { res_desc_uninit.assume_init_mut() };
        res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;
        let mut tex_desc_uninit = MaybeUninit::<CUDA_TEXTURE_DESC>::uninit();
        // SAFETY: Zeroing raw bytes is safe; we set all necessary fields before reading.
        unsafe {
            ptr::write_bytes(tex_desc_uninit.as_mut_ptr(), 0, 1);
        }
        // SAFETY: All bytes are zeroed and we set required fields before use.
        let tex_desc = unsafe { tex_desc_uninit.assume_init_mut() };
        tex_desc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER;
        tex_desc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER;
        tex_desc.addressMode[2] = CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER;
        tex_desc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
        tex_desc.flags = 0;
        let mut texture: CUtexObject = 0;
        // SAFETY: cuTexObjectCreate creates a texture object from the resource and texture descriptors.
        let tex_result = unsafe {
            cuTexObjectCreate(
                ptr::from_mut(&mut texture),
                ptr::from_ref(res_desc),
                ptr::from_ref(tex_desc),
                ptr::null(),
            )
        };
        if tex_result != CUresult::CUDA_SUCCESS {
            // SAFETY: array was successfully created above.
            unsafe { cuArrayDestroy(array); }
            return Err(anyhow::anyhow!(
                "cuTexObjectCreate failed with error code: {tex_result:?}"
            ));
        }
        Ok(Self { array, texture })
    }
}
impl Drop for DensityTexture {
    fn drop(&mut self) {
        // SAFETY: texture was successfully created in new().
        unsafe { cuTexObjectDestroy(self.texture); }
        // SAFETY: array was successfully created in new().
        unsafe { cuArrayDestroy(self.array); }
    }
}
pub(crate) struct GpuResources {
    pub(crate) density_texture: DensityTexture,
    pub(crate) material_buffer: CudaSlice<GpuVoxelMaterial>,
    pub(crate) framebuffer: CudaSlice<Rgba>,
    pub(crate) accumulator: CudaSlice<Rgba>,
    pub(crate) voxel_params: CudaSlice<GpuVoxelGridParams>,
    pub(crate) render_params: CudaSlice<GpuRenderParams>,
    pub(crate) width: u32,
    pub(crate) height: u32,
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

        let density_texture = DensityTexture::new(
            voxel_data,
            [voxel_params.dim_x, voxel_params.dim_y, voxel_params.dim_z],
        )?;
        let gpu_materials: Vec<GpuVoxelMaterial> = voxel_data
            .iter()
            .map(|voxel| GpuVoxelMaterial {
                albedo: voxel.albedo,
                sigma_t: voxel.sigma_t,
                anisotropy: voxel.anisotropy,
                ior: voxel.ior,
            })
            .collect();
        let material_buffer = stream.clone_htod(&gpu_materials)?;
        let framebuffer = stream.alloc_zeros::<Rgba>(pixel_count)?;
        let accumulator = stream.alloc_zeros::<Rgba>(pixel_count)?;
        let voxel_params_gpu = stream.clone_htod(&[*voxel_params])?;
        let render_params = stream.alloc_zeros::<GpuRenderParams>(1)?;
        Ok(Self {
            density_texture,
            material_buffer,
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
