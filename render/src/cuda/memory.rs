extern crate alloc;

use alloc::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;

use scene_box::Voxel;

use crate::ffi::{GpuRenderParams, GpuVoxel, GpuVoxelGridParams, Rgba};

pub(crate) struct GpuResources {
    pub voxel_buffer: CudaSlice<GpuVoxel>,
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

        let gpu_voxels: Vec<GpuVoxel> = voxel_data
            .iter()
            .map(|voxel| GpuVoxel {
                intensity: voxel.intensity,
                sigma_a: voxel.sigma_a,
                sigma_s: voxel.sigma_s,
                anisotropy: voxel.anisotropy,
                ior: voxel.ior,
            })
            .collect();
        let voxel_buffer = stream.clone_htod(&gpu_voxels)?;
        let framebuffer = stream.alloc_zeros::<Rgba>(pixel_count)?;
        let accumulator = stream.alloc_zeros::<Rgba>(pixel_count)?;
        let voxel_params_gpu = stream.clone_htod(&[*voxel_params])?;
        let render_params = stream.alloc_zeros::<GpuRenderParams>(1)?;

        Ok(Self {
            voxel_buffer,
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
