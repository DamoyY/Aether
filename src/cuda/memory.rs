extern crate alloc;

use alloc::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream};

use crate::{
    ffi::{GpuRenderParams, GpuVoxelGridParams, Rgba},
    voxel::Voxel,
};

pub(crate) struct GpuResources {
    pub voxels: CudaSlice<Voxel>,
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

        let voxels = stream.clone_htod(voxel_data)?;
        let framebuffer = stream.alloc_zeros::<Rgba>(pixel_count)?;
        let accumulator = stream.alloc_zeros::<Rgba>(pixel_count)?;
        let voxel_params = stream.alloc_zeros::<GpuVoxelGridParams>(1)?;
        let render_params = stream.alloc_zeros::<GpuRenderParams>(1)?;

        Ok(Self {
            voxels,
            framebuffer,
            accumulator,
            voxel_params,
            render_params,
            width,
            height,
        })
    }

    pub(crate) fn update_voxel_params(
        &mut self,
        stream: &Arc<CudaStream>,
        params: &GpuVoxelGridParams,
    ) -> Result<()> {
        self.voxel_params = stream.clone_htod(&[*params])?;
        Ok(())
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
