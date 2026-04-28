use bytemuck::{Pod, Zeroable};
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub(crate) struct Rgba {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub alpha: f32,
}
unsafe impl DeviceRepr for Rgba {}
unsafe impl ValidAsZeroBits for Rgba {}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVoxelMaterial {
    pub albedo: [f32; 3],
    pub sigma_t: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
}
unsafe impl DeviceRepr for GpuVoxelMaterial {}
unsafe impl ValidAsZeroBits for GpuVoxelMaterial {}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVoxelGridParams {
    pub dim_x: u32,
    pub dim_y: u32,
    pub dim_z: u32,
    pub voxel_size: f32,
}
unsafe impl DeviceRepr for GpuVoxelGridParams {}
unsafe impl ValidAsZeroBits for GpuVoxelGridParams {}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuRenderParams {
    pub width: u32,
    pub height: u32,
    pub _pad0: [u32; 2],
    pub camera_pos: [f32; 3],
    pub _pad1: f32,
    pub camera_forward: [f32; 3],
    pub _pad2: f32,
    pub camera_right: [f32; 3],
    pub _pad3: f32,
    pub camera_up: [f32; 3],
    pub fov: f32,
    pub light_pos: [f32; 3],
    pub _pad4: f32,
    pub light_color: [f32; 3],
    pub light_intensity: f32,
    pub samples_per_pixel: u32,
    pub current_sample: u32,
    pub majorant: f32,
    pub seed: u32,
    pub background: [f32; 3],
    pub _pad5: f32,
}
unsafe impl DeviceRepr for GpuRenderParams {}
unsafe impl ValidAsZeroBits for GpuRenderParams {}
