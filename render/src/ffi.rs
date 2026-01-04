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
// SAFETY: Rgba is #[repr(C)] and contains only f32 fields which are valid for GPU transfer.
unsafe impl DeviceRepr for Rgba {}
// SAFETY: Rgba contains only f32 fields, and zero-initialized f32 (0.0) is valid.
unsafe impl ValidAsZeroBits for Rgba {}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVoxelGridParams {
    pub dim_x: u32,
    pub dim_y: u32,
    pub dim_z: u32,
    pub voxel_size: f32,
}
// SAFETY: GpuVoxelGridParams is #[repr(C)] and contains only primitive types valid for GPU transfer.
unsafe impl DeviceRepr for GpuVoxelGridParams {}
// SAFETY: GpuVoxelGridParams contains only u32 and f32 fields, zero-initialized values are valid.
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
    pub _pad5: u32,
    pub sigma_a: [f32; 3],
    pub _pad6: f32,
    pub sigma_s: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
    pub seed: u32,
    pub _pad7: [u32; 2],
    pub background: [f32; 3],
    pub _pad8: f32,
}
// SAFETY: GpuRenderParams is #[repr(C)] and contains only primitive types valid for GPU transfer.
unsafe impl DeviceRepr for GpuRenderParams {}
// SAFETY: GpuRenderParams contains only primitives and arrays of primitives, zero-initialized values are valid.
unsafe impl ValidAsZeroBits for GpuRenderParams {}
