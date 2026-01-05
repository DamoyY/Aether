use core::ops::{Add as _, Div as _, Mul as _, Sub as _};

use anyhow::Result;
use cudarc::driver::{LaunchConfig, PushKernelArg as _};
use glam::Vec3;
use scene_box::SceneData;

use crate::{
    config::Config,
    cuda::{context::Gpu, memory::GpuResources},
    ffi::GpuRenderParams,
    ffi::GpuVoxelGridParams,
};

pub(crate) struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub fov: f32,
}
impl Camera {
    pub(crate) fn new(position: Vec3, target: Vec3, world_up: Vec3, fov: f32) -> Self {
        let forward = target.sub(position).normalize();
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward).normalize();

        Self {
            position,
            forward,
            right,
            up,
            fov,
        }
    }
}
pub(crate) struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}
pub(crate) struct Renderer {
    ctx: Gpu,
    resources: GpuResources,
    camera: Camera,
    light: PointLight,
    config: Config,
    majorant: f32,
    background: [f32; 3],
    current_sample: u32,
}
fn aces_tonemap(x: f32) -> f32 {
    const ACES_A: f32 = 2.51;
    const ACES_B: f32 = 0.03;
    const ACES_C: f32 = 2.43;
    const ACES_D: f32 = 0.59;
    const ACES_E: f32 = 0.14;
    let numerator = x.mul(x.mul(ACES_A).add(ACES_B));
    let denominator = x.mul(x.mul(ACES_C).add(ACES_D)).add(ACES_E);
    numerator.div(denominator).clamp(0.0, 1.0)
}
fn gamma_correct_channel(value: f32) -> u8 {
    let tonemapped = aces_tonemap(value);
    let gamma = 1.0_f32.div(2.2);
    let corrected = tonemapped.powf(gamma).mul(255.0);
    f32_to_u8_clamped(corrected)
}
fn f32_to_u8_clamped(value: f32) -> u8 {
    if value <= 0.0 {
        return 0;
    }
    if value >= 255.0 {
        return 255;
    }
    let mut low: u8 = 0;
    let mut high: u8 = 255;
    while low < high {
        let mid = low.wrapping_add(high.wrapping_sub(low).wrapping_div(2));
        if f32::from(mid) < value {
            low = mid.saturating_add(1);
        } else {
            high = mid;
        }
    }
    low
}
fn compute_majorant(voxels: &[scene_box::Voxel]) -> f32 {
    voxels
        .iter()
        .filter(|voxel| voxel.intensity > 0.0)
        .map(|voxel| {
            let sigma_t = [
                voxel.sigma_a[0].add(voxel.sigma_s[0]),
                voxel.sigma_a[1].add(voxel.sigma_s[1]),
                voxel.sigma_a[2].add(voxel.sigma_s[2]),
            ];
            sigma_t[0].max(sigma_t[1]).max(sigma_t[2])
        })
        .fold(0.0_f32, f32::max)
}
impl Renderer {
    pub(crate) fn new(config: Config, scene_data: &SceneData) -> Result<Self> {
        let ctx = Gpu::new()?;
        let voxel_params = GpuVoxelGridParams {
            dim_x: scene_data.dimensions[0],
            dim_y: scene_data.dimensions[1],
            dim_z: scene_data.dimensions[2],
            voxel_size: scene_data.voxel_size,
        };
        let resources = GpuResources::new(
            &ctx.stream,
            &scene_data.voxels,
            &voxel_params,
            config.render.width,
            config.render.height,
        )?;
        let camera = Camera::new(
            Vec3::from_array(scene_data.camera.position),
            Vec3::from_array(scene_data.camera.target),
            Vec3::Y,
            scene_data.camera.fov,
        );
        let light = PointLight {
            position: Vec3::from_array(scene_data.light.position),
            color: Vec3::from_array(scene_data.light.color),
            intensity: scene_data.light.intensity,
        };
        let majorant = compute_majorant(&scene_data.voxels);
        Ok(Self {
            ctx,
            resources,
            camera,
            light,
            config,
            majorant,
            background: scene_data.background,
            current_sample: 0,
        })
    }

    pub(crate) fn clear_accumulator(&mut self) -> Result<()> {
        let block_size = (16_u32, 16_u32, 1_u32);
        let grid_size = (
            self.resources.width.div_ceil(16),
            self.resources.height.div_ceil(16),
            1_u32,
        );
        let cfg = LaunchConfig {
            block_dim: block_size,
            grid_dim: grid_size,
            shared_mem_bytes: 0,
        };
        let mut builder = self.ctx.stream.launch_builder(&self.ctx.clear_fn);
        builder.arg(&mut self.resources.accumulator);
        builder.arg(&self.resources.width);
        builder.arg(&self.resources.height);
        // SAFETY: The kernel arguments match the expected types and the launch configuration is valid.
        unsafe { builder.launch(cfg) }?;
        self.ctx.stream.synchronize()?;
        self.current_sample = 0;
        Ok(())
    }

    pub(crate) fn render_progressive(&mut self) -> Result<()> {
        let block_size = (16_u32, 16_u32, 1_u32);
        let grid_size = (
            self.resources.width.div_ceil(16),
            self.resources.height.div_ceil(16),
            1_u32,
        );
        let cfg = LaunchConfig {
            block_dim: block_size,
            grid_dim: grid_size,
            shared_mem_bytes: 0,
        };
        let samples_per_frame = self.config.render.samples_per_frame;
        let remaining = self
            .config
            .render
            .target_samples
            .saturating_sub(self.current_sample);
        let batch_size = samples_per_frame.min(remaining);
        for _ in 0..batch_size {
            let render_params = GpuRenderParams {
                width: self.config.render.width,
                height: self.config.render.height,
                _pad0: [0; 2],
                camera_pos: self.camera.position.to_array(),
                _pad1: 0.0,
                camera_forward: self.camera.forward.to_array(),
                _pad2: 0.0,
                camera_right: self.camera.right.to_array(),
                _pad3: 0.0,
                camera_up: self.camera.up.to_array(),
                fov: self.camera.fov,
                light_pos: self.light.position.to_array(),
                _pad4: 0.0,
                light_color: self.light.color.to_array(),
                light_intensity: self.light.intensity,
                samples_per_pixel: 1,
                current_sample: self.current_sample,
                majorant: self.majorant,
                seed: rand::random(),
                background: self.background,
                _pad5: 0.0,
            };
            self.resources
                .update_render_params(&self.ctx.stream, &render_params)?;
            let mut builder = self.ctx.stream.launch_builder(&self.ctx.render_fn);
            builder.arg(&mut self.resources.framebuffer);
            builder.arg(&mut self.resources.accumulator);
            builder.arg(&self.resources.density_texture.texture);
            builder.arg(&self.resources.material_buffer);
            builder.arg(&self.resources.voxel_params);
            builder.arg(&self.resources.render_params);
            // SAFETY: The kernel arguments match the expected types and the launch configuration is valid.
            unsafe { builder.launch(cfg) }?;
            self.current_sample = self.current_sample.saturating_add(1);
        }

        self.ctx.stream.synchronize()?;
        Ok(())
    }

    pub(crate) fn get_framebuffer(&self) -> Result<Vec<u8>> {
        let float_data = self.resources.read_framebuffer(&self.ctx.stream)?;
        let capacity = float_data
            .len()
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("framebuffer capacity overflow"))?;
        let mut output = Vec::with_capacity(capacity);
        for pixel in float_data {
            let red = gamma_correct_channel(pixel.red);
            let green = gamma_correct_channel(pixel.green);
            let blue = gamma_correct_channel(pixel.blue);
            output.extend_from_slice(&[red, green, blue, 255]);
        }
        Ok(output)
    }

    pub(crate) const fn sample_count(&self) -> u32 {
        self.current_sample
    }

    pub(crate) const fn target_samples(&self) -> u32 {
        self.config.render.target_samples
    }

    pub(crate) fn window_title(&self) -> &str {
        &self.config.window.title
    }
}
