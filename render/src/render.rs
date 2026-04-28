use crate::{
    config::Config,
    cuda::{context::Gpu, memory::GpuResources},
    ffi::{GpuRenderParams, GpuVoxelGridParams},
};
use anyhow::Result;
use core::ops::{Add as _, Div as _, Mul as _, Sub as _};
use cudarc::driver::{LaunchConfig, PushKernelArg as _};
use glam::Vec3;
use scene_box::SceneData;
pub(crate) struct Camera {
    position: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    fov: f32,
    yaw: f32,
    pitch: f32,
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
            yaw: forward.z.atan2(forward.x),
            pitch: forward.y.asin(),
        }
    }
    fn rebuild_basis(&mut self) {
        const EPS: f32 = 1.0e-6;
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let forward_raw = Vec3::new(cos_pitch.mul(cos_yaw), sin_pitch, cos_pitch.mul(sin_yaw));
        let forward = if forward_raw.length_squared() > EPS {
            forward_raw.normalize()
        } else {
            Vec3::Z
        };
        let world_up = Vec3::Y;
        let right_unnorm = forward.cross(world_up);
        let right = if right_unnorm.length_squared() > EPS {
            right_unnorm.normalize()
        } else {
            Vec3::X
        };
        let up_unnorm = right.cross(forward);
        let up = if up_unnorm.length_squared() > EPS {
            up_unnorm.normalize()
        } else {
            world_up
        };
        self.forward = forward;
        self.right = right;
        self.up = up;
    }
    fn look(&mut self, yaw_delta: f32, pitch_delta: f32) {
        let max_pitch = core::f32::consts::FRAC_PI_2.sub(0.01);
        let min_pitch = 0.0_f32.sub(max_pitch);
        self.yaw = self.yaw.add(yaw_delta);
        self.pitch = self.pitch.add(pitch_delta).clamp(min_pitch, max_pitch);
        self.rebuild_basis();
    }
    fn translate(&mut self, delta: Vec3) {
        self.position = self.position.add(delta);
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
        .map(|voxel| voxel.sigma_t[0].max(voxel.sigma_t[1]).max(voxel.sigma_t[2]))
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
    pub(crate) const fn camera_position(&self) -> [f32; 3] {
        [
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
        ]
    }
    pub(crate) const fn camera_forward(&self) -> [f32; 3] {
        [
            self.camera.forward.x,
            self.camera.forward.y,
            self.camera.forward.z,
        ]
    }
    pub(crate) fn apply_camera_input(
        &mut self,
        move_right: f32,
        move_forward: f32,
        move_up: f32,
        yaw_delta: f32,
        pitch_delta: f32,
    ) {
        const EPS: f32 = 1.0e-6;
        if yaw_delta != 0.0 || pitch_delta != 0.0 {
            self.camera.look(yaw_delta, pitch_delta);
        }
        let mut forward_flat = Vec3::new(self.camera.forward.x, 0.0, self.camera.forward.z);
        if forward_flat.length_squared() > EPS {
            forward_flat = forward_flat.normalize();
        } else {
            forward_flat = self.camera.forward;
        }
        let mut right_flat = Vec3::new(self.camera.right.x, 0.0, self.camera.right.z);
        if right_flat.length_squared() > EPS {
            right_flat = right_flat.normalize();
        } else {
            right_flat = self.camera.right;
        }
        let up = Vec3::Y;
        let delta = right_flat
            .mul(move_right)
            .add(forward_flat.mul(move_forward))
            .add(up.mul(move_up));
        if delta != Vec3::ZERO {
            self.camera.translate(delta);
        }
    }
}
