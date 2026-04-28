extern crate alloc;
mod config;
mod cuda;
mod ffi;
mod render;
use alloc::sync::Arc;
use anyhow::Result;
use config::Config;
use core::{
    ops::{Div as _, Mul as _, Sub as _},
    sync::atomic::{AtomicBool, Ordering},
};
use image::{ImageBuffer, Rgba};
use log::info;
use mimalloc::MiMalloc;
use pixels::{Pixels, SurfaceTexture};
use render::Renderer;
use scene_box::SceneData;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
const MOVE_STEPS: f32 = 20.0;
const LOOK_STEPS: f32 = 30.0;
struct App {
    config: Config,
    scene_data: Option<SceneData>,
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    renderer: Option<Renderer>,
    saved: bool,
    interrupted: Arc<AtomicBool>,
    move_step_right: f32,
    move_step_up: f32,
    move_step_forward: f32,
    look_step: f32,
}
impl App {
    fn new(config: Config, scene_data: SceneData, interrupted: Arc<AtomicBool>) -> Self {
        let dim_right = u16::try_from(scene_data.dimensions[0]).unwrap_or(u16::MAX);
        let dim_up = u16::try_from(scene_data.dimensions[1]).unwrap_or(u16::MAX);
        let dim_forward = u16::try_from(scene_data.dimensions[2]).unwrap_or(u16::MAX);
        let move_step_right = f32::from(dim_right)
            .mul(scene_data.voxel_size)
            .div(MOVE_STEPS);
        let move_step_up = f32::from(dim_up).mul(scene_data.voxel_size).div(MOVE_STEPS);
        let move_step_forward = f32::from(dim_forward)
            .mul(scene_data.voxel_size)
            .div(MOVE_STEPS);
        let look_step = core::f32::consts::TAU.div(LOOK_STEPS);
        Self {
            config,
            scene_data: Some(scene_data),
            window: None,
            pixels: None,
            renderer: None,
            saved: false,
            interrupted,
            move_step_right,
            move_step_up,
            move_step_forward,
            look_step,
        }
    }
    fn apply_camera_step(&mut self, code: KeyCode) -> bool {
        let mut move_right = 0.0_f32;
        let mut move_forward = 0.0_f32;
        let mut move_up = 0.0_f32;
        let mut yaw = 0.0_f32;
        let mut pitch = 0.0_f32;
        if code == KeyCode::KeyW {
            move_forward = self.move_step_forward;
        } else if code == KeyCode::KeyS {
            move_forward = 0.0_f32.sub(self.move_step_forward);
        } else if code == KeyCode::KeyD {
            move_right = self.move_step_right;
        } else if code == KeyCode::KeyA {
            move_right = 0.0_f32.sub(self.move_step_right);
        } else if code == KeyCode::Space {
            move_up = self.move_step_up;
        } else if code == KeyCode::ControlLeft || code == KeyCode::ControlRight {
            move_up = 0.0_f32.sub(self.move_step_up);
        } else if code == KeyCode::ArrowLeft {
            yaw = 0.0_f32.sub(self.look_step);
        } else if code == KeyCode::ArrowRight {
            yaw = self.look_step;
        } else if code == KeyCode::ArrowUp {
            pitch = self.look_step;
        } else if code == KeyCode::ArrowDown {
            pitch = 0.0_f32.sub(self.look_step);
        } else {
            return false;
        }
        let Some(renderer) = self.renderer.as_mut() else {
            return false;
        };
        renderer.apply_camera_input(move_right, move_forward, move_up, yaw, pitch);
        let pos = renderer.camera_position();
        let fwd = renderer.camera_forward();
        println!(
            "Camera: pos=({:.3}, {:.3}, {:.3}), forward=({:.3}, {:.3}, {:.3})",
            pos[0], pos[1], pos[2], fwd[0], fwd[1], fwd[2]
        );
        if let Err(err) = renderer.clear_accumulator() {
            log::error!("Failed to clear accumulator: {err}");
        }
        self.saved = false;
        true
    }
}
fn save_image(renderer: &Renderer, output_path: &str, width: u32, height: u32) {
    match renderer.get_framebuffer() {
        Ok(framebuffer) => {
            if let Some(img) = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, framebuffer) {
                match img.save(output_path) {
                    Ok(()) => info!("Saved output to {output_path}"),
                    Err(err) => log::error!("Failed to save output: {err}"),
                }
            } else {
                log::error!("Failed to create image buffer: framebuffer size mismatch");
            }
        }
        Err(err) => log::error!("Failed to get framebuffer: {err}"),
    }
}
fn handle_redraw_requested(
    saved: &mut bool,
    config: &Config,
    window: &Window,
    pixels: &mut Pixels<'static>,
    renderer: &mut Renderer,
    event_loop: &ActiveEventLoop,
) {
    if renderer.sample_count() < renderer.target_samples() {
        if let Err(err) = renderer.render_progressive() {
            log::error!("Render error: {err}");
            event_loop.exit();
            return;
        }
        if let Ok(framebuffer) = renderer.get_framebuffer() {
            pixels.frame_mut().copy_from_slice(&framebuffer);
        }
        let sample_count_u16 = u16::try_from(renderer.sample_count()).unwrap_or(u16::MAX);
        let target_samples_u16 = u16::try_from(renderer.target_samples()).unwrap_or(u16::MAX);
        let sample_count = f32::from(sample_count_u16);
        let target_samples = f32::from(target_samples_u16);
        let progress = sample_count.div(target_samples).mul(100.0);
        window.set_title(&format!(
            "{} - {:.1}% ({} samples)",
            renderer.window_title(),
            progress,
            renderer.sample_count()
        ));
    } else if !*saved {
        *saved = true;
        if let Some(output_path) = config.render.output.as_ref() {
            save_image(
                renderer,
                output_path,
                config.render.width,
                config.render.height,
            );
        }
        window.set_title(&format!(
            "{} - Done ({} samples)",
            renderer.window_title(),
            renderer.sample_count()
        ));
    }
    if pixels.render().is_err() {
        event_loop.exit();
    }
}
impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {}
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title(&self.config.window.title)
            .with_inner_size(PhysicalSize::new(
                self.config.render.width,
                self.config.render.height,
            ))
            .with_resizable(false)
            .with_visible(false);
        let window = match event_loop.create_window(window_attributes) {
            Ok(w) => Arc::new(w),
            Err(err) => {
                log::error!("Failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };
        let window_size = window.inner_size();
        let surface_texture =
            SurfaceTexture::new(window_size.width, window_size.height, Arc::clone(&window));
        let pixels = match Pixels::new(
            self.config.render.width,
            self.config.render.height,
            surface_texture,
        ) {
            Ok(px) => px,
            Err(err) => {
                log::error!("Failed to create pixels: {err}");
                event_loop.exit();
                return;
            }
        };
        let Some(scene_data) = self.scene_data.take() else {
            log::error!("Scene data already consumed");
            event_loop.exit();
            return;
        };
        let mut renderer = match Renderer::new(self.config.clone(), &scene_data) {
            Ok(rend) => rend,
            Err(err) => {
                log::error!("Failed to create renderer: {err}");
                event_loop.exit();
                return;
            }
        };
        if let Err(err) = renderer.clear_accumulator() {
            log::error!("Failed to clear accumulator: {err}");
            event_loop.exit();
            return;
        }
        info!("Renderer initialized");
        window.set_visible(true);
        self.window = Some(window);
        self.pixels = Some(pixels);
        self.renderer = Some(renderer);
    }
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {}
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        _event: DeviceEvent,
    ) {
    }
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {}
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    match key_event.state {
                        ElementState::Pressed => {
                            if code == KeyCode::Escape {
                                event_loop.exit();
                                return;
                            }
                            if key_event.repeat {
                                return;
                            }
                            self.apply_camera_step(code);
                        }
                        ElementState::Released => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                let Some(pixels) = self.pixels.as_mut() else {
                    return;
                };
                if pixels.resize_surface(size.width, size.height).is_err() {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                let saved = &mut self.saved;
                let config = &self.config;
                let Some(window) = self.window.as_ref() else {
                    return;
                };
                let Some(pixels) = self.pixels.as_mut() else {
                    return;
                };
                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                handle_redraw_requested(saved, config, window, pixels, renderer, event_loop);
            }
            WindowEvent::ActivationTokenDone { .. }
            | WindowEvent::Moved(_)
            | WindowEvent::Destroyed
            | WindowEvent::DroppedFile(_)
            | WindowEvent::HoveredFile(_)
            | WindowEvent::HoveredFileCancelled
            | WindowEvent::Focused(_)
            | WindowEvent::ModifiersChanged(_)
            | WindowEvent::Ime(_)
            | WindowEvent::CursorMoved { .. }
            | WindowEvent::CursorEntered { .. }
            | WindowEvent::CursorLeft { .. }
            | WindowEvent::MouseWheel { .. }
            | WindowEvent::MouseInput { .. }
            | WindowEvent::PinchGesture { .. }
            | WindowEvent::PanGesture { .. }
            | WindowEvent::DoubleTapGesture { .. }
            | WindowEvent::RotationGesture { .. }
            | WindowEvent::TouchpadPressure { .. }
            | WindowEvent::AxisMotion { .. }
            | WindowEvent::Touch(_)
            | WindowEvent::ScaleFactorChanged { .. }
            | WindowEvent::ThemeChanged(_)
            | WindowEvent::Occluded(_) => {}
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.interrupted.load(Ordering::Relaxed) && !self.saved {
            self.saved = true;
            info!("Interrupted, saving image...");
            if let (Some(renderer), Some(window)) = (self.renderer.as_ref(), self.window.as_ref()) {
                if let Some(output_path) = self.config.render.output.as_ref() {
                    save_image(
                        renderer,
                        output_path,
                        self.config.render.width,
                        self.config.render.height,
                    );
                }
                window.set_title(&format!(
                    "{} - Interrupted ({} samples)",
                    renderer.window_title(),
                    renderer.sample_count()
                ));
            }
            event_loop.exit();
            return;
        }
        event_loop.set_control_flow(ControlFlow::Poll);
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {}
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {}
}
fn main() -> Result<()> {
    env_logger::init();
    let config = Config::load("render/render.yaml")?;
    info!("Configuration loaded");
    let scene_data = scene_box::generate(&config.scene_path)?;
    info!("Scene generated: {:?}", scene_data.dimensions);
    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = Arc::clone(&interrupted);
    ctrlc::set_handler(move || {
        interrupted_clone.store(true, Ordering::Relaxed);
    })?;
    let event_loop = EventLoop::new()?;
    let mut app = App::new(config, scene_data, interrupted);
    event_loop.run_app(&mut app)?;
    Ok(())
}
