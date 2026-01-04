extern crate alloc;

mod config;
mod cuda;
mod ffi;
mod render;

use alloc::sync::Arc;
use core::ops::{Div as _, Mul as _};

use anyhow::Result;
use config::Config;
use log::info;
use pixels::{Pixels, SurfaceTexture};
use render::Renderer;
use scene_box::SceneData;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    config: Config,
    scene_data: Option<SceneData>,
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    renderer: Option<Renderer>,
}

impl App {
    const fn new(config: Config, scene_data: SceneData) -> Self {
        Self {
            config,
            scene_data: Some(scene_data),
            window: None,
            pixels: None,
            renderer: None,
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {}

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title(&self.config.window.title)
            .with_inner_size(LogicalSize::new(
                self.config.render.width,
                self.config.render.height,
            ))
            .with_resizable(false);

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
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(pixels) = self.pixels.as_mut() else {
            return;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if pixels.resize_surface(size.width, size.height).is_err() {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                if renderer.sample_count() < renderer.target_samples() {
                    if let Err(err) = renderer.render_progressive() {
                        log::error!("Render error: {err}");
                        event_loop.exit();
                        return;
                    }

                    if let Ok(framebuffer) = renderer.get_framebuffer() {
                        pixels.frame_mut().copy_from_slice(&framebuffer);
                    }

                    let sample_count_u16 =
                        u16::try_from(renderer.sample_count()).unwrap_or(u16::MAX);
                    let target_samples_u16 =
                        u16::try_from(renderer.target_samples()).unwrap_or(u16::MAX);
                    let sample_count = f32::from(sample_count_u16);
                    let target_samples = f32::from(target_samples_u16);
                    let progress = sample_count.div(target_samples).mul(100.0);
                    window.set_title(&format!(
                        "{} - {:.1}% ({} samples)",
                        renderer.window_title(),
                        progress,
                        renderer.sample_count()
                    ));
                }

                if pixels.render().is_err() {
                    event_loop.exit();
                }
            }
            WindowEvent::ActivationTokenDone { .. }
            | WindowEvent::Moved(_)
            | WindowEvent::Destroyed
            | WindowEvent::DroppedFile(_)
            | WindowEvent::HoveredFile(_)
            | WindowEvent::HoveredFileCancelled
            | WindowEvent::Focused(_)
            | WindowEvent::KeyboardInput { .. }
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

    let event_loop = EventLoop::new()?;
    let mut app = App::new(config, scene_data);

    event_loop.run_app(&mut app)?;

    Ok(())
}
