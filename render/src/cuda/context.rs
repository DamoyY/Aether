extern crate alloc;

use alloc::sync::Arc;
use std::env;
use std::path::PathBuf;

use anyhow::Result;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream},
    nvrtc::Ptx,
};

pub(crate) struct Gpu {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) render_fn: CudaFunction,
    pub(crate) clear_fn: CudaFunction,
    _ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
}

impl Gpu {
    pub(crate) fn new() -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream(); 
        let ptx = PathBuf::from(env::var("OUT_DIR")?).join("path_trace.ptx");
        let module = ctx.load_module(Ptx::from_file(ptx))?;
        let render_fn = module.load_function("render_kernel")?;
        let clear_fn = module.load_function("clear_accumulator")?;

        Ok(Self {
            stream,
            render_fn,
            clear_fn,
            _ctx: ctx,
            _module: module,
        })
    }
}
