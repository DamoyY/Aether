use std::{env, path::PathBuf, process::Command};
fn main() -> Result<(), Box<dyn core::error::Error>> {
    println!("cargo:rerun-if-changed=cuda/");
    let cuda_path = env::var("CUDA_PATH")?;
    let nvcc = format!("{cuda_path}/bin/nvcc");
    let ptx_output = PathBuf::from(env::var("OUT_DIR")?).join("path_trace.ptx");
    let ptx_output_str = ptx_output.to_str().ok_or("Invalid PTX output path")?;
    Command::new(&nvcc)
        .args([
            "-ptx",
            "-arch=sm_86",
            "-O3",
            "--use_fast_math",
            "-allow-unsupported-compiler",
            "-I",
            "cuda/include",
            "-o",
            ptx_output_str,
            "cuda/kernels/path_trace.cu",
        ])
        .status()?;
    println!("cargo:rustc-link-search={cuda_path}/lib/x64");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    Ok(())
}
