use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn compile_cuda(nvcc: &str, src: &Path, out: &Path) -> Result<(), Box<dyn core::error::Error>> {
    let src_str = src.to_str().ok_or("Invalid CUDA source path")?;
    let out_str = out.to_str().ok_or("Invalid PTX output path")?;
    println!("cargo:rerun-if-changed={src_str}");
    let output = Command::new(nvcc)
        .args([
            "-ptx",
            "-arch=sm_86",
            "-O3",
            "--use_fast_math",
            "-allow-unsupported-compiler",
            "-o",
            out_str,
            src_str,
        ])
        .output()?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "CUDA compilation failed for {src_str}:\nstdout: {stdout}\nstderr: {stderr}"
        )
        .into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let cuda_path = env::var("CUDA_PATH")?;
    let nvcc = format!("{cuda_path}/bin/nvcc");
    let kernels_dir = manifest_dir.join("src/scenes/atom/cuda/kernels");
    compile_cuda(
        &nvcc,
        &kernels_dir.join("density.cu"),
        &out_dir.join("density.ptx"),
    )?;
    compile_cuda(
        &nvcc,
        &kernels_dir.join("postprocess.cu"),
        &out_dir.join("postprocess.ptx"),
    )?;
    println!("cargo:rustc-link-search={cuda_path}/lib/x64");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    Ok(())
}
