use std::{env, path::PathBuf, process::Command};
fn main() -> Result<(), Box<dyn core::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let cuda_src = manifest_dir.join("src/scenes/atom/cuda/kernels/atom_density.cu");
    let cuda_src_str = cuda_src.to_str().ok_or("Invalid CUDA source path")?;
    println!("cargo:rerun-if-changed={cuda_src_str}");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let cuda_path = env::var("CUDA_PATH")?;
    let nvcc = format!("{cuda_path}/bin/nvcc");
    let ptx_output = out_dir.join("atom_density.ptx");
    let ptx_output_str = ptx_output.to_str().ok_or("Invalid PTX output path")?;
    let output = Command::new(&nvcc)
        .args([
            "-ptx",
            "-arch=sm_86",
            "-O3",
            "--use_fast_math",
            "-allow-unsupported-compiler",
            "-o",
            ptx_output_str,
            cuda_src_str,
        ])
        .output()?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("CUDA compilation failed:\nstdout: {stdout}\nstderr: {stderr}").into());
    }
    println!("cargo:rustc-link-search={cuda_path}/lib/x64");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    Ok(())
}
