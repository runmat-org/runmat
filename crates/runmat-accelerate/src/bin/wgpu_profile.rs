//! Native filesystem-based WGPU profiling command.

#[cfg(not(target_arch = "wasm32"))]
#[path = "wgpu_profile/native.rs"]
mod native;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::run()
}

#[cfg(target_arch = "wasm32")]
fn main() -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "wgpu_profile is a native command-line profiler and cannot run on WASM",
    ))
}
