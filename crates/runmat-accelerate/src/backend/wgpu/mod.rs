pub mod autotune;
pub mod bindings;
pub mod cache;
pub mod compat;
pub mod config;
pub mod debug;
pub mod dispatch;
pub mod metrics;
pub mod params;
pub mod pipelines;
pub mod provider;
pub mod residency;
pub mod resources;
pub mod shaders;
pub(crate) mod shader_compilation;
pub mod types;
// Browser startup has no process-local filesystem cache to replay.
#[cfg(not(target_arch = "wasm32"))]
pub mod warmup;

#[cfg(test)]
mod tests;
