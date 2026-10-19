//! Shared ownership of a WGPU buffer across provider and rendering consumers.
//!
//! Native WGPU buffers can cross threads; browser WGPU buffers remain local to
//! the browser execution lane. This pointer does not grant device or allocation
//! authority: callers still need the owning handle and an emission permit.

#[cfg(not(target_arch = "wasm32"))]
pub type WgpuBufferOwner = std::sync::Arc<wgpu::Buffer>;

#[cfg(target_arch = "wasm32")]
pub type WgpuBufferOwner = std::rc::Rc<wgpu::Buffer>;
