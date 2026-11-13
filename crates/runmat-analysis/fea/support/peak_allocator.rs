//! Diagnostic-binary-local Rust heap peak measurement; native allocations are not included.

#[path = "peak_allocator/counting.rs"]
mod counting;

#[path = "process_memory.rs"]
mod process_memory;

pub use counting::{begin_measurement, end_measurement};
pub use process_memory::process_peak_rss_bytes;
