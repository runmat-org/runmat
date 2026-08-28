mod descriptor;
mod fallback;

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

pub use descriptor::{SharedMemoryDescriptor, SharedMemoryKind};
pub use fallback::FileBackedSharedMemory;
