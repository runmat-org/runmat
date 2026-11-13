mod export;
#[cfg(not(target_arch = "wasm32"))]
mod filesystem;
mod import;
mod namespace;

pub use export::CacheExport;
#[cfg(not(target_arch = "wasm32"))]
pub use filesystem::FilesystemObjectStore;
pub use import::{copy_verified_bounded_with_admission, import_verified_object, CacheImport};
pub use namespace::CacheNamespace;

mod memory;
pub use memory::MemoryObjectStore;
