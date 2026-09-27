//! One platform threading contract for provider and allocation authorities.
/// Native authorities are shareable across execution threads. Browser workers
/// retain their own authorities; JS-backed payloads gain no unsafe Send/Sync.
#[cfg(not(target_arch = "wasm32"))]
pub trait AcceleratorThreadBound: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> AcceleratorThreadBound for T {}

#[cfg(target_arch = "wasm32")]
pub trait AcceleratorThreadBound {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> AcceleratorThreadBound for T {}
