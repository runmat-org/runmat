//! The generic source constraint for language-error wrappers. Use this bound
//! when forwarding an owned cause to RuntimeErrorBuilder, rather than repeating
//! a native-only Send + Sync constraint at each domain boundary.

/// An owned cause accepted by `RuntimeErrorBuilder::with_source`.
///
/// Native causes remain safe for worker transport. Browser causes may retain
/// worker-local resources; wrappers must keep the actual source, not its text.
#[cfg(not(target_arch = "wasm32"))]
pub trait RuntimeErrorSourceBound: std::error::Error + Send + Sync + 'static {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: std::error::Error + Send + Sync + 'static + ?Sized> RuntimeErrorSourceBound for T {}

#[cfg(target_arch = "wasm32")]
pub trait RuntimeErrorSourceBound: std::error::Error + 'static {}
#[cfg(target_arch = "wasm32")]
impl<T: std::error::Error + 'static + ?Sized> RuntimeErrorSourceBound for T {}
