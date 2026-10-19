//! Provider-state references follow the backend execution lane. Device
//! resources remain separately owned; only state that can retain GPU handles
//! crosses this captured allocation-owner boundary.

#[cfg(not(target_arch = "wasm32"))]
pub(super) type OwnerShared<T> = std::sync::Arc<T>;
#[cfg(target_arch = "wasm32")]
pub(super) type OwnerShared<T> = std::rc::Rc<T>;

#[cfg(not(target_arch = "wasm32"))]
pub(super) type OwnerWeak<T> = std::sync::Weak<T>;
#[cfg(target_arch = "wasm32")]
pub(super) type OwnerWeak<T> = std::rc::Weak<T>;
