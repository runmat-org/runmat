//! Reference counting follows the provider's platform ownership boundary.
//! Browser owners are local; native owners retain thread-safe weak acquisition.

#[cfg(target_arch = "wasm32")]
pub(super) use std::rc::{Rc as Shared, Weak as WeakShared};
#[cfg(not(target_arch = "wasm32"))]
pub(super) use std::sync::{Arc as Shared, Weak as WeakShared};
