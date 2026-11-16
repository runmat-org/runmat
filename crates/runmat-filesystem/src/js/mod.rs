//! JavaScript filesystem callbacks shared by browser execution and tooling.
// Each browser worker owns a separate Wasm instance and callback table. Moving
// these JS values between Rust threads in a shared-memory module is unsupported.
#[cfg(target_feature = "atomics")]
compile_error!("The JavaScript filesystem adapter requires non-shared Wasm memory; install a separate provider in each browser worker");

mod bindings;
mod convert;
mod handle;
mod provider;

pub use provider::JsFsProvider;
