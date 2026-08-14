mod value_arena;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use value_arena::ScopedValueRoots;
pub use value_arena::ValueArena;
