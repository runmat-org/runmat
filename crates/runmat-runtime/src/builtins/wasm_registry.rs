#[cfg(target_arch = "wasm32")]
mod generated {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/runmat_wasm_registry.rs"
    ));
}

#[cfg(target_arch = "wasm32")]
pub fn register_all() {
    generated::register_all();
}

#[cfg(not(target_arch = "wasm32"))]
pub fn register_all() {}

