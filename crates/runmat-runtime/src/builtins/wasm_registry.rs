#[cfg(target_arch = "wasm32")]
use log::info;
#[cfg(target_arch = "wasm32")]
use std::sync::Once;

#[cfg(target_arch = "wasm32")]
pub(crate) mod generated {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/runmat_wasm_registry.rs"
    ));
}

#[cfg(target_arch = "wasm32")]
static WASM_REGISTRY_ONCE: Once = Once::new();

#[cfg(target_arch = "wasm32")]
pub fn register_all() {
    WASM_REGISTRY_ONCE.call_once(|| {
        info!("runmat-runtime: executing wasm builtin registry");
        generated::register_all();
        info!(
            "runmat-runtime: registered {} wasm builtins",
            runmat_builtins::builtin_functions().len()
        );
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn register_all() {}
