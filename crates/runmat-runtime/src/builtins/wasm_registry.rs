#[cfg(target_arch = "wasm32")]
use log::info;
#[cfg(target_arch = "wasm32")]
use std::sync::Once;

#[cfg(all(target_arch = "wasm32", not(runmat_generating_wasm_registry)))]
pub(crate) mod generated {
    include!("generated_wasm_registry.rs");
}

#[cfg(all(target_arch = "wasm32", runmat_generating_wasm_registry))]
pub(crate) mod generated {
    include!(env!("RUNMAT_WASM_REGISTRY_OUT"));
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
        runmat_builtins::wasm_registry::mark_registered();
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn register_all() {}
