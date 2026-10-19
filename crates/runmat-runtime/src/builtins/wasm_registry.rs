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
        let manifest = crate::builtin::migration_inventory::migration_inventory()
            .snapshot
            .observed
            .registration_manifest;
        assert_eq!(
            manifest.digest,
            generated::REGISTRY_MANIFEST_DIGEST,
            "generated WASM registration rows differ from the live WASM registry"
        );
        assert_eq!(
            manifest.entries.len(),
            generated::REGISTRY_ENTRY_COUNT,
            "generated WASM registration entry count differs from the live WASM registry"
        );
        assert_eq!(
            manifest.counts.builtin,
            generated::REGISTRY_BUILTIN_COUNT,
            "generated WASM builtin count differs from the live WASM registry"
        );
        assert_eq!(
            manifest.counts.constant,
            generated::REGISTRY_CONSTANT_COUNT,
            "generated WASM constant count differs from the live WASM registry"
        );
        assert_eq!(
            manifest.counts.gpu_spec,
            generated::REGISTRY_GPU_SPEC_COUNT,
            "generated WASM GPU-spec count differs from the live WASM registry"
        );
        assert_eq!(
            manifest.counts.fusion_spec,
            generated::REGISTRY_FUSION_SPEC_COUNT,
            "generated WASM fusion-spec count differs from the live WASM registry"
        );
        info!(
            "runmat-runtime: registered {} wasm builtins",
            runmat_builtins::builtin_functions().len()
        );
        runmat_builtins::wasm_registry::mark_registered();
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn register_all() {}
