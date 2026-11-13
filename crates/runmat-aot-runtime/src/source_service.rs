//! Optional compiler-bearing source service in the same selected runtime
//! archive. Pure AOT builds do not compile or link Core.

use std::ffi::c_void;

use runmat_execution::host::VerifiedNativeServiceIssuer;
use runmat_execution::host::{native_source_evaluation_service_abi, NativeDeploymentServiceKind};
use runmat_native_codegen::aot::LinkedNativeProduct;
use runmat_runtime::context::RuntimeContext;

/// The generated selected launcher references this symbol only when the
/// program reaches SourceEvaluation. A pure runtime archive cannot satisfy
/// that link. The qualified linker verifies this archive's exact bytes before
/// producing the image; the symbol check below is a linked-code check, not an
/// attempt to hash a loaded static library at runtime.
/// Construct the real compiler-bearing owner from the already-admitted
/// selected process Runtime. This ABI is private to the qualified linked
/// image and is never called from a decoded manifest or user-supplied host.
///
/// # Safety
/// `runtime` must point to the live RuntimeContext created by this exact
/// linked AOT runtime. The caller takes ownership of the returned allocation.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_source_evaluation_component_v2(
    runtime: *const RuntimeContext,
) -> *mut c_void {
    let runtime = unsafe { &*runtime };
    Box::into_raw(Box::new(runmat_core::SelectedNativeSourceEvaluator::new(
        runtime,
    )))
    .cast()
}

pub(super) fn admit(
    linked: &LinkedNativeProduct,
    factory: *const c_void,
) -> Result<VerifiedNativeServiceIssuer, String> {
    let expected = runmat_aot_source_evaluation_component_v2 as *const () as *const c_void;
    if factory.is_null() || factory != expected {
        return Err("selected source-evaluation factory is absent from this linked image".into());
    }
    let binding = linked.binding();
    let service = binding
        .services
        .iter()
        .find(|service| service.kind == NativeDeploymentServiceKind::SourceEvaluation)
        .ok_or("selected source-evaluation implementation is absent")?;
    if service.abi != native_source_evaluation_service_abi()
        || service.implementation != binding.runtime_component
    {
        return Err(
            "linked source-evaluation ABI or component differs from the selected runtime".into(),
        );
    }
    // SAFETY: this function is called only from the generated linked entry
    // after the actual optional symbol matches this crate's implementation.
    // The qualified linker verified the runtime archive (also the service
    // component) bytes against the selected component digest.
    unsafe {
        VerifiedNativeServiceIssuer::from_verified_product(
            binding,
            NativeDeploymentServiceKind::SourceEvaluation,
        )
    }
    .map_err(|error| error.to_string())
}

pub(super) fn install(
    runtime: RuntimeContext,
    issuer: VerifiedNativeServiceIssuer,
) -> Result<(RuntimeContext, runmat_core::SelectedNativeSourceEvaluator), String> {
    // SAFETY: admit matched this exact linked symbol and the selected archive
    // bytes before the host reached this point. The returned Box is uniquely
    // owned by this invocation and uses this image's single Core/Runtime ABI.
    let owner = *unsafe {
        Box::from_raw(
            runmat_aot_source_evaluation_component_v2(&runtime)
                .cast::<runmat_core::SelectedNativeSourceEvaluator>(),
        )
    };
    let ports = runtime
        .service_ports()
        .clone()
        .with_base_workspace(owner.base_workspace_service())
        .with_selected_source_evaluation(owner.service(), issuer)
        .map_err(|error| error.to_string())?;
    Ok((runtime.with_service_ports(ports), owner))
}
