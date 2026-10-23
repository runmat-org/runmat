//! Exact linked constructor for native invocation-local services.
//! A decoded product and input never enter this factory without the linked
//! product admission and per-effect issuers checked by its caller.

use runmat_execution::host::{
    NativeDeploymentServiceKind, NativeLocalServiceInputs, NativeSelectedProductBinding,
    VerifiedNativeServiceIssuer,
};
use runmat_execution::Digest;
use runmat_runtime::context::RuntimeSelectedLocalServices;
use std::ffi::c_void;

type LocalOwner = Result<RuntimeSelectedLocalServices, runmat_runtime::RuntimeError>;

/// # Safety
/// All pointers are live values in this exact linked Rust Runtime ABI. The
/// caller uniquely owns the returned allocation and has already verified the
/// selected product, issuer inventory, and invocation input digest.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_local_component_v1(
    product: *const NativeSelectedProductBinding,
    invocation: *const Digest,
    issuers: *const Vec<VerifiedNativeServiceIssuer>,
    inputs: *const Option<NativeLocalServiceInputs>,
) -> *mut c_void {
    let product = unsafe { &*product };
    let invocation = unsafe { *invocation };
    let issuers = unsafe { &*issuers };
    let inputs = unsafe { &*inputs };
    let owner = RuntimeSelectedLocalServices::materialize(
        product,
        invocation,
        issuers.iter().cloned(),
        inputs.as_ref(),
    );
    Box::into_raw(Box::new(owner)).cast()
}

pub(super) fn admit(
    linked: &runmat_native_codegen::aot::LinkedNativeProduct,
    kind: NativeDeploymentServiceKind,
) -> Result<VerifiedNativeServiceIssuer, String> {
    let binding = linked.binding();
    let service = binding
        .services
        .iter()
        .find(|service| service.kind == kind)
        .ok_or("selected local service is absent")?;
    if service.abi != runmat_execution::host::native_local_service_abi()
        || service.implementation != binding.runtime_component
    {
        return Err("selected local ABI/component differs from linked factory".into());
    }
    // SAFETY: linked-image admission authenticated the runtime archive and
    // this factory is a strong symbol in that same retained archive.
    unsafe { VerifiedNativeServiceIssuer::from_verified_product(binding, kind) }
        .map_err(|error| error.to_string())
}

pub(super) fn install(
    runtime: runmat_runtime::context::RuntimeContext,
    product: &NativeSelectedProductBinding,
    invocation: Digest,
    issuers: Vec<VerifiedNativeServiceIssuer>,
    inputs: &Option<NativeLocalServiceInputs>,
) -> Result<runmat_runtime::context::RuntimeContext, String> {
    runmat_execution::host::validate_native_local_inputs(Some(product), inputs.as_ref())
        .map_err(|error| error.to_string())?;
    // SAFETY: admit joined every issuer to the exact linked product and pack
    // production verified this symbol's presence in the target runtime.
    let owner = *unsafe {
        Box::from_raw(
            runmat_aot_local_component_v1(product, &invocation, &issuers, inputs)
                .cast::<LocalOwner>(),
        )
    };
    let ports = runtime
        .service_ports()
        .clone()
        .with_selected_local_services(owner.map_err(|error| error.to_string())?);
    Ok(runtime.with_service_ports(ports))
}
