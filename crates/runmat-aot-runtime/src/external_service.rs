//! Optional host-owned ports from the exact target-built selected archive.
//! Stock factories supply console, callback and immutable builtin owners.
//! A custom embedding supplies matching factories and live owners for its reached kinds.

use runmat_execution::host::{
    native_host_callback_service_abi, native_hosted_dynamic_dispatch_abi, native_ui_service_abi,
    NativeDeploymentProfile, NativeDeploymentServiceBehavior, NativeDeploymentServiceKind,
    NativeSelectedProductBinding, VerifiedNativeServiceIssuer,
};
use runmat_execution::Digest;
use runmat_native_codegen::aot::LinkedNativeProduct;
use runmat_runtime::context::{
    NativeLinkedCallbackPort, NativeLinkedDynamicPort, NativeLinkedHostPorts, RuntimeContext,
};
use std::ffi::c_void;

#[cfg(feature = "stock-hosted-factories")]
#[path = "external_service/stock.rs"]
mod stock;

type HostFactory = unsafe extern "C" fn(
    *const RuntimeContext,
    *const VerifiedNativeServiceIssuer,
    *const Digest,
) -> *mut c_void;

pub(super) fn admit(
    linked: &LinkedNativeProduct,
    kind: NativeDeploymentServiceKind,
    factory: *const c_void,
) -> Result<VerifiedNativeServiceIssuer, String> {
    if factory.is_null() {
        return Err(format!("linked {kind:?} factory is absent"));
    }
    let expected_abi = match kind {
        NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::UserInterface) => {
            native_ui_service_abi()
        }
        NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::HostCallback) => {
            native_host_callback_service_abi()
        }
        NativeDeploymentServiceKind::HostedDynamicDispatch => native_hosted_dynamic_dispatch_abi(),
        _ => return Err("linked host factory has a non-hosted service kind".into()),
    };
    let binding = linked.binding();
    let service = binding
        .services
        .iter()
        .find(|service| service.kind == kind)
        .ok_or_else(|| format!("selected {kind:?} implementation is absent"))?;
    if service.abi != expected_abi
        || service.implementation != binding.runtime_component
        || binding.profile != NativeDeploymentProfile::HostedNative
        || service.behavior != NativeDeploymentServiceBehavior::HostedAmbient
    {
        return Err(format!(
            "linked {kind:?} ABI/component/profile differs from selection"
        ));
    }
    // The launcher imported the exact kind-specific symbol and the target-pack
    // producer verified that symbol in the same retained archive bytes.
    unsafe { VerifiedNativeServiceIssuer::from_verified_product(binding, kind) }
        .map_err(|error| error.to_string())
}

fn verify_live(
    binding: &NativeSelectedProductBinding,
    invocation: Digest,
    issuer: &VerifiedNativeServiceIssuer,
    kind: NativeDeploymentServiceKind,
    factory: *const c_void,
) -> Result<(), String> {
    if invocation.bytes().iter().all(|byte| *byte == 0)
        || issuer.binding().kind != kind
        || issuer.product() != binding.fingerprint().map_err(|error| error.to_string())?
        || factory.is_null()
    {
        return Err(format!("live {kind:?} issuer/invocation/factory mismatch"));
    }
    Ok(())
}

/// The linked factory returns `Box<Result<NativeLinkedHostPorts, RuntimeError>>`.
/// That private Rust carrier preserves typed failure sources across the linked
/// boundary and never serializes a trait object or ambient host pointer.
pub(super) fn install_host(
    runtime: RuntimeContext,
    binding: &NativeSelectedProductBinding,
    invocation: Digest,
    issuer: &VerifiedNativeServiceIssuer,
    factory: *const c_void,
) -> Result<(RuntimeContext, NativeLinkedHostPorts), String> {
    let kind = issuer.binding().kind;
    if kind != NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::UserInterface) {
        return Err("linked host installer received another service kind".into());
    }
    verify_live(binding, invocation, issuer, kind, factory)?;
    // SAFETY: `factory` is the exact kind-specific symbol imported into the
    // generated launcher from the verified target-built archive, with the
    // private ABI identified by `native_*_service_abi`.
    let factory: HostFactory = unsafe { std::mem::transmute(factory) };
    let pointer = unsafe { factory(&runtime, issuer, &invocation) };
    if pointer.is_null() {
        return Err(format!("linked {kind:?} factory returned no host owner"));
    }
    // SAFETY: this private ABI returns one owned Box with this exact layout.
    let owner = unsafe {
        Box::from_raw(pointer.cast::<Result<NativeLinkedHostPorts, runmat_runtime::RuntimeError>>())
    };
    let owner = (*owner).map_err(|error| error.to_string())?;
    let ports = runtime
        .service_ports()
        .clone()
        .with_selected_linked_host(&runtime, issuer.clone(), invocation, &owner)
        .map_err(|error| error.to_string())?;
    Ok((runtime.with_service_ports(ports), owner))
}

/// A callable owner is distinct from console, interaction, and graphics.
/// The stock runtime supplies its existing invocation-local call router;
/// an embedding archive may instead supply a qualified router of its own.
pub(super) fn install_callback(
    runtime: RuntimeContext,
    binding: &NativeSelectedProductBinding,
    invocation: Digest,
    issuer: &VerifiedNativeServiceIssuer,
    factory: *const c_void,
) -> Result<RuntimeContext, String> {
    let kind = NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::HostCallback);
    verify_live(binding, invocation, issuer, kind, factory)?;
    // SAFETY: the target-built archive was checked for the distinct callback
    // symbol and ABI before the generated launcher imported this pointer.
    let factory: HostFactory = unsafe { std::mem::transmute(factory) };
    let pointer = unsafe { factory(&runtime, issuer, &invocation) };
    if pointer.is_null() {
        return Err("linked callback factory returned no callable owner".into());
    }
    let owner = unsafe {
        Box::from_raw(
            pointer.cast::<Result<NativeLinkedCallbackPort, runmat_runtime::RuntimeError>>(),
        )
    };
    let owner = (*owner).map_err(|error| error.to_string())?;
    let ports = runtime
        .service_ports()
        .clone()
        .with_selected_linked_callback(issuer.clone(), invocation, owner)
        .map_err(|error| error.to_string())?;
    Ok(runtime.with_service_ports(ports))
}

/// The linked factory returns `Box<Result<NativeLinkedDynamicPort, RuntimeError>>`.
pub(super) fn install_dynamic(
    runtime: RuntimeContext,
    binding: &NativeSelectedProductBinding,
    invocation: Digest,
    issuer: &VerifiedNativeServiceIssuer,
    factory: *const c_void,
) -> Result<RuntimeContext, String> {
    let kind = NativeDeploymentServiceKind::HostedDynamicDispatch;
    verify_live(binding, invocation, issuer, kind, factory)?;
    // SAFETY: same qualified private ABI as above, with a distinct kind symbol.
    let factory: HostFactory = unsafe { std::mem::transmute(factory) };
    let pointer = unsafe { factory(&runtime, issuer, &invocation) };
    if pointer.is_null() {
        return Err("linked dynamic binding factory returned no owner".into());
    }
    let owner = unsafe {
        Box::from_raw(
            pointer.cast::<Result<NativeLinkedDynamicPort, runmat_runtime::RuntimeError>>(),
        )
    };
    let owner = (*owner).map_err(|error| error.to_string())?;
    let ports = runtime
        .service_ports()
        .clone()
        .with_selected_linked_dynamic(issuer.clone(), invocation, owner)
        .map_err(|error| error.to_string())?;
    Ok(runtime.with_service_ports(ports))
}
