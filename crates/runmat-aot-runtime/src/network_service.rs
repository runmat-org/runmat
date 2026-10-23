//! Linked native HTTP/TCP owner for a selected Network effect.
//! Neither transport starts I/O at construction. The current invocation's
//! Tokio reactor is retained by both real production services.

use runmat_execution::host::{
    native_network_service_abi, NativeDeploymentProfile, NativeDeploymentServiceBehavior,
    NativeDeploymentServiceKind, NativeSelectedProductBinding, VerifiedNativeServiceIssuer,
};
use runmat_execution::Digest;
use runmat_native_codegen::aot::LinkedNativeProduct;
use runmat_runtime::context::{RuntimeContext, RuntimeHttpService, RuntimeTcpService};
use std::{ffi::c_void, rc::Rc};

struct NativeNetworkPorts {
    http: Rc<dyn RuntimeHttpService>,
    tcp: Rc<dyn RuntimeTcpService>,
}

/// # Safety
/// The pointer is a live Handle from this exact linked AOT runtime, retained
/// through the selected invocation. The returned box has one caller owner.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_network_component_v1(
    handle: *const tokio::runtime::Handle,
) -> *mut c_void {
    let handle = unsafe { &*handle };
    let ports = NativeNetworkPorts {
        http: Rc::new(runmat_runtime::http::NativeHttpService::new(handle.clone())),
        tcp: Rc::new(runmat_runtime::tcp::NativeTcpService::new(handle.clone())),
    };
    Box::into_raw(Box::new(ports)).cast()
}

pub(super) fn admit(
    linked: &LinkedNativeProduct,
    factory: *const c_void,
) -> Result<VerifiedNativeServiceIssuer, String> {
    let expected = runmat_aot_network_component_v1 as *const () as *const c_void;
    if factory.is_null() || factory != expected {
        return Err("selected Network factory is absent from this linked image".into());
    }
    let binding = linked.binding();
    let kind = NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Network);
    let service = binding
        .services
        .iter()
        .find(|service| service.kind == kind)
        .ok_or("selected Network implementation is absent")?;
    if service.abi != native_network_service_abi()
        || service.implementation != binding.runtime_component
        || binding.profile != NativeDeploymentProfile::HostedNative
        || service.behavior != NativeDeploymentServiceBehavior::HostedAmbient
    {
        return Err("selected Network ABI/component differs from linked factory".into());
    }
    // SAFETY: the linked image authenticated the archive and the exact
    // generated-launcher factory pointer above.
    unsafe { VerifiedNativeServiceIssuer::from_verified_product(binding, kind) }
        .map_err(|error| error.to_string())
}

pub(super) fn install(
    runtime: RuntimeContext,
    binding: &NativeSelectedProductBinding,
    invocation: Digest,
    issuer: &VerifiedNativeServiceIssuer,
    factory: *const c_void,
    handle: &tokio::runtime::Handle,
) -> Result<RuntimeContext, String> {
    if invocation.bytes().iter().all(|byte| *byte == 0)
        || issuer.binding().kind
            != NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Network)
        || issuer.product() != binding.fingerprint().map_err(|error| error.to_string())?
        || factory != runmat_aot_network_component_v1 as *const () as *const c_void
    {
        return Err("live Network issuer/invocation/factory mismatch".into());
    }
    // SAFETY: admit checked this exact linked factory. The reactor is owned by
    // the same AOT invocation through execution and final service teardown.
    let owner = *unsafe {
        Box::from_raw(
            runmat_aot_network_component_v1(handle as *const _).cast::<NativeNetworkPorts>(),
        )
    };
    let ports = runtime
        .service_ports()
        .clone()
        .with_selected_network(issuer.clone(), invocation, owner.http, owner.tcp)
        .map_err(|error| error.to_string())?;
    Ok(runtime.with_service_ports(ports))
}
