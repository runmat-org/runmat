//! Hosted-native console, callable and builtin owners for the stock linked runtime.
//! Interaction and graphics are deliberately absent; their operation-local
//! ports return unavailable instead of misrepresenting a console as a GUI.

use super::*;
use runmat_runtime::context::{
    PortableForwardingConsoleHost, RuntimeCallRouter, RuntimeHostService,
};
use std::rc::Rc;

fn invalid_stock_factory() -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error("stock linked factory received a mismatched invocation")
        .with_identifier("RunMat:NativeSelected:ServiceMismatch")
        .build()
}

/// Stock console-only UI. An embedding archive can replace this symbol by
/// building without `stock-hosted-factories` and returning an interactive
/// `NativeLinkedHostPorts` with real presentation and callback owners.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_ui_component_v2(
    runtime: *const RuntimeContext,
    issuer: *const VerifiedNativeServiceIssuer,
    invocation: *const Digest,
) -> *mut c_void {
    let result = (|| {
        let (runtime, issuer, invocation) =
            unsafe { (runtime.as_ref(), issuer.as_ref(), invocation.as_ref()) };
        let (Some(runtime), Some(issuer), Some(invocation)) = (runtime, issuer, invocation) else {
            return Err(invalid_stock_factory());
        };
        if issuer.binding().kind
            != NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::UserInterface)
            || invocation.bytes().iter().all(|byte| *byte == 0)
        {
            return Err(invalid_stock_factory());
        }
        let _ = runtime;
        let host: Rc<dyn RuntimeHostService> = Rc::new(PortableForwardingConsoleHost);
        Ok(NativeLinkedHostPorts::console(host))
    })();
    Box::into_raw(Box::new(result)).cast::<c_void>()
}

/// Stock callback routing is the existing invocation-local call router, not
/// a graphical callback queue or an ambient dynamic-binding grant.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_host_callback_component_v2(
    runtime: *const RuntimeContext,
    issuer: *const VerifiedNativeServiceIssuer,
    invocation: *const Digest,
) -> *mut c_void {
    let result = (|| {
        let (runtime, issuer, invocation) =
            unsafe { (runtime.as_ref(), issuer.as_ref(), invocation.as_ref()) };
        let (Some(runtime), Some(issuer), Some(invocation)) = (runtime, issuer, invocation) else {
            return Err(invalid_stock_factory());
        };
        if issuer.binding().kind
            != NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::HostCallback)
            || invocation.bytes().iter().all(|byte| *byte == 0)
        {
            return Err(invalid_stock_factory());
        }
        let _ = runtime;
        Ok(NativeLinkedCallbackPort::new(Rc::new(RuntimeCallRouter)))
    })();
    Box::into_raw(Box::new(result)).cast::<c_void>()
}

/// Stock hosted dispatch owns an immutable binding generation. Preserve an
/// already scoped table rather than broadening an exact compiled selection.
///
/// # Safety
/// Each non-null argument must point to its declared live Rust type for this
/// call. The returned pointer owns a boxed `Result<NativeLinkedDynamicPort,
/// RuntimeError>` and must be consumed exactly once through the matching
/// linked-runtime ABI.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_hosted_dynamic_component_v1(
    runtime: *const RuntimeContext,
    issuer: *const VerifiedNativeServiceIssuer,
    invocation: *const Digest,
) -> *mut c_void {
    let result = (|| {
        let (runtime, issuer, invocation) =
            unsafe { (runtime.as_ref(), issuer.as_ref(), invocation.as_ref()) };
        let (Some(runtime), Some(issuer), Some(invocation)) = (runtime, issuer, invocation) else {
            return Err(invalid_stock_factory());
        };
        if issuer.binding().kind != NativeDeploymentServiceKind::HostedDynamicDispatch
            || invocation.bytes().iter().all(|byte| *byte == 0)
        {
            return Err(invalid_stock_factory());
        }
        let service = match runtime.service_ports().builtin() {
            Some(service) => service.clone(),
            None => crate::builtin::hosted_bindings().map_err(|reason| {
                runmat_runtime::build_runtime_error(reason)
                    .with_identifier("RunMat:NativeSelected:BindingInventory")
                    .build()
            })?,
        };
        Ok(NativeLinkedDynamicPort::new(service))
    })();
    Box::into_raw(Box::new(result)).cast::<c_void>()
}
