//! Real linked clock factory. Its strong symbol and Runtime archive bytes are
//! verified by the selected pack/link owner, not by a decoded declaration.
use runmat_execution::host::{
    native_clock_service_abi, NativeClockInputs, NativeClockInvocationIdentity,
    NativeDeploymentServiceKind, VerifiedNativeServiceIssuer,
};
use runmat_native_codegen::aot::LinkedNativeProduct;
use runmat_runtime::context::{RuntimeContext, RuntimeTimingService};
use std::{ffi::c_void, rc::Rc};

type ClockOwner = Result<Rc<dyn RuntimeTimingService>, runmat_runtime::RuntimeError>;

/// # Safety
/// Both pointers address live values from this exact linked Rust Runtime ABI.
/// The caller uniquely owns the returned ClockOwner allocation. Inputs and
/// behavior must already have passed selected program/request admission.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_clock_component_v1(
    runtime: *const RuntimeContext,
    inputs: *const Option<NativeClockInputs>,
) -> *mut c_void {
    let runtime = unsafe { &*runtime };
    let inputs = unsafe { &*inputs };
    let owner: ClockOwner = match inputs {
        Some(inputs) => runmat_runtime::context::DeterministicTimingService::new(
            inputs.clone(),
            runtime.cancellation(),
        )
        .map(|service| Rc::new(service) as Rc<dyn RuntimeTimingService>),
        // This factory is entered only during fresh standalone setup, before
        // timing injection. Preserve that actual host clock, including CPU
        // sampling and timezone policy, rather than inventing virtual inputs.
        None => runtime.service_ports().timing().map(Rc::clone),
    };
    Box::into_raw(Box::new(owner)).cast()
}

pub(super) fn admit(linked: &LinkedNativeProduct) -> Result<VerifiedNativeServiceIssuer, String> {
    let binding = linked.binding();
    let kind = NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Clock);
    let service = binding
        .services
        .iter()
        .find(|service| service.kind == kind)
        .ok_or("selected Clock implementation is absent")?;
    if service.abi != native_clock_service_abi()
        || service.implementation != binding.runtime_component
    {
        return Err("selected Clock ABI/component differs from the linked timing factory".into());
    }
    // SAFETY: the verified launcher authenticated this exact Runtime archive.
    // The direct strong factory reference below links its real implementation
    // in this same image; pack production verifies that target-code symbol.
    unsafe { VerifiedNativeServiceIssuer::from_verified_product(binding, kind) }
        .map_err(|error| error.to_string())
}

pub(super) fn install(
    runtime: RuntimeContext,
    issuer: VerifiedNativeServiceIssuer,
    identity: NativeClockInvocationIdentity,
    inputs: Option<NativeClockInputs>,
) -> Result<RuntimeContext, String> {
    if identity.inputs()
        != inputs
            .as_ref()
            .map(NativeClockInputs::fingerprint)
            .transpose()
            .map_err(|error| error.to_string())?
    {
        return Err("clock factory inputs differ from the exact admitted invocation".into());
    }
    // SAFETY: admission checked the exact factory ABI/component and policy;
    // the unique boxed result uses this image's single Runtime ABI.
    let owner = *unsafe {
        Box::from_raw(runmat_aot_clock_component_v1(&runtime, &inputs).cast::<ClockOwner>())
    };
    let ports = runtime
        .service_ports()
        .clone()
        .with_selected_timing(owner.map_err(|error| error.to_string())?, issuer, identity)
        .map_err(|error| error.to_string())?;
    Ok(runtime.with_service_ports(ports))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_factory_preserves_hosted_clock_and_constructs_explicit_virtual_clock() {
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        // Exercise the real factory ABI, without manufacturing a selected
        // issuer or claiming qualification from synthetic product metadata.
        let ambient_result = unsafe {
            Box::from_raw(runmat_aot_clock_component_v1(&runtime, &None).cast::<ClockOwner>())
        };
        let ambient = (*ambient_result).unwrap();
        assert!(Rc::ptr_eq(
            &ambient,
            runtime.service_ports().timing().unwrap()
        ));
        let inputs = Some(NativeClockInputs {
            schema_version: runmat_execution::host::NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
            initial_monotonic_nanos: 71,
            initial_wall_unix_seconds: 1_700_000_000,
            initial_wall_nanosecond: 0,
            read_step_nanos: 1_000,
            local_offset_seconds: 0,
            process_cpu: None,
        });
        let virtual_result = unsafe {
            Box::from_raw(runmat_aot_clock_component_v1(&runtime, &inputs).cast::<ClockOwner>())
        };
        let virtual_clock = (*virtual_result).unwrap();
        assert!(!Rc::ptr_eq(&virtual_clock, &ambient));
        assert_eq!(
            virtual_clock.monotonic_elapsed(),
            std::time::Duration::from_nanos(71)
        );
        assert_eq!(
            virtual_clock.monotonic_elapsed(),
            std::time::Duration::from_nanos(1_071)
        );
        assert!(virtual_clock.process_cpu_seconds().is_none());
        assert!(virtual_clock.command_thread_cpu_sampler().is_none());
        let weak = Rc::downgrade(&virtual_clock);
        let pending = virtual_clock.sleep(std::time::Duration::from_secs(1));
        drop(virtual_clock);
        assert!(
            weak.upgrade().is_none(),
            "a wait retains clock state, not its service or RuntimeContext"
        );
        drop(pending);
    }
}
