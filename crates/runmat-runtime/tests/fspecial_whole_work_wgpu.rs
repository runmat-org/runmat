//! Required hardware witness for the optional resident fspecial route. The
//! service and builtin use the same runtime library, and no adapter absence is
//! interpreted as a passing test.
#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use futures::executor::block_on;
use runmat_accelerate::{backend::wgpu::provider::WgpuProvider, placement::PlacementSession};
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, ProviderPrecision, ThreadProviderGuard,
};
use runmat_runtime::{
    call_builtin_async_in_context,
    context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    },
};
use runmat_value::Value;
use std::rc::Rc;

struct HostBudget(u64);

impl RuntimeParallelService for HostBudget {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(self.0),
            epoch: 0,
        }
    }
}

fn context(session: Rc<PlacementSession>, host_bytes: u64) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session)
            .with_parallel(Rc::new(HostBudget(host_bytes))),
    )
}

struct FspecialEnvRestore(Option<std::ffi::OsString>);
impl Drop for FspecialEnvRestore {
    fn drop(&mut self) {
        match &self.0 {
            Some(value) => std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", value),
            None => std::env::remove_var("RUNMAT_ACCEL_FSPECIAL_DEVICE"),
        }
    }
}

#[test]
#[ignore = "requires actual WGPU adapter; run explicitly in F32 and F64 image-filter gates"]
fn finite_host_workspace_and_completed_fspecial_publication() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let _restore = FspecialEnvRestore(std::env::var_os("RUNMAT_ACCEL_FSPECIAL_DEVICE"));
    std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", "1");
    let provider: &'static WgpuProvider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required"),
    ));
    let _owner = ThreadProviderGuard::set(Some(provider));
    match std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref() {
        Ok("f32") => assert_eq!(provider.precision(), ProviderPrecision::F32),
        Ok("f64") => assert_eq!(provider.precision(), ProviderPrecision::F64),
        _ => panic!("set RUNMAT_WGPU_FORCE_PRECISION=f32 or f64 for this hardware gate"),
    }
    let args = [Value::from("average"), Value::Num(2.0)];
    let session = Rc::new(PlacementSession::new(Default::default()));
    let published = block_on(call_builtin_async_in_context(
        &context(session.clone(), 32),
        "fspecial",
        &args,
    ))
    .expect("exact 32-byte host workspace must be admitted");
    let Value::GpuTensor(output) = published else {
        panic!("supported average form must publish resident output")
    };
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(output.device_id(), provider.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(provider.precision())
    );
    let numeric = block_on(provider.download_numeric(&output)).expect("read resident kernel");
    match numeric.data {
        HostNumericDataOwned::F32(values) => assert_eq!(values, vec![0.25_f32; 4]),
        HostNumericDataOwned::F64(values) => assert_eq!(values, vec![0.25_f64; 4]),
        other => panic!("unexpected resident kernel payload: {other:?}"),
    }
    let completed = (0..1000).any(|_| {
        if session.profile_snapshot().feedback.iter().any(|sample| {
            sample.signature.operation == "fspecial" && sample.samples == 1 && sample.failures == 0
        }) {
            true
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
            false
        }
    });
    assert!(
        completed,
        "whole-call feedback must await the WGPU completion fence"
    );

    let before = provider.placement_resources().live_bytes;
    let telemetry = provider.telemetry_snapshot();
    let rejected = block_on(call_builtin_async_in_context(
        &context(Rc::new(PlacementSession::new(Default::default())), 31),
        "fspecial",
        &args,
    ))
    .expect_err("one byte below the host tensor bound must reject before provider work");
    assert!(rejected
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
    assert_eq!(provider.placement_resources().live_bytes, before);
    let after = provider.telemetry_snapshot();
    assert_eq!(after.upload_bytes, telemetry.upload_bytes);
    assert_eq!(after.download_bytes, telemetry.download_bytes);
    output
        .allocation_lease()
        .validate_live()
        .expect("published sibling must survive rejection");
}
