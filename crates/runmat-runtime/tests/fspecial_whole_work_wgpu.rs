//! Required hardware witness for the optional resident fspecial route. The
//! service and builtin use the same runtime library, and no adapter absence is
//! interpreted as a passing test.
#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use futures::executor::block_on;
use runmat_accelerate::{backend::wgpu::provider::WgpuProvider, placement::PlacementSession};
use runmat_accelerate_api::{
    observation::{ProviderObservationCapability, ProviderObservationSupport},
    AccelDownloadFuture, AccelProvider, FspecialRequest, GpuTensorHandle, HostNumericDataOwned,
    HostTensorView, ProviderCapabilitySnapshot, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderPrecision, ThreadProviderGuard,
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
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session)
            .with_parallel(Rc::new(HostBudget(host_bytes))),
    );
    context.set_runmat_extensions_enabled(true);
    context
}

/// Keep WGPU's actual capability, resource and observation contracts while
/// forcing the one legal provider-hook Unsupported outcome. No synthetic GPU
/// submission is counted as selected-kernel work.
struct UnsupportedFspecial<'a>(&'a WgpuProvider);

impl AccelProvider for UnsupportedFspecial<'_> {
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unsupported fspecial fixture must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("unsupported fspecial fixture must not download")
    }
    fn precision(&self) -> ProviderPrecision {
        self.0.precision()
    }
    fn device_info(&self) -> String {
        self.0.device_info()
    }
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        self.0.capability_snapshot()
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.0.query_feasibility(query)
    }
    fn operation_observation_capability(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationCapability {
        self.0.operation_observation_capability(query)
    }
    fn begin_operation_observation(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationSupport {
        self.0.begin_operation_observation(query)
    }
    fn placement_resources(&self) -> runmat_execution::ProviderResourceSnapshot {
        self.0.placement_resources()
    }
    fn fspecial(&self, _: &FspecialRequest) -> anyhow::Result<GpuTensorHandle> {
        Err(runmat_accelerate_api::unsupported_provider_operation(
            "fixture: provider route became unavailable",
        ))
    }
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
    let Value::GpuTensor(output) = published.value() else {
        panic!("supported average form must publish resident output")
    };
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(output.device_id(), provider.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_precision(output),
        Some(provider.precision())
    );
    let numeric = block_on(provider.download_numeric(output)).expect("read resident kernel");
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

#[test]
#[ignore = "requires actual WGPU adapter; run explicitly in the image-filter hardware gate"]
fn provider_unsupported_host_fallback_is_one_completed_conditional_route() {
    let _restore = FspecialEnvRestore(std::env::var_os("RUNMAT_ACCEL_FSPECIAL_DEVICE"));
    std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", "1");
    let backend: &'static WgpuProvider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required"),
    ));
    let facade: &'static UnsupportedFspecial<'static> =
        Box::leak(Box::new(UnsupportedFspecial(backend)));
    let _owner = ThreadProviderGuard::set(Some(facade));
    let session = Rc::new(PlacementSession::new(Default::default()));
    let args = [Value::from("average"), Value::Num(2.0)];
    let output = block_on(call_builtin_async_in_context(
        &context(session.clone(), 32),
        "fspecial",
        &args,
    ))
    .expect("actual Unsupported hook permits host fallback");
    let Value::Tensor(output) = output.value() else {
        panic!("unsupported hook must publish the host double tensor")
    };
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(output.materialize_f64(), vec![0.25; 4]);
    let samples = session
        .profile_snapshot()
        .feedback
        .iter()
        .filter(|sample| sample.signature.operation == "fspecial")
        .map(|sample| (sample.samples, sample.failures))
        .collect::<Vec<_>>();
    assert_eq!(samples, vec![(1, 0)]);
}
