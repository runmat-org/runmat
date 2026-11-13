//! Consumes the selected compound through the original numerical/publication
//! owners. No hook redispatch, fresh placement decision, or ambient plan lookup.
use super::*;
use crate::context::preparation::AdmittedRealUnaryCompound;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, ProviderRealUnaryExecution, ProviderRealUnaryPublication,
};

pub(crate) trait BeforeHost:
    FnOnce(&GpuTensorHandle, RealUnaryClassification) -> BuiltinResult<()>
{
}
impl<T> BeforeHost for T where
    T: FnOnce(&GpuTensorHandle, RealUnaryClassification) -> BuiltinResult<()>
{
}

pub(crate) trait HostPublication:
    FnOnce(&'static dyn AccelProvider, &GpuTensorHandle, Value) -> BuiltinResult<Value>
{
}
impl<T> HostPublication for T where
    T: FnOnce(&'static dyn AccelProvider, &GpuTensorHandle, Value) -> BuiltinResult<Value>
{
}

pub(crate) trait DevicePublication:
    FnOnce(&'static dyn AccelProvider, &GpuTensorHandle, GpuTensorHandle) -> BuiltinResult<Value>
{
}
impl<T> DevicePublication for T where
    T: FnOnce(
        &'static dyn AccelProvider,
        &GpuTensorHandle,
        GpuTensorHandle,
    ) -> BuiltinResult<Value>
{
}

pub(crate) struct RealUnaryAdapter<B, H, R> {
    pub internal_error: &'static runmat_builtins::BuiltinErrorDescriptor,
    pub before_host: B,
    pub publish_host: H,
    pub publish_real: R,
}

pub(super) async fn execute<F, Fut, B, H, R>(
    runtime: Option<&RuntimeContext>,
    plan: crate::context::preparation::AdmittedRealUnaryCompound,
    value: Value,
    body: F,
    adapter: RealUnaryAdapter<B, H, R>,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
    B: BeforeHost,
    H: HostPublication,
    R: DevicePublication,
{
    let result = execute_inner(runtime, &plan, &value, body, adapter).await;
    result.map_err(|mut cause| {
        cause.gpu_gather_retry = crate::GpuGatherRetry::Never;
        cause
    })
}

async fn execute_inner<F, Fut, B, H, R>(
    runtime: Option<&RuntimeContext>,
    plan: &crate::context::preparation::AdmittedRealUnaryCompound,
    value: &Value,
    body: F,
    adapter: RealUnaryAdapter<B, H, R>,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
    B: BeforeHost,
    H: HostPublication,
    R: DevicePublication,
{
    let recorder = runmat_accelerate_api::observation::current_provider_observation();
    if recorder
        .as_ref()
        .is_some_and(|recorder| !recorder.expects_real_unary(plan.physical()))
    {
        return Err(AdmittedRealUnaryCompound::invalid_continuation());
    }
    plan.preflight_execution(runtime, &[value], recorder.is_some())?;
    let source = plan.physical().source();
    let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
        .ok_or_else(AdmittedRealUnaryCompound::invalid_continuation)?;
    let cancellation = runtime.map(RuntimeContext::cancellation);
    let evidence = plan
        .physical()
        .probe(cancellation.as_ref())
        .await
        .map_err(|cause| plan.provider_execution_error(runtime, adapter.internal_error, cause))?;
    plan.validate_current(runtime, &[value])?;
    let execution = plan
        .physical()
        .consume(evidence, cancellation.as_ref())
        .await
        .map_err(|cause| plan.provider_execution_error(runtime, adapter.internal_error, cause))?;
    plan.validate_current(runtime, &[value])?;
    match execution {
        ProviderRealUnaryExecution::Real(executed) => {
            let (output, evidence) = executed.into_parts();
            let output = (adapter.publish_real)(owner, source, output)?;
            let output = RootedValue::new(output).map_err(|cause| cause.into_runtime_error())?;
            plan.validate_current(runtime, &[value])?;
            if let Some(recorder) = recorder {
                recorder.confirm_real_unary_execution(&evidence);
            }
            Ok(output)
        }
        ProviderRealUnaryExecution::Continue(evidence) => {
            if !plan.physical().authenticates_continuation(&evidence) {
                return Err(AdmittedRealUnaryCompound::invalid_continuation());
            }
            (adapter.before_host)(source, evidence.classification())?;
            if let Some(recorder) = &recorder {
                recorder.begin_real_unary_continuation(&evidence);
            }
            let proof = plan.select_host_branch(runtime)?;
            let host = crate::builtins::common::gpu_helpers::gather_value_async(value).await?;
            plan.validate_current(runtime, &[value])?;
            let route = crate::builtins::math::host_route::canonical(
                runtime,
                crate::context::VmFusionHostInstruction::ElementwiseCall,
                std::iter::once(&host),
            );
            let numerical = super::classify_value(
                &host,
                plan.physical().predicate(),
                runtime,
                plan.admission().binding.builtin.name,
            )?;
            let output = body(host).await?;
            if let (Some(route), Some(runtime)) = (route, runtime) {
                route.confirm_numerical_host_branch(
                    runtime,
                    crate::context::VmFusionHostInstruction::ElementwiseCall,
                    numerical,
                );
            }
            if proof.as_ref().is_some_and(|proof| !proof.is_confirmed()) {
                return Err(AdmittedRealUnaryCompound::invalid_continuation());
            }
            plan.validate_current(runtime, &[value])?;
            let domain = match output.value() {
                Value::Num(_) | Value::Tensor(_) => runmat_types::NumericDomain::Real,
                Value::Complex(_, _) | Value::ComplexTensor(_) => {
                    runmat_types::NumericDomain::Complex
                }
                _ => return Err(AdmittedRealUnaryCompound::invalid_continuation()),
            };
            let publication = if plan.unary().publication().publishes_host(source, domain) {
                ProviderRealUnaryPublication::Host
            } else {
                ProviderRealUnaryPublication::ProviderUpload
            };
            if let Some(recorder) = &recorder {
                recorder.begin_real_unary_publication(&evidence, domain);
            }
            let output = output.try_map_owned(
                |output| (adapter.publish_host)(owner, source, output),
                |cause| cause.into_runtime_error(),
            )?;
            let published_host = !crate::dispatcher::value_contains_gpu(output.value());
            if published_host != (publication == ProviderRealUnaryPublication::Host) {
                return Err(AdmittedRealUnaryCompound::invalid_continuation());
            }
            plan.validate_current(runtime, &[value])?;
            if let Some(recorder) = recorder {
                recorder.confirm_real_unary_continuation(&evidence, publication);
            }
            Ok(output)
        }
    }
}
