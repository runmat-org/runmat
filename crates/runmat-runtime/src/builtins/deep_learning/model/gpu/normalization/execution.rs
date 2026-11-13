use super::super::stage_failure::StageExecutionFailure;
use super::super::*;
use super::validation::{validate_output, validate_statistics};
use crate::builtins::common::gpu_helpers::{
    propagate_output_provenance, reject_gpu_output, reject_gpu_outputs,
};
use crate::builtins::deep_learning::model::plan::normalization::NormalizationStage;
use runmat_accelerate_api::{DeepLearningLayerNormCapability, DeepLearningLayerNormRequest};

pub(in crate::builtins::deep_learning) struct NormalizationRecord {
    pub layer_index: usize,
    pub scale: GpuTensorHandle,
    pub offset: GpuTensorHandle,
    pub statistics: GpuTensorHandle,
    pub request: DeepLearningLayerNormRequest,
}

pub(in crate::builtins::deep_learning) struct RecordedNormalization {
    pub output: GpuTensorHandle,
    pub record: NormalizationRecord,
}

enum ForwardMode {
    Ordinary,
    Record,
}

struct Execution {
    output: GpuTensorHandle,
    record: Option<NormalizationRecord>,
}

fn parameter(layer: &ObjectInstance, role: ParameterRole) -> Option<&GpuTensorHandle> {
    resident_parameter_handle(layer, role)
}

async fn prepare_parameter(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    layer: &ObjectInstance,
    role: ParameterRole,
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> Result<GpuTensorHandle, StageExecutionFailure> {
    if let Some(handle) = parameter(layer, role) {
        return Ok(handle.clone());
    }
    let host = parameters::prepare_host_parameter(layer, role, function).await?;
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    Ok(parameters::upload_parameter(
        provider, host, input, protected, function,
    )?)
}

pub(in crate::builtins::deep_learning) async fn forward(
    network: &ObjectInstance,
    input: &GpuTensorHandle,
    stage: &NormalizationStage,
    target_axes: &AxisPlan,
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> Result<Result<GpuTensorHandle, GpuCapabilityReason>, StageExecutionFailure> {
    Ok(execute(
        network,
        input,
        stage,
        target_axes,
        ForwardMode::Ordinary,
        cancel,
        function,
    )
    .await?
    .map(|execution| execution.output))
}

pub(in crate::builtins::deep_learning) async fn forward_recorded(
    network: &ObjectInstance,
    input: &GpuTensorHandle,
    stage: &NormalizationStage,
    target_axes: &AxisPlan,
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> Result<Result<RecordedNormalization, GpuCapabilityReason>, StageExecutionFailure> {
    match execute(
        network,
        input,
        stage,
        target_axes,
        ForwardMode::Record,
        cancel,
        function,
    )
    .await?
    {
        Ok(execution) => {
            let record = execution.record.ok_or_else(|| {
                deep_learning_error(function, "recorded normalization omitted statistics")
            })?;
            Ok(Ok(RecordedNormalization {
                output: execution.output,
                record,
            }))
        }
        Err(reason) => Ok(Err(reason)),
    }
}

async fn execute(
    network: &ObjectInstance,
    input: &GpuTensorHandle,
    stage: &NormalizationStage,
    target_axes: &AxisPlan,
    mode: ForwardMode,
    cancel: &dyn Fn() -> BuiltinResult<()>,
    function: &'static str,
) -> Result<Result<Execution, GpuCapabilityReason>, StageExecutionFailure> {
    // A later composed stage/presentation plan can pre-admit device permutes.
    // This route deliberately declines them before any provider effect.
    if target_axes != &stage.geometry.axes {
        return Ok(Err(GpuCapabilityReason::AxisLayout));
    }
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| {
            crate::builtins::common::gpu_helpers::ResidentComputeDecline::InvalidOwner
                .with_domain_error(unsupported_error(
                    function,
                    "no provider owns the GPU dlarray input",
                ))
        })?;
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        unreachable!("host plan admitted network Layers")
    };
    let Some(Value::Object(layer)) = layers.data().get(stage.layer_index) else {
        return Err(deep_learning_error(
            function,
            "admitted normalization stage no longer matches network Layers",
        )
        .into());
    };
    let parameters = match super::admit_parameters(layer, provider, input, function)? {
        Ok(parameters) => parameters,
        Err(reason) => return Ok(Err(reason)),
    };
    let scale = parameters.scale;
    let offset = parameters.offset;
    let request = stage.geometry.resident_request();
    let capability = provider
        .deep_learning_layer_norm_capability(input, scale, offset, &request)
        .map_err(|error| {
            provider_failure(function, "resident normalization admission", error.into())
        })?;
    match capability {
        DeepLearningLayerNormCapability::Ready => {}
        DeepLearningLayerNormCapability::UnsupportedOperation => {
            return Ok(Err(GpuCapabilityReason::ProviderOperation))
        }
        DeepLearningLayerNormCapability::ResourceLimit => {
            return Ok(Err(GpuCapabilityReason::ResourceLimit))
        }
    }
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    let scale = prepare_parameter(
        provider,
        layer,
        ParameterRole::Scale,
        input,
        &[input],
        cancel,
        function,
    )
    .await?;
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    let offset = prepare_parameter(
        provider,
        layer,
        ParameterRole::Offset,
        input,
        &[input, &scale],
        cancel,
        function,
    )
    .await?;
    cancel().map_err(StageExecutionFailure::Abandoned)?;
    let (output, statistics) = match mode {
        ForwardMode::Ordinary => (
            provider
                .deep_learning_layer_norm(input, &scale, &offset, &request)
                .await
                .map_err(|error| {
                    provider_failure(function, "resident layer normalization", error.into())
                })?,
            None,
        ),
        ForwardMode::Record => {
            let recorded = provider
                .deep_learning_layer_norm_recorded(input, &scale, &offset, &request)
                .await
                .map_err(|error| {
                    provider_failure(function, "recorded resident normalization", error.into())
                })?;
            (recorded.output, Some(recorded.statistics))
        }
    };
    if let Err(error) = cancel() {
        return Err(StageExecutionFailure::Abandoned(reject_gpu_outputs(
            std::iter::once(output).chain(statistics),
            error,
        )));
    }
    let mut output = match validate_output(
        output,
        input,
        &scale,
        &offset,
        provider,
        stage.geometry.axes.shape(),
        function,
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(StageExecutionFailure::Operational(reject_gpu_outputs(
                statistics, error,
            )));
        }
    };
    let statistics = if let Some(statistics) = statistics {
        match validate_statistics(
            statistics,
            input,
            &scale,
            &offset,
            &output,
            provider,
            stage.geometry.groups,
            function,
        ) {
            Ok(statistics) => Some(statistics),
            Err(error) => {
                return Err(StageExecutionFailure::Operational(reject_gpu_output(
                    output, error,
                )));
            }
        }
    } else {
        None
    };
    if let Err(error) = propagate_output_provenance(&mut output, [input, &scale, &offset]) {
        return Err(StageExecutionFailure::Operational(reject_gpu_outputs(
            std::iter::once(output).chain(statistics),
            error,
        )));
    }
    let record = statistics.map(|statistics| NormalizationRecord {
        layer_index: stage.layer_index,
        scale,
        offset,
        statistics,
        request,
    });
    Ok(Ok(Execution { output, record }))
}
