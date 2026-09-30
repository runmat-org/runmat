//! Canonical tape bindings for the immutable resident route that actually ran.
//! Parameter identities come from the model's borrowed table view, never
//! from a second layer-property parser.

use super::super::*;
use model::execution::NetworkCaller;
use model::gpu::mixed::ResidentTrace;
use model::parameters::ParameterRole;
use model::plan::axes::AxisPlan;
use runmat_accelerate_api::{handle_precision, GpuTensorHandle, ProviderPrecision};
use std::collections::HashMap;

pub(super) struct Parent {
    pub node: usize,
    pub shape: Vec<usize>,
    pub dtype: NumericDType,
}

pub(in crate::builtins::deep_learning::autodiff) struct ResidentTapeRecord {
    pub trace: ResidentTrace,
    input: Option<Parent>,
    parameters: HashMap<(usize, ParameterRole), Parent>,
    presented_axes: AxisPlan,
}

impl ResidentTapeRecord {
    pub(in crate::builtins::deep_learning::autodiff) fn presented_axes_shape(&self) -> &[usize] {
        self.presented_axes.shape()
    }

    pub(in crate::builtins::deep_learning::autodiff) fn parent_bindings(
        &self,
    ) -> impl Iterator<Item = (usize, &[usize], NumericDType)> + '_ {
        self.input
            .iter()
            .chain(self.parameters.values())
            .map(|parent| (parent.node, parent.shape.as_slice(), parent.dtype))
    }
}

fn parent(value: &Value) -> BuiltinResult<Option<Parent>> {
    let Some(node) = dlarray_node_id(value) else {
        return Ok(None);
    };
    let payload = super::super::tape::node_value(node)?;
    let dtype = payload.numeric_dtype().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: traced parent has no floating class",
        )
    })?;
    Ok(Some(Parent {
        node,
        shape: payload.shape().to_vec(),
        dtype,
    }))
}

fn resident_dtype(handle: &GpuTensorHandle) -> BuiltinResult<NumericDType> {
    match handle_precision(handle) {
        Some(ProviderPrecision::F32) => Ok(NumericDType::F32),
        Some(ProviderPrecision::F64) => Ok(NumericDType::F64),
        None => Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident network tape requires single or double storage",
        )),
    }
}

/// Pure preflight, before uploads or provider work. A cross-precision parent
/// needs a general resident cast operation; this slice does not invent one.
pub(in crate::builtins::deep_learning) fn admit_parents(
    network: &ObjectInstance,
    input: &Value,
    resident: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<()> {
    let execution_dtype = resident_dtype(resident)?;
    let input_parent = parent(input)?;
    if input_parent
        .as_ref()
        .is_some_and(|parent| parent.dtype != execution_dtype)
    {
        return Err(unsupported_error(
            function,
            format!("{function}: resident tape input class differs from execution precision"),
        ));
    }
    for row in model::parameters::view(network, function)? {
        if let Some(parent) = row.value.map(parent).transpose()?.flatten() {
            if parent.dtype != execution_dtype {
                return Err(unsupported_error(
                    function,
                    format!(
                        "{function}: resident tape parameter class requires a provider precision cast"
                    ),
                ));
            }
        }
    }
    if !model::state::view(network, function)?.is_empty() {
        return Err(unsupported_error(
            function,
            format!("{function}: resident recurrent State adjoints are not yet available"),
        ));
    }
    Ok(())
}

pub(in crate::builtins::deep_learning) fn record(
    network: &ObjectInstance,
    input: &Value,
    output: &GpuTensorHandle,
    trace: ResidentTrace,
    presented_axes: AxisPlan,
    caller: NetworkCaller,
) -> BuiltinResult<super::executed::NetworkOutputNodes> {
    let function = caller.function();
    let mut nodes = super::executed::NetworkOutputNodes {
        prediction: None,
        state: Vec::new(),
    };
    if !tape_is_active() {
        return Ok(nodes);
    }
    if trace.output.view_identity() != output.view_identity() {
        return Err(deep_learning_error(
            function,
            format!("{function}: presented resident output differs from executed trace"),
        ));
    }
    let input = parent(input)?;
    let mut parameters = HashMap::new();
    for row in model::parameters::view(network, function)? {
        if let Some(binding) = row.value.map(parent).transpose()?.flatten() {
            parameters.insert((row.key.layer_index, row.key.role), binding);
        }
    }
    if input.is_none() && parameters.is_empty() {
        return Ok(nodes);
    }
    let execution = Rc::new(ResidentTapeRecord {
        trace,
        input,
        parameters,
        presented_axes,
    });
    nodes.prediction = Some(super::super::tape::push_resident_node(
        output.clone(),
        NodeKind::ResidentNetwork { execution },
    )?);
    Ok(nodes)
}

pub(in crate::builtins::deep_learning::autodiff) async fn reverse(
    execution: &ResidentTapeRecord,
    seed: GpuTensorHandle,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<(usize, GpuTensorHandle)>> {
    super::super::gradient::resident::admit_cancellation(runtime)?;
    if seed.shape() != execution.presented_axes.shape() {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident seed differs from presented network axes",
        ));
    }
    if handle_precision(&seed) != handle_precision(&execution.trace.output)
        || crate::builtins::common::gpu_helpers::exact_provider_for_binary_inputs(
            &seed,
            &execution.trace.output,
        )
        .is_err()
    {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident seed class or owner differs from executed output",
        ));
    }
    let cancel = || super::super::gradient::resident::admit_cancellation(runtime);
    let result = model::gpu::mixed::reverse_resident(&execution.trace, seed, &cancel).await?;
    super::super::gradient::resident::admit_cancellation(runtime)?;
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(&result.input)
        .ok_or_else(|| {
        deep_learning_error("dlgradient", "dlgradient: resident owner retired")
    })?;
    let mut output = Vec::new();
    if let Some(parent) = &execution.input {
        output.push((parent.node, restore_parent(provider, result.input, parent)?));
    }
    for (layer_index, role, gradient) in result.parameters {
        if let Some(parent) = execution.parameters.get(&(layer_index, role)) {
            output.push((parent.node, restore_parent(provider, gradient, parent)?));
        }
    }
    Ok(output)
}

fn restore_parent(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    gradient: GpuTensorHandle,
    parent: &Parent,
) -> BuiltinResult<GpuTensorHandle> {
    let dtype = resident_dtype(&gradient)?;
    if dtype != parent.dtype {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident adjoint class differs from its admitted parent",
        ));
    }
    let expected = parent
        .shape
        .iter()
        .try_fold(1usize, |count, &size| count.checked_mul(size));
    let actual = gradient
        .shape()
        .iter()
        .try_fold(1usize, |count, &size| count.checked_mul(size));
    if expected.is_none() || expected != actual {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident adjoint element count differs from its parent",
        ));
    }
    if gradient.shape() == parent.shape {
        return Ok(gradient);
    }
    let restored = provider
        .reshape(&gradient, &parent.shape)
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                deep_learning_error(
                    "dlgradient",
                    "dlgradient: provider failed to restore parent shape",
                ),
                error.into(),
            )
        })?;
    use crate::builtins::common::gpu_helpers::{
        reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy,
        UnaryGpuOutputContract,
    };
    if unary_gpu_output_matches_shape(
        &restored,
        &gradient,
        provider,
        &parent.shape,
        UnaryGpuOutputContract {
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision: handle_precision(&gradient),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::AllowInput,
        },
    ) {
        Ok(restored)
    } else {
        Err(reject_gpu_output(
            restored,
            deep_learning_error(
                "dlgradient",
                "dlgradient: invalid restored parent descriptor",
            ),
        ))
    }
}
