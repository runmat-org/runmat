//! Canonical tape bindings for the immutable resident route that actually ran.
//! Parameter identities come from the model's borrowed table view, never
//! from a second layer-property parser.

use super::super::*;
use model::execution::NetworkCaller;
use model::gpu::mixed::ResidentTrace;
use model::parameters::ParameterRole;
use model::plan::axes::AxisPlan;
use runmat_accelerate_api::{handle_precision, GpuTensorHandle, ProviderPrecision};
use std::collections::{HashMap, HashSet};

mod cast;

struct Parent {
    node: usize,
    shape: Vec<usize>,
    precision: ProviderPrecision,
}

/// Immutable tape bindings recorded before forward effects. A later gradient
/// request selects which parent publication casts require admission.
pub(in crate::builtins::deep_learning) struct ResidentTapeAdmission {
    input: Option<Parent>,
    parameters: HashMap<(usize, ParameterRole), Parent>,
}

pub(in crate::builtins::deep_learning::autodiff) struct ResidentTapeRecord {
    pub trace: ResidentTrace,
    input: Option<Parent>,
    parameters: HashMap<(usize, ParameterRole), Parent>,
    presented_axes: AxisPlan,
}

/// A request-specific, cast-only resource proof for the parent gradients that
/// this dlgradient call will actually publish or propagate further upstream.
pub(in crate::builtins::deep_learning::autodiff) struct ResidentReversePlan {
    required_parents: HashSet<usize>,
    input_cast: Option<cast::CastRequirement>,
    parameter_casts: HashMap<(usize, ParameterRole), cast::CastRequirement>,
    cast_budget: cast::CastBudget,
}

impl ResidentTapeRecord {
    pub(in crate::builtins::deep_learning::autodiff) fn presented_axes_shape(&self) -> &[usize] {
        self.presented_axes.shape()
    }

    pub(in crate::builtins::deep_learning::autodiff) fn parent_bindings(
        &self,
    ) -> impl Iterator<Item = (usize, &[usize], NumericDType, NumericDType)> + '_ {
        self.input
            .iter()
            .chain(self.parameters.values())
            .map(|parent| {
                let dtype = match parent.precision {
                    ProviderPrecision::F32 => NumericDType::F32,
                    ProviderPrecision::F64 => NumericDType::F64,
                };
                let source = match handle_precision(&self.trace.output) {
                    Some(ProviderPrecision::F32) => NumericDType::F32,
                    Some(ProviderPrecision::F64) => NumericDType::F64,
                    None => dtype,
                };
                (parent.node, parent.shape.as_slice(), dtype, source)
            })
    }
}

fn parent(value: &Value) -> BuiltinResult<Option<Parent>> {
    let Some(node) = dlarray_node_id(value) else {
        return Ok(None);
    };
    let payload = super::super::tape::node_value(node)?;
    let precision = match payload.numeric_dtype() {
        Some(NumericDType::F32) => ProviderPrecision::F32,
        Some(NumericDType::F64) => ProviderPrecision::F64,
        _ => {
            return Err(unsupported_error(
                "dlgradient",
                "dlgradient: resident traced parent requires single or double storage",
            ))
        }
    };
    Ok(Some(Parent {
        node,
        shape: payload.shape().to_vec(),
        precision,
    }))
}

fn resident_precision(handle: &GpuTensorHandle) -> BuiltinResult<ProviderPrecision> {
    match handle_precision(handle) {
        Some(precision) => Ok(precision),
        None => Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident network tape requires single or double storage",
        )),
    }
}

/// Pure structural preflight before uploads or provider work. This must not
/// require casts for future gradient targets that a caller never requests.
pub(in crate::builtins::deep_learning) fn admit_parents(
    network: &ObjectInstance,
    input: &Value,
    resident: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<ResidentTapeAdmission> {
    resident_precision(resident)?;
    crate::builtins::common::gpu_helpers::exact_provider_for_handle(resident).ok_or_else(|| {
        crate::builtins::common::gpu_helpers::ResidentComputeDecline::InvalidOwner
            .with_domain_error(deep_learning_error(
                function,
                "resident network owner retired",
            ))
    })?;
    let input_parent = parent(input)?;
    let mut parameters = HashMap::new();
    for row in model::parameters::view(network, function)? {
        if let Some(parent) = row.value.map(parent).transpose()?.flatten() {
            parameters.insert((row.key.layer_index, row.key.role), parent);
        }
    }
    if !model::state::view(network, function)?.is_empty() {
        return Err(unsupported_error(
            function,
            format!("{function}: resident recurrent State adjoints are not yet available"),
        ));
    }
    Ok(ResidentTapeAdmission {
        input: input_parent,
        parameters,
    })
}

pub(in crate::builtins::deep_learning) fn record(
    admission: ResidentTapeAdmission,
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
    if admission.input.is_none() && admission.parameters.is_empty() {
        return Ok(nodes);
    }
    let execution = Rc::new(ResidentTapeRecord {
        trace,
        input: admission.input,
        parameters: admission.parameters,
        presented_axes,
    });
    nodes.prediction = Some(super::super::tape::push_resident_node(
        output.clone(),
        NodeKind::ResidentNetwork { execution },
    )?);
    Ok(nodes)
}

pub(in crate::builtins::deep_learning::autodiff) fn plan_reverse(
    execution: &ResidentTapeRecord,
    required_nodes: &HashSet<usize>,
) -> BuiltinResult<ResidentReversePlan> {
    let function = "dlgradient";
    let source = resident_precision(&execution.trace.output)?;
    let provider =
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&execution.trace.output)
            .ok_or_else(|| {
                crate::builtins::common::gpu_helpers::ResidentComputeDecline::InvalidOwner
                    .with_domain_error(deep_learning_error(
                        function,
                        "dlgradient: resident owner retired",
                    ))
            })?;
    let mut cast_budget = cast::CastBudget::default();
    let mut required_parents = HashSet::new();
    let input_cast = if let Some(parent) = execution
        .input
        .as_ref()
        .filter(|parent| required_nodes.contains(&parent.node))
    {
        required_parents.insert(parent.node);
        let required = cast::admit(provider, source, parent.precision, &parent.shape, function)?;
        if let Some(required) = required {
            cast_budget.include(required, function)?;
        }
        required
    } else {
        None
    };
    let mut parameter_casts = HashMap::new();
    for (key, parent) in &execution.parameters {
        if !required_nodes.contains(&parent.node) {
            continue;
        }
        required_parents.insert(parent.node);
        if let Some(required) =
            cast::admit(provider, source, parent.precision, &parent.shape, function)?
        {
            cast_budget.include(required, function)?;
            parameter_casts.insert(*key, required);
        }
    }
    Ok(ResidentReversePlan {
        required_parents,
        input_cast,
        parameter_casts,
        cast_budget,
    })
}

pub(in crate::builtins::deep_learning::autodiff) async fn reverse(
    execution: &ResidentTapeRecord,
    plan: &ResidentReversePlan,
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
    let mut returned = HashSet::with_capacity(result.parameters.len());
    for (layer_index, role, _) in &result.parameters {
        let key = (*layer_index, *role);
        if !returned.insert(key) {
            return Err(deep_learning_error(
                "dlgradient",
                "resident reverse returned a parameter adjoint more than once",
            ));
        }
    }
    if execution
        .parameters
        .keys()
        .any(|key| !returned.contains(key))
    {
        return Err(deep_learning_error(
            "dlgradient",
            "resident reverse omitted a traced parameter adjoint",
        ));
    }
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(&result.input)
        .ok_or_else(|| {
        crate::builtins::common::gpu_helpers::ResidentComputeDecline::InvalidOwner
            .with_domain_error(deep_learning_error(
                "dlgradient",
                "dlgradient: resident owner retired",
            ))
    })?;
    let mut borrowed = Vec::with_capacity(result.parameters.len() + 2);
    borrowed.push(execution.trace.output.clone());
    borrowed.push(result.input.clone());
    borrowed.extend(
        result
            .parameters
            .iter()
            .map(|(_, _, gradient)| gradient.clone()),
    );
    let mut remaining_cast_budget = plan.cast_budget;
    let mut output = Vec::new();
    if let Some(parent) = execution
        .input
        .as_ref()
        .filter(|parent| plan.required_parents.contains(&parent.node))
    {
        output.push((
            parent.node,
            restore_parent(
                provider,
                result.input,
                parent,
                plan.input_cast,
                &borrowed,
                &mut remaining_cast_budget,
                runtime,
            )
            .await?,
        ));
    }
    for (layer_index, role, gradient) in result.parameters {
        if let Some(parent) = execution
            .parameters
            .get(&(layer_index, role))
            .filter(|parent| plan.required_parents.contains(&parent.node))
        {
            output.push((
                parent.node,
                restore_parent(
                    provider,
                    gradient,
                    parent,
                    plan.parameter_casts.get(&(layer_index, role)).copied(),
                    &borrowed,
                    &mut remaining_cast_budget,
                    runtime,
                )
                .await?,
            ));
        }
    }
    Ok(output)
}

async fn restore_parent(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    mut gradient: GpuTensorHandle,
    parent: &Parent,
    admitted_cast: Option<cast::CastRequirement>,
    borrowed: &[GpuTensorHandle],
    remaining_cast_budget: &mut cast::CastBudget,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<GpuTensorHandle> {
    super::super::gradient::resident::admit_cancellation(runtime)?;
    let precision = resident_precision(&gradient)?;
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
    if crate::builtins::common::gpu_helpers::exact_provider_for_handle(&gradient)
        .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: resident adjoint owner differs from admitted provider",
        ));
    }
    // The cast was admitted for the parent's shape. Restore that view before
    // rechecking physical cast demand, rather than querying a different shape
    // produced by an otherwise valid adjoint.
    if gradient.shape() != parent.shape {
        super::super::gradient::resident::admit_cancellation(runtime)?;
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
        if !unary_gpu_output_matches_shape(
            &restored,
            &gradient,
            provider,
            &parent.shape,
            UnaryGpuOutputContract {
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
                precision: Some(precision),
                integer: None,
                logical: false,
                alias: GpuOutputAliasPolicy::AllowInput,
            },
        ) {
            return Err(reject_gpu_output(
                restored,
                deep_learning_error(
                    "dlgradient",
                    "dlgradient: invalid restored parent descriptor",
                ),
            ));
        }
        gradient = restored;
        super::super::gradient::resident::admit_cancellation(runtime)?;
    }
    if precision != parent.precision {
        let requirement = admitted_cast.ok_or_else(|| {
            deep_learning_error(
                "dlgradient",
                "resident parent precision changed after tape admission",
            )
        })?;
        gradient = cast::execute(
            provider,
            gradient,
            parent.precision,
            borrowed,
            requirement,
            remaining_cast_budget,
            "dlgradient",
        )
        .await?;
    } else if admitted_cast.is_some() {
        return Err(deep_learning_error(
            "dlgradient",
            "resident parent precision differs from admitted cast route",
        ));
    }
    super::super::gradient::resident::admit_cancellation(runtime)?;
    Ok(gradient)
}
