//! Request-bound facts for the exact selected image source transcript.
//!
//! This uses the existing fusion semantic resolver. It cannot mint a provider
//! arithmetic proof: full-stage numerical equivalence, effects, and resources
//! remain separate admissions.

use runmat_accelerate_api::{
    GpuTensorHandle, ImageNormalizeDescriptor, ProviderImageMomentProgram,
    ProviderImageOutputOperand, ProviderImageOutputProgram, ProviderImageStageWorkload,
    ProviderPrecision,
};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::FusionPowerArithmeticReceipt;
use runmat_types::ValueFact;
use runmat_value::{NumericDType, Tensor, Value};
use std::sync::Arc;

use crate::fusion::{FusionGroupPlan, FusionOp, FusionPattern, ImageNormalizePattern, ImageScalar};
use crate::graph::{AccelGraph, AccelNodeLabel, NodeId, ValueId};
use crate::placement::PlacementCorrelationId;

use super::{ImageMeanAndPowerSource, ImageSourceLeaf, ImageSourceTranscript};

#[path = "source_invocation/moment_bindings.rs"]
mod moment_bindings;
#[path = "source_invocation/binary_operations.rs"]
mod binary_operations;
#[path = "source_invocation/coverage.rs"]
mod coverage;
#[path = "source_invocation/real_output_operations.rs"]
mod real_output_operations;
#[path = "source_invocation/power_program.rs"]
mod power_program;
#[path = "source_invocation/publications.rs"]
mod publications;
pub use publications::ImageSourcePublication;
#[path = "source_invocation/sigma_raw.rs"]
mod sigma_raw;
#[path = "source_invocation/sigma_operations.rs"]
mod sigma_operations;
pub(crate) use binary_operations::{ImageSelectedBinaryCall, ImageSelectedBinaryCalls};
pub(crate) use moment_bindings::{ImageSelectedMomentBindings, ImageSelectedMomentConversion};
pub(crate) use real_output_operations::{ImageSelectedRealCall, ImageSelectedRealCalls};
pub(crate) use sigma_operations::ImageSelectedSigmaOperations;
pub(crate) use sigma_raw::ImageSelectedSigmaRaw;

#[derive(Debug, Clone)]
struct BoundImageStage {
    input: BoundImageInput,
    desc: ImageNormalizeDescriptor,
    source: ImageMeanAndPowerSource,
    sigma_control: Option<sigma_operations::BoundSigmaAddControl>,
    output_binary: Vec<super::source_output_hooks::SelectedBinaryStep>,
    output_real: Vec<super::source_output_hooks::SelectedRealStep>,
}

#[derive(Debug, Clone)]
enum BoundImageInput {
    Resident(GpuTensorHandle),
    HostSingle(Tensor),
}

impl BoundImageInput {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Resident(handle) => handle.shape(),
            Self::HostSingle(tensor) => tensor.shape(),
        }
    }
}

/// Source-bound arithmetic for one selected power producer. This does not
/// authorize the legacy F32 staged hook to execute mixed arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSelectedPower {
    pub output: ValueId,
    pub arithmetic: FusionPowerArithmeticReceipt,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageSourceAdmissionFailure {
    #[error("image source graph or selected binding is not admitted")]
    Source,
    #[error("image source invocation inputs are malformed")]
    Inputs,
    #[error("image source operation {0} differs from the selected graph")]
    Operation(NodeId),
    #[error("image source leaf {0} is missing from this invocation")]
    MissingLeaf(ValueId),
    #[error("image source constant {0} differs from its exact selected value")]
    ConstantChanged(ValueId),
    #[error("image source canonical inference or domain is unproven: {0}")]
    CanonicalUnproven(#[source] anyhow::Error),
    #[error("image source canonical provider admission failed: {0}")]
    CanonicalExecution(#[source] runmat_accelerate_api::ProviderExecutionError),
}

impl ImageSourceAdmissionFailure {
    fn after_observed_probe(self) -> Self {
        Self::CanonicalExecution(
            runmat_accelerate_api::ProviderExecutionError::from_source(self)
                .with_unproven_effects(),
        )
    }
}

fn after_probe_if(
    error: ImageSourceAdmissionFailure,
    observed: bool,
) -> ImageSourceAdmissionFailure {
    if observed {
        error.after_observed_probe()
    } else {
        error
    }
}

fn canonical_failure(
    failure: crate::fusion::semantics::FusionResolveFailure,
) -> ImageSourceAdmissionFailure {
    match failure.into_execution_or_structural() {
        Ok(structural) => ImageSourceAdmissionFailure::CanonicalUnproven(structural),
        Err(execution) => ImageSourceAdmissionFailure::CanonicalExecution(execution),
    }
}

fn source_value_ids(
    source: &ImageSourceTranscript,
) -> Result<Vec<ValueId>, ImageSourceAdmissionFailure> {
    let upper = source
        .steps
        .iter()
        .try_fold(source.leaves.len(), |count, step| {
            count.checked_add(step.inputs.len())?.checked_add(1)
        })
        .ok_or(ImageSourceAdmissionFailure::Inputs)?;
    let mut ids = Vec::new();
    ids.try_reserve_exact(upper).map_err(|error| {
        ImageSourceAdmissionFailure::CanonicalExecution(
            runmat_accelerate_api::ProviderExecutionError::from_source(error),
        )
    })?;
    ids.extend(source.leaves.iter().map(|leaf| match leaf {
        ImageSourceLeaf::Constant { id, .. } | ImageSourceLeaf::Live { id } => *id,
    }));
    ids.extend(source.steps.iter().flat_map(|step| {
        step.inputs
            .iter()
            .copied()
            .chain(std::iter::once(step.output))
    }));
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

/// Fixed-capacity request lookup. The selected input IDs are sorted once;
/// each later lookup is a binary search and does not allocate a hash table.
pub(super) struct ImageValuePositions(Vec<(ValueId, usize)>);

impl ImageValuePositions {
    pub(super) fn new(ids: &[ValueId]) -> Option<Self> {
        Self::try_new(ids).ok().flatten()
    }

    fn try_new(ids: &[ValueId]) -> Result<Option<Self>, std::collections::TryReserveError> {
        let mut positions = Vec::new();
        positions.try_reserve_exact(ids.len())?;
        positions.extend(
            ids.iter()
                .copied()
                .enumerate()
                .map(|(index, id)| (id, index)),
        );
        positions.sort_unstable_by_key(|(id, _)| *id);
        if positions.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Ok(None);
        }
        Ok(Some(Self(positions)))
    }

    pub(super) fn get(&self, id: &ValueId) -> Option<&usize> {
        self.0
            .binary_search_by_key(id, |(key, _)| *key)
            .ok()
            .map(|index| &self.0[index].1)
    }

    fn contains_key(&self, id: &ValueId) -> bool {
        self.get(id).is_some()
    }

    fn ids(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.0.iter().map(|(id, _)| *id)
    }
}

fn bound_scalar(
    scalar: &ImageScalar,
    plan: &FusionGroupPlan,
    inputs: &[Value],
    positions: &ImageValuePositions,
) -> Option<f64> {
    match scalar {
        ImageScalar::Constant(value) => Some(*value),
        ImageScalar::Value(id) => {
            let value = positions
                .get(id)
                .and_then(|position| inputs.get(*position))
                .or_else(|| plan.const_values.get(id))?;
            match super::source_output_hooks::scalar(value)? {
                ProviderImageOutputOperand::ScalarF32Bits(bits) => {
                    Some(f64::from(f32::from_bits(bits)))
                }
                ProviderImageOutputOperand::ScalarF64Bits(bits) => Some(f64::from_bits(bits)),
                _ => None,
            }
        }
    }
}

fn same_scalar(left: &ImageScalar, right: &ImageScalar) -> bool {
    match (left, right) {
        (ImageScalar::Constant(a), ImageScalar::Constant(b)) => a.to_bits() == b.to_bits(),
        (ImageScalar::Value(a), ImageScalar::Value(b)) => a == b,
        _ => false,
    }
}

fn same_optional_scalar(left: &Option<ImageScalar>, right: &Option<ImageScalar>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => same_scalar(a, b),
        _ => false,
    }
}

pub(super) fn same_image_pattern(
    left: &ImageNormalizePattern,
    right: &ImageNormalizePattern,
) -> bool {
    left.input == right.input
        && same_scalar(&left.epsilon, &right.epsilon)
        && same_optional_scalar(&left.gain, &right.gain)
        && same_optional_scalar(&left.bias, &right.bias)
        && same_optional_scalar(&left.gamma, &right.gamma)
        && left.clamp_zero == right.clamp_zero
        && left.clamp_zero_value == right.clamp_zero_value
        && left.bound_mean_axes == right.bound_mean_axes
        && left.requires_single_input == right.requires_single_input
        && left.reduction_routes == right.reduction_routes
}

fn matches_selected_image_group(graph: &AccelGraph, plan: &FusionGroupPlan) -> bool {
    super::source_extension::matches_extended_group(graph, &plan.group)
}

fn selected_stage(
    graph: &AccelGraph,
    plan: &FusionGroupPlan,
    inputs: &[Value],
    positions: &ImageValuePositions,
) -> Option<BoundImageStage> {
    let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref()? else {
        return None;
    };
    if !super::bound_image_controls_match(pattern, |id| {
        positions
            .get(&id)
            .and_then(|position| inputs.get(*position))
    }) {
        return None;
    }
    let routes = pattern.reduction_routes.as_ref()?;
    let input = match inputs.get(*positions.get(&pattern.input)?)? {
        Value::GpuTensor(handle) => BoundImageInput::Resident(handle.clone()),
        Value::Tensor(tensor)
            if tensor.numeric_dtype() == NumericDType::F32 && tensor.as_f32_slice().is_some() =>
        {
            BoundImageInput::HostSingle(tensor.clone())
        }
        _ => return None,
    };
    let [batch, height, width]: [usize; 3] = input.shape().try_into().ok()?;
    if [batch, height, width].contains(&0) {
        return None;
    }
    let control = |scalar: &ImageScalar| bound_scalar(scalar, plan, inputs, positions);
    let desc = ImageNormalizeDescriptor {
        batch,
        height,
        width,
        epsilon: control(&pattern.epsilon)?,
        gain: match &pattern.gain {
            Some(value) => Some(control(value)?),
            None => None,
        },
        bias: match &pattern.bias {
            Some(value) => Some(control(value)?),
            None => None,
        },
        gamma: match &pattern.gamma {
            Some(value) => Some(control(value)?),
            None => None,
        },
        clamp_zero: pattern.clamp_zero,
    };
    let bound_final_exponent = match &pattern.gamma {
        Some(ImageScalar::Value(id)) => Some(power_program::bound_final_exponent(
            *id,
            desc.gamma?,
            inputs,
            positions,
        )?),
        _ => None,
    };
    let source = ImageMeanAndPowerSource::from_selected_graph(
        graph,
        routes,
        [batch, height, width],
        bound_final_exponent,
    )?;
    Some(BoundImageStage {
        input,
        desc,
        source,
        sigma_control: sigma_operations::freeze_control(plan, inputs, positions),
        output_binary: Vec::new(),
        output_real: Vec::new(),
    })
}

#[derive(Debug, Clone)]
pub struct ImageSourceInvocation {
    pub(super) source: Arc<ImageSourceTranscript>,
    pub(super) facts: Vec<(ValueId, ValueFact)>,
    semantic_identity: Digest,
    output_program: Option<ProviderImageOutputProgram>,
    pub(super) moment_program: Option<ProviderImageMomentProgram>,
    workload: Option<ProviderImageStageWorkload>,
    selected_stage: Option<BoundImageStage>,
    selected_powers: Vec<ImageSelectedPower>,
    probe_observed: bool,
    publications_bound: bool,
}

impl ImageSourceInvocation {
    /// Bind live external leaves from *this* request. A plan's captured
    /// constant remains in structural identity, but never shadows an actual
    /// input ValueId. Exact source constants cannot be silently rebound by a
    /// caller-supplied input either.
    pub async fn bind(
        graph: &AccelGraph,
        plan: &FusionGroupPlan,
        inputs: &[Value],
        precision: ProviderPrecision,
        placement: Option<PlacementCorrelationId>,
    ) -> Result<Self, ImageSourceAdmissionFailure> {
        let source = ImageSourceTranscript::from_graph(graph, &plan.group)
            .ok_or(ImageSourceAdmissionFailure::Source)?;
        if !matches_selected_image_group(graph, plan) {
            return Err(ImageSourceAdmissionFailure::Source);
        }
        Self::bind_source(
            Arc::new(source),
            graph,
            plan,
            inputs,
            precision,
            placement,
            false,
        )
        .await
    }

    /// The selected bytecode owner already sealed the exact graph transcript.
    /// Borrow that immutable source through finite preflight and retain its
    /// shared descriptor for the asynchronous staged execution session.
    pub(crate) async fn bind_selected(
        source: Arc<ImageSourceTranscript>,
        graph: &AccelGraph,
        plan: &FusionGroupPlan,
        inputs: &[Value],
        precision: ProviderPrecision,
        placement: Option<PlacementCorrelationId>,
        finite: bool,
    ) -> Result<Self, ImageSourceAdmissionFailure> {
        if source.steps.len() != plan.operations.len()
            || source.steps.iter().any(|step| {
                graph.node(step.node).is_none_or(|node| {
                    node.inputs != step.inputs
                        || node.outputs.as_slice() != [step.output]
                        || node.label != step.label
                        || node.span != step.pc
                })
            })
        {
            return Err(ImageSourceAdmissionFailure::Source);
        }
        Self::bind_source(source, graph, plan, inputs, precision, placement, finite).await
    }

    async fn bind_source(
        source: Arc<ImageSourceTranscript>,
        graph: &AccelGraph,
        plan: &FusionGroupPlan,
        inputs: &[Value],
        precision: ProviderPrecision,
        placement: Option<PlacementCorrelationId>,
        finite: bool,
    ) -> Result<Self, ImageSourceAdmissionFailure> {
        if plan.inputs.len() != inputs.len() || plan.group.nodes.len() != plan.operations.len() {
            return Err(ImageSourceAdmissionFailure::Inputs);
        }
        let input_positions = ImageValuePositions::try_new(&plan.inputs)
            .map_err(|error| {
                ImageSourceAdmissionFailure::CanonicalExecution(
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            })?
            .ok_or(ImageSourceAdmissionFailure::Inputs)?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(source.steps.len())
            .map_err(|error| {
                ImageSourceAdmissionFailure::CanonicalExecution(
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            })?;
        selected.extend(source.steps.iter().map(|step| (step.node, step)));
        selected.sort_unstable_by_key(|(node, _)| *node);
        if selected.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(ImageSourceAdmissionFailure::Source);
        }
        for (&node, operation) in plan.group.nodes.iter().zip(&plan.operations) {
            let step = selected
                .binary_search_by_key(&node, |(key, _)| *key)
                .ok()
                .map(|index| selected[index].1)
                .ok_or(ImageSourceAdmissionFailure::Operation(node))?;
            let same = match (operation, &step.label) {
                (
                    FusionOp::Primitive { op, inputs, output },
                    AccelNodeLabel::Primitive(selected),
                ) => op == selected && inputs == &step.inputs && *output == Some(step.output),
                (
                    FusionOp::Builtin {
                        name,
                        origin,
                        inputs,
                        output,
                    },
                    AccelNodeLabel::Builtin {
                        name: selected,
                        origin: selected_origin,
                    },
                ) => {
                    name == selected
                        && origin == selected_origin
                        && inputs == &step.inputs
                        && *output == Some(step.output)
                }
                _ => false,
            };
            if !same {
                return Err(ImageSourceAdmissionFailure::Operation(node));
            }
        }
        let mut source_ids = Vec::new();
        source_ids
            .try_reserve_exact(source.leaves.len())
            .map_err(|error| {
                ImageSourceAdmissionFailure::CanonicalExecution(
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            })?;
        for leaf in &source.leaves {
            let id = match leaf {
                ImageSourceLeaf::Constant { id, .. } | ImageSourceLeaf::Live { id } => *id,
            };
            source_ids.push(id);
            let actual = input_positions
                .get(&id)
                .and_then(|position| inputs.get(*position))
                .or_else(|| plan.const_values.get(&id))
                .ok_or(ImageSourceAdmissionFailure::MissingLeaf(id))?;
            if let ImageSourceLeaf::Constant { digest, .. } = leaf {
                let actual_digest = crate::fusion::semantics::exact_constant_digest(actual)
                    .map_err(|_| ImageSourceAdmissionFailure::ConstantChanged(id))?;
                if actual_digest != *digest {
                    return Err(ImageSourceAdmissionFailure::ConstantChanged(id));
                }
            } else if !input_positions.contains_key(&id) {
                // No saved compile-time value can stand in for a live source.
                return Err(ImageSourceAdmissionFailure::MissingLeaf(id));
            }
        }
        source_ids.sort_unstable();
        if source_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ImageSourceAdmissionFailure::Source);
        }
        if input_positions
            .ids()
            .any(|id| source_ids.binary_search(&id).is_err())
        {
            return Err(ImageSourceAdmissionFailure::Inputs);
        }
        let fact_ids = source_value_ids(&source)?;
        let mut facts = Vec::new();
        facts.try_reserve_exact(fact_ids.len()).map_err(|error| {
            ImageSourceAdmissionFailure::CanonicalExecution(
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
        let power_count = source
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    &step.label,
                    AccelNodeLabel::Primitive(crate::graph::PrimitiveOp::ElemPow)
                )
            })
            .count();
        let mut selected_powers = Vec::new();
        selected_powers
            .try_reserve_exact(power_count)
            .map_err(|error| {
                ImageSourceAdmissionFailure::CanonicalExecution(
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            })?;
        let resolved = if finite {
            crate::fusion::semantics::ResolvedProgram::resolve_finite_image(
                plan, inputs, precision, placement,
            )
            .await
        } else {
            crate::fusion::semantics::ResolvedProgram::resolve(plan, inputs, precision, placement)
                .await
        }
        .map_err(canonical_failure)?;
        let probe_observed = resolved.observed_provider_probe();
        // Retain the canonical operand facts while the exact request is bound.
        // Equal scalar values of different classes can select different
        // arithmetic, so output facts alone cannot describe this program.
        for id in fact_ids {
            let fact = resolved.fact(id).cloned().map_err(|error| {
                after_probe_if(
                    ImageSourceAdmissionFailure::CanonicalUnproven(error),
                    probe_observed,
                )
            })?;
            facts.push((id, fact));
        }
        let semantic_identity = resolved.cache_identity().map_err(|error| {
            after_probe_if(
                ImageSourceAdmissionFailure::CanonicalUnproven(error),
                probe_observed,
            )
        })?;
        for step in &source.steps {
            if matches!(
                &step.label,
                AccelNodeLabel::Primitive(crate::graph::PrimitiveOp::ElemPow)
            ) {
                let arithmetic = resolved
                    .power_arithmetic(step.output)
                    .ok_or(ImageSourceAdmissionFailure::Operation(step.node))
                    .map_err(|error| after_probe_if(error, probe_observed))?;
                selected_powers.push(ImageSelectedPower {
                    output: step.output,
                    arithmetic,
                });
            }
        }
        let mut invocation = Self {
            source,
            facts,
            semantic_identity,
            output_program: None,
            moment_program: None,
            workload: None,
            selected_stage: None,
            selected_powers,
            probe_observed,
            publications_bound: false,
        };
        if let Some((moments, preparation)) = invocation
            .project_moment_program(plan, inputs, precision, placement)
            .await
            .map_err(|error| after_probe_if(error, probe_observed))?
        {
            invocation.moment_program = Some(moments);
            let (binary_steps, real_steps) =
                if let Some(projected) = invocation.project_output_program(plan, inputs) {
                    invocation.output_program = Some(projected.program);
                    (projected.binary_steps, projected.real_steps)
                } else {
                    (Vec::new(), Vec::new())
                };
            if let Some(stage) = selected_stage(graph, plan, inputs, &input_positions) {
                let stage = {
                    let mut stage = stage;
                    // These companions have one selected owner. Move them
                    // into the held stage rather than clone their nested
                    // source operand vectors during preparation.
                    stage.output_binary = binary_steps;
                    stage.output_real = real_steps;
                    stage
                };
                let powers = invocation
                    .project_power_program(&stage.source)
                    .ok_or(ImageSourceAdmissionFailure::Source)
                    .map_err(|error| after_probe_if(error, probe_observed))?;
                let epsilon = stage
                    .sigma_control
                    .as_ref()
                    .ok_or(ImageSourceAdmissionFailure::Source)
                    .map_err(|error| after_probe_if(error, probe_observed))?
                    .literal();
                let mut workload = stage
                    .source
                    .staged_workload_selected(&stage.desc, powers, epsilon)
                    .ok_or(ImageSourceAdmissionFailure::Source)
                    .map_err(|error| after_probe_if(error, probe_observed))?;
                workload.moment_program = invocation.moment_program.clone();
                workload.output_program = invocation.output_program.clone();
                if workload.output_program.is_some() && workload.validate() {
                    // Preparation and all additional calls belong to this exact
                    // request identity; old arithmetic/resource receipts cannot
                    // authenticate the expanded source program.
                    invocation.semantic_identity =
                        Digest::sha256_serialized(&(semantic_identity, &workload, preparation))
                            .map_err(|error| {
                                after_probe_if(
                                    ImageSourceAdmissionFailure::CanonicalUnproven(error.into()),
                                    probe_observed,
                                )
                            })?;
                    invocation.workload = Some(workload);
                    invocation.selected_stage = Some(stage);
                }
            }
        }
        Ok(invocation)
    }

    pub fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }

    pub(crate) fn observed_provider_probe(&self) -> bool {
        self.probe_observed
    }

    pub(crate) fn selected_publications_bound(&self) -> bool {
        self.publications_bound
    }

    /// Canonical request-bound evidence, not a certificate that a provider
    /// hook implements the same arithmetic or that a value is initialized.
    pub fn source_fact(&self, id: ValueId) -> Option<&ValueFact> {
        self.facts
            .binary_search_by_key(&id, |(value, _)| *value)
            .ok()
            .map(|index| &self.facts[index].1)
    }

    pub fn selected_power(&self, output: ValueId) -> Option<ImageSelectedPower> {
        self.selected_powers
            .iter()
            .find(|power| power.output == output)
            .copied()
    }

    /// This frozen request is projected during binding from the same exact
    /// invocation; callers cannot substitute later plan or control values.
    pub fn requested_output_program(&self) -> Option<&ProviderImageOutputProgram> {
        self.output_program.as_ref()
    }

    pub fn requested_staged_workload(&self) -> Option<&ProviderImageStageWorkload> {
        self.workload.as_ref()
    }

    /// Only the exact GPU view and controls admitted by this same request can
    /// supply the held stage's mean/power transcript. A coincident shape or a
    /// separately supplied source plan is never selection authority.
    pub fn selected_staged_mean_power(
        &self,
        input: &GpuTensorHandle,
    ) -> Option<(&ImageNormalizeDescriptor, &ImageMeanAndPowerSource)> {
        let selected = self.selected_stage.as_ref()?;
        let BoundImageInput::Resident(bound) = &selected.input else {
            return None;
        };
        bound.allocation_lease().validate_live().ok()?;
        input.allocation_lease().validate_live().ok()?;
        (bound.view_identity() == input.view_identity())
            .then_some((&selected.desc, &selected.source))
    }

    pub(crate) fn selected_staged_mean_power_host(
        &self,
        input: &Tensor,
    ) -> Option<(&ImageNormalizeDescriptor, &ImageMeanAndPowerSource)> {
        let selected = self.selected_stage.as_ref()?;
        let BoundImageInput::HostSingle(bound) = &selected.input else {
            return None;
        };
        let bound_words = bound.as_f32_slice()?;
        let input_words = input.as_f32_slice()?;
        (bound.shape() == input.shape()
            && bound_words.len() == input_words.len()
            && std::ptr::eq(bound_words.as_ptr(), input_words.as_ptr()))
        .then_some((&selected.desc, &selected.source))
    }

    pub(crate) fn selected_staged_mean_power_for_value(
        &self,
        input: &Value,
    ) -> Option<(&ImageNormalizeDescriptor, &ImageMeanAndPowerSource)> {
        match input {
            Value::GpuTensor(handle) => self.selected_staged_mean_power(handle),
            Value::Tensor(tensor) => self.selected_staged_mean_power_host(tensor),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::semantics::FusionResolveFailure;
    use anyhow::anyhow;
    use runmat_accelerate_api::test_support::TestAllocationOwner;
    use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn selected_fact_inventory_retains_leaves_operands_and_outputs_once() {
        use crate::fusion::image_reduction::ImageSourceStep;
        use crate::graph::{InstrSpan, PrimitiveOp};

        let source = ImageSourceTranscript {
            leaves: vec![
                ImageSourceLeaf::Live { id: 9 },
                ImageSourceLeaf::Constant {
                    id: 2,
                    digest: Digest::sha256("double-control"),
                },
            ],
            steps: vec![ImageSourceStep {
                node: 0,
                pc: InstrSpan { start: 1, end: 1 },
                label: AccelNodeLabel::Primitive(PrimitiveOp::ElemPow),
                inputs: vec![9, 2, 9],
                output: 4,
                selected_binding: None,
            }],
            checkpoints: vec![4],
        };
        assert_eq!(source_value_ids(&source).unwrap(), vec![2, 4, 9]);
        let double = runmat_runtime::value_fact::value_fact(&Value::Num(2.0));
        let single = runmat_runtime::value_fact::value_fact(&Value::Tensor(
            runmat_value::Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap(),
        ));
        let invocation = ImageSourceInvocation {
            source: Arc::new(source),
            facts: vec![
                (2, double.clone()),
                (4, single.clone()),
                (9, single.clone()),
            ],
            semantic_identity: Digest::sha256("typed-request"),
            output_program: None,
            moment_program: None,
            workload: None,
            selected_stage: None,
            selected_powers: vec![],
            probe_observed: false,
            publications_bound: false,
        };
        assert_eq!(invocation.source_fact(2), Some(&double));
        assert_eq!(invocation.source_fact(9), Some(&single));
        assert_ne!(
            double.numeric().unwrap().class,
            single.numeric().unwrap().class
        );
        assert!(invocation.source_fact(3).is_none());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn canonical_runtime_failure_retains_typed_provider_cause() {
        let runtime = runmat_runtime::build_runtime_error("selected image probe failed")
            .with_identifier("RunMat:image:Internal")
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .build();
        let failure = canonical_failure(FusionResolveFailure::from(runtime));
        let ImageSourceAdmissionFailure::CanonicalExecution(execution) = failure else {
            panic!("runtime admission must stay terminal and typed");
        };
        let original = std::error::Error::source(&execution)
            .and_then(|source| source.downcast_ref::<runmat_runtime::RuntimeError>())
            .expect("original runtime error is retained");
        assert_eq!(original.identifier(), Some("RunMat:image:Internal"));
        assert_eq!(
            original.gpu_gather_retry(),
            runmat_runtime::GpuGatherRetry::Never
        );
        assert!(!execution.is_unsupported());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn canonical_structural_failure_remains_structural() {
        let failure = canonical_failure(FusionResolveFailure::Structural(anyhow!(
            "bad selected graph"
        )));
        assert!(matches!(
            failure,
            ImageSourceAdmissionFailure::CanonicalUnproven(_)
        ));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn selected_stage_rejects_same_shape_other_view_and_invalidated_source() {
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(
                vec![2, 4, 5],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let other = owner
            .allocate_zeroed(
                vec![2, 4, 5],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let source = ImageMeanAndPowerSource {
            mu: super::super::ImageMeanProviderPlan { calls: vec![] },
            variance: super::super::ImageMeanProviderPlan { calls: vec![] },
            square: super::super::ImageSourcePower {
                node: 0,
                base: 0,
                exponent_value: 0,
                represented_exponent: 2.0,
            },
            final_power: super::super::ImageSourcePower {
                node: 0,
                base: 0,
                exponent_value: 0,
                represented_exponent: 1.0,
            },
        };
        let invocation = ImageSourceInvocation {
            source: Arc::new(ImageSourceTranscript {
                steps: vec![],
                leaves: vec![],
                checkpoints: vec![],
            }),
            facts: vec![],
            semantic_identity: Digest::sha256("selected-stage-test"),
            output_program: None,
            moment_program: None,
            workload: None,
            selected_stage: Some(BoundImageStage {
                sigma_control: None,
                output_binary: Vec::new(),
                output_real: Vec::new(),
                input: BoundImageInput::Resident(input.handle.clone()),
                desc: ImageNormalizeDescriptor {
                    batch: 2,
                    height: 4,
                    width: 5,
                    epsilon: 0.25,
                    gain: None,
                    bias: None,
                    gamma: None,
                    clamp_zero: false,
                },
                source,
            }),
            selected_powers: vec![],
            probe_observed: false,
            publications_bound: false,
        };
        assert!(invocation
            .selected_staged_mean_power(&input.handle)
            .is_some());
        assert!(invocation
            .selected_staged_mean_power(&other.handle)
            .is_none());
        input.probe.mark_unavailable();
        assert!(invocation
            .selected_staged_mean_power(&input.handle)
            .is_none());
        assert_eq!(other.probe.retirements(), 0);
    }
}
