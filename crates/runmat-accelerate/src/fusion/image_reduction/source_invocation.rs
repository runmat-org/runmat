//! Request-bound facts for the exact selected image source transcript.
//!
//! This uses the existing fusion semantic resolver. It cannot mint a provider
//! arithmetic proof: full-stage numerical equivalence, effects, and resources
//! remain separate admissions.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use runmat_accelerate_api::{
    GpuTensorHandle, ImageNormalizeDescriptor, ProviderImageMomentProgram,
    ProviderImageOutputOperand, ProviderImageOutputProgram, ProviderImageStageWorkload,
    ProviderPrecision,
};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::FusionPowerArithmeticReceipt;
use runmat_types::ValueFact;
use runmat_value::Value;

use crate::fusion::{FusionGroupPlan, FusionOp, FusionPattern, ImageNormalizePattern, ImageScalar};
use crate::graph::{AccelGraph, AccelNodeLabel, NodeId, ValueId};
use crate::placement::PlacementCorrelationId;

use super::{ImageMeanAndPowerSource, ImageSourceLeaf, ImageSourceTranscript};

#[path = "source_invocation/moment_bindings.rs"]
mod moment_bindings;
#[path = "source_invocation/binary_operations.rs"]
mod binary_operations;
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
    input: GpuTensorHandle,
    desc: ImageNormalizeDescriptor,
    source: ImageMeanAndPowerSource,
    sigma_control: Option<sigma_operations::BoundSigmaAddControl>,
    output_binary: Vec<super::source_output_hooks::SelectedBinaryStep>,
    output_real: Vec<super::source_output_hooks::SelectedRealStep>,
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

fn canonical_failure(
    failure: crate::fusion::semantics::FusionResolveFailure,
) -> ImageSourceAdmissionFailure {
    match failure.into_execution_or_structural() {
        Ok(structural) => ImageSourceAdmissionFailure::CanonicalUnproven(structural),
        Err(execution) => ImageSourceAdmissionFailure::CanonicalExecution(execution),
    }
}

fn source_value_ids(source: &ImageSourceTranscript) -> BTreeSet<ValueId> {
    source
        .leaves
        .iter()
        .map(|leaf| match leaf {
            ImageSourceLeaf::Constant { id, .. } | ImageSourceLeaf::Live { id } => *id,
        })
        .chain(source.steps.iter().flat_map(|step| {
            step.inputs
                .iter()
                .copied()
                .chain(std::iter::once(step.output))
        }))
        .collect()
}

fn bound_scalar(
    scalar: &ImageScalar,
    plan: &FusionGroupPlan,
    inputs: &[Value],
    positions: &HashMap<ValueId, usize>,
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
    positions: &HashMap<ValueId, usize>,
) -> Option<BoundImageStage> {
    let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref()? else {
        return None;
    };
    let routes = pattern.reduction_routes.as_ref()?;
    let Value::GpuTensor(input) = inputs.get(*positions.get(&pattern.input)?)? else {
        return None;
    };
    let [batch, height, width]: [usize; 3] = input.shape().try_into().ok()?;
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
    let source = ImageMeanAndPowerSource::from_graph(graph, routes, [batch, height, width])?;
    source.staged_workload(&desc)?;
    Some(BoundImageStage {
        input: input.clone(),
        desc,
        source,
        sigma_control: sigma_operations::freeze_control(plan, inputs, positions),
        output_binary: Vec::new(),
        output_real: Vec::new(),
    })
}

#[derive(Debug, Clone)]
pub struct ImageSourceInvocation {
    pub(super) source: ImageSourceTranscript,
    pub(super) facts: Vec<(ValueId, ValueFact)>,
    semantic_identity: Digest,
    output_program: Option<ProviderImageOutputProgram>,
    pub(super) moment_program: Option<ProviderImageMomentProgram>,
    workload: Option<ProviderImageStageWorkload>,
    selected_stage: Option<BoundImageStage>,
    selected_powers: Vec<ImageSelectedPower>,
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
        Self::bind_source(source, graph, plan, inputs, precision, placement).await
    }

    /// The selected bytecode sealed this exact source graph once. Recheck
    /// its live graph/plan join without rebuilding a second transcript.
    pub async fn bind_selected(
        source: Arc<ImageSourceTranscript>,
        graph: &AccelGraph,
        plan: &FusionGroupPlan,
        inputs: &[Value],
        precision: ProviderPrecision,
        placement: Option<PlacementCorrelationId>,
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
        let source = Arc::try_unwrap(source).unwrap_or_else(|shared| (*shared).clone());
        Self::bind_source(source, graph, plan, inputs, precision, placement).await
    }

    async fn bind_source(
        source: ImageSourceTranscript,
        graph: &AccelGraph,
        plan: &FusionGroupPlan,
        inputs: &[Value],
        precision: ProviderPrecision,
        placement: Option<PlacementCorrelationId>,
    ) -> Result<Self, ImageSourceAdmissionFailure> {
        if !matches_selected_image_group(graph, plan) {
            return Err(ImageSourceAdmissionFailure::Source);
        }
        if plan.inputs.len() != inputs.len() || plan.group.nodes.len() != plan.operations.len() {
            return Err(ImageSourceAdmissionFailure::Inputs);
        }
        let mut input_positions = HashMap::with_capacity(plan.inputs.len());
        for (position, &id) in plan.inputs.iter().enumerate() {
            if input_positions.insert(id, position).is_some() {
                return Err(ImageSourceAdmissionFailure::Inputs);
            }
        }
        let selected: HashMap<_, _> = source.steps.iter().map(|step| (step.node, step)).collect();
        if selected.len() != source.steps.len() {
            return Err(ImageSourceAdmissionFailure::Source);
        }
        for (&node, operation) in plan.group.nodes.iter().zip(&plan.operations) {
            let step = selected
                .get(&node)
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
        let mut source_ids = HashSet::new();
        for leaf in &source.leaves {
            let id = match leaf {
                ImageSourceLeaf::Constant { id, .. } | ImageSourceLeaf::Live { id } => *id,
            };
            if !source_ids.insert(id) {
                return Err(ImageSourceAdmissionFailure::Source);
            }
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
        if input_positions.keys().any(|id| !source_ids.contains(id)) {
            return Err(ImageSourceAdmissionFailure::Inputs);
        }
        let resolved =
            crate::fusion::semantics::ResolvedProgram::resolve(plan, inputs, precision, placement)
                .await
                .map_err(canonical_failure)?;
        // Retain the canonical operand facts while the exact request is bound.
        // Equal scalar values of different classes can select different
        // arithmetic, so output facts alone cannot describe this program.
        let facts = source_value_ids(&source)
            .into_iter()
            .map(|id| {
                resolved
                    .fact(id)
                    .cloned()
                    .map(|fact| (id, fact))
                    .map_err(ImageSourceAdmissionFailure::CanonicalUnproven)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let semantic_identity = resolved
            .cache_identity()
            .map_err(ImageSourceAdmissionFailure::CanonicalUnproven)?;
        let selected_powers = source
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    &step.label,
                    AccelNodeLabel::Primitive(crate::graph::PrimitiveOp::ElemPow)
                )
            })
            .map(|step| {
                resolved
                    .power_arithmetic(step.output)
                    .map(|arithmetic| ImageSelectedPower {
                        output: step.output,
                        arithmetic,
                    })
                    .ok_or(ImageSourceAdmissionFailure::Operation(step.node))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut invocation = Self {
            source,
            facts,
            semantic_identity,
            output_program: None,
            moment_program: None,
            workload: None,
            selected_stage: None,
            selected_powers,
            publications_bound: false,
        };
        if let Some((moments, preparation)) = invocation
            .project_moment_program(plan, inputs, precision, placement)
            .await?
        {
            invocation.moment_program = Some(moments);
            let projected_output = invocation.project_output_program(plan, inputs);
            invocation.output_program = projected_output
                .as_ref()
                .map(|output| output.program.clone());
            if let Some(stage) = selected_stage(graph, plan, inputs, &input_positions) {
                let stage = {
                    let mut stage = stage;
                    if let Some(projected) = projected_output.as_ref() {
                        stage.output_binary = projected.binary_steps.clone();
                        stage.output_real = projected.real_steps.clone();
                    }
                    stage
                };
                let powers = invocation
                    .project_power_program(&stage.source)
                    .ok_or(ImageSourceAdmissionFailure::Source)?;
                let mut workload = stage
                    .source
                    .staged_workload_selected(&stage.desc, powers)
                    .expect("selected stage validated");
                workload.moment_program = invocation.moment_program.clone();
                workload.output_program = invocation.output_program.clone();
                if workload.output_program.is_some() && workload.validate() {
                    // Preparation and all additional calls belong to this exact
                    // request identity; old arithmetic/resource receipts cannot
                    // authenticate the expanded source program.
                    invocation.semantic_identity = Digest::sha256(
                        serde_json::to_vec(&(semantic_identity, &workload, preparation)).map_err(
                            |error| ImageSourceAdmissionFailure::CanonicalUnproven(error.into()),
                        )?,
                    );
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
        selected.input.allocation_lease().validate_live().ok()?;
        input.allocation_lease().validate_live().ok()?;
        (selected.input.view_identity() == input.view_identity())
            .then_some((&selected.desc, &selected.source))
    }

    /// Every selected graph operation must have exactly one staged owner, in
    /// source order. A matching pattern and final value alone do not prove
    /// that intermediate casts or a source-visible output were preserved.
    pub(crate) fn selected_stage_covers_source(&self) -> bool {
        let Some(moments) = self.moment_program.as_ref() else {
            return false;
        };
        let Some(binary) = self.selected_binary_calls() else {
            return false;
        };
        let Some(sigma) = self.selected_sigma_operations() else {
            return false;
        };
        let Some(real) = self.selected_real_output_calls() else {
            return false;
        };
        let Some(powers) = self
            .requested_staged_workload()
            .and_then(|workload| workload.selected_powers.as_ref())
        else {
            return false;
        };
        let mut outputs = Vec::with_capacity(self.source.steps.len());
        let append_moments =
            |outputs: &mut Vec<u64>,
             calls: &[runmat_accelerate_api::ProviderImageMomentPublication]| {
                for call in calls {
                    outputs.push(call.raw_output);
                    outputs.extend(call.conversions.iter().map(|cast| cast.source_output));
                }
            };
        append_moments(&mut outputs, &moments.mu);
        outputs.push(binary.sigma_center().output());
        outputs.push(u64::from(powers.square.output));
        append_moments(&mut outputs, &moments.variance);
        outputs.push(sigma.source_ids().2);
        append_moments(&mut outputs, std::slice::from_ref(&moments.sigma));
        let Some(output_program) = self.output_program.as_ref() else {
            return false;
        };
        if output_program.calls.len() != real.calls().len() + 3
            || output_program.calls.first().map(|call| call.source_output)
                != Some(binary.output_sub().output())
            || output_program.calls.get(1).map(|call| call.source_output)
                != Some(binary.output_div().output())
            || output_program
                .calls
                .iter()
                .filter(|call| {
                    matches!(
                        call.hook,
                        runmat_accelerate_api::ProviderImageOutputHook::ElemPow
                    )
                })
                .map(|call| call.source_output)
                .collect::<Vec<_>>()
                != [u64::from(powers.final_power.output)]
            || output_program
                .calls
                .iter()
                .filter(|call| {
                    matches!(
                        call.hook,
                        runmat_accelerate_api::ProviderImageOutputHook::ScalarMul
                            | runmat_accelerate_api::ProviderImageOutputHook::ScalarAdd
                            | runmat_accelerate_api::ProviderImageOutputHook::ScalarMax
                            | runmat_accelerate_api::ProviderImageOutputHook::CastSingle
                            | runmat_accelerate_api::ProviderImageOutputHook::CastDouble
                    )
                })
                .map(|call| call.source_output)
                .ne(real.calls().iter().map(ImageSelectedRealCall::output))
        {
            return false;
        }
        outputs.extend(output_program.calls.iter().map(|call| call.source_output));
        outputs.len() == self.source.steps.len()
            && outputs.iter().copied().eq(self
                .source
                .steps
                .iter()
                .map(|step| u64::from(step.output)))
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
        assert_eq!(source_value_ids(&source), BTreeSet::from([2, 4, 9]));
        let double = runmat_runtime::value_fact::value_fact(&Value::Num(2.0));
        let single = runmat_runtime::value_fact::value_fact(&Value::Tensor(
            runmat_value::Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap(),
        ));
        let invocation = ImageSourceInvocation {
            source,
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
            source: ImageSourceTranscript {
                steps: vec![],
                leaves: vec![],
                checkpoints: vec![],
            },
            facts: vec![],
            semantic_identity: Digest::sha256("selected-stage-test"),
            output_program: None,
            moment_program: None,
            workload: None,
            selected_stage: Some(BoundImageStage {
                sigma_control: None,
                output_binary: Vec::new(),
                output_real: Vec::new(),
                input: input.handle.clone(),
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
