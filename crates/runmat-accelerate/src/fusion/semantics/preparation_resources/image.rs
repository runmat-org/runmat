use super::*;
use runmat_types::LiteralValue;

/// The image source has reductions, POWER and conversions that are not pure
/// unary/binary declarations. This is a separate typed shape inventory over
/// the already sealed source transcript, not a widened pure-op certificate.
pub(crate) struct ImageSourceGeometry {
    core: PureSourceGeometry,
    operand_ids: usize,
    source_leaves: usize,
    source_steps: usize,
    source_edges: usize,
    power_steps: usize,
    real_unary_steps: usize,
    source_fact_ids_upper: usize,
    power_identity_bytes: u64,
    literal_bytes: u64,
}

impl ImageSourceGeometry {
    pub(crate) fn stage_cardinality(&self) -> Result<ImageStageCardinality, PreparationBoundError> {
        let mut sigma_facts = RequestedBytes::default();
        // The source-selected sigma call owns two canonical facts. Its held
        // physical plan clones that pair while the original is still live.
        for _ in 0..4 {
            sigma_facts.fact_payload(self.core.max_rank)?;
        }
        Ok(ImageStageCardinality {
            source_steps: self.source_steps,
            power_steps: self.power_steps,
            power_identity_bytes: self.power_identity_bytes,
            moment_cast_operation_bytes: runmat_builtins::SINGLE_CATALOG_ENTRY
                .identity
                .name
                .len()
                .max(runmat_builtins::DOUBLE_CATALOG_ENTRY.identity.name.len()),
            sigma_fact_payload_bytes: sigma_facts.0,
            moment_casts_upper: self
                .source_steps
                .checked_mul(5)
                .ok_or(PreparationBoundError::Overflow)?,
            rank_upper: self.core.max_rank.max(3),
        })
    }

    pub fn admit(
        plan: &FusionGroupPlan,
        inputs: &[Value],
        source: &crate::fusion::image_reduction::ImageSourceTranscript,
    ) -> Result<Self, PreparationBoundError> {
        if plan.inputs.len() != inputs.len()
            || plan.operations.len() != source.steps.len()
            || plan.group.nodes.len() != source.steps.len()
        {
            return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
        }
        let mut max_rank = 2;
        let mut resident_inputs = 0usize;
        for value in inputs
            .iter()
            .chain(plan.const_values.values())
            .chain(plan.constants.values())
        {
            let rank = match value {
                Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => 2,
                Value::Tensor(tensor) => tensor.shape().len(),
                Value::StringArray(array) => array.shape.len(),
                Value::CharArray(array) => array.shape().len(),
                Value::GpuTensor(_) => source_rank(value)
                    .ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?,
                _ => return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier),
            };
            max_rank = max_rank.max(rank);
            if let Value::GpuTensor(handle) = value {
                // An owned snapshot API may build and clone arbitrary nested
                // provider metadata. A finite request needs the provider's
                // immutable, lifetime-owned inventory before any local
                // source vectors or provider probe are created.
                if runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
                    .and_then(|provider| provider.capability_snapshot_ref())
                    .is_none()
                {
                    return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
                }
            }
        }
        if plan
            .const_values
            .values()
            .chain(plan.constants.values())
            .any(|value| matches!(value, Value::GpuTensor(_)))
        {
            return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
        }
        for value in inputs {
            resident_inputs = resident_inputs
                .checked_add(usize::from(matches!(value, Value::GpuTensor(_))))
                .ok_or(PreparationBoundError::Overflow)?;
        }
        let mut operand_ids = 0usize;
        let mut add_sign_operations = 0usize;
        let mut real_unary_steps = 0usize;
        for operation in &plan.operations {
            let (count, add, real_unary) = match operation {
                FusionOp::Primitive { op, inputs, .. } => {
                    if !matches!(
                        op,
                        PrimitiveOp::Add
                            | PrimitiveOp::Sub
                            | PrimitiveOp::Mul
                            | PrimitiveOp::ElemMul
                            | PrimitiveOp::ElemDiv
                            | PrimitiveOp::ElemPow
                            | PrimitiveOp::Neg
                            | PrimitiveOp::UPlus
                    ) {
                        return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
                    }
                    (inputs.len(), *op == PrimitiveOp::Add, false)
                }
                FusionOp::Builtin { inputs, .. } => {
                    let binding = SemanticBinding::capture_typed(operation)
                        .map_err(|_| PreparationBoundError::UnsupportedDeclarationOrCarrier)?
                        .ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?;
                    if !runmat_builtins::finite_fusion_fact_route(binding.entry, inputs.len()) {
                        return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
                    }
                    (
                        inputs.len(),
                        false,
                        binding
                            .declaration
                            .semantics
                            .is_some_and(|semantics| semantics.required_real_unary.is_some()),
                    )
                }
            };
            if count > 4 {
                return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
            }
            operand_ids = operand_ids
                .checked_add(count)
                .ok_or(PreparationBoundError::Overflow)?;
            add_sign_operations = add_sign_operations
                .checked_add(usize::from(add))
                .ok_or(PreparationBoundError::Overflow)?;
            real_unary_steps = real_unary_steps
                .checked_add(usize::from(real_unary))
                .ok_or(PreparationBoundError::Overflow)?;
        }
        let source_leaves_upper = inputs
            .len()
            .checked_add(plan.const_values.len())
            .ok_or(PreparationBoundError::Overflow)?;
        // Every sealed leaf must bind to a live request input or an exact
        // source constant. Reject an unmatched transcript before pricing the
        // canonical resolver from the plan's smaller leaf cardinality.
        if source.leaves.len() > source_leaves_upper {
            return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
        }
        let source_operand_ids = source
            .steps
            .iter()
            .try_fold(0usize, |count, step| count.checked_add(step.inputs.len()))
            .ok_or(PreparationBoundError::Overflow)?;
        let source_fact_ids_upper = source
            .leaves
            .len()
            .checked_add(source_operand_ids)
            .and_then(|count| count.checked_add(source.steps.len()))
            .ok_or(PreparationBoundError::Overflow)?;
        let power_steps = source
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    &step.label,
                    crate::graph::AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)
                )
            })
            .count();
        // Primitive POWER has no source-step builtin binding; its canonical
        // arithmetic receipt supplies the sole POWER binding at resolution.
        let [power_variant] = runmat_builtins::POWER_CATALOG_ENTRY.bindings else {
            return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
        };
        let power_binding = runmat_builtins::POWER_CATALOG_ENTRY.binding_identity(power_variant);
        let power_identity_bytes = power_binding
            .builtin
            .name
            .len()
            .checked_add(power_binding.variant.len())
            .and_then(|len| len.checked_mul(power_steps))
            .and_then(|len| u64::try_from(len).ok())
            .ok_or(PreparationBoundError::Overflow)?;
        let fact_slots_upper = source_leaves_upper
            .checked_add(plan.operations.len())
            .ok_or(PreparationBoundError::Overflow)?;
        let mut literal = RequestedBytes::default();
        // The canonical literal projector admits at most sixteen elements per
        // numeric control. Count its fixed enum storage and every source text
        // byte in each operation that could borrow that control.
        literal.vector::<LiteralValue>(operand_ids)?;
        let mut largest_literal = 0_u64;
        for value in inputs
            .iter()
            .chain(plan.const_values.values())
            .chain(plan.constants.values())
        {
            let mut one = RequestedBytes::default();
            match value {
                Value::String(text) => one.add(
                    u64::try_from(text.as_str().map_or(0, str::len))
                        .map_err(|_| PreparationBoundError::Overflow)?,
                )?,
                Value::StringArray(array) => {
                    one.vector::<usize>(array.shape.len())?;
                    one.vector::<runmat_value::StringElement>(array.data().len())?;
                    for text in array.data() {
                        one.add(
                            u64::try_from(text.as_str().map_or(0, str::len))
                                .map_err(|_| PreparationBoundError::Overflow)?,
                        )?;
                    }
                }
                Value::CharArray(array) => {
                    one.vector::<usize>(array.shape().len())?;
                    one.vector::<char>(array.data().len())?;
                }
                Value::Tensor(tensor) if tensor.len() <= 16 => {
                    one.vector::<usize>(tensor.shape().len())?;
                    one.vector::<LiteralValue>(tensor.len())?;
                    // All supported fixed-width integer/floating decimal
                    // spellings fit in forty bytes per scalar.
                    one.add(
                        u64::try_from(tensor.len())
                            .ok()
                            .and_then(|count| count.checked_mul(40))
                            .ok_or(PreparationBoundError::Overflow)?,
                    )?;
                }
                Value::Int(_) | Value::Num(_) => one.add(40)?,
                _ => {}
            }
            largest_literal = largest_literal.max(one.0);
        }
        literal.add(
            largest_literal
                .checked_mul(
                    u64::try_from(operand_ids).map_err(|_| PreparationBoundError::Overflow)?,
                )
                .ok_or(PreparationBoundError::Overflow)?,
        )?;
        Ok(Self {
            core: PureSourceGeometry {
                inputs: inputs.len(),
                constants: plan.const_values.len(),
                slot_constants: plan.constants.len(),
                operations: plan.operations.len(),
                source_leaves_upper,
                fact_slots_upper,
                resident_inputs,
                source_operand_ids: operand_ids,
                liveness_value_ids_upper: plan
                    .inputs
                    .len()
                    .checked_add(usize::from(plan.output.is_some()))
                    .and_then(|count| count.checked_add(plan.materialized_stores.len()))
                    .and_then(|count| count.checked_add(operand_ids))
                    .and_then(|count| count.checked_add(plan.operations.len()))
                    .ok_or(PreparationBoundError::Overflow)?,
                add_sign_operations,
                max_rank,
                // This image owner inventories domain proof stages separately
                // in projected_program_bytes(), not via PureSourceGeometry's
                // selected-operation admission path.
                owned_admission_bytes: 0,
            },
            operand_ids,
            source_leaves: source.leaves.len(),
            source_steps: source.steps.len(),
            source_edges: source_operand_ids,
            power_steps,
            real_unary_steps,
            source_fact_ids_upper,
            power_identity_bytes,
            literal_bytes: literal.0,
        })
    }

    /// Bind-source vectors created before canonical numerical resolution.
    /// All counts come from the sealed transcript and live input IDs; no
    /// provider output or workspace payload is mirrored to size this phase.
    pub fn source_binding_bytes(&self) -> Result<u64, PreparationBoundError> {
        use crate::fusion::image_reduction::{ImageSelectedPower, ImageSourceStep};

        let mut bytes = RequestedBytes::default();
        bytes.vector::<(ValueId, usize)>(self.core.inputs)?;
        bytes.vector::<(crate::graph::NodeId, &ImageSourceStep)>(self.source_steps)?;
        bytes.vector::<ValueId>(self.source_leaves)?;
        bytes.vector::<ValueId>(self.source_fact_ids_upper)?;
        bytes.vector::<(ValueId, ValueFact)>(self.source_fact_ids_upper)?;
        bytes.vector::<ImageSelectedPower>(self.power_steps)?;
        for _ in 0..self.source_fact_ids_upper {
            bytes.fact_payload(self.core.max_rank)?;
        }
        Ok(bytes.0)
    }

    /// Selected-source projection and the copies retained by the frozen
    /// workload/session join. The copies correspond to `ImageSourceInvocation`,
    /// its workload, publication rebinding, and the later session's frozen,
    /// rebuilt and comparison programs; they are summed rather than assumed
    /// to reuse an allocator block. Physical proposal payloads have a
    /// separate owner below this source projection.
    pub fn projected_program_bytes(&self) -> Result<u64, PreparationBoundError> {
        use crate::fusion::image_reduction::{
            ImageMeanProviderCall, ImageSelectedBinaryCall, ImageSelectedMomentConversion,
            ImageSelectedRealCall, ImageSourceStep, SelectedBinaryStep, SelectedRealStep,
        };
        use runmat_accelerate_api::{
            ProviderImageMeanCall, ProviderImageMomentConversion, ProviderImageMomentPublication,
            ProviderImageOutputCall, ProviderImageOutputOperand, ProviderImagePowerCall,
        };

        let mut bytes = RequestedBytes::default();
        let rank = self.core.max_rank;
        let calls = self.source_steps;
        let edges = self.source_edges;
        let leaves = self.core.source_leaves_upper;
        let moment_calls = 5usize; // two mu, two variance, one sigma
        let moment_casts = moment_calls
            .checked_mul(calls)
            .ok_or(PreparationBoundError::Overflow)?;

        // `project_moment_program`: the sorted source pairs, leaves and node
        // index coexist with a cloned canonical fact table. Two route walks
        // may each hold a selected-node list while finished publications live.
        bytes.vector::<(ValueId, &Value, bool)>(leaves)?;
        bytes.vector::<(ValueId, &Value)>(leaves)?;
        bytes.vector::<(crate::graph::NodeId, &ImageSourceStep)>(calls)?;
        bytes.vector::<(ValueId, ValueFact)>(self.source_fact_ids_upper)?;
        for _ in 0..self.source_fact_ids_upper {
            bytes.fact_payload(rank)?;
        }
        bytes.vector::<(ValueId, FusionAdmissionWork)>(moment_calls)?;
        bytes.vector::<runmat_runtime::builtins::common::spec::FusionProofStage>(moment_calls)?;
        bytes.vector::<&runmat_accelerate_api::ProviderImageMeanCall>(4)?;
        bytes.vector::<crate::graph::NodeId>(
            calls
                .checked_mul(3)
                .ok_or(PreparationBoundError::Overflow)?,
        )?;
        bytes.vector::<ProviderImageMomentConversion>(moment_casts)?;
        bytes.vector::<u64>(
            moment_calls
                .checked_add(moment_casts)
                .ok_or(PreparationBoundError::Overflow)?,
        )?;

        // The retained moment program is copied into the workload, rebound
        // session, sigma recipe and equality projection. Each copy owns its
        // conversion Vec and the cloned canonical fact shape/provider text.
        for _ in 0..6 {
            bytes.vector::<ProviderImageMomentPublication>(moment_calls)?;
            bytes.vector::<ProviderImageMomentConversion>(moment_casts)?;
            for _ in 0..moment_calls
                .checked_add(moment_casts)
                .ok_or(PreparationBoundError::Overflow)?
            {
                bytes.fact_payload(rank)?;
            }
        }

        // `project_output_program`: ordered lookup, produced IDs, output
        // calls, and their source-selected binary/real companions overlap.
        bytes.vector::<(ValueId, usize)>(self.core.inputs)?;
        bytes.vector::<ValueId>(calls)?;
        bytes.vector::<SelectedBinaryStep>(calls)?;
        bytes.vector::<SelectedRealStep>(calls)?;
        bytes.vector::<ValueId>(edges)?;
        bytes.vector::<ProviderImageOutputOperand>(
            edges
                .checked_mul(2)
                .ok_or(PreparationBoundError::Overflow)?,
        )?;
        bytes.vector::<(u64, usize)>(calls)?; // output-program validation
        bytes.vector::<ImageSelectedBinaryCall>(3)?;
        bytes.vector::<ImageSelectedRealCall>(calls)?;
        let canonical_real_shapes = calls
            .checked_mul(4)
            .and_then(|count| count.checked_add(12))
            .ok_or(PreparationBoundError::Overflow)?;
        bytes.vector::<u64>(
            rank.checked_mul(canonical_real_shapes)
                .ok_or(PreparationBoundError::Overflow)?,
        )?; // binary/real selected facts and their plan copies
        bytes.vector::<ImageSelectedMomentConversion>(moment_casts)?;
        // Two selected moment routes retain outer mu/variance vectors, each
        // with two conversion vectors, in addition to the elements above.
        bytes.vector::<Vec<ImageSelectedMomentConversion>>(4)?;

        // The publication join, session begin and physical selector retain
        // separate output-program copies. All nested operand vectors belong
        // to those copies, not a fabricated full-image numeric payload.
        for _ in 0..6 {
            bytes.vector::<ProviderImageOutputCall>(calls)?;
            bytes.vector::<ProviderImageOutputOperand>(edges)?;
            bytes.vector::<(u64, usize)>(calls)?; // validate while copy lives
        }

        // Two selected POWER calls own exact exponent shapes and source
        // binding/operation identity strings. The workload and session may
        // hold four physical copies during equality validation.
        for _ in 0..4 {
            bytes.vector::<ProviderImagePowerCall>(2)?;
            bytes.vector::<u64>(rank.checked_mul(2).ok_or(PreparationBoundError::Overflow)?)?;
            bytes.add(self.power_identity_bytes)?;
        }
        for _ in 0..6 {
            bytes.vector::<ProviderImageMeanCall>(4)?;
        }
        bytes.vector::<ImageMeanProviderCall>(4)?;
        Ok(bytes.0)
    }

    pub fn resolution_bytes(&self) -> Result<u64, PreparationBoundError> {
        // The async request and a later static placement projection can own
        // independent semantic programs at the same time. Do not assume the
        // static projection reuses the first program's buffers.
        let source = self.core.source_fixed_bytes()?;
        let mut bytes = RequestedBytes(
            source
                .checked_mul(2)
                .ok_or(PreparationBoundError::Overflow)?,
        );
        // Reduction controls can turn a projected at-most-sixteen-element
        // literal axis vector into KnownAxes and clone it once during the
        // canonical shape inference. The three rank vectors in each shared
        // source envelope already cover known_dims, the reduced shape and
        // the resulting fact; these two axis vectors are distinct from the
        // retained LiteralValue projection below. Both async and static
        // resolution may hold their own copy.
        let axis_slots = self
            .source_steps
            .checked_mul(16)
            .and_then(|slots| slots.checked_mul(2))
            .and_then(|slots| slots.checked_mul(2))
            .ok_or(PreparationBoundError::Overflow)?;
        bytes.vector::<usize>(axis_slots)?;
        bytes.add(self.source_binding_bytes()?)?;
        bytes.add(self.projected_program_bytes()?)?;
        if self.real_unary_steps != 0 {
            let mut domain = RequestedBytes::default();
            domain.add(
                crate::fusion::real_domain::resolver_vector_upper_bytes(
                    self.core.source_leaves_upper,
                    self.core.operations,
                    self.operand_ids,
                )
                .ok_or(PreparationBoundError::Overflow)?,
            )?;
            let stages = crate::fusion::real_domain::proof_stage_upper(
                self.core.source_leaves_upper,
                self.core.operations,
                self.real_unary_steps,
            )
            .ok_or(PreparationBoundError::Overflow)?;
            // The first GraphIndex slot may overlap the larger exact reserve
            // made after the graph walk, while the old one-slot Vec retires.
            domain.vector::<runmat_runtime::builtins::common::spec::FusionProofStage>(
                stages
                    .checked_add(1)
                    .ok_or(PreparationBoundError::Overflow)?,
            )?;
            // Each resident leaf can contribute four shape-bearing provider
            // observations to one selected real-domain proof.
            let observed_shapes = self
                .core
                .resident_inputs
                .checked_mul(4)
                .and_then(|count| count.checked_mul(self.real_unary_steps))
                .ok_or(PreparationBoundError::Overflow)?;
            domain.vector::<u64>(
                observed_shapes
                    .checked_mul(self.core.max_rank)
                    .ok_or(PreparationBoundError::Overflow)?,
            )?;
            let probe_visits = self
                .core
                .resident_inputs
                .checked_mul(self.real_unary_steps)
                .ok_or(PreparationBoundError::Overflow)?;
            let transient =
                runmat_runtime::builtins::common::gpu_helpers::scalar_probe_host_upper_bytes(
                    self.core.max_rank,
                )
                .ok_or(PreparationBoundError::Overflow)?;
            domain.add(
                transient
                    .checked_mul(
                        u64::try_from(probe_visits).map_err(|_| PreparationBoundError::Overflow)?,
                    )
                    .ok_or(PreparationBoundError::Overflow)?,
            )?;
            // The request-bound program can coexist with the later static
            // projection. Both may retain separate range caches, work-stage
            // vectors and observed representation metadata; neither borrows
            // the other's resolved facts.
            bytes.add(
                domain
                    .0
                    .checked_mul(2)
                    .ok_or(PreparationBoundError::Overflow)?,
            )?;
        }
        let paired_representations = self
            .power_steps
            .checked_mul(2)
            .ok_or(PreparationBoundError::Overflow)?;
        bytes.vector::<runmat_accelerate_api::ProviderRepresentation>(paired_representations)?;
        bytes.vector::<u64>(
            paired_representations
                .checked_mul(self.core.max_rank)
                .ok_or(PreparationBoundError::Overflow)?,
        )?;
        bytes.add(self.literal_bytes)?;
        // Exact selected bindings can enter one canonical inference at a time.
        // Its request argument and one returned fact coexist with the source
        // operands already counted in the retained semantic program.
        bytes.vector::<ValueFact>(self.operand_ids)?;
        bytes.vector::<LiteralValue>(self.operand_ids)?;
        Ok(bytes.0)
    }
}

/// Authenticated by `ImageSourceGeometry::admit`; no second source parser or
/// class-string projection is allowed at the stage metadata owner.
#[derive(Clone, Copy)]
pub(crate) struct ImageStageCardinality {
    pub(crate) source_steps: usize,
    pub(crate) power_steps: usize,
    pub(crate) power_identity_bytes: u64,
    pub(crate) moment_cast_operation_bytes: usize,
    pub(crate) sigma_fact_payload_bytes: u64,
    pub(crate) moment_casts_upper: usize,
    pub(crate) rank_upper: usize,
}
