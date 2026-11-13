//! Checked request-local source geometry for finite semantic preparation.
//!
//! A bound is meaningful only for the selected declaration and exact live
//! carrier. Unknown declarations do not inherit the simple real-numeric
//! recipe. This module does not authenticate or allocate a provider buffer.

use std::mem::size_of;

use runmat_accelerate_api::{GpuTensorLayout, GpuTensorStorage, NumericElementType};
use runmat_builtins::{BuiltinInferenceRule, MathInferenceRule};
use runmat_runtime::builtins::common::spec::{
    FusionAdmissionWork, FusionPreparationClass, FusionSignEvidence,
};
use runmat_runtime::context::host_allocation::HostAllocationPlan;
use runmat_types::{DimensionFact, OperatorKind, ScalarOperation, ValueFact};
use runmat_value::{NumericDType, Value};

use super::{
    semantic_bindings::SemanticBinding, ExactProductReceipt, FusionArithmeticEvidence,
    FusionGroupPlan, FusionOp, SourceOperationIdentity,
};
use crate::graph::{PrimitiveOp, ValueId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparationBoundError {
    UnsupportedDeclarationOrCarrier,
    Overflow,
}

impl std::fmt::Display for PreparationBoundError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDeclarationOrCarrier => formatter.write_str(
                "finite fusion preparation has no proven bound for the selected declaration or carrier",
            ),
            Self::Overflow => formatter.write_str("finite fusion preparation byte count overflows"),
        }
    }
}

impl std::error::Error for PreparationBoundError {}

#[derive(Default)]
struct RequestedBytes(u64);

impl RequestedBytes {
    fn add(&mut self, bytes: u64) -> Result<(), PreparationBoundError> {
        self.0 = self
            .0
            .checked_add(bytes)
            .ok_or(PreparationBoundError::Overflow)?;
        Ok(())
    }

    fn vector<T>(&mut self, count: usize) -> Result<(), PreparationBoundError> {
        let bytes = count
            .checked_mul(size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(PreparationBoundError::Overflow)?;
        self.add(bytes)
    }

    fn fact_payload(&mut self, rank: usize) -> Result<(), PreparationBoundError> {
        self.vector::<DimensionFact>(rank)?;
        // Runtime's authenticated device ValueFact has `device:` plus the
        // decimal u32 device ID. Reserve the maximum requested String bytes
        // for either host or resident facts without guessing an allocator's
        // capacity rounding.
        self.add(17)
    }
}

fn source_rank(value: &Value) -> Option<usize> {
    match value {
        Value::Num(_) => Some(2),
        Value::Tensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) =>
        {
            Some(tensor.shape().len())
        }
        Value::GpuTensor(handle)
            if handle.physical().storage() == GpuTensorStorage::Real
                && matches!(
                    handle.physical().element_type(),
                    NumericElementType::F32 | NumericElementType::F64
                )
                && handle.view().layout() == GpuTensorLayout::Contiguous
                && !handle.view().is_logical()
                && handle.allocation_lease().validate_live().is_ok()
                && handle.view().validate(handle.physical()).is_ok()
                && runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(
                    handle,
                )
                .is_some_and(|owner| owner.capability_snapshot_ref().is_some()) =>
        {
            Some(handle.shape().len())
        }
        _ => None,
    }
}

/// Same shape/origin payload used by every source-fact preparation owner.
pub(crate) fn source_fact_payload_bytes(rank: usize) -> Result<u64, PreparationBoundError> {
    let mut bytes = RequestedBytes::default();
    bytes.fact_payload(rank)?;
    Ok(bytes.0)
}

fn selected_geometry(operation: &FusionOp) -> Result<(usize, bool), PreparationBoundError> {
    if let FusionOp::Primitive { op, inputs, .. } = operation {
        // VM primitives use the already-selected typed operator rule rather
        // than a named catalog binding. Power has a separate numerical-domain
        // admission, while comparisons and transpose do not produce this
        // cohort's real floating scalar result.
        let arity = match op {
            PrimitiveOp::Neg | PrimitiveOp::UPlus => 1,
            PrimitiveOp::Add
            | PrimitiveOp::Sub
            | PrimitiveOp::Mul
            | PrimitiveOp::ElemMul
            | PrimitiveOp::ElemDiv
            | PrimitiveOp::ElemLeftDiv => 2,
            _ => return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier),
        };
        if inputs.len() != arity {
            return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
        }
        return Ok((arity, *op == PrimitiveOp::Add));
    }
    let Some(binding) = SemanticBinding::capture_typed(operation)
        .map_err(|_| PreparationBoundError::UnsupportedDeclarationOrCarrier)?
    else {
        return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
    };
    let declaration = binding.declaration;
    if declaration.reduction.is_some()
        || declaration.structural.is_some()
        || declaration.epilogue.is_some()
    {
        return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
    }
    let class = declaration
        .semantics
        .and_then(|semantics| semantics.finite_preparation_class())
        .ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?;
    let supported_fact_route = matches!(
        binding.entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(_))
            | BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(_))
    );
    if !supported_fact_route {
        return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
    }
    let input_count = match operation {
        FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs.len(),
    };
    let arity = match class {
        FusionPreparationClass::PureRealUnary => 1,
        FusionPreparationClass::PureRealBinary => 2,
    };
    if input_count != arity {
        return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
    }
    let add = declaration
        .semantics
        .and_then(|semantics| semantics.scalar_operation)
        == Some(ScalarOperation::Operator(OperatorKind::Add));
    Ok((arity, add))
}

/// The exact source cardinalities before any operation may suspend. This is
/// allocation-free; phase-specific requested-byte additions are applied by
/// the execution owner before constructing the leaf/binding vectors.
pub(crate) struct PureSourceGeometry {
    pub inputs: usize,
    pub constants: usize,
    pub slot_constants: usize,
    pub operations: usize,
    pub source_leaves_upper: usize,
    pub fact_slots_upper: usize,
    pub resident_inputs: usize,
    pub source_operand_ids: usize,
    pub liveness_value_ids_upper: usize,
    pub add_sign_operations: usize,
    pub max_rank: usize,
    owned_admission_bytes: u64,
}

impl PureSourceGeometry {
    pub fn admit(plan: &FusionGroupPlan, inputs: &[Value]) -> Result<Self, PreparationBoundError> {
        Self::admit_iter(plan, inputs.iter())
    }

    pub(crate) fn admit_borrowed(
        plan: &FusionGroupPlan,
        inputs: &[&Value],
    ) -> Result<Self, PreparationBoundError> {
        Self::admit_iter(plan, inputs.iter().copied())
    }

    fn admit_iter<'a>(
        plan: &FusionGroupPlan,
        inputs: impl ExactSizeIterator<Item = &'a Value> + Clone,
    ) -> Result<Self, PreparationBoundError> {
        if inputs.len() != plan.inputs.len() {
            return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
        }
        let mut max_rank = 2;
        let mut resident_inputs = 0usize;
        for value in inputs.clone() {
            max_rank = max_rank.max(
                source_rank(value).ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?,
            );
            resident_inputs += usize::from(matches!(value, Value::GpuTensor(_)));
        }
        for value in plan.const_values.values().chain(plan.constants.values()) {
            max_rank = max_rank.max(
                source_rank(value).ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?,
            );
            // Unbound resident constants have no request-input lease in the
            // current invocation-leaf owner. Do not certify them here.
            if matches!(value, Value::GpuTensor(_)) {
                return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
            }
        }
        let mut source_operand_ids = 0usize;
        let mut add_sign_operations = 0usize;
        let mut owned_admission_bytes = 0u64;
        for operation in &plan.operations {
            let (arity, add) = match selected_geometry(operation) {
                Ok(pure) => pure,
                Err(PreparationBoundError::UnsupportedDeclarationOrCarrier) => {
                    // An operation-owned finite quote is NOT a pure-class or
                    // numerical permission. Actual semantic admission and the
                    // checked scalar product still enforce their exact theorem.
                    let binding = SemanticBinding::capture_typed(operation)
                        .map_err(|_| PreparationBoundError::UnsupportedDeclarationOrCarrier)?
                        .ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?;
                    let owner = binding
                        .declaration
                        .semantics
                        .and_then(|semantics| semantics.admit)
                        .ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?;
                    let arity = owner
                        .preparation_arity()
                        .filter(|arity| *arity <= 2)
                        .ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?;
                    let ids = match operation {
                        FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => {
                            inputs
                        }
                    };
                    if ids.len() != arity {
                        return Err(PreparationBoundError::UnsupportedDeclarationOrCarrier);
                    }
                    let mut operands = [None; 2];
                    for (slot, &id) in ids.iter().enumerate() {
                        // Exact invocation leaf first. A computed SSA operand
                        // never borrows an ancestor's value or cached fallback.
                        operands[slot] = if let Some(position) =
                            plan.inputs.iter().position(|&candidate| candidate == id)
                        {
                            inputs.clone().nth(position)
                        } else {
                            plan.const_values.get(&id)
                        };
                    }
                    let quote = owner.preparation_resources(
                        runmat_runtime::builtins::common::spec::FusionAdmissionPreparationQuery {
                            operands: &operands[..arity], max_rank,
                        }).ok_or(PreparationBoundError::UnsupportedDeclarationOrCarrier)?;
                    owned_admission_bytes = owned_admission_bytes
                        .checked_add(quote.scratch_bytes)
                        .and_then(|bytes| bytes.checked_add(quote.retained_bytes))
                        .ok_or(PreparationBoundError::Overflow)?;
                    (arity, false)
                }
                Err(error) => return Err(error),
            };
            source_operand_ids = source_operand_ids
                .checked_add(arity)
                .ok_or(PreparationBoundError::Overflow)?;
            if add {
                add_sign_operations += 1;
            }
        }
        let source_leaves_upper = inputs
            .len()
            .checked_add(plan.const_values.len())
            .ok_or(PreparationBoundError::Overflow)?;
        let fact_slots_upper = source_leaves_upper
            .checked_add(plan.operations.len())
            .ok_or(PreparationBoundError::Overflow)?;
        let liveness_value_ids_upper = plan
            .inputs
            .len()
            .checked_add(usize::from(plan.output.is_some()))
            .and_then(|count| count.checked_add(plan.materialized_stores.len()))
            .and_then(|count| count.checked_add(source_operand_ids))
            .and_then(|count| count.checked_add(plan.operations.len()))
            .ok_or(PreparationBoundError::Overflow)?;
        Ok(Self {
            inputs: inputs.len(),
            constants: plan.const_values.len(),
            slot_constants: plan.constants.len(),
            operations: plan.operations.len(),
            source_leaves_upper,
            fact_slots_upper,
            resident_inputs,
            source_operand_ids,
            liveness_value_ids_upper,
            add_sign_operations,
            max_rank,
            owned_admission_bytes,
        })
    }

    pub fn source_fixed_bytes(&self) -> Result<u64, PreparationBoundError> {
        let mut bytes = RequestedBytes::default();
        bytes.add(self.owned_admission_bytes)?;
        // Captured selected declarations and borrowed request inputs.
        bytes.add(
            crate::fusion_exec::binding_storage_bytes(self.operations)
                .ok_or(PreparationBoundError::Overflow)?,
        )?;
        bytes.vector::<Option<SemanticBinding>>(self.operations)?;
        bytes.vector::<&Value>(self.inputs)?;
        // FusionInvocationLeaves: one sorted ID vector, one final pair vector,
        // one producer check vector, and admitted resident slots.
        bytes.vector::<ValueId>(self.inputs)?;
        bytes.vector::<(ValueId, &Value)>(self.source_leaves_upper)?;
        bytes.vector::<ValueId>(self.operations)?;
        bytes.add(super::invocation_leaves::resident_slot_bytes(self.inputs)?)?;
        // Retained source facts and source topology.
        bytes.vector::<ValueFact>(self.inputs)?;
        bytes.vector::<(ValueId, Option<ValueFact>)>(self.fact_slots_upper)?;
        bytes.vector::<SourceOperationIdentity>(self.operations)?;
        bytes.vector::<ValueId>(self.source_operand_ids)?;
        bytes.vector::<ValueId>(self.inputs)?;
        bytes.vector::<ValueId>(self.inputs)?;
        bytes.vector::<(ValueId, runmat_execution::Digest)>(self.constants)?;
        bytes.vector::<(usize, runmat_execution::Digest)>(self.slot_constants)?;
        bytes.vector::<SemanticBinding>(self.operations)?;
        // Phase-two physical-ABI preflight determines both possible floating
        // recipes after semantic resolution. Its two sorted control vectors
        // and excluded operand lists are created while this phase-one receipt
        // is live, then dropped before the disjoint producer receipt starts.
        for _ in 0..2 {
            bytes.vector::<ValueId>(
                self.operations
                    .checked_mul(2)
                    .ok_or(PreparationBoundError::Overflow)?,
            )?;
            bytes.vector::<ValueId>(self.source_operand_ids)?;
            bytes.vector::<usize>(self.inputs)?;
        }
        // The phase-two geometry walk must apply the same local represented
        // output facts as emission before it can count a downstream binary
        // expression. Its sorted projection and cloned fact payloads exist
        // before the additive producer receipt; charge them to phase one.
        bytes.vector::<(ValueId, ValueFact)>(self.operations)?;
        for _ in 0..self.operations {
            bytes.fact_payload(self.max_rank)?;
        }
        // All six evidence tables are seeded at the same exact ValueId set,
        // even when no operation later populates a particular table.
        bytes.vector::<(ValueId, Option<SemanticBinding>)>(self.fact_slots_upper)?;
        bytes.vector::<(
            ValueId,
            Option<runmat_accelerate_api::NumericalDomainEvidence>,
        )>(self.fact_slots_upper)?;
        bytes.vector::<(ValueId, Option<FusionSignEvidence>)>(self.fact_slots_upper)?;
        bytes.vector::<(ValueId, Option<FusionArithmeticEvidence>)>(self.fact_slots_upper)?;
        bytes.vector::<(ValueId, Option<ExactProductReceipt>)>(self.fact_slots_upper)?;
        bytes.vector::<(ValueId, Option<FusionAdmissionWork>)>(self.fact_slots_upper)?;
        // Every potential producer reserves one deterministic scalar slot.
        bytes.vector::<(ValueId, Option<Value>)>(self.operations)?;
        // Resident authentication retains the representation's u64 shape in
        // addition to its authenticated ValueFact shape/provider identity.
        for _ in 0..self.resident_inputs {
            bytes.vector::<u64>(self.max_rank)?;
        }
        // The source facts, final fact slots, and resident authenticated facts
        // each retain their own shape and provider-origin payloads.
        for count in [self.inputs, self.fact_slots_upper, self.resident_inputs] {
            for _ in 0..count {
                bytes.fact_payload(self.max_rank)?;
            }
        }
        // The one active inferred operation owns one fact vector and one
        // represented-value vector. Summing each operation's requested bytes
        // is conservative across both retained-program and static passes.
        bytes.vector::<ValueFact>(self.source_operand_ids)?;
        bytes.vector::<Option<&Value>>(self.source_operand_ids)?;
        for _ in 0..self.source_operand_ids {
            bytes.fact_payload(self.max_rank)?;
        }
        // Binary broadcast constructs aligned left/right rank vectors and an
        // output rank vector. This also covers the single-output trig clone.
        for _ in 0..self.operations {
            for _ in 0..3 {
                bytes.vector::<DimensionFact>(self.max_rank)?;
            }
        }
        // Add's optional sign proof makes fresh actual facts for both inputs
        // and one infer_binary result, with a second broadcast projection.
        for _ in 0..self.add_sign_operations {
            for _ in 0..3 {
                bytes.fact_payload(self.max_rank)?;
                bytes.vector::<DimensionFact>(self.max_rank)?;
            }
        }
        // A produced Single or dense Double scalar can retain one exact
        // one-element numeric buffer plus its two-dimensional Tensor shape.
        // Num output is smaller; count the Tensor path for every producer.
        for _ in 0..self.operations {
            bytes.add(
                HostAllocationPlan::numeric_bytes(NumericDType::F64, 1)
                    .map_err(|_| PreparationBoundError::Overflow)?,
            )?;
            bytes.vector::<usize>(2)?;
            bytes.fact_payload(2)?;
        }
        Ok(bytes.0)
    }

    /// The retained provider-effective program and a later host-static
    /// program can coexist while a candidate is priced. Summing both source
    /// construction envelopes also covers their temporary inference peaks;
    /// no allocator reuse or scheduling order is assumed.
    pub fn overlapping_preparation_bytes(
        &self,
        host_uses: Option<(usize, usize)>,
    ) -> Result<u64, PreparationBoundError> {
        let source = self.source_fixed_bytes()?;
        let candidate = host_uses
            .map(|(entry_uses, load_uses)| {
                crate::fusion_exec::host_readback::candidate_preparation_bytes(
                    self.resident_inputs,
                    entry_uses,
                    load_uses,
                    self.operations,
                    self.source_operand_ids,
                    self.liveness_value_ids_upper,
                    self.max_rank,
                )
                .ok_or(PreparationBoundError::Overflow)
            })
            .transpose()?
            .unwrap_or(0);
        source
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(candidate))
            .ok_or(PreparationBoundError::Overflow)
    }
}

#[path = "preparation_resources/image.rs"]
mod image;
pub(crate) use image::{ImageSourceGeometry, ImageStageCardinality};
