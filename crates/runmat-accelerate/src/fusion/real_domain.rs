//! Request-local real range proof for selected unary fusion requirements.
//! Unsupported source operations stay unknown; no range is inferred from a
//! graph's labels or from an ancestor of an unevaluated intermediate.

use std::collections::HashMap;

use anyhow::anyhow;
use runmat_accelerate_api::{NumericalDomainEvidence, ProviderPrecision};
use runmat_builtins::BuiltinBindingIdentity;
use runmat_runtime::builtins::common::{
    gpu_helpers::{self, RealProbeStage},
    spec::{FusionAdmissionWork, FusionOperandDomainEvidence, FusionProofStage},
};
use runmat_types::{
    real_floating_add_mode, NumericClass, NumericDomain, OperatorKind, RealFloatingAddMode,
    RealUnaryClassification, RealUnaryPredicate, ScalarOperation, StorageFact, ValueFact,
};
use runmat_value::{NumericScalar, Value};

use super::semantics::{invocation_leaves::FusionInvocationLeaves, FusionResolveFailure};
use super::{FusionGroupPlan, FusionOp};
use crate::graph::{PrimitiveOp, ValueId};

#[derive(Clone, Copy, Debug)]
pub(super) struct RealRange {
    minimum: f64,
    maximum: f64,
    class: NumericClass,
    // A resident leaf gets this authority only after its exact owner,
    // physical precision, NaN mask and extrema have been validated. Derived
    // ranges retain it through selected, represented arithmetic.
    resident_arithmetic: bool,
}

pub(super) struct RealRangeResolver<'a> {
    operations: Vec<(ValueId, usize)>,
    leaves: Vec<(ValueId, &'a Value)>,
    ranges: Vec<(ValueId, Option<Option<RealRange>>)>,
    ids: Vec<ValueId>,
    visit_marks: Vec<(u32, u8)>,
    visit_generation: u32,
    visit_order: Vec<ValueId>,
    valid: bool,
    indexed_operations: usize,
    indexed_leaves: usize,
}

impl<'a> RealRangeResolver<'a> {
    pub(super) fn new(plan: &'a FusionGroupPlan, invocation: &FusionInvocationLeaves<'a>) -> Self {
        let mut operations = Vec::with_capacity(plan.operations.len());
        let leaves = invocation.values().to_vec();
        let indexed_leaves = invocation.unique_leaf_count();
        let mut valid = true;
        for (index, operation) in plan.operations.iter().enumerate() {
            let output = match operation {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
            };
            if let Some(output) = output {
                valid &= lookup(&leaves, *output).is_none();
                operations.push((*output, index));
            }
        }
        operations.sort_unstable_by_key(|(id, _)| *id);
        valid &= !operations.windows(2).any(|pair| pair[0].0 == pair[1].0);
        let mut ids = Vec::with_capacity(leaves.len() + operations.len());
        ids.extend(leaves.iter().map(|(id, _)| *id));
        ids.extend(operations.iter().map(|(id, _)| *id));
        ids.sort_unstable();
        ids.dedup();
        let ranges = ids.iter().map(|&id| (id, None)).collect();
        let visit_marks = vec![(0, 0); ids.len()];
        let visit_order = Vec::with_capacity(ids.len());
        Self {
            operations,
            leaves,
            ranges,
            ids,
            visit_marks,
            visit_generation: 0,
            visit_order,
            valid,
            indexed_operations: plan.operations.len(),
            indexed_leaves,
        }
    }

    pub(super) fn is_valid(&self) -> bool {
        self.valid
    }

    pub(super) fn report_index(
        &self,
        work: &mut FusionAdmissionWork,
        binding: BuiltinBindingIdentity,
        provider_precision: ProviderPrecision,
    ) {
        append_stage(
            work,
            binding,
            provider_precision,
            FusionProofStage::GraphIndex {
                operations: self.indexed_operations as u64,
                leaves: self.indexed_leaves as u64,
            },
        );
    }
}

fn lookup<T>(entries: &[(ValueId, T)], id: ValueId) -> Option<&T> {
    entries
        .binary_search_by_key(&id, |(key, _)| *key)
        .ok()
        .map(|index| &entries[index].1)
}

fn range_at(
    ranges: &[(ValueId, Option<Option<RealRange>>)],
    id: ValueId,
) -> Option<Option<RealRange>> {
    lookup(ranges, id).copied().flatten()
}

trait RangeLookup {
    fn known(&self, id: ValueId) -> Option<Option<RealRange>>;
}

impl RangeLookup for Vec<(ValueId, Option<Option<RealRange>>)> {
    fn known(&self, id: ValueId) -> Option<Option<RealRange>> {
        range_at(self, id)
    }
}

#[cfg(test)]
impl RangeLookup for HashMap<ValueId, Option<RealRange>> {
    fn known(&self, id: ValueId) -> Option<Option<RealRange>> {
        self.get(&id).copied()
    }
}

pub(super) trait ValueFactLookup {
    fn fact(&self, id: ValueId) -> Option<&ValueFact>;
}

impl ValueFactLookup for HashMap<ValueId, ValueFact> {
    fn fact(&self, id: ValueId) -> Option<&ValueFact> {
        self.get(&id)
    }
}

impl ValueFactLookup for super::semantics::fact_index::FactIndex {
    fn fact(&self, id: ValueId) -> Option<&ValueFact> {
        self.get(&id)
    }
}

fn floating_class(facts: &dyn ValueFactLookup, id: ValueId) -> Option<NumericClass> {
    let numeric = facts.fact(id)?.numeric()?;
    (numeric.domain == NumericDomain::Real
        && matches!(numeric.class, NumericClass::Single | NumericClass::Double))
    .then_some(numeric.class)
}

fn represented(value: f64, class: NumericClass) -> f64 {
    match class {
        NumericClass::Single => f64::from(value as f32),
        _ => value,
    }
}

/// Keep the language's represented-add policy as the arithmetic authority.
/// The generic fact intentionally describes GPU storage as opaque; only a
/// request-local range backed by an exact-owner resident probe may project
/// that carrier as dense numerical data for this fusion operation. A derived
/// range retains the same certificate, but sparse and unrelated unknown facts
/// do not gain numerical authority.
fn represented_add_mode(
    left_fact: &ValueFact,
    right_fact: &ValueFact,
    left: RealRange,
    right: RealRange,
) -> Option<RealFloatingAddMode> {
    fn admitted(fact: &ValueFact, range: RealRange) -> Option<ValueFact> {
        let numeric = fact.numeric()?;
        if numeric.domain != NumericDomain::Real || numeric.class != range.class {
            return None;
        }
        let mut fact = fact.clone();
        if range.resident_arithmetic
            && matches!(fact.storage, StorageFact::Opaque | StorageFact::Unknown)
        {
            fact.storage = StorageFact::Dense;
        }
        Some(fact)
    }
    real_floating_add_mode(&admitted(left_fact, left)?, &admitted(right_fact, right)?)
}

fn append_stage(
    work: &mut FusionAdmissionWork,
    binding: BuiltinBindingIdentity,
    provider_precision: ProviderPrecision,
    stage: FusionProofStage,
) {
    if matches!(work, FusionAdmissionWork::None) {
        *work = FusionAdmissionWork::DomainProof {
            binding,
            provider_precision,
            stages: Vec::new(),
        };
    }
    if let FusionAdmissionWork::DomainProof { stages, .. } = work {
        stages.push(stage);
    }
}

impl RealRangeResolver<'_> {
    fn visit(&mut self, id: ValueId, plan: &FusionGroupPlan) -> (Option<()>, u64) {
        let Self {
            operations,
            leaves,
            ranges: known,
            ids,
            visit_marks: marks,
            visit_generation: generation,
            visit_order: order,
            ..
        } = self;
        order.clear();
        if *generation == u32::MAX {
            marks.fill((0, 0));
            *generation = 0;
        }
        *generation += 1;
        let current_generation = *generation;
        let status = |marks: &[(u32, u8)], index: usize| {
            if marks[index].0 == current_generation {
                marks[index].1
            } else {
                0
            }
        };
        let mut stack = vec![(id, false)];
        let mut examined = 0u64;
        while let Some((current, expanded)) = stack.pop() {
            examined = examined.saturating_add(1);
            if range_at(known, current).is_some() {
                continue;
            }
            let Ok(index) = ids.binary_search(&current) else {
                return (None, examined);
            };
            if status(marks, index) == 2 {
                continue;
            }
            if expanded {
                marks[index] = (current_generation, 2);
                order.push(current);
                continue;
            }
            if status(marks, index) == 1 {
                return (None, examined);
            }
            marks[index] = (current_generation, 1);
            stack.push((current, true));
            if lookup(leaves, current).is_some() {
                continue;
            }
            let Some(operation) =
                lookup(operations, current).and_then(|index| plan.operations.get(*index))
            else {
                return (None, examined);
            };
            let inputs = match operation {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
            };
            for input in inputs.iter().rev() {
                let Some(index) = ids.binary_search(input).ok() else {
                    return (None, examined);
                };
                if status(marks, index) == 1 {
                    return (None, examined);
                }
                if range_at(known, *input).is_none() && status(marks, index) != 2 {
                    stack.push((*input, false));
                }
            }
        }
        (Some(()), examined)
    }
}

async fn leaf_range(
    value: &Value,
    class: NumericClass,
    binding: BuiltinBindingIdentity,
    internal_error: &'static runmat_builtins::BuiltinErrorDescriptor,
    provider_precision: ProviderPrecision,
    work: &mut FusionAdmissionWork,
) -> std::result::Result<Option<RealRange>, FusionResolveFailure> {
    match value {
        Value::Num(number) => {
            append_stage(
                work,
                binding,
                provider_precision,
                FusionProofStage::HostScan { elements: 1 },
            );
            let number = represented(*number, class);
            Ok((!number.is_nan()).then_some(RealRange {
                minimum: number,
                maximum: number,
                class,
                resident_arithmetic: false,
            }))
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            if tensor.is_empty() {
                return Ok(None);
            }
            append_stage(
                work,
                binding,
                provider_precision,
                FusionProofStage::HostScan {
                    elements: u64::try_from(tensor.len()).map_err(anyhow::Error::from)?,
                },
            );
            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;
            for index in 0..tensor.len() {
                let value = match tensor.numeric_value_at(index) {
                    Some(NumericScalar::F64(value)) => value,
                    Some(NumericScalar::F32(value)) => f64::from(value),
                    _ => return Ok(None),
                };
                let value = represented(value, class);
                if value.is_nan() {
                    return Ok(None);
                }
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
            Ok(Some(RealRange {
                minimum,
                maximum,
                class,
                resident_arithmetic: false,
            }))
        }
        Value::GpuTensor(handle) => {
            let provider = gpu_helpers::exact_provider_for_handle(handle)
                .ok_or_else(|| anyhow!("fusion range input has no exact provider owner"))?;
            if provider.precision() != provider_precision {
                return Ok(None);
            }
            let builtin = gpu_helpers::RealProbeDiagnostic {
                builtin: binding.builtin,
                internal_error,
            };
            let mut observe = |stage: RealProbeStage| {
                let stage = match stage {
                    RealProbeStage::Provider { operation, input } => {
                        FusionProofStage::Provider { operation, input }
                    }
                    RealProbeStage::ScalarReadback => {
                        FusionProofStage::ScalarReadback { elements: 1 }
                    }
                };
                append_stage(work, binding, provider_precision, stage);
            };
            if gpu_helpers::probe_real_nan_absence_observed(provider, handle, builtin, &mut observe)
                .await
                .map_err(FusionResolveFailure::from)?
                != Some(true)
            {
                return Ok(None);
            }
            let Some(extrema) = gpu_helpers::probe_real_extrema_observed(
                provider,
                handle,
                true,
                builtin,
                &mut observe,
            )
            .await
            .map_err(FusionResolveFailure::from)?
            else {
                return Ok(None);
            };
            if extrema.maximum.is_none() {
                return Ok(None);
            }
            let minimum = represented(extrema.minimum, class);
            let maximum = represented(extrema.maximum.unwrap(), class);
            Ok(
                (!minimum.is_nan() && !maximum.is_nan() && minimum <= maximum).then_some(
                    RealRange {
                        minimum,
                        maximum,
                        class,
                        resident_arithmetic: true,
                    },
                ),
            )
        }
        _ => Ok(None),
    }
}

fn operation_range(
    operation: &FusionOp,
    selected: Option<super::semantic_bindings::SemanticBinding>,
    facts: &dyn ValueFactLookup,
    ranges: &dyn RangeLookup,
    output: ValueId,
) -> Option<RealRange> {
    let class = floating_class(facts, output)?;
    let selected_operation = selected.and_then(|binding| {
        binding
            .declaration
            .semantics
            .and_then(|semantics| semantics.scalar_operation)
    });
    match (operation, selected_operation) {
        (
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs,
                ..
            },
            None,
        )
        | (FusionOp::Builtin { inputs, .. }, Some(ScalarOperation::Operator(OperatorKind::Add))) => {
            let [left_id, right_id] = inputs.as_slice() else {
                return None;
            };
            let left_class = floating_class(facts, *left_id)?;
            let right_class = floating_class(facts, *right_id)?;
            let left = ranges.known(*left_id)??;
            let right = ranges.known(*right_id)??;
            if left.class != left_class || right.class != right_class {
                return None;
            }
            let mode =
                represented_add_mode(facts.fact(*left_id)?, facts.fact(*right_id)?, left, right)?;
            if mode.result_class() != class {
                return None;
            }
            let minimum = mode.add_represented(left.minimum, right.minimum);
            let maximum = mode.add_represented(left.maximum, right.maximum);
            (!minimum.is_nan() && !maximum.is_nan() && minimum <= maximum).then_some(RealRange {
                minimum,
                maximum,
                class,
                resident_arithmetic: left.resident_arithmetic || right.resident_arithmetic,
            })
        }
        (FusionOp::Builtin { inputs, .. }, Some(ScalarOperation::Cast(target))) => {
            if target != class {
                return None;
            }
            let [input] = inputs.as_slice() else {
                return None;
            };
            let source = ranges.known(*input)??;
            let minimum = represented(source.minimum, class);
            let maximum = represented(source.maximum, class);
            (!minimum.is_nan() && !maximum.is_nan() && minimum <= maximum).then_some(RealRange {
                minimum,
                maximum,
                class,
                resident_arithmetic: source.resident_arithmetic,
            })
        }
        _ => None,
    }
}

fn classification(range: RealRange, predicate: RealUnaryPredicate) -> RealUnaryClassification {
    let inside = match predicate {
        RealUnaryPredicate::ClosedUnitInterval => range.minimum >= -1.0 && range.maximum <= 1.0,
        RealUnaryPredicate::AtLeastOne => range.minimum >= 1.0,
        RealUnaryPredicate::Nonnegative => range.minimum >= 0.0,
        RealUnaryPredicate::AtLeastMinusOne => range.minimum >= -1.0,
    };
    if inside {
        RealUnaryClassification::Inside
    } else {
        RealUnaryClassification::Indeterminate
    }
}

pub(super) struct RealDomainProofContext<'a> {
    pub plan: &'a FusionGroupPlan,
    pub snapshots: &'a [Option<super::semantic_bindings::SemanticBinding>],
    pub facts: &'a dyn ValueFactLookup,
    pub binding: BuiltinBindingIdentity,
    pub internal_error: &'static runmat_builtins::BuiltinErrorDescriptor,
    pub provider_precision: ProviderPrecision,
}

pub(super) async fn prove(
    context: RealDomainProofContext<'_>,
    resolver: &mut RealRangeResolver<'_>,
    operand_id: ValueId,
    predicate: RealUnaryPredicate,
    work: &mut FusionAdmissionWork,
) -> std::result::Result<Option<FusionOperandDomainEvidence>, FusionResolveFailure> {
    let RealDomainProofContext {
        plan,
        snapshots,
        facts,
        binding,
        internal_error,
        provider_precision,
    } = context;
    let (order, examined) = resolver.visit(operand_id, plan);
    append_stage(
        work,
        binding,
        provider_precision,
        FusionProofStage::GraphWalk { examined },
    );
    let Some(()) = order else {
        return Ok(None);
    };
    for &id in &resolver.visit_order {
        if range_at(&resolver.ranges, id).is_some() {
            continue;
        }
        let range = if let Some(value) = lookup(&resolver.leaves, id) {
            match floating_class(facts, id) {
                Some(class) => {
                    leaf_range(
                        value,
                        class,
                        binding,
                        internal_error,
                        provider_precision,
                        work,
                    )
                    .await?
                }
                None => None,
            }
        } else {
            let Some((index, operation)) = lookup(&resolver.operations, id).and_then(|index| {
                plan.operations
                    .get(*index)
                    .map(|operation| (*index, operation))
            }) else {
                return Ok(None);
            };
            let operands = match operation {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => {
                    inputs.len()
                }
            };
            append_stage(
                work,
                binding,
                provider_precision,
                FusionProofStage::RangePropagation {
                    operands: u64::try_from(operands).map_err(anyhow::Error::from)?,
                },
            );
            operation_range(
                operation,
                snapshots.get(index).copied().flatten(),
                facts,
                &resolver.ranges,
                id,
            )
        };
        let index = resolver
            .ranges
            .binary_search_by_key(&id, |(key, _)| *key)
            .map_err(|_| anyhow!("fusion range output has no source value ID"))?;
        resolver.ranges[index].1 = Some(range);
    }
    let Some(range) = range_at(&resolver.ranges, operand_id).flatten() else {
        return Ok(None);
    };
    Ok(Some(FusionOperandDomainEvidence {
        value_id: operand_id,
        numerical_domain: NumericalDomainEvidence::RealUnary {
            predicate,
            classification: classification(range, predicate),
        },
        sign: None,
    }))
}

#[cfg(test)]
#[path = "real_domain/tests.rs"]
mod tests;
