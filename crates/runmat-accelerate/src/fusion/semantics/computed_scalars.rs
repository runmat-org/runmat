//! Exact request-local scalar results of selected, pure fusion operations.
//!
//! These are admission evidence for an operation's own produced ValueId, not
//! source leaves or catalog literals. An unsupported operation remains absent.

use runmat_time::{duration_ns_saturating, Instant};
use runmat_types::{
    NumericClass, NumericDomain, OperatorKind, RealFloatingRecipe, ScalarOperation, StorageFact,
    ValueFact,
};
use runmat_value::{NumericScalar, Tensor, Value};

use crate::graph::{PrimitiveOp, ValueId};
use crate::placement::{FusionPlacementObserver, FusionScalarEvidenceWork, PlacementCorrelationId};

use super::assembly::InferredOperation;
use super::{FusionGroupPlan, FusionOp, SemanticBinding};

pub(super) struct ComputedScalars {
    // Source assembly has already rejected duplicate producers. A fixed
    // request-local slot table makes the proof payload bound independent of
    // HashMap growth and keeps every lookup logarithmic in producer count.
    values: Vec<(ValueId, Option<Value>)>,
}

/// Measures only exact scalar evaluation and publication. Awaited domain
/// probes and unrelated semantic resolution are outside these timed windows.
pub(super) struct ScalarEvidenceObservation {
    observer: FusionPlacementObserver,
    work: FusionScalarEvidenceWork,
    duration_ns: u64,
    completed: bool,
}

impl ScalarEvidenceObservation {
    pub(super) fn new(correlation: Option<PlacementCorrelationId>) -> Self {
        Self {
            observer: FusionPlacementObserver::new(correlation),
            work: FusionScalarEvidenceWork::default(),
            duration_ns: 0,
            completed: false,
        }
    }

    pub(super) fn evaluate(
        &mut self,
        operation: &FusionOp,
        binding: Option<SemanticBinding>,
        values: &[Option<&Value>],
        inferred: &InferredOperation<'_>,
    ) -> Option<Value> {
        if !ComputedScalars::eligible(&inferred.fact) {
            return None;
        }
        self.work.attempted = self.work.attempted.saturating_add(1);
        let started = Instant::now();
        let result = ComputedScalars::evaluate(operation, binding, values, inferred);
        self.duration_ns = self
            .duration_ns
            .saturating_add(duration_ns_saturating(started.elapsed()));
        result
    }

    pub(super) fn record(
        &mut self,
        scalars: &mut ComputedScalars,
        id: ValueId,
        value: Option<Value>,
        fact: Option<&ValueFact>,
    ) {
        if value.is_none() {
            return;
        }
        let started = Instant::now();
        let produced = scalars.record(id, value, fact);
        self.duration_ns = self
            .duration_ns
            .saturating_add(duration_ns_saturating(started.elapsed()));
        self.work.produced = self.work.produced.saturating_add(u64::from(produced));
    }

    pub(super) fn completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for ScalarEvidenceObservation {
    fn drop(&mut self) {
        self.observer
            .scalar_evidence_preparation(self.duration_ns, self.work, self.completed);
    }
}

impl ComputedScalars {
    pub(super) fn from_plan(plan: &FusionGroupPlan) -> Self {
        let mut values = Vec::with_capacity(plan.operations.len());
        for operation in &plan.operations {
            let output = match operation {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
            };
            if let Some(id) = output {
                values.push((*id, None));
            }
        }
        values.sort_unstable_by_key(|(id, _)| *id);
        Self { values }
    }

    fn eligible(fact: &ValueFact) -> bool {
        fact.numeric().is_some_and(|output| {
            fact.shape.element_count() == Some(1)
                && output.domain == NumericDomain::Real
                && matches!(output.class, NumericClass::Single | NumericClass::Double)
                && matches!(
                    fact.storage,
                    StorageFact::Scalar | StorageFact::Dense | StorageFact::Unknown
                )
        })
    }

    pub(super) fn get(&self, id: ValueId) -> Option<&Value> {
        let index = self
            .values
            .binary_search_by_key(&id, |(slot, _)| *slot)
            .ok()?;
        self.values[index].1.as_ref()
    }

    pub(super) fn evaluate(
        operation: &FusionOp,
        binding: Option<SemanticBinding>,
        values: &[Option<&Value>],
        inferred: &InferredOperation<'_>,
    ) -> Option<Value> {
        let output = inferred.fact.numeric()?;
        if !Self::eligible(&inferred.fact) {
            return None;
        }
        if values.len() != inferred.arguments.len() {
            return None;
        }
        let instruction = match (operation, binding) {
            (_, Some(binding)) => binding.declaration.semantics?.scalar_operation?,
            (FusionOp::Primitive { op, .. }, None) => match op {
                PrimitiveOp::Neg => ScalarOperation::Operator(OperatorKind::UnaryMinus),
                PrimitiveOp::UPlus => ScalarOperation::Operator(OperatorKind::UnaryPlus),
                PrimitiveOp::Add
                | PrimitiveOp::Sub
                | PrimitiveOp::Mul
                | PrimitiveOp::ElemMul
                | PrimitiveOp::ElemDiv
                | PrimitiveOp::ElemLeftDiv => {
                    let operator = if *op == PrimitiveOp::Mul {
                        OperatorKind::ElementwiseMultiply
                    } else {
                        op.operator_kind()
                    };
                    ScalarOperation::RealArithmetic(RealFloatingRecipe::for_facts(
                        operator,
                        inferred.arguments.first()?,
                        inferred.arguments.get(1)?,
                    )?)
                }
                _ => return None,
            },
            _ => return None,
        };
        // Named plus/minus/times/rdivide/ldivide and scalar mtimes declare
        // their selected operator, while the checked scalar producer lowers
        // that operator through this same represented real recipe. Preserve
        // its mixed-class computation and result rounding here as well.
        let instruction = match instruction {
            ScalarOperation::Operator(
                operator @ (OperatorKind::Add
                | OperatorKind::Subtract
                | OperatorKind::MatrixMultiply
                | OperatorKind::ElementwiseMultiply
                | OperatorKind::ElementwiseDivide
                | OperatorKind::ElementwiseLeftDivide),
            ) => ScalarOperation::RealArithmetic(RealFloatingRecipe::for_facts(
                operator,
                inferred.arguments.first()?,
                inferred.arguments.get(1)?,
            )?),
            other => other,
        };
        if instruction.arity()? != values.len() {
            return None;
        }
        let operand = |index: usize| {
            scalar(
                values.get(index).copied().flatten(),
                inferred.arguments.get(index)?,
            )
        };
        let represented = match instruction {
            ScalarOperation::Cast(NumericClass::Single) if output.class == NumericClass::Single => {
                f64::from(operand(0)? as f32)
            }
            ScalarOperation::Cast(NumericClass::Double) if output.class == NumericClass::Double => {
                operand(0)?
            }
            ScalarOperation::RealArithmetic(recipe) if recipe.result_class() == output.class => {
                recipe.evaluate(operand(0)?, operand(1)?)?
            }
            ScalarOperation::Operator(OperatorKind::UnaryPlus)
                if inferred.arguments[0].numeric()?.class == output.class =>
            {
                operand(0)?
            }
            ScalarOperation::Operator(OperatorKind::UnaryMinus)
                if inferred.arguments[0].numeric()?.class == output.class =>
            {
                -operand(0)?
            }
            _ => return None,
        };
        if !represented.is_finite() {
            return None;
        }
        match output.class {
            NumericClass::Single => Some(Value::Tensor(
                Tensor::from_f32(vec![represented as f32], vec![1, 1]).ok()?,
            )),
            NumericClass::Double => {
                if inferred.fact.storage == StorageFact::Dense {
                    Some(Value::Tensor(
                        Tensor::new(vec![represented], vec![1, 1]).ok()?,
                    ))
                } else {
                    Some(Value::Num(represented))
                }
            }
            _ => None,
        }
    }

    pub(super) fn record(
        &mut self,
        id: ValueId,
        value: Option<Value>,
        fact: Option<&ValueFact>,
    ) -> bool {
        let (Some(value), Some(fact)) = (value, fact) else {
            return false;
        };
        let actual = runmat_runtime::value_fact::value_fact(&value);
        // Binary scalar inference uses the abstract Scalar storage fact even
        // for Single. The runtime has no scalar Single Value variant, so its
        // exact one-element F32 carrier is necessarily a dense Tensor.
        let single_scalar_carrier = fact.storage == StorageFact::Scalar
            && actual.storage == StorageFact::Dense
            && fact
                .numeric()
                .is_some_and(|numeric| numeric.class == NumericClass::Single)
            && actual.shape.element_count() == Some(1);
        if actual.numeric() != fact.numeric()
            || !actual.shape.is_proven_equivalent(&fact.shape)
            || (fact.storage != StorageFact::Unknown
                && actual.storage != fact.storage
                && !single_scalar_carrier)
        {
            return false;
        }
        let Ok(index) = self.values.binary_search_by_key(&id, |(slot, _)| *slot) else {
            return false;
        };
        self.values[index].1 = Some(value);
        true
    }
}

fn scalar(value: Option<&Value>, fact: &ValueFact) -> Option<f64> {
    let numeric = fact.numeric()?;
    if fact.shape.element_count() != Some(1) || numeric.domain != NumericDomain::Real {
        return None;
    }
    let represented = match value? {
        Value::Num(value) if numeric.class == NumericClass::Double => *value,
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor.numeric_value_at(0)? {
            NumericScalar::F32(value) if numeric.class == NumericClass::Single => f64::from(value),
            NumericScalar::F64(value) if numeric.class == NumericClass::Double => value,
            _ => return None,
        },
        _ => return None,
    };
    represented.is_finite().then_some(represented)
}
