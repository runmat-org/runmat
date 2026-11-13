//! Request-local algorithmic work. Helpers are shared shader declarations, but
//! each operation executes independently and contributes its own work.
use anyhow::{anyhow, Result};
use runmat_accelerate_api::ProviderPrecision;
use runmat_runtime::builtins::common::spec::FusionMathHelper;
use runmat_types::{rules::RealFloatingRecipe, NumericClass, ScalarOperation};

use super::{semantics::ResolvedProgram, FusionGroupPlan, FusionOp};

mod implementation;

impl ResolvedProgram {
    /// Ordinary bytecode materializes each operation at its own result shape;
    /// no final-shape broadcast or GPU scalar program is needed to price it.
    pub(crate) fn host_scalar_work_units(&self, plan: &FusionGroupPlan) -> Result<u64> {
        total_work(
            1,
            plan.operations.iter().map(|operation| {
                let output = match operation {
                    FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
                }
                .ok_or_else(|| anyhow!("host work requires a resolved output"))?;
                let elements = self
                    .fact(output)?
                    .shape
                    .element_count()
                    .and_then(|n| u64::try_from(n).ok())
                    .ok_or_else(|| anyhow!("host work requires a bounded output shape"))?;
                elements
                    .checked_mul(self.operation_work_units(operation)?)
                    .ok_or_else(|| anyhow!("host scalar work overflows"))
            }),
        )
    }

    pub(crate) fn scalar_work_units(&self, plan: &FusionGroupPlan, elements: u64) -> Result<u64> {
        let operations = plan
            .operations
            .iter()
            .map(|operation| self.operation_work_units(operation));
        total_work(elements, operations)
    }

    /// Price the physical implementations actually emitted by the selected
    /// lowerer. Host execution continues to use its original semantic facts.
    /// Elementwise coverage validation and reduction source mapping supply a
    /// complete emitted operation plan here, not an unsliced source graph.
    pub(crate) fn physical_scalar_work_units(
        &self,
        plan: &FusionGroupPlan,
        elements: u64,
        emitted: &[runmat_accelerate_api::scalar_program::EmittedScalarImplementation],
    ) -> Result<u64> {
        if emitted.is_empty() {
            return self.scalar_work_units(plan, elements);
        }
        implementation::work_units(self, plan, elements, emitted)
    }

    /// Consume semantic facts, not GPU implementation admission: a valid host
    /// candidate must remain priceable when no shader can execute its recipe.
    fn operation_work_units(&self, operation: &FusionOp) -> Result<u64> {
        let (inputs, output) = match operation {
            FusionOp::Primitive { inputs, output, .. }
            | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
        };
        let output = output.ok_or_else(|| anyhow!("fusion work requires a resolved output"))?;
        let result_class = self.floating_class(output)?;
        let semantics = self.declaration(output).and_then(|entry| entry.semantics);
        let operator = match operation {
            FusionOp::Primitive { op, .. } => Some(op.operator_kind()),
            FusionOp::Builtin { .. } => match semantics.and_then(|entry| entry.scalar_operation) {
                Some(ScalarOperation::Operator(operator)) => Some(operator),
                _ => None,
            },
        };
        let recipe = match (operator, inputs.as_slice()) {
            (Some(operator), [left, right]) => {
                RealFloatingRecipe::for_facts(operator, self.fact(*left)?, self.fact(*right)?)
            }
            _ => None,
        };
        let mut computation_class = result_class;
        let mut conversions = 0;
        if let Some(recipe) = recipe {
            if recipe.result_class() != result_class {
                return Err(anyhow!(
                    "arithmetic work recipe disagrees with result class"
                ));
            }
            computation_class = recipe.computation_class();
            for &input in inputs {
                conversions += u64::from(self.floating_class(input)? != computation_class);
            }
            conversions += u64::from(computation_class != result_class);
        }
        let precision = match computation_class {
            NumericClass::Single => ProviderPrecision::F32,
            NumericClass::Double => ProviderPrecision::F64,
            _ => return Err(anyhow!("fusion work requires a floating computation class")),
        };
        operation_units(precision, semantics.map_or(&[], |entry| entry.helpers))
            .checked_add(conversions)
            .ok_or_else(|| anyhow!("arithmetic conversion work overflows"))
    }
}

fn operation_units(precision: ProviderPrecision, helpers: &[FusionMathHelper]) -> u64 {
    // Preserve the existing unit-operation prior for ordinary arithmetic,
    // RealPower and Round. It is not a full instruction bound for those forms.
    // A remainder operation's owning bound already includes its full body.
    if helpers.contains(&FusionMathHelper::FloatingRemainder) {
        crate::kernel_math::floating_remainder::scalar_work_units(precision)
    } else {
        1
    }
}

fn total_work(elements: u64, operations: impl IntoIterator<Item = Result<u64>>) -> Result<u64> {
    operations
        .into_iter()
        .try_fold(0_u64, |sum, units| {
            sum.checked_add(units?)
                .ok_or_else(|| anyhow!("fusion scalar work sum overflow"))
        })?
        .checked_mul(elements)
        .ok_or_else(|| anyhow!("fusion scalar work product overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_remainders_count_per_operation_at_resolved_precision() {
        let single = operation_units(
            ProviderPrecision::F32,
            &[FusionMathHelper::FloatingRemainder],
        );
        let double = operation_units(
            ProviderPrecision::F64,
            &[FusionMathHelper::FloatingRemainder],
        );
        assert!(double > single);
        assert_eq!(
            total_work(7, [Ok(single), Ok(single), Ok(double), Ok(1)]).unwrap(),
            7 * (2 * single + double + 1)
        );
        assert_eq!(
            operation_units(ProviderPrecision::F64, &[FusionMathHelper::RealPower]),
            1
        );
        assert_eq!(
            operation_units(ProviderPrecision::F64, &[FusionMathHelper::Round]),
            1
        );
        assert_eq!(operation_units(ProviderPrecision::F32, &[]), 1);
    }

    #[test]
    fn missing_and_overflowed_work_never_become_zero_cost() {
        assert!(total_work(1, [Err(anyhow!("missing operation"))]).is_err());
        assert!(total_work(1, [Ok(u64::MAX), Ok(1)]).is_err());
        assert!(total_work(2, [Ok(u64::MAX)]).is_err());
        assert_eq!(total_work(0, [Ok(3)]).unwrap(), 0);
    }
}
