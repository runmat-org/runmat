//! Request-local algorithmic work. Helpers are shared shader declarations, but
//! each operation executes independently and contributes its own work.
use anyhow::{anyhow, Result};
use runmat_accelerate_api::ProviderPrecision;
use runmat_runtime::builtins::common::spec::FusionMathHelper;
use runmat_types::NumericClass;

use super::{semantics::ResolvedProgram, FusionGroupPlan, FusionOp};

impl ResolvedProgram {
    pub(crate) fn scalar_work_units(&self, plan: &FusionGroupPlan, elements: u64) -> Result<u64> {
        let operations = plan.operations.iter().map(|operation| {
            let output = match operation {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
            }
            .ok_or_else(|| anyhow!("fusion work requires a resolved output"))?;
            let precision = match self.floating_class(output)? {
                NumericClass::Single => ProviderPrecision::F32,
                NumericClass::Double => ProviderPrecision::F64,
                _ => return Err(anyhow!("fusion work requires a floating class")),
            };
            let helpers = self
                .declaration(output)
                .and_then(|declaration| declaration.semantics)
                .map_or(&[][..], |semantics| semantics.helpers);
            Ok(operation_units(precision, helpers))
        });
        total_work(elements, operations)
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
