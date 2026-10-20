//! One typed Cell traversal for integer and logical extension roles.

use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct NumericRoles {
    pub(super) integer: bool,
    pub(super) logical: bool,
}

struct NumericRoleScanner<'a> {
    pending: Vec<std::slice::Iter<'a, Value>>,
    roles: NumericRoles,
}

impl<'a> NumericRoleScanner<'a> {
    fn new(value: &'a Value) -> Self {
        Self {
            pending: vec![std::slice::from_ref(value).iter()],
            roles: NumericRoles::default(),
        }
    }

    fn step(&mut self) -> BuiltinResult<bool> {
        if self.roles.integer && self.roles.logical {
            return Ok(false);
        }
        while let Some(frame) = self.pending.last_mut() {
            let Some(value) = frame.next() else {
                self.pending.pop();
                continue;
            };
            self.roles.integer |= value_is_integer_numeric(value);
            self.roles.logical |= value_is_logical_numeric(value);
            if let Value::Cell(cell) = value {
                self.pending.try_reserve(1).map_err(|error| {
                    allocation_failure(
                        "boxplot: extension metadata exceeds available memory",
                        error,
                    )
                })?;
                self.pending.push(cell.data().iter());
            }
            return Ok(true);
        }
        Ok(false)
    }
}

pub(super) fn numeric_roles(value: &Value) -> BuiltinResult<NumericRoles> {
    let mut scanner = NumericRoleScanner::new(value);
    while scanner.step()? {}
    Ok(scanner.roles)
}

pub(super) async fn numeric_roles_cooperative(
    value: &Value,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<NumericRoles> {
    cooperation.completed(0).await?;
    let mut scanner = NumericRoleScanner::new(value);
    while scanner.step()? {
        cooperation.completed(1).await?;
    }
    Ok(scanner.roles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::{CellArray, IntValue, LogicalArray};
    use std::num::NonZeroU64;
    use std::rc::Rc;

    #[test]
    fn one_typed_scanner_agrees_across_nested_sync_and_cooperative_drivers() {
        let group = Value::Cell(
            CellArray::new(
                vec![
                    Value::Cell(
                        CellArray::new(vec![Value::Int(IntValue::U64(u64::MAX))], 1, 1).unwrap(),
                    ),
                    Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let sync = numeric_roles(&group).unwrap();
        let mut cooperation = super::super::cooperation::BoxplotCooperation::new(None);
        let asynchronous =
            futures::executor::block_on(numeric_roles_cooperative(&group, &mut cooperation))
                .unwrap();
        assert!(sync.integer && sync.logical);
        assert_eq!(sync.integer, asynchronous.integer);
        assert_eq!(sync.logical, asynchronous.logical);
    }

    #[test]
    fn nested_extension_scan_cancels_before_finishing_a_large_host_cell() {
        let group = Value::Cell(CellArray::new(vec![Value::Bool(false); 1024], 1, 1024).unwrap());
        let source = Rc::new(CancellationSource::new());
        let cancel_on_turn = source.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                cancel_on_turn.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = super::super::cooperation::BoxplotCooperation::with_budget(budget);
        let error =
            futures::executor::block_on(numeric_roles_cooperative(&group, &mut cooperation))
                .expect_err("cancelled extension scan cannot traverse the rest of the cell");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    }
}
