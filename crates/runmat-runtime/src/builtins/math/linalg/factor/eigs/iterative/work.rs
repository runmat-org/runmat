//! Original receipt plus the canonical bounded operator-work quantum.
use crate::{
    builtins::math::linalg::ops::linear_operator::OperatorWork,
    context::{host_allocation::HostAllocationReceipt, RuntimeContext},
    BuiltinResult,
};
pub(super) struct Work<'a> {
    pub runtime: Option<&'a RuntimeContext>,
    pub receipt: Option<&'a HostAllocationReceipt>,
    operator: Option<OperatorWork<'a>>,
}
impl<'a> Work<'a> {
    #[cfg(test)]
    pub fn with_test_budget(
        runtime: &'a RuntimeContext,
        receipt: &'a HostAllocationReceipt,
        budget: crate::execution::CpuCooperationBudget,
        quantum: std::num::NonZeroU64,
    ) -> BuiltinResult<Self> {
        let operator = OperatorWork::with_test_budget(runtime, budget, quantum)
            .map_err(|e| super::operator_error(Some(runtime), e))?;
        let result = Self {
            runtime: Some(runtime),
            receipt: Some(receipt),
            operator: Some(operator),
        };
        result.check()?;
        Ok(result)
    }
    pub fn new(
        runtime: Option<&'a RuntimeContext>,
        receipt: Option<&'a HostAllocationReceipt>,
    ) -> BuiltinResult<Self> {
        let operator = runtime
            .map(OperatorWork::new)
            .transpose()
            .map_err(|e| super::operator_error(runtime, e))?;
        let result = Self {
            runtime,
            receipt,
            operator,
        };
        result.check()?;
        Ok(result)
    }
    pub fn check(&self) -> BuiltinResult<()> {
        match (self.runtime, self.receipt) {
            (Some(runtime), Some(receipt)) => receipt
                .checkpoint(runtime)
                .map_err(|e| super::admission(self.runtime, e)),
            _ => crate::builtins::math::linalg::factor::iterative::copy::unbounded_checkpoint(
                self.runtime,
                "eigs",
            ),
        }
    }
    pub async fn tick(&mut self) -> BuiltinResult<()> {
        let boundary = self
            .operator
            .as_ref()
            .is_some_and(|work| work.capacity() == 1);
        if boundary {
            self.check()?;
        }
        if let Some(work) = self.operator.as_mut() {
            work.tick()
                .await
                .map_err(|e| super::operator_error(self.runtime, e))?;
        }
        if boundary {
            self.check()?;
        }
        Ok(())
    }
    pub async fn flush(&mut self) -> BuiltinResult<()> {
        self.check()?;
        if let Some(work) = self.operator.as_mut() {
            work.flush()
                .await
                .map_err(|e| super::operator_error(self.runtime, e))?;
        }
        self.check()
    }
}
