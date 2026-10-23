//! Original factor receipt fenced at the canonical OperatorWork boundaries.
use super::{
    DenseFactorError, HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
    RuntimeContext,
};
use crate::builtins::math::linalg::ops::linear_operator::{CooperativeOperatorError, OperatorWork};
impl From<CooperativeOperatorError> for DenseFactorError {
    fn from(error: CooperativeOperatorError) -> Self {
        match error {
            CooperativeOperatorError::Execution(e) => Self::Execution(e),
            CooperativeOperatorError::Operator(
                crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError::Admission(
                    e,
                ),
            ) => Self::Admission(e),
            CooperativeOperatorError::Operator(e) => Self::Operator(e),
        }
    }
}
pub(super) struct Work<'a> {
    runtime: Option<&'a RuntimeContext>,
    receipt: Option<&'a HostAllocationReceipt>,
    operator: Option<OperatorWork<'a>>,
}
impl<'a> Work<'a> {
    pub fn new(
        runtime: Option<&'a RuntimeContext>,
        receipt: Option<&'a HostAllocationReceipt>,
    ) -> Result<Self, DenseFactorError> {
        let result = Self {
            runtime,
            receipt,
            operator: runtime.map(OperatorWork::new).transpose()?,
        };
        result.check()?;
        Ok(result)
    }
    fn check(&self) -> Result<(), DenseFactorError> {
        match (self.runtime, self.receipt) {
            (Some(runtime), Some(receipt)) => receipt.checkpoint(runtime)?,
            (_, None) => {
                if HostAllocationReceipt::policy(self.runtime)? != HostAllocationPolicy::Unbounded {
                    return Err(HostAllocationError::StaleAuthority.into());
                }
            }
            _ => return Err(HostAllocationError::MissingManagedOwner.into()),
        }
        Ok(())
    }
    pub async fn tick(&mut self) -> Result<(), DenseFactorError> {
        let boundary = self
            .operator
            .as_ref()
            .is_some_and(|work| work.capacity() == 1);
        if boundary {
            self.check()?;
        }
        if let Some(work) = self.operator.as_mut() {
            work.tick().await?;
        }
        if boundary {
            self.check()?;
        }
        Ok(())
    }
    pub async fn flush(&mut self) -> Result<(), DenseFactorError> {
        self.check()?;
        if let Some(work) = self.operator.as_mut() {
            work.flush().await?;
        }
        self.check()
    }
}
