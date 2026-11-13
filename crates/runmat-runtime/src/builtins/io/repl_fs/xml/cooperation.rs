//! XML work units use the invocation's execution service for yield/cancel policy.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use runmat_builtins::{XmlDiagnosticKind, XmlOperation};

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct XmlCooperation {
    budget: Option<XmlBudgetOwner>,
    operation: XmlOperation,
}

pub(super) type XmlBudgetOwner = Rc<RefCell<Option<CpuCooperationBudget>>>;

struct BudgetReturn {
    owner: XmlBudgetOwner,
    budget: Option<CpuCooperationBudget>,
}

impl BudgetReturn {
    fn take(owner: XmlBudgetOwner) -> Self {
        let budget = owner
            .borrow_mut()
            .take()
            .expect("XML budget has one active operation");
        Self {
            owner,
            budget: Some(budget),
        }
    }

    fn budget_mut(&mut self) -> &mut CpuCooperationBudget {
        self.budget
            .as_mut()
            .expect("budget retained until guard drop")
    }

    fn budget(&self) -> &CpuCooperationBudget {
        self.budget
            .as_ref()
            .expect("budget retained until guard drop")
    }
}

impl Drop for BudgetReturn {
    fn drop(&mut self) {
        *self.owner.borrow_mut() = self.budget.take();
    }
}

impl XmlCooperation {
    pub(super) fn operation(&self) -> XmlOperation {
        self.operation
    }

    pub(super) fn is_execution_error(&self, error: &RuntimeError) -> bool {
        let mut source = std::error::Error::source(error);
        while let Some(cause) = source {
            if cause.downcast_ref::<ExecutionServiceError>().is_some() {
                return true;
            }
            source = cause.source();
        }
        false
    }
    pub(super) fn new(runtime: Option<&RuntimeContext>, operation: XmlOperation) -> Self {
        Self {
            budget: runtime.map(|runtime| Rc::new(RefCell::new(Some(runtime.cpu_cooperation())))),
            operation,
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget, operation: XmlOperation) -> Self {
        Self {
            budget: Some(Rc::new(RefCell::new(Some(budget)))),
            operation,
        }
    }

    pub(super) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        self.raw_checkpoint_future(units as u64)
            .await
            .map_err(|err| self.map_error(err))?;
        Ok(())
    }

    pub(super) fn raw_checkpoint_future(
        &self,
        units: u64,
    ) -> Pin<Box<dyn Future<Output = Result<(), ExecutionServiceError>>>> {
        let Some(owner) = self.budget.as_ref() else {
            return Box::pin(async { Ok(()) });
        };
        checkpoint_owner(owner.clone(), units)
    }

    pub(super) async fn await_io<T, E, F>(&mut self, future: F) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        // Check the invocation before polling an I/O future, including an
        // implementation that would otherwise complete in its first poll.
        self.completed(0).await?;
        match self.budget.as_ref() {
            Some(owner) => {
                let retained = BudgetReturn::take(owner.clone());
                retained
                    .budget()
                    .await_cancellable(future)
                    .await
                    .map_err(|err| self.map_error(err))
            }
            None => Ok(future.await),
        }
    }

    pub(super) fn budget_owner(&self) -> Option<XmlBudgetOwner> {
        self.budget.clone()
    }

    pub(super) fn execution_error(&self, error: ExecutionServiceError) -> RuntimeError {
        self.map_error(error)
    }

    fn map_error(&self, error: ExecutionServiceError) -> RuntimeError {
        match error {
            cancelled @ ExecutionServiceError::Cancelled => build_runtime_error(format!(
                "{}: execution was cancelled",
                self.operation.name()
            ))
            .with_identifier(self.operation.identifier(XmlDiagnosticKind::Cancelled))
            .with_builtin(self.operation.name())
            .with_source(cancelled)
            .build(),
            other => {
                let message = format!(
                    "{}: execution service failed: {other}",
                    self.operation.name()
                );
                build_runtime_error(message)
                    .with_identifier(self.operation.identifier(XmlDiagnosticKind::Execution))
                    .with_builtin(self.operation.name())
                    .with_source(other)
                    .build()
            }
        }
    }
}

pub(super) fn checkpoint_owner(
    owner: XmlBudgetOwner,
    units: u64,
) -> Pin<Box<dyn Future<Output = Result<(), ExecutionServiceError>>>> {
    let mut retained = BudgetReturn::take(owner);
    Box::pin(async move { retained.budget_mut().checkpoint(units).await })
}
