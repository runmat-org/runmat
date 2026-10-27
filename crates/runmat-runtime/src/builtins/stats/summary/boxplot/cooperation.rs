//! Invocation-owned cooperation for potentially large boxplot factor products.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{BuiltinResult, RuntimeError};

use super::{internal_with_source, NAME};

pub(super) struct BoxplotCooperation {
    budget: Option<CpuCooperationBudget>,
    runtime: Option<RuntimeContext>,
}

impl BoxplotCooperation {
    pub(super) fn new(runtime: Option<&RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(|runtime| {
                CpuCooperationBudget::new(runtime.execution().clone(), runtime.cancellation())
            }),
            runtime: runtime.cloned(),
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget) -> Self {
        Self {
            budget: Some(budget),
            runtime: None,
        }
    }

    pub(super) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        let result = budget.checkpoint(units).await;
        result.map_err(|error| self.service_error(error))
    }

    pub(super) fn budget(&mut self) -> Option<&mut CpuCooperationBudget> {
        self.budget.as_mut()
    }

    pub(super) fn service_error(&self, error: ExecutionServiceError) -> RuntimeError {
        match error {
            ExecutionServiceError::Cancelled => {
                let mut error = ExecutionServiceError::Cancelled
                    .into_runtime_error_in_context(self.runtime.as_ref());
                error.context = error.context.with_builtin(NAME);
                error
            }
            other => internal_with_source(
                "boxplot: execution service failed during preparation",
                other.into_runtime_error(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::CpuCooperationPolicy;
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::{CellArray, StringArray, Tensor, Value};
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    #[test]
    fn active_expansion_retains_original_invocation_token() {
        let original = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            original.token(),
        );
        runtime.set_error_namespace("BoxplotInvocation");
        let mut cooperation = BoxplotCooperation::new(Some(&runtime));
        let replacement = CancellationSource::new();
        runtime.replace_cancellation_token(replacement.token());
        replacement.cancel(CancellationReason::User);
        futures::executor::block_on(cooperation.completed(0))
            .expect("rebinding the ambient token must not cancel existing work");
        original.cancel(CancellationReason::User);
        let error = futures::executor::block_on(cooperation.completed(0))
            .expect_err("the invocation token remains authoritative");
        assert_eq!(error.identifier(), Some("BoxplotInvocation:Cancelled"));
        assert_eq!(error.context.builtin.as_deref(), Some(NAME));
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
        assert!(std::error::Error::source(&error)
            .is_some_and(|source| source.downcast_ref::<ExecutionServiceError>().is_some()));
    }

    #[test]
    fn registered_boxplot_rejects_cancelled_invocation_before_plot_effects() {
        let source = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        source.cancel(CancellationReason::User);
        let error = block_on(super::super::boxplot_builtin_registered(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                Value::from("FullFactors"),
                Value::from("on"),
            ]),
        ))
        .expect_err("registered invocation uses its own cancellation token");
        assert!(error
            .identifier()
            .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
    }

    #[test]
    fn full_factor_expansion_charges_turns_and_stops_before_publication() {
        let factors = || {
            let first = Value::StringArray(
                StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
            );
            let second = Value::StringArray(
                StringArray::new(vec!["one".into(), "two".into()], vec![2, 1]).unwrap(),
            );
            Value::Cell(CellArray::new(vec![first, second], 1, 2).unwrap())
        };
        let data = || Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap());
        let options = super::super::BoxplotOptions {
            full_factors: true,
            ..Default::default()
        };
        let source = CancellationSource::new();
        let turns = Rc::new(Cell::new(0));
        let counted = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                counted.set(counted.get() + 1);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = BoxplotCooperation::with_budget(budget);
        let layout = block_on(super::super::groups::build_box_layout_cooperative(
            data(),
            Some(factors()),
            &options,
            &mut cooperation,
        ))
        .unwrap();
        assert_eq!(layout.slots.len(), 4);
        assert!(turns.get() >= layout.slots.len());
        let turns_after_layout = turns.get();
        block_on(super::super::plot::prepare_box_layout_cooperative(
            &layout,
            &options,
            &mut cooperation,
        ))
        .expect("plot-plan preparation also cooperates before effects");
        assert!(turns.get() > turns_after_layout);

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
        let mut cooperation = BoxplotCooperation::with_budget(budget);
        let error = block_on(super::super::groups::build_box_layout_cooperative(
            data(),
            Some(factors()),
            &options,
            &mut cooperation,
        ))
        .expect_err("cancelled expansion cannot publish geometry");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
        assert!(std::error::Error::source(&error)
            .is_some_and(|source| source.downcast_ref::<ExecutionServiceError>().is_some()));

        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || Box::pin(async { Err(ExecutionServiceError::Infrastructure("turn failed".into())) }),
        );
        let mut cooperation = BoxplotCooperation::with_budget(budget);
        let error = block_on(super::super::groups::build_box_layout_cooperative(
            data(),
            Some(factors()),
            &options,
            &mut cooperation,
        ))
        .expect_err("service failure retains its original cause");
        assert_eq!(
            error.identifier(),
            super::super::BOXPLOT_ERROR_INTERNAL.identifier
        );
        assert!(error.source.is_some());
    }
}
