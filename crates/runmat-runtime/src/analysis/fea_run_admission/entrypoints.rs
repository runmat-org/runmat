//! Admission at synchronous public execution roots, before mesh resolution or solver effects.
use super::{FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope};
use crate::analysis::*;

pub(crate) fn run<T>(
    kind: FeaRunKind,
    identity: runmat_execution::Digest,
    operation: &str,
    version: &str,
    context: &OperationContext,
    work: impl FnOnce() -> Result<T, OperationErrorEnvelope>,
) -> Result<T, OperationErrorEnvelope> {
    run_classified(kind, identity, operation, version, context, work, |_| {
        FeaRunOutcome::Completed
    })
}

pub(crate) fn run_classified<T>(
    kind: FeaRunKind,
    identity: runmat_execution::Digest,
    operation: &str,
    version: &str,
    context: &OperationContext,
    work: impl FnOnce() -> Result<T, OperationErrorEnvelope>,
    success_outcome: impl FnOnce(&T) -> FeaRunOutcome,
) -> Result<T, OperationErrorEnvelope> {
    let allocation = super::current_fea_cpu_capacity_millicores()
        .unwrap_or_else(|| (runmat_cpu_execution::current_threads() as u64).saturating_mul(1000));
    let mut request = FeaRunRequest::local(kind, allocation, identity)
        .map_err(|denied| denied.into_operation_error(operation, version, context))?;
    request.site = FeaRunScope::admission_site()
        .map_err(|denied| denied.into_operation_error(operation, version, context))?;
    let scope = FeaRunScope::admit(request)
        .map_err(|denied| denied.into_operation_error(operation, version, context))?;
    scope
        .check()
        .map_err(|denied| denied.into_operation_error(operation, version, context))?;
    let result = scope.run_sync(work);
    let outcome = match &result {
        Ok(value) => success_outcome(value),
        Err(error)
            if error.error_type == crate::operations::OperationErrorType::Cancelled
                || crate::interrupt::is_cancelled() =>
        {
            FeaRunOutcome::Cancelled
        }
        Err(_) => FeaRunOutcome::Failed,
    };
    settle_result(scope, result, outcome, operation, version, context)
}

pub(crate) fn settle_result<T>(
    scope: FeaRunScope,
    result: Result<T, OperationErrorEnvelope>,
    outcome: FeaRunOutcome,
    operation: &str,
    version: &str,
    context: &OperationContext,
) -> Result<T, OperationErrorEnvelope> {
    let (result, outcome) = match result {
        Ok(value) => match scope.check_execution() {
            Ok(()) => (Ok(value), outcome),
            Err(failure) => {
                let stopped = failure.outcome();
                (
                    Err(failure.into_operation_error(operation, version, context)),
                    stopped,
                )
            }
        },
        Err(original) => (Err(original), outcome),
    };
    let lease_error = scope
        .settle(outcome)
        .err()
        .map(|denied| denied.into_operation_error(operation, version, context));
    merge_lease_error(result, lease_error)
}

/// Keep the domain's original typed cause. The context boundary carries one
/// complete secondary envelope, rather than concatenating diagnostic strings.
pub fn merge_lease_error<T>(
    result: Result<T, OperationErrorEnvelope>,
    lease_error: Option<OperationErrorEnvelope>,
) -> Result<T, OperationErrorEnvelope> {
    match (result, lease_error) {
        (Ok(_), Some(lease)) => Err(lease),
        (Err(mut original), Some(lease)) => {
            original
                .context
                .entry("fea_lease_error_code".into())
                .or_insert_with(|| lease.error_code.clone());
            if let Ok(encoded) = serde_json::to_string(&lease) {
                original
                    .context
                    .entry("fea_lease_error".into())
                    .or_insert(encoded);
            }
            Err(original)
        }
        (result, None) => result,
    }
}

pub(crate) fn sweep_outcome(data: &AnalysisStudySweepData) -> FeaRunOutcome {
    if data.failure_entries.iter().any(|entry| {
        matches!(
            entry.kind,
            AnalysisStudySweepFailureKind::Failed | AnalysisStudySweepFailureKind::Indeterminate
        )
    }) {
        FeaRunOutcome::Failed
    } else if data
        .failure_entries
        .iter()
        .any(|entry| entry.kind == AnalysisStudySweepFailureKind::Cancelled)
    {
        FeaRunOutcome::Cancelled
    } else if data.failed_count > 0 || data.success_count != data.study_count {
        FeaRunOutcome::Failed
    } else {
        FeaRunOutcome::Completed
    }
}

pub(in crate::analysis) fn run_model<T>(
    model: &AnalysisModel,
    operation: AnalysisOperation,
    context: &OperationContext,
    work: impl FnOnce() -> Result<T, OperationErrorEnvelope>,
) -> Result<T, OperationErrorEnvelope> {
    // This is a pure trust boundary, not solver execution. Reuse the owning
    // domain validators before encoding identity or requesting a paid lease;
    // malformed physical quantities are not an account/license defect.
    if matches!(operation, AnalysisOperation::Thermal) {
        crate::analysis::validate_thermal_run_model(model, context)?;
    }
    runmat_analysis_core::validate_model(model).map_err(|error| {
        crate::analysis::map_fea_run_error(
            operation,
            model,
            context,
            runmat_analysis_fea::FeaRunError::InvalidModel(error.to_string()),
        )
    })?;
    // Model serialization is canonical for these typed, ordered contracts.
    let bytes = serde_json::to_vec(model).map_err(|error| {
        crate::analysis::map_fea_run_error(
            operation,
            model,
            context,
            runmat_analysis_fea::FeaRunError::InvalidModel(error.to_string()),
        )
    })?;
    run(
        FeaRunKind::Solve,
        runmat_execution::Digest::sha256(bytes),
        operation.name(),
        operation.version(),
        context,
        work,
    )
}

macro_rules! standalone {
    ($name:ident, $options:ty, $operation:ident, $owner:ident) => {
        pub fn $name(
            model: &AnalysisModel,
            backend: ComputeBackend,
            options: $options,
            context: OperationContext,
        ) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
            run_model(model, AnalysisOperation::$operation, &context, || {
                crate::analysis::$owner::$name(model, backend, options, context.clone())
            })
        }
    };
}

standalone!(
    analysis_run_modal_with_options_op,
    AnalysisModalRunOptions,
    Modal,
    standalone_publication
);
standalone!(
    analysis_run_acoustic_with_options_op,
    AnalysisAcousticRunOptions,
    Acoustic,
    standalone_publication
);
standalone!(
    analysis_run_transient_with_options_op,
    AnalysisTransientRunOptions,
    Transient,
    standalone_publication
);
standalone!(
    analysis_run_nonlinear_with_options_op,
    AnalysisNonlinearRunOptions,
    Nonlinear,
    standalone_publication
);
standalone!(
    analysis_run_thermal_with_options_op,
    AnalysisThermalRunOptions,
    Thermal,
    thermal_execution
);

/// Explicit host fixture for numerical tests. Production admission still fails closed;
/// tests must install this factory's result under their shared host-state lock.
#[cfg(test)]
pub fn test_admission() -> std::sync::Arc<dyn super::FeaRunAdmission> {
    struct TestAdmission;
    struct TestPermit {
        budget: super::FeaRunBudget,
    }
    impl super::FeaRunAdmission for TestAdmission {
        fn admit(
            &self,
            _: &FeaRunRequest,
        ) -> Result<Box<dyn super::FeaRunPermit>, super::FeaRunDenied> {
            Ok(Box::new(TestPermit {
                budget: super::FeaRunBudget {
                    budget_ops: runmat_simulation_usage::OperationCount::from_operations(u64::MAX),
                    granted_ops: runmat_simulation_usage::OperationCount::from_operations(u64::MAX),
                    expires_at_unix_millis: (runmat_time::unix_timestamp_ms() as u64)
                        .saturating_add(3_600_000),
                    heartbeat_millis: 30_000,
                },
            }))
        }
    }
    impl super::FeaRunPermit for TestPermit {
        fn budget(&self) -> super::FeaRunBudget {
            self.budget
        }
        fn heartbeat(
            &self,
            _: &super::FeaRunUsage,
        ) -> Result<super::FeaRunBudget, super::FeaRunDenied> {
            Ok(self.budget())
        }
        fn settle(
            self: Box<Self>,
            _: FeaRunOutcome,
            _: &super::FeaRunUsage,
        ) -> Result<(), super::FeaRunDenied> {
            Ok(())
        }
    }
    std::sync::Arc::new(TestAdmission)
}
