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
    let allocation = (runmat_cpu_execution::current_threads() as u64).saturating_mul(1000);
    let request = FeaRunRequest::local(kind, allocation, identity)
        .map_err(|denied| denied.into_operation_error(operation, version, context))?;
    let scope = FeaRunScope::admit(request)
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
    scope.settle(outcome);
    result
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
    // Model serialization is canonical for these typed, ordered contracts.
    let bytes = serde_json::to_vec(model).map_err(|_| {
        super::FeaRunDenied::LicenseInvalid.into_operation_error(
            operation.name(),
            operation.version(),
            context,
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
    struct TestPermit;
    impl super::FeaRunAdmission for TestAdmission {
        fn admit(
            &self,
            _: &FeaRunRequest,
        ) -> Result<Box<dyn super::FeaRunPermit>, super::FeaRunDenied> {
            Ok(Box::new(TestPermit))
        }
    }
    impl super::FeaRunPermit for TestPermit {
        fn settle(
            self: Box<Self>,
            _: FeaRunOutcome,
            _: &runmat_execution::resource::ResourceUsage,
        ) {
        }
    }
    std::sync::Arc::new(TestAdmission)
}
