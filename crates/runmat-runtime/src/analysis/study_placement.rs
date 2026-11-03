use std::future::Future;
use std::pin::Pin;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{
    analysis_complete_structural_study_op, analysis_complete_study_sweep_op,
    analysis_prepare_structural_study_op, analysis_study_sweep_identity, fea_run_admission,
    AnalysisStudyRunData, AnalysisStudySpec, AnalysisStudySweepData, AnalysisStudySweepRunOutcome,
    AnalysisStudySweepSpec, PreparedStructuralStudy, StructuralStudyExecutionFailure,
    StructuralStudyExecutionOutcome, ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION,
    ANALYSIS_RUN_STUDY_SWEEP_OPERATION, ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
};

pub type StructuralStudyPlacementFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<StructuralStudyPlacementResult>, String>> + Send + 'a>>;

/// One runtime-prepared study exposed to a placement adapter.
///
/// The adapter may inspect the immutable study and path-free numerical inputs, but the runtime
/// retains ownership of the prepared value so only it can admit and publish a returned result.
pub struct StructuralStudyPlacementEntry<'a> {
    pub study_index: usize,
    pub study: &'a AnalysisStudySpec,
    pub prepared: &'a PreparedStructuralStudy,
}

#[derive(Debug, Clone)]
pub struct StructuralStudyPlacementResult {
    pub study_index: usize,
    pub outcome: StructuralStudyExecutionOutcome,
}

/// Placement-only execution boundary for already validated structural studies.
///
/// Implementations choose process, durable, or remote placement. They do not validate authored
/// semantics, apply sweep policy, persist runs, or reorder results for publication.
pub trait StructuralStudyPlacementExecutor: Send + Sync {
    /// Aggregate CPU capacity reserved by this placement, not per-task CPU time.
    fn allocated_cpu_millicores(&self, study_count: usize) -> u32;

    fn run_site(&self) -> super::fea_run_admission::FeaRunSite {
        super::fea_run_admission::FeaRunScope::admission_site()
            .unwrap_or(super::fea_run_admission::FeaRunSite::Local)
    }

    fn execute<'a>(
        &'a self,
        entries: &'a [StructuralStudyPlacementEntry<'a>],
    ) -> StructuralStudyPlacementFuture<'a>;
}

pub async fn analysis_execute_structural_study_with_placement(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    executor: &dyn StructuralStudyPlacementExecutor,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let scope = admit_placement(
        super::analysis_study_content_digest(spec),
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &context,
        executor,
        1,
    )?;
    let result = match scope
        .run_checked_result_future(execute_study_admitted(spec, context.clone(), executor))
        .await
    {
        Ok(result) => result,
        Err(failure) => {
            let classification = failure.outcome();
            return fea_run_admission::entrypoints::settle_result(
                scope,
                Err(failure.into_operation_error(
                    ANALYSIS_RUN_STUDY_OPERATION,
                    ANALYSIS_RUN_STUDY_OP_VERSION,
                    &context,
                )),
                classification,
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &context,
            );
        }
    };
    let outcome = placement_outcome(&result);
    super::fea_run_admission::entrypoints::settle_result(
        scope,
        result,
        outcome,
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &context,
    )
}

async fn execute_study_admitted(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    executor: &dyn StructuralStudyPlacementExecutor,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let prepared = analysis_prepare_structural_study_op(spec, context.clone())?;
    let entries = [StructuralStudyPlacementEntry {
        study_index: 0,
        study: spec,
        prepared: &prepared,
    }];
    let mut results = executor
        .execute(&entries)
        .await
        .map_err(|message| placement_error(spec, &context, message))?;
    if results.len() != 1 || results[0].study_index != 0 {
        return Err(placement_error(
            spec,
            &context,
            "structural study placement returned an invalid result inventory",
        ));
    }
    analysis_complete_structural_study_op(prepared, results.remove(0).outcome)
}

pub async fn analysis_execute_structural_sweep_with_placement(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
    executor: &dyn StructuralStudyPlacementExecutor,
) -> Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope> {
    let identity = analysis_study_sweep_identity(spec, &context)?;
    let scope = admit_placement(
        identity.digest(),
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        &context,
        executor,
        if spec.fail_fast {
            1
        } else {
            spec.studies.len()
        },
    )?;
    let result = match scope
        .run_checked_result_future(execute_sweep_admitted(spec, context.clone(), executor))
        .await
    {
        Ok(result) => result,
        Err(failure) => {
            let classification = failure.outcome();
            return fea_run_admission::entrypoints::settle_result(
                scope,
                Err(failure.into_operation_error(
                    ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                    ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                    &context,
                )),
                classification,
                ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                &context,
            );
        }
    };
    let outcome = match &result {
        Ok(value) => super::fea_run_admission::entrypoints::sweep_outcome(&value.data),
        Err(_) => placement_outcome(&result),
    };
    super::fea_run_admission::entrypoints::settle_result(
        scope,
        result,
        outcome,
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        &context,
    )
}

async fn execute_sweep_admitted(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
    executor: &dyn StructuralStudyPlacementExecutor,
) -> Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope> {
    analysis_study_sweep_identity(spec, &context)?;
    let mut prepared = spec
        .studies
        .iter()
        .map(
            |study| match analysis_prepare_structural_study_op(study, context.clone()) {
                Ok(prepared) => (Some(prepared), None),
                Err(error) => (
                    None,
                    Some(
                        if error.error_type == OperationErrorType::Cancelled
                            || crate::interrupt::is_cancelled()
                        {
                            AnalysisStudySweepRunOutcome::Cancelled(error)
                        } else {
                            AnalysisStudySweepRunOutcome::Failed(error)
                        },
                    ),
                ),
            },
        )
        .collect::<Vec<_>>();

    if !(spec.fail_fast && prepared.iter().any(|(_, outcome)| outcome.is_some())) {
        if spec.fail_fast {
            execute_fail_fast(spec, &mut prepared, context.clone(), executor).await?;
        } else {
            let entries = prepared
                .iter()
                .enumerate()
                .filter_map(|(study_index, (prepared, _))| {
                    prepared
                        .as_ref()
                        .map(|prepared| StructuralStudyPlacementEntry {
                            study_index,
                            study: &spec.studies[study_index],
                            prepared,
                        })
                })
                .collect::<Vec<_>>();
            if !entries.is_empty() {
                let results = executor
                    .execute(&entries)
                    .await
                    .map_err(|message| sweep_placement_error(spec, &context, message))?;
                apply_placement_results(spec, &mut prepared, results, &context)?;
            }
        }
    }

    let outcomes = prepared
        .into_iter()
        .map(|(_, outcome)| outcome.unwrap_or(AnalysisStudySweepRunOutcome::NotRun))
        .collect();
    analysis_complete_study_sweep_op(spec, context, outcomes)
}

fn admit_placement(
    identity: runmat_execution::Digest,
    operation: &str,
    version: &str,
    context: &OperationContext,
    executor: &dyn StructuralStudyPlacementExecutor,
    study_count: usize,
) -> Result<super::fea_run_admission::FeaRunScope, OperationErrorEnvelope> {
    use super::fea_run_admission::{FeaRunKind, FeaRunRequest, FeaRunScope};
    let mut request = FeaRunRequest::local(
        FeaRunKind::MeshAndSolve,
        u64::from(executor.allocated_cpu_millicores(study_count)),
        identity,
    )
    .map_err(|e| e.into_operation_error(operation, version, context))?;
    request.site = executor.run_site();
    FeaRunScope::admit(request).map_err(|e| e.into_operation_error(operation, version, context))
}

fn placement_outcome<T>(
    result: &Result<T, OperationErrorEnvelope>,
) -> super::fea_run_admission::FeaRunOutcome {
    use super::fea_run_admission::FeaRunOutcome;
    match result {
        Ok(_) => FeaRunOutcome::Completed,
        Err(error)
            if error.error_type == OperationErrorType::Cancelled
                || crate::interrupt::is_cancelled() =>
        {
            FeaRunOutcome::Cancelled
        }
        Err(_) => FeaRunOutcome::Failed,
    }
}

async fn execute_fail_fast(
    spec: &AnalysisStudySweepSpec,
    prepared: &mut [(
        Option<PreparedStructuralStudy>,
        Option<AnalysisStudySweepRunOutcome>,
    )],
    context: OperationContext,
    executor: &dyn StructuralStudyPlacementExecutor,
) -> Result<(), OperationErrorEnvelope> {
    for study_index in 0..prepared.len() {
        let mut results = {
            let Some(prepared_study) = prepared[study_index].0.as_ref() else {
                break;
            };
            let entries = [StructuralStudyPlacementEntry {
                study_index,
                study: &spec.studies[study_index],
                prepared: prepared_study,
            }];
            executor
                .execute(&entries)
                .await
                .map_err(|message| sweep_placement_error(spec, &context, message))?
        };
        if results.len() != 1 || results[0].study_index != study_index {
            return Err(sweep_placement_error(
                spec,
                &context,
                "fail-fast structural sweep placement returned an invalid result inventory",
            ));
        }
        let terminal = terminal_kind(&results[0].outcome);
        apply_placement_result(spec, prepared, results.remove(0), &context)?;
        if !matches!(terminal, PlacementTerminalKind::Completed) {
            break;
        }
    }
    Ok(())
}

fn apply_placement_results(
    spec: &AnalysisStudySweepSpec,
    prepared: &mut [(
        Option<PreparedStructuralStudy>,
        Option<AnalysisStudySweepRunOutcome>,
    )],
    results: Vec<StructuralStudyPlacementResult>,
    context: &OperationContext,
) -> Result<(), OperationErrorEnvelope> {
    let expected = prepared.iter().filter(|(entry, _)| entry.is_some()).count();
    if results.len() != expected {
        return Err(sweep_placement_error(
            spec,
            context,
            "structural sweep placement returned the wrong result count",
        ));
    }
    for result in results {
        apply_placement_result(spec, prepared, result, context)?;
    }
    if prepared
        .iter()
        .any(|(entry, outcome)| entry.is_some() && outcome.is_none())
    {
        return Err(sweep_placement_error(
            spec,
            context,
            "structural sweep placement omitted a prepared study",
        ));
    }
    Ok(())
}

fn apply_placement_result(
    spec: &AnalysisStudySweepSpec,
    prepared: &mut [(
        Option<PreparedStructuralStudy>,
        Option<AnalysisStudySweepRunOutcome>,
    )],
    result: StructuralStudyPlacementResult,
    context: &OperationContext,
) -> Result<(), OperationErrorEnvelope> {
    let slot = prepared.get_mut(result.study_index).ok_or_else(|| {
        sweep_placement_error(
            spec,
            context,
            "structural sweep placement returned an out-of-range study index",
        )
    })?;
    if slot.1.is_some() {
        return Err(sweep_placement_error(
            spec,
            context,
            "structural sweep placement returned a duplicate or unrequested study index",
        ));
    }
    let prepared_study = slot.0.take().ok_or_else(|| {
        sweep_placement_error(
            spec,
            context,
            "structural sweep placement returned an unprepared study index",
        )
    })?;
    let terminal_kind = terminal_kind(&result.outcome);
    let completed = analysis_complete_structural_study_op(prepared_study, result.outcome);
    slot.1 = Some(match (terminal_kind, completed) {
        (_, Ok(envelope)) => AnalysisStudySweepRunOutcome::Succeeded(Box::new(envelope)),
        (PlacementTerminalKind::Failed, Err(error)) => AnalysisStudySweepRunOutcome::Failed(error),
        (PlacementTerminalKind::Cancelled, Err(error)) => {
            AnalysisStudySweepRunOutcome::Cancelled(error)
        }
        (PlacementTerminalKind::Indeterminate, Err(error)) => {
            AnalysisStudySweepRunOutcome::Indeterminate(error)
        }
        (PlacementTerminalKind::Completed, Err(error)) => {
            AnalysisStudySweepRunOutcome::Failed(error)
        }
    });
    Ok(())
}

#[derive(Clone, Copy)]
enum PlacementTerminalKind {
    Completed,
    Failed,
    Cancelled,
    Indeterminate,
}

fn terminal_kind(outcome: &StructuralStudyExecutionOutcome) -> PlacementTerminalKind {
    match outcome {
        Ok(_) => PlacementTerminalKind::Completed,
        Err(StructuralStudyExecutionFailure::Failed(_)) => PlacementTerminalKind::Failed,
        Err(StructuralStudyExecutionFailure::Cancelled) => PlacementTerminalKind::Cancelled,
        Err(StructuralStudyExecutionFailure::Indeterminate(_)) => {
            PlacementTerminalKind::Indeterminate
        }
    }
}

fn placement_error(
    spec: &AnalysisStudySpec,
    context: &OperationContext,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.PLACEMENT_PROTOCOL_FAILED",
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message,
        std::collections::BTreeMap::from([("study_id".to_string(), spec.study_id.clone())]),
    )
}

fn sweep_placement_error(
    spec: &AnalysisStudySweepSpec,
    context: &OperationContext,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY_SWEEP.PLACEMENT_PROTOCOL_FAILED",
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message,
        std::collections::BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
    )
}
