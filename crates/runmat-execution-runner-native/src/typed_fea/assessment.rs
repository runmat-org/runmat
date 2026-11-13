//! Frozen imported assessments use the ordinary bounded portable task host.
use crate::analysis_task_pool::{
    begin_analysis_batch, AnalysisTaskPool, LocalAnalysisTaskPool, RemoteAnalysisTaskPool,
};
use runmat_analysis_execution::assessment_program::{
    build_assessment_task_submission, prepare_assessment_program, AdmittedAssessmentExecution,
    FrozenAssessmentCallable, PreparedAssessmentProgram,
};
use runmat_analysis_execution::named_callable::NamedCallableExecutionContext;
use runmat_runtime::analysis::{
    assessment::PublishedStudyAssessment, assessment_evidence_identity, AssessmentExecutionReceipt,
    PreparedStudyPublication,
};
use std::collections::BTreeSet;

struct NativeAssessmentReceipt(AdmittedAssessmentExecution);
impl AssessmentExecutionReceipt for NativeAssessmentReceipt {
    fn assessment_id(&self) -> &str {
        self.0.assessment_id()
    }
    fn program_identity(&self) -> runmat_execution::Digest {
        self.0.program_identity()
    }
    fn diagnostic(&self) -> Option<&str> {
        self.0.diagnostic()
    }
    fn validate(
        &self,
        report: &PublishedStudyAssessment,
    ) -> Result<runmat_analysis_core::AssessmentFunctionEvaluation, String> {
        let measurement = report
            .result
            .measured
            .ok_or_else(|| "assessment has no measured value".to_owned())?;
        self.0
            .validate_binding(
                &report.definition,
                measurement.canonical_value(),
                report.study_definition,
                assessment_evidence_identity(report)?,
            )
            .map_err(|error| error.to_string())
    }
}

pub async fn complete(
    source: &runmat_core::CheckedFeaSessionSource,
    publication: PreparedStudyPublication,
) -> Result<
    runmat_runtime::analysis::AnalysisStudyRunData,
    runmat_runtime::analysis::storage::AnalysisPublicationError,
> {
    if publication.function_inputs().next().is_none() {
        return publication
            .complete(Vec::<NativeAssessmentReceipt>::new())
            .map(|envelope| envelope.data);
    }
    let (mut pool, resources) = local_pool()?;
    complete_with_pool(source, publication, &mut pool, resources).await
}

pub async fn freeze_functions(
    source: &runmat_core::CheckedFeaSessionSource,
    assessments: &[runmat_analysis_core::AuthoredAssessment],
) -> Result<Vec<FrozenAssessmentCallable>, String> {
    let mut functions = Vec::new();
    for assessment in assessments {
        let runmat_analysis_core::AssessmentRule::Function { function, .. } = &assessment.rule
        else {
            continue;
        };
        let digest = runmat_execution::Digest::from_bytes(function.source_digest).to_string();
        let imported = super::callable::compile_imported_function(
            source,
            &function.module_path,
            &function.symbol,
            digest.strip_prefix("sha256:"),
        )
        .await?;
        functions.push(
            FrozenAssessmentCallable::new(
                assessment.clone(),
                imported.revision,
                imported.callable,
                imported.executable,
            )
            .map_err(|error| error.to_string())?,
        );
    }
    Ok(functions)
}

/// Complete a detached numerical solve with the callable inventory frozen at submission.
pub async fn complete_frozen_locally(
    publication: PreparedStudyPublication,
    functions: &[FrozenAssessmentCallable],
) -> Result<
    runmat_runtime::analysis::AnalysisStudyRunData,
    runmat_runtime::analysis::storage::AnalysisPublicationError,
> {
    let mut ids = BTreeSet::new();
    for function in functions {
        function.validate().map_err(|error| error.to_string())?;
        if !ids.insert(&function.assessment().assessment_id) {
            return Err("frozen assessment callable inventory has duplicate criteria".into());
        }
    }
    if publication.function_inputs().next().is_none() {
        return publication
            .complete(Vec::<NativeAssessmentReceipt>::new())
            .map(|envelope| envelope.data);
    }
    let (mut pool, resources) = local_pool()?;
    let mut receipts = Vec::new();
    for report in publication.function_inputs() {
        if runmat_runtime::interrupt::is_cancelled() {
            return Err("assessment execution was cancelled".into());
        }
        let frozen = functions
            .iter()
            .find(|function| function.assessment().assessment_id == report.definition.assessment_id)
            .ok_or_else(|| "detached solve omitted a frozen assessment callable".to_string())?;
        let measured = report
            .result
            .measured
            .ok_or_else(|| "assessment has no admitted measurement".to_string())?;
        let program = frozen
            .prepare(
                &report.definition,
                measured.canonical_value(),
                report.study_definition,
                assessment_evidence_identity(report)?,
            )
            .map_err(|error| error.to_string())?;
        receipts.push(execute_program(program, &mut pool, resources.clone()).await?);
    }
    publication.complete(receipts).map(|envelope| envelope.data)
}

fn local_pool() -> Result<
    (
        LocalAnalysisTaskPool,
        runmat_execution::resource::ResourceRequest,
    ),
    String,
> {
    let mut config = crate::NativeExecutionConfig::for_current_executable()
        .map_err(|error| error.to_string())?;
    config.max_workers = 1;
    let resources = runmat_execution::resource::ResourceRequest {
        cpu_millicores: config.worker_cpu_millicores,
        memory_bytes: config.worker_memory_bytes,
        scratch_bytes: crate::process_output::RawOutputBudget::local_scratch_bytes(0, None)
            .map_err(|error| error.to_string())?,
        max_wall_millis: 30_000,
        max_artifact_bytes: 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([
            runmat_execution::resource::Capability::ProcessIsolation,
        ]),
    };
    let session = crate::NativeProgramSession::new(config).map_err(|error| error.to_string())?;
    let pool = LocalAnalysisTaskPool::new(session);
    Ok((pool, resources))
}

/// Remote workers use the same request preparation, receipt admission and final publication.
/// The caller owns the remote pool and supplies its explicit resource envelope.
pub async fn complete_remotely(
    source: &runmat_core::CheckedFeaSessionSource,
    publication: PreparedStudyPublication,
    pool: std::sync::Arc<crate::RemotePoolDriver>,
    resources: runmat_execution::resource::ResourceRequest,
) -> Result<
    runmat_runtime::analysis::AnalysisStudyRunData,
    runmat_runtime::analysis::storage::AnalysisPublicationError,
> {
    let mut pool = RemoteAnalysisTaskPool::new(pool);
    complete_with_pool(source, publication, &mut pool, resources).await
}

async fn complete_with_pool<P: AnalysisTaskPool>(
    source: &runmat_core::CheckedFeaSessionSource,
    publication: PreparedStudyPublication,
    pool: &mut P,
    resources: runmat_execution::resource::ResourceRequest,
) -> Result<
    runmat_runtime::analysis::AnalysisStudyRunData,
    runmat_runtime::analysis::storage::AnalysisPublicationError,
> {
    let receipts =
        execute_functions(source, publication.function_inputs(), pool, resources).await?;
    publication.complete(receipts).map(|envelope| envelope.data)
}

async fn execute_functions<'a, P: AnalysisTaskPool>(
    source: &runmat_core::CheckedFeaSessionSource,
    inputs: impl Iterator<Item = &'a PublishedStudyAssessment>,
    pool: &mut P,
    resources: runmat_execution::resource::ResourceRequest,
) -> Result<Vec<NativeAssessmentReceipt>, String> {
    resources.validate().map_err(|error| error.to_string())?;
    let mut receipts = Vec::new();
    for report in inputs {
        if runmat_runtime::interrupt::is_cancelled() {
            return Err("assessment execution was cancelled".into());
        }
        let runmat_analysis_core::AssessmentRule::Function { function, .. } =
            &report.definition.rule
        else {
            return Err("assessment execution input does not name a function".into());
        };
        let source_digest =
            runmat_execution::Digest::from_bytes(function.source_digest).to_string();
        let imported = super::callable::compile_imported_function(
            source,
            &function.module_path,
            &function.symbol,
            source_digest.strip_prefix("sha256:"),
        )
        .await?;
        let measured = report
            .result
            .measured
            .ok_or_else(|| "assessment has no admitted measurement".to_owned())?;
        let program = prepare_assessment_program(
            &report.definition,
            measured.canonical_value(),
            report.study_definition,
            assessment_evidence_identity(report)?,
            runmat_analysis_execution::named_callable::NamedCallableSourceRef {
                program_revision: &imported.revision,
                callable: &imported.callable,
                compiled_binding: &imported.compiled_binding,
            },
            imported.executable,
        )
        .map_err(|error| error.to_string())?;
        receipts.push(execute_program(program, pool, resources.clone()).await?);
    }
    if runmat_runtime::interrupt::is_cancelled() {
        return Err("assessment execution was cancelled".into());
    }
    Ok(receipts)
}

/// Re-evaluate compatible immutable saved outputs without re-solving or changing the source.
pub async fn reassess_saved_locally(
    source: &runmat_core::CheckedFeaSessionSource,
    definition: runmat_execution::Digest,
    manifest_path: &std::path::Path,
    study: &runmat_analysis_core::AuthoredStudy,
    limits: runmat_runtime::analysis::assessment::AssessmentEvaluationLimits,
    maximum_manifest_bytes: u64,
) -> Result<runmat_runtime::analysis::assessment::PublishedStudyReassessment, String> {
    let publication =
        runmat_runtime::analysis::assessment::prepare_saved_study_reassessment_for_source(
            definition,
            manifest_path,
            study,
            limits,
            maximum_manifest_bytes,
            runmat_runtime::interrupt::is_cancelled,
        )
        .map_err(|error| error.to_string())?;
    if publication.function_inputs().next().is_none() {
        return publication
            .complete(
                Vec::<NativeAssessmentReceipt>::new(),
                runmat_runtime::interrupt::is_cancelled,
            )
            .map_err(|error| error.to_string());
    }
    let (mut pool, resources) = local_pool()?;
    let receipts =
        execute_functions(source, publication.function_inputs(), &mut pool, resources).await?;
    publication
        .complete(receipts, runmat_runtime::interrupt::is_cancelled)
        .map_err(|error| error.to_string())
}

async fn execute_program<P: AnalysisTaskPool>(
    program: PreparedAssessmentProgram,
    pool: &mut P,
    resources: runmat_execution::resource::ResourceRequest,
) -> Result<NativeAssessmentReceipt, String> {
    let planned_task = runmat_execution::TaskId::derive(&[program.identity().bytes()]);
    let admission = begin_analysis_batch(pool, &[planned_task])
        .await
        .map_err(|error| error.to_string())?;
    let task = build_assessment_task_submission(
        &program,
        BTreeSet::new(),
        NamedCallableExecutionContext {
            scope_id: pool.scope_id(),
            pool_id: pool.pool_id(),
            host: crate::program_host::program_host(
                program.request(),
                pool.permitted_trust_tiers(),
            )
            .map_err(|error| error.to_string())?,
            resources: resources.clone(),
            retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .map_err(|error| error.to_string())?;
    let batch = admission
        .admit(vec![(program.request().clone(), task)])
        .map_err(|error| error.to_string())?;
    let response = match pool.execute_batch(batch).await {
        Ok(_) if runmat_runtime::interrupt::is_cancelled() => {
            return Err("assessment execution was cancelled".into())
        }
        Ok(mut responses) if responses.len() == 1 => responses.remove(0).response,
        Ok(_) => return Err("assessment host returned the wrong number of receipts".into()),
        Err(crate::NativeExecutionError::Cancelled) => {
            return Err("assessment execution was cancelled".into())
        }
        Err(error) => runmat_execution_artifact::ProgramExecutionResponse::Failure {
            message: error.to_string(),
        },
    };
    Ok(NativeAssessmentReceipt(
        program
            .admit_response(&response)
            .map_err(|error| error.to_string())?,
    ))
}
