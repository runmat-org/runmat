//! Numerical admission and whole-study publication are distinct completion phases.
use super::{
    assessment::PublishedStudyAssessment, structural_study::publication::DeferredStudyRun,
    study_result_manifest::StudyResultManifest, AnalysisStudyRunData,
};
use crate::operations::OperationEnvelope;
use runmat_analysis_core::{evaluate_function_assessment, AssessmentIssue, AssessmentMeasurement};
use runmat_execution::Digest;

/// Family-owned manifest types remain captured until final publication. The host can inspect
/// only authenticated assessment inputs; it cannot substitute numerical results or a manifest.
pub struct PreparedStudyPublication {
    completion: DeferredStudyRun,
    manifest: serde_json::Value,
}

/// Execution hosts authenticate portable task results before implementing this boundary.
/// This process-local interface deliberately cannot be deserialized from user/UI data.
pub trait AssessmentExecutionReceipt {
    fn assessment_id(&self) -> &str;
    fn program_identity(&self) -> Digest;
    fn diagnostic(&self) -> Option<&str> {
        None
    }
    fn validate(
        &self,
        report: &PublishedStudyAssessment,
    ) -> Result<runmat_analysis_core::AssessmentFunctionEvaluation, String>;
}

impl PreparedStudyPublication {
    pub(super) fn new<W, N>(
        mut completion: DeferredStudyRun,
        (mut manifest, manifest_copies): (
            StudyResultManifest<W, N>,
            super::storage::publication::OwnedCopyReservation,
        ),
    ) -> Result<Self, super::storage::AnalysisPublicationError>
    where
        W: serde::Serialize + Send + 'static,
        N: serde::Serialize + Send + 'static,
    {
        completion.published.data.assessments = super::assessment::evaluate_with_source(
            manifest.compiled(),
            &completion.published.data,
            &manifest.compiled().assessments,
            super::assessment::AssessmentEvaluationLimits::default(),
            crate::interrupt::is_cancelled,
            &completion.staging,
        )
        .map_err(failure::assessment_failure)?;
        completion
            .staging
            .admit_manifest_copy(&completion.published.data)?;
        manifest.set_public_run(completion.published.data.clone());
        let projection = completion.staging.serialize_retained_manifest(&manifest)?;
        // Serialization returned the JSON owner; the typed clone no longer
        // coexists with staging/commit. Preserve original and staged charges.
        drop(manifest);
        completion.staging.release_owned_copies(manifest_copies)?;
        let manifest = projection;
        Ok(Self {
            completion,
            manifest,
        })
    }

    /// Applies the placement the host admitted before submission: strict placement fails
    /// closed when a requested accelerator step ran on `cpu_reference`; a study planned on
    /// `cpu_reference` publishes its one fallback warning.
    pub fn admit_accelerator_placement(
        &mut self,
        placement: &runmat_analysis_fea::accelerator_admission::AcceleratorPlacement,
    ) -> Result<(), runmat_analysis_fea::accelerator_admission::AcceleratorRequiredError> {
        let data = &mut self.completion.published.data;
        let fallback = placement.settle_study(data.steps.iter().map(|step| {
            (
                step.procedure_step.step_id.as_str(),
                step.provenance.requested_backend,
                step.provenance.execution_backend,
            )
        }))?;
        if let Some(fallback) = fallback {
            if !data
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == fallback.code)
            {
                data.diagnostics.push(fallback);
            }
        }
        Ok(())
    }

    pub fn function_inputs(&self) -> impl Iterator<Item = &PublishedStudyAssessment> {
        self.completion
            .published
            .data
            .assessments
            .iter()
            .filter(|report| {
                report.result.issue == Some(AssessmentIssue::FunctionExecutionUnavailable)
            })
    }

    /// A capability-limited host explicitly retains indeterminate function assessments.
    pub fn publish_without_function_execution(
        self,
    ) -> Result<OperationEnvelope<AnalysisStudyRunData>, super::storage::AnalysisPublicationError>
    {
        self.persist()
    }

    pub fn complete<R: AssessmentExecutionReceipt>(
        mut self,
        executions: Vec<R>,
    ) -> Result<OperationEnvelope<AnalysisStudyRunData>, super::storage::AnalysisPublicationError>
    {
        complete_assessment_reports(&mut self.completion.published.data.assessments, executions)?;
        self.persist()
    }

    fn persist(
        mut self,
    ) -> Result<OperationEnvelope<AnalysisStudyRunData>, super::storage::AnalysisPublicationError>
    {
        self.completion
            .staging
            .update_manifest_public_run(&mut self.manifest, &self.completion.published.data)?;
        self.completion.commit(self.manifest)
    }
}

#[cfg(test)]
mod tests;

pub(super) fn complete_assessment_reports<R: AssessmentExecutionReceipt>(
    reports: &mut [PublishedStudyAssessment],
    executions: Vec<R>,
) -> Result<(), String> {
    let mut by_id = std::collections::BTreeMap::new();
    for execution in executions {
        if by_id
            .insert(execution.assessment_id().to_owned(), execution)
            .is_some()
        {
            return Err("assessment completion contains duplicate receipts".into());
        }
    }
    for report in reports
        .iter_mut()
        .filter(|report| report.result.issue == Some(AssessmentIssue::FunctionExecutionUnavailable))
    {
        let execution = by_id
            .remove(&report.definition.assessment_id)
            .ok_or_else(|| {
                "assessment completion is missing a required function receipt".to_owned()
            })?;
        let quantity = report
            .result
            .measured
            .ok_or_else(|| "assessment has no admitted measurement".to_owned())?;
        let outcome = execution.validate(report)?;
        report.function_execution_identity = Some(execution.program_identity());
        report.function_execution_diagnostic = execution
            .diagnostic()
            .map(|message| message.chars().take(1024).collect());
        report.result = evaluate_function_assessment(
            &report.definition,
            Some(AssessmentMeasurement {
                quantity,
                complete: true,
                numerically_acceptable: true,
                absolute_error_bound: None,
            }),
            outcome,
        );
    }
    if !by_id.is_empty() {
        return Err("assessment completion contains unexpected or duplicate receipts".into());
    }
    Ok(())
}

/// Canonical evidence inventory is authenticated by the numerical store before this boundary.
/// Include run and study identities so equal scalar values cannot exchange receipts.
pub fn assessment_evidence_identity(report: &PublishedStudyAssessment) -> Result<Digest, String> {
    let evidence = report
        .evidence
        .as_ref()
        .ok_or_else(|| "assessment has no admitted scalar evidence".to_owned())?;
    serde_json::to_vec(&(report.study_definition, &report.run_id, evidence))
        .map(Digest::sha256)
        .map_err(|error| error.to_string())
}

mod failure;
pub(super) use failure::{publication_failure, publication_failure_at};
