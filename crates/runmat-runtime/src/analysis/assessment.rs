//! Assess checked acceptance rules against immutable, bounded result metadata.
//! This owner does not execute solvers or reinterpret observation reductions.
use super::{storage, AnalysisStudyRunData};
use runmat_analysis_core::{
    evaluate_assessment, lower_authored_observations, AssessmentMeasurement, AssessmentQuantity,
    AssessmentResult, AuthoredAssessment,
};
use runmat_analysis_fea::{
    results::assessment::ScalarObservationEvidence, study::CompiledAnalysisStudy,
};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedStudyAssessment {
    pub definition: AuthoredAssessment,
    pub result: AssessmentResult,
    pub study_definition: Digest,
    pub run_id: Option<String>,
    pub evidence: Option<ScalarObservationEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_execution_identity: Option<Digest>,
    /// Bounded execution/return-contract diagnostic, separate from the engineering verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_execution_diagnostic: Option<String>,
}

/// Sweep case attribution is retained outside the immutable per-study report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedCaseAssessments {
    pub case_id: String,
    pub assessments: Vec<PublishedStudyAssessment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedStudyReassessment {
    pub study_definition: Digest,
    pub assessments: Vec<PublishedStudyAssessment>,
    pub evidence_artifact_path: String,
}

/// Host request anchored to both the current checked source and the original publication.
/// Criteria and callable definitions are resolved from checked source, never supplied by the UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaReassessmentRequest {
    pub source_path: String,
    pub study_id: String,
    pub checked_revision: super::SourceDigest,
    pub original_definition: Digest,
    pub result_manifest_path: String,
}

mod reassessment;
pub use reassessment::{
    prepare_saved_study_reassessment, prepare_saved_study_reassessment_for_source,
    reassess_saved_study, PreparedStudyReassessment,
};
#[cfg(test)]
use reassessment::{publish_study_reassessment, validate_reassessment_outputs};

#[derive(Debug, Clone, Copy)]
pub struct AssessmentEvaluationLimits {
    pub maximum_assessments: usize,
    pub maximum_runs: usize,
    pub maximum_run_artifact_bytes: u64,
}

impl Default for AssessmentEvaluationLimits {
    fn default() -> Self {
        Self {
            maximum_assessments: 4096,
            maximum_runs: 4096,
            maximum_run_artifact_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AssessmentEvaluationError {
    #[error("assessment study identity is invalid: {0}")]
    InvalidStudy(String),
    #[error("assessment criteria are invalid: {0}")]
    InvalidCriteria(String),
    #[error("assessment evidence is invalid: {0}")]
    InvalidEvidence(String),
    #[error("assessment evaluation was cancelled")]
    Cancelled,
    #[error("assessment evaluation exceeds the configured resource limit")]
    ResourceLimit,
}

/// Evaluate only the authored outputs belonging to this checked study and publication.
/// Missing evidence remains indeterminate; storage corruption is a publication error.
pub fn assess_study_results(
    compiled: &CompiledAnalysisStudy,
    published: &AnalysisStudyRunData,
) -> Result<Vec<PublishedStudyAssessment>, String> {
    reassess_study_results(compiled, published, &compiled.assessments)
}

/// Evaluate revised criteria over existing outputs without executing or changing the study.
/// `compiled` is the original checked definition that produced `published`, not a replacement
/// definition with its revision fence removed. Each report retains the revised criterion and
/// the original study and numerical evidence identities.
pub fn reassess_study_results(
    compiled: &CompiledAnalysisStudy,
    published: &AnalysisStudyRunData,
    assessments: &[AuthoredAssessment],
) -> Result<Vec<PublishedStudyAssessment>, String> {
    reassess_study_results_with_limits(
        compiled,
        published,
        assessments,
        AssessmentEvaluationLimits::default(),
        || false,
    )
    .map_err(|e| e.to_string())
}

/// Bounded evaluation limits the criteria and metadata reads. Cancellation
/// aborts the operation; a partial prefix is never published as a complete assessment set.
pub fn reassess_study_results_with_limits(
    compiled: &CompiledAnalysisStudy,
    published: &AnalysisStudyRunData,
    assessments: &[AuthoredAssessment],
    limits: AssessmentEvaluationLimits,
    is_cancelled: impl Fn() -> bool,
) -> Result<Vec<PublishedStudyAssessment>, AssessmentEvaluationError> {
    let source =
        storage::capture_artifact_store().map_err(AssessmentEvaluationError::InvalidEvidence)?;
    evaluate_with_source(
        compiled,
        published,
        assessments,
        limits,
        is_cancelled,
        source.as_ref(),
    )
}

mod workspace;

pub(in crate::analysis) trait ScalarEvidenceSource {
    /// Prepared publication supplies a captured cumulative host allowance. Standalone
    /// reassessment retains its existing explicit artifact/count limits.
    fn workspace_limit(&self) -> Result<Option<u64>, storage::AssessmentStoreError> {
        Ok(None)
    }

    fn load_scalar_evidence(
        &self,
        run_id: &str,
        maximum_bytes: u64,
        retained_workspace_bytes: u64,
    ) -> Result<Option<storage::AssessmentRunEvidence>, storage::AssessmentStoreError>;
}
impl<T: storage::AnalysisArtifactStore + ?Sized> ScalarEvidenceSource for T {
    fn load_scalar_evidence(
        &self,
        run_id: &str,
        maximum_bytes: u64,
        _retained_workspace_bytes: u64,
    ) -> Result<Option<storage::AssessmentRunEvidence>, storage::AssessmentStoreError> {
        self.load_assessment_metadata(run_id, maximum_bytes)
    }
}
impl ScalarEvidenceSource for storage::AnalysisPublicationBuilder {
    fn workspace_limit(&self) -> Result<Option<u64>, storage::AssessmentStoreError> {
        self.assessment_workspace_limit()
            .map(Some)
            .map_err(Into::into)
    }
    fn load_scalar_evidence(
        &self,
        run_id: &str,
        maximum_bytes: u64,
        retained_workspace_bytes: u64,
    ) -> Result<Option<storage::AssessmentRunEvidence>, storage::AssessmentStoreError> {
        self.load_assessment_metadata_admitted(run_id, maximum_bytes, retained_workspace_bytes)
    }
}

pub(in crate::analysis) fn evaluate_with_source(
    compiled: &CompiledAnalysisStudy,
    published: &AnalysisStudyRunData,
    assessments: &[AuthoredAssessment],
    limits: AssessmentEvaluationLimits,
    is_cancelled: impl Fn() -> bool,
    source: &(impl ScalarEvidenceSource + ?Sized),
) -> Result<Vec<PublishedStudyAssessment>, AssessmentEvaluationError> {
    use AssessmentEvaluationError as Error;
    if is_cancelled() {
        return Err(Error::Cancelled);
    }
    if assessments.len() > limits.maximum_assessments {
        return Err(Error::ResourceLimit);
    }
    let study_definition = compiled
        .definition_digest()
        .map_err(|e| Error::InvalidStudy(e.to_string()))?;
    if published.study_fingerprint != study_definition.to_string() {
        return Err(Error::InvalidStudy(
            "evidence belongs to another checked study revision".into(),
        ));
    }
    if compiled.study_id != published.study_id || compiled.model.model_id.0 != published.model_id {
        return Err(Error::InvalidStudy(
            "publication belongs to another study".into(),
        ));
    }
    let workspace = workspace::Workspace::capture(source)?;
    workspace.admit(&(&compiled.outputs, assessments))?;
    let output_types = runmat_analysis_core::authored_output_types(
        &compiled.outputs,
        &compiled.model.sequence.steps,
    )
    .map_err(|e| Error::InvalidStudy(e.to_string()))?;
    runmat_analysis_core::validate_assessments(assessments, &output_types)
        .map_err(Error::InvalidCriteria)?;
    if assessments.is_empty() {
        return Ok(Vec::new());
    }
    let observations =
        lower_authored_observations(&compiled.outputs, &compiled.model.sequence.steps)
            .map_err(|e| Error::InvalidStudy(e.to_string()))?;
    let mut reports = Vec::new();
    let mut metadata_by_run = std::collections::BTreeMap::new();
    for definition in assessments {
        if is_cancelled() {
            return Err(Error::Cancelled);
        }
        let producer = observations
            .iter()
            .find(|o| o.request.observation_id().0 == definition.output_id)
            .ok_or_else(|| {
                Error::InvalidCriteria("checked output has no observation producer".into())
            })?;
        let step = published
            .steps
            .iter()
            .find(|s| s.procedure_step.step_id == producer.step_id);
        let mut evidence = None;
        let mut measurement = None;
        if let Some(step) = step {
            if !metadata_by_run.contains_key(step.run_id.as_str())
                && metadata_by_run.len() >= limits.maximum_runs
            {
                return Err(Error::ResourceLimit);
            }
            let retained_workspace =
                workspace.admit(&(&output_types, &observations, &metadata_by_run, &reports))?;
            if !metadata_by_run.contains_key(step.run_id.as_str()) {
                // The original store authenticates this projection; cumulative admission
                // occurs before its copy and before joining the existing cache.
                let loaded = source
                    .load_scalar_evidence(
                        &step.run_id,
                        workspace.run_artifact_bytes(limits.maximum_run_artifact_bytes),
                        retained_workspace,
                    )
                    .map_err(|error| match error {
                        storage::AssessmentStoreError::ResourceLimit => Error::ResourceLimit,
                        storage::AssessmentStoreError::Invalid(message) => {
                            Error::InvalidEvidence(message)
                        }
                    })?;
                metadata_by_run.insert(step.run_id.as_str(), loaded);
            }
            let metadata = &metadata_by_run[step.run_id.as_str()];
            if let Some(metadata) = metadata.as_ref() {
                if metadata.identity != step.identity {
                    return Err(Error::InvalidEvidence(
                        "evidence belongs to another numerical run".into(),
                    ));
                }
                let mut candidates = metadata
                    .observation_scalars
                    .iter()
                    .filter(|e| e.observation_id.0 == definition.output_id);
                if let Some(candidate) = candidates.next() {
                    if candidates.next().is_some() {
                        return Err(Error::InvalidEvidence(
                            "output has ambiguous scalar evidence".into(),
                        ));
                    }
                    let dimension = runmat_analysis_core::assessment_output_descriptor(
                        output_types.get(&definition.output_id),
                    )
                    .ok_or_else(|| {
                        Error::InvalidCriteria("output has no scalar physical dimension".into())
                    })?;
                    measurement = Some(AssessmentMeasurement {
                        quantity: AssessmentQuantity::try_from_descriptor(
                            (dimension).into(),
                            candidate.value,
                        )
                        .map_err(|error| {
                            Error::InvalidEvidence(format!("invalid assessment quantity: {error}"))
                        })?,
                        complete: true,
                        numerically_acceptable: step.numerically_acceptable
                            && metadata.numerically_acceptable,
                        absolute_error_bound: None,
                    });
                    // Reserve the upcoming evidence copy while the authenticated cache lives.
                    workspace.admit(&(
                        &output_types,
                        &observations,
                        &reports,
                        &metadata_by_run,
                        candidate,
                    ))?;
                    evidence = Some(candidate.clone());
                }
            }
        }
        // Definition plus result identifiers may both be copied into the report. Use the
        // shared structural profile before growth; existing reports/cache remain charged.
        workspace.admit(&(
            &output_types,
            &observations,
            &metadata_by_run,
            &reports,
            definition,
            definition,
            &evidence,
            step.map(|step| &step.run_id),
        ))?;
        reports.push(PublishedStudyAssessment {
            definition: definition.clone(),
            result: evaluate_assessment(definition, measurement),
            study_definition,
            run_id: step.map(|s| s.run_id.clone()),
            evidence,
            function_execution_identity: None,
            function_execution_diagnostic: None,
        });
    }
    if is_cancelled() {
        return Err(Error::Cancelled);
    }
    Ok(reports)
}

#[cfg(test)]
pub(super) mod tests;
