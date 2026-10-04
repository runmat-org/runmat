//! Authenticated saved-study reassessment and receipt completion.
use super::*;
#[derive(Serialize)]
struct StudyReassessmentEvidence<'a> {
    schema_version: u16,
    study_definition: Digest,
    assessments: &'a [PublishedStudyAssessment],
}

/// Reassess a saved publication by its immutable identity. The original checked study and
/// numerical run inventory come from the authenticated store, not caller-supplied run metadata.
pub fn reassess_saved_study(
    definition: Digest,
    manifest_path: &std::path::Path,
    assessments: &[AuthoredAssessment],
    limits: AssessmentEvaluationLimits,
    maximum_manifest_bytes: u64,
    is_cancelled: impl Fn() -> bool,
) -> Result<PublishedStudyReassessment, AssessmentEvaluationError> {
    prepare_saved_study_reassessment(
        definition,
        manifest_path,
        assessments,
        limits,
        maximum_manifest_bytes,
        &is_cancelled,
    )?
    .publish_without_function_execution(is_cancelled)
}

pub fn prepare_saved_study_reassessment(
    definition: Digest,
    manifest_path: &std::path::Path,
    assessments: &[AuthoredAssessment],
    limits: AssessmentEvaluationLimits,
    maximum_manifest_bytes: u64,
    is_cancelled: impl Fn() -> bool,
) -> Result<PreparedStudyReassessment, AssessmentEvaluationError> {
    prepare_saved_reassessment(
        definition,
        manifest_path,
        assessments,
        limits,
        maximum_manifest_bytes,
        is_cancelled,
        |_| Ok(()),
    )
}

/// A source-facing host must preserve the meaning of every output being reassessed.
/// Reusing an output name after changing its quantity, region, samples or producer is not
/// evidence that the saved numerical publication evaluated the new expression.
pub fn prepare_saved_study_reassessment_for_source(
    definition: Digest,
    manifest_path: &std::path::Path,
    study: &runmat_analysis_core::AuthoredStudy,
    limits: AssessmentEvaluationLimits,
    maximum_manifest_bytes: u64,
    is_cancelled: impl Fn() -> bool,
) -> Result<PreparedStudyReassessment, AssessmentEvaluationError> {
    prepare_saved_reassessment(
        definition,
        manifest_path,
        &study.assessments,
        limits,
        maximum_manifest_bytes,
        is_cancelled,
        |compiled| validate_reassessment_source(compiled, study),
    )
}

fn validate_reassessment_source(
    compiled: &CompiledAnalysisStudy,
    study: &runmat_analysis_core::AuthoredStudy,
) -> Result<(), AssessmentEvaluationError> {
    validate_reassessment_outputs(
        compiled,
        &study.study_id,
        &study.outputs,
        &study.sequence.steps,
        &study.assessments,
    )
}

pub(super) fn validate_reassessment_outputs(
    compiled: &CompiledAnalysisStudy,
    study_id: &str,
    outputs: &[runmat_analysis_core::AuthoredOutput],
    steps: &[runmat_analysis_core::ProcedureStep],
    assessments: &[AuthoredAssessment],
) -> Result<(), AssessmentEvaluationError> {
    use AssessmentEvaluationError as Error;
    if compiled.study_id != study_id {
        return Err(Error::InvalidCriteria(
            "saved results belong to another study".into(),
        ));
    }
    let original = lower_authored_observations(&compiled.outputs, &compiled.model.sequence.steps)
        .map_err(|error| Error::InvalidStudy(error.to_string()))?;
    let revised = lower_authored_observations(outputs, steps)
        .map_err(|error| Error::InvalidCriteria(error.to_string()))?;
    for assessment in assessments {
        let current = revised
            .iter()
            .find(|output| output.request.observation_id().0 == assessment.output_id);
        let saved = original
            .iter()
            .find(|output| output.request.observation_id().0 == assessment.output_id);
        if current.is_none() || current != saved {
            return Err(Error::InvalidCriteria(format!(
                "output '{}' changed; run the revised study to produce its evidence",
                assessment.output_id
            )));
        }
        let producer = &current.expect("checked above").step_id;
        let current_step = steps.iter().find(|step| &step.step_id == producer);
        let saved_step = compiled
            .model
            .sequence
            .steps
            .iter()
            .find(|step| &step.step_id == producer);
        if !matches!((current_step, saved_step), (Some(current), Some(saved)) if
            current.family == saved.family && current.procedure == saved.procedure && current.independent_domain == saved.independent_domain)
        {
            return Err(Error::InvalidCriteria("output producer or independent domain changed; run the revised study to produce its evidence".into()));
        }
    }
    Ok(())
}

fn prepare_saved_reassessment(
    definition: Digest,
    manifest_path: &std::path::Path,
    assessments: &[AuthoredAssessment],
    limits: AssessmentEvaluationLimits,
    maximum_manifest_bytes: u64,
    is_cancelled: impl Fn() -> bool,
    validate_source: impl FnOnce(&CompiledAnalysisStudy) -> Result<(), AssessmentEvaluationError>,
) -> Result<PreparedStudyReassessment, AssessmentEvaluationError> {
    if is_cancelled() {
        return Err(AssessmentEvaluationError::Cancelled);
    }
    if assessments.len() > limits.maximum_assessments {
        return Err(AssessmentEvaluationError::ResourceLimit);
    }
    let (compiled, published) = crate::analysis::study_result_manifest::load_for_reassessment(
        definition,
        manifest_path,
        maximum_manifest_bytes,
    )
    .map_err(|error| match error {
        crate::analysis::structural_study::StudyEvidenceReadError::ResourceLimit => {
            AssessmentEvaluationError::ResourceLimit
        }
        crate::analysis::structural_study::StudyEvidenceReadError::Invalid(message) => {
            AssessmentEvaluationError::InvalidEvidence(message)
        }
    })?;
    validate_source(&compiled)?;
    let reports = reassess_study_results_with_limits(
        &compiled,
        &published,
        assessments,
        limits,
        &is_cancelled,
    )?;
    if is_cancelled() {
        return Err(AssessmentEvaluationError::Cancelled);
    }
    Ok(PreparedStudyReassessment {
        study_definition: definition,
        reports,
    })
}

/// Publish a separate immutable assessment report. Neither the original solve publication nor
/// its authored criteria are rewritten. Storage uses the same provider-backed evidence owner
/// as whole-study results and addresses each revision by its content.
#[cfg(test)]
pub(super) fn publish_study_reassessment(
    compiled: &CompiledAnalysisStudy,
    published: &AnalysisStudyRunData,
    assessments: &[AuthoredAssessment],
    limits: AssessmentEvaluationLimits,
    is_cancelled: impl Fn() -> bool,
) -> Result<PublishedStudyReassessment, AssessmentEvaluationError> {
    let reports = reassess_study_results_with_limits(
        compiled,
        published,
        assessments,
        limits,
        &is_cancelled,
    )?;
    if is_cancelled() {
        return Err(AssessmentEvaluationError::Cancelled);
    }
    let study_definition = compiled
        .definition_digest()
        .map_err(|error| AssessmentEvaluationError::InvalidStudy(error.to_string()))?;
    PreparedStudyReassessment {
        study_definition,
        reports,
    }
    .publish_without_function_execution(is_cancelled)
}

/// Saved numerical evidence is admitted before any custom function tasks execute.
pub struct PreparedStudyReassessment {
    study_definition: Digest,
    reports: Vec<PublishedStudyAssessment>,
}
impl PreparedStudyReassessment {
    pub fn function_inputs(&self) -> impl Iterator<Item = &PublishedStudyAssessment> {
        self.reports.iter().filter(|report| {
            report.result.issue
                == Some(runmat_analysis_core::AssessmentIssue::FunctionExecutionUnavailable)
        })
    }
    pub fn complete<R: crate::analysis::AssessmentExecutionReceipt>(
        mut self,
        executions: Vec<R>,
        is_cancelled: impl Fn() -> bool,
    ) -> Result<PublishedStudyReassessment, AssessmentEvaluationError> {
        crate::analysis::prepared_study_publication::complete_assessment_reports(
            &mut self.reports,
            executions,
        )
        .map_err(AssessmentEvaluationError::InvalidEvidence)?;
        self.publish_without_function_execution(is_cancelled)
    }
    pub fn publish_without_function_execution(
        self,
        is_cancelled: impl Fn() -> bool,
    ) -> Result<PublishedStudyReassessment, AssessmentEvaluationError> {
        if is_cancelled() {
            return Err(AssessmentEvaluationError::Cancelled);
        }
        let Self {
            study_definition,
            reports,
        } = self;
        let payload = serde_json::to_value(StudyReassessmentEvidence {
            schema_version: 1,
            study_definition,
            assessments: &reports,
        })
        .map_err(|error| AssessmentEvaluationError::InvalidEvidence(error.to_string()))?;
        let evidence_artifact_path =
            crate::analysis::structural_study::persist_content_addressed_study_evidence(
                &study_definition.to_string(),
                "study-reassessment",
                payload,
            )
            .map_err(AssessmentEvaluationError::InvalidEvidence)?;
        Ok(PublishedStudyReassessment {
            study_definition,
            assessments: reports,
            evidence_artifact_path,
        })
    }
}
