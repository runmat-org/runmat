//! Project completed numerical result variants into shared publication facts.
use super::*;

pub(super) fn completed_run_facts(run: CompletedStudyStepRun) -> CompletedRunFacts {
    match run {
        CompletedStudyStepRun::Standard(run)
        | CompletedStudyStepRun::OriginalFields { run, .. } => {
            let run = *run;
            let data = run.data;
            CompletedRunFacts {
                identity: data.run.identity.clone(),
                operation: run.operation,
                op_version: run.op_version,
                run_id: data.run_id.clone(),
                run_status: data.run_status,
                numerically_acceptable: data.numerically_acceptable,
                solver_convergence: data.solver_convergence,
                result_quality: data.result_quality,
                quality_reasons: data.quality_reasons.clone(),
                provenance: data.provenance.clone(),
                numerical_result: CompletedNumericalResult::Published(
                    crate::analysis::AnalysisStudyStepNumericalResultData::Standard,
                ),
                standard_persistence: Some(data),
                frequency_persistence: None,
                acoustic_frequency_persistence: None,
            }
        }
        CompletedStudyStepRun::AcousticFrequency(run) => {
            let run = *run;
            let operation = run_operation_for_identity(&run.identity)
                .expect("admitted acoustic-frequency identity has a runtime operation");
            CompletedRunFacts {
                identity: run.identity,
                operation: operation.name().to_string(),
                op_version: operation.version().to_string(),
                run_id: run.run_id,
                run_status: run.run_status,
                numerically_acceptable: run.numerically_acceptable,
                solver_convergence: run.solver_convergence,
                result_quality: run.result_quality,
                quality_reasons: run.quality_reasons,
                provenance: run.provenance,
                numerical_result: CompletedNumericalResult::AcousticFrequency(run.completion),
                standard_persistence: None,
                frequency_persistence: None,
                acoustic_frequency_persistence: Some(run.persistence),
            }
        }
        CompletedStudyStepRun::StructuralFrequency(run) => {
            let run = *run;
            let operation = run_operation_for_identity(&run.identity)
                .expect("admitted structural-frequency identity has a runtime operation");
            CompletedRunFacts {
                identity: run.identity,
                operation: operation.name().to_string(),
                op_version: operation.version().to_string(),
                run_id: run.run_id,
                run_status: run.run_status,
                numerically_acceptable: run.numerically_acceptable,
                solver_convergence: run.solver_convergence,
                result_quality: run.result_quality,
                quality_reasons: run.quality_reasons,
                provenance: run.provenance,
                numerical_result: CompletedNumericalResult::Published(run.numerical_result),
                standard_persistence: None,
                frequency_persistence: Some(run.persistence),
                acoustic_frequency_persistence: None,
            }
        }
        CompletedStudyStepRun::StructuralResponseSpectrum(run) => {
            let run = *run;
            let operation = run_operation_for_identity(&run.identity)
                .expect("admitted response-spectrum identity has a runtime operation");
            CompletedRunFacts {
                identity: run.identity,
                operation: operation.name().to_string(),
                op_version: operation.version().to_string(),
                run_id: run.run_id,
                run_status: run.run_status,
                numerically_acceptable: run.numerically_acceptable,
                solver_convergence: run.solver_convergence,
                result_quality: run.result_quality,
                quality_reasons: run.quality_reasons,
                provenance: run.provenance,
                numerical_result: CompletedNumericalResult::Published(
                    crate::analysis::AnalysisStudyStepNumericalResultData::StructuralResponseSpectrum(Box::new(
                        run.numerical_result,
                    )),
                ),
                standard_persistence: Some(run.persistence),
                frequency_persistence: None,
                acoustic_frequency_persistence: None,
            }
        }
        CompletedStudyStepRun::StructuralRandomResponse(run) => {
            let run = *run;
            let operation = run_operation_for_identity(&run.identity)
                .expect("admitted random-response identity has a runtime operation");
            CompletedRunFacts {
                identity: run.identity,
                operation: operation.name().to_string(),
                op_version: operation.version().to_string(),
                run_id: run.run_id,
                run_status: run.run_status,
                numerically_acceptable: run.numerically_acceptable,
                solver_convergence: run.solver_convergence,
                result_quality: run.result_quality,
                quality_reasons: run.quality_reasons,
                provenance: run.provenance,
                numerical_result: CompletedNumericalResult::Published(
                    crate::analysis::AnalysisStudyStepNumericalResultData::StructuralRandomResponse(
                        Box::new(run.numerical_result),
                    ),
                ),
                standard_persistence: Some(run.persistence),
                frequency_persistence: None,
                acoustic_frequency_persistence: None,
            }
        }
    }
}
