//! Project completed numerical result variants into shared publication facts.
use super::*;

pub(super) fn completed_run_facts(run: CompletedStudyStepRun) -> CompletedRunFacts {
    match run {
        CompletedStudyStepRun::Standard(run) => {
            let run = *run;
            let data = run.data;
            CompletedRunFacts {
                identity: data.run.identity.clone(),
                operation: run.operation,
                op_version: run.op_version,
                run_id: data.run_id.clone(),
                run_status: data.run_status,
                numerically_acceptable: data.numerically_acceptable,
                production_publishable: data.production_publishable,
                solver_convergence: data.solver_convergence,
                result_quality: data.result_quality,
                quality_reasons: data.quality_reasons.clone(),
                production_publication_reasons: data.production_publication_reasons.clone(),
                provenance: data.provenance.clone(),
                numerical_result: crate::analysis::AnalysisStudyStepNumericalResultData::Standard,
                standard_persistence: Some(data),
                frequency_persistence: None,
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
                production_publishable: run.production_publishable,
                solver_convergence: run.solver_convergence,
                result_quality: run.result_quality,
                quality_reasons: run.quality_reasons,
                production_publication_reasons: run.production_publication_reasons,
                provenance: run.provenance,
                numerical_result: run.numerical_result,
                standard_persistence: None,
                frequency_persistence: Some(run.persistence),
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
                production_publishable: run.production_publishable,
                solver_convergence: run.solver_convergence,
                result_quality: run.result_quality,
                quality_reasons: run.quality_reasons,
                production_publication_reasons: run.production_publication_reasons,
                provenance: run.provenance,
                numerical_result:
                    crate::analysis::AnalysisStudyStepNumericalResultData::StructuralResponseSpectrum(
                        Box::new(run.numerical_result),
                    ),
                standard_persistence: Some(run.persistence),
                frequency_persistence: None,
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
                production_publishable: run.production_publishable,
                solver_convergence: run.solver_convergence,
                result_quality: run.result_quality,
                quality_reasons: run.quality_reasons,
                production_publication_reasons: run.production_publication_reasons,
                provenance: run.provenance,
                numerical_result:
                    crate::analysis::AnalysisStudyStepNumericalResultData::StructuralRandomResponse(
                        Box::new(run.numerical_result),
                    ),
                standard_persistence: Some(run.persistence),
                frequency_persistence: None,
            }
        }
    }
}
