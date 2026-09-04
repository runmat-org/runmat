use runmat_analysis_core::FormulationMaturity;
use runmat_analysis_fea::FeaRunResult;

use super::contracts::{
    ProductionPublicationReason, ProductionPublicationReasonCode, QualityGate, RunStatus,
};

pub(super) fn evaluate_production_publication(
    run: &FeaRunResult,
    numerically_acceptable: bool,
) -> (bool, Vec<ProductionPublicationReason>) {
    let mut reasons = Vec::new();
    if !numerically_acceptable {
        reasons.push(ProductionPublicationReason {
            code: ProductionPublicationReasonCode::NumericalAcceptanceFailed,
            detail: "the run did not satisfy its numerical acceptance policy".to_string(),
        });
    }
    if run.identity.qualification() != FormulationMaturity::Production {
        reasons.push(ProductionPublicationReason {
            code: ProductionPublicationReasonCode::FormulationNotProductionQualified,
            detail: format!(
                "the admitted {:?}/{:?} formulation is {:?}, not production-qualified",
                run.identity.family(),
                run.identity.procedure(),
                run.identity.qualification(),
            ),
        });
    }
    (reasons.is_empty(), reasons)
}

pub(super) fn classify_run_status(
    result_quality: QualityGate,
    numerically_acceptable: bool,
) -> RunStatus {
    if result_quality == QualityGate::Fail {
        RunStatus::Rejected
    } else if numerically_acceptable {
        RunStatus::Completed
    } else {
        RunStatus::Degraded
    }
}
