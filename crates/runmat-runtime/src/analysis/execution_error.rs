use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, AnalysisProcedure, PhysicsFamily};
use runmat_analysis_fea::planning::resource::ResourcePlanningError;
use runmat_analysis_fea::{AnalysisRunIdentity, FeaRunError};

use crate::operations::{
    operation_error, OperationContext, OperationErrorEnvelope, OperationErrorSeverity,
    OperationErrorSpec, OperationErrorType,
};

use super::execution_dispatch::AnalysisOperation;
use super::formulation_admission::{admit_runtime_proxy, RuntimeProxyFormulation};

/// Strict accelerator placement (`accelerators = required`) could not run on the accelerator.
pub const ACCELERATOR_REQUIRED_ERROR_CODE: &str = "RM.FEA.ACCELERATOR_REQUIRED";

pub(super) fn map_fea_run_error(
    operation: AnalysisOperation,
    model: &AnalysisModel,
    context: &OperationContext,
    err: FeaRunError,
) -> OperationErrorEnvelope {
    let model_details = || {
        BTreeMap::from([
            ("analysis_model_id".to_string(), model.model_id.0.clone()),
            ("geometry_id".to_string(), model.geometry_id.clone()),
        ])
    };
    let cancelled = || {
        operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.cancelled_error_code(),
                error_type: OperationErrorType::Cancelled,
                retryable: false,
                severity: OperationErrorSeverity::Warning,
            },
            "FEA run cancelled by user",
            model_details(),
        )
    };
    match err {
        FeaRunError::Usage(cause) => super::fea_run_admission::FeaRunDenied::Accounting(cause)
            .into_operation_error(operation.name(), operation.version(), context),
        FeaRunError::Cancelled => cancelled(),
        FeaRunError::InvalidModel(message) | FeaRunError::Observation(message) => operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.invalid_model_error_code(),
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            message,
            model_details(),
        ),
        FeaRunError::FrequencyPlan(error) => operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.invalid_model_error_code(),
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            error.to_string(),
            model_details(),
        ),
        FeaRunError::Resource(error) => {
            let error_type = match error {
                ResourcePlanningError::Cancelled => return cancelled(),
                ResourcePlanningError::Accounting(cause) => {
                    return super::fea_run_admission::FeaRunDenied::Accounting(cause)
                        .into_operation_error(operation.name(), operation.version(), context);
                }
                ResourcePlanningError::CanonicalWireCapacity { .. }
                | ResourcePlanningError::MemoryBudget { .. }
                | ResourcePlanningError::AccuracyCeiling { .. }
                | ResourcePlanningError::StaticSolveCeiling { .. }
                | ResourcePlanningError::ThermalSolveCeiling { .. }
                | ResourcePlanningError::Inventory(_)
                | ResourcePlanningError::Capability(_)
                | ResourcePlanningError::Accelerator { .. } => OperationErrorType::Capacity,
                ResourcePlanningError::Invalid(_)
                | ResourcePlanningError::Overflow(_)
                | ResourcePlanningError::Identity
                | ResourcePlanningError::Encoding(_) => OperationErrorType::Contract,
            };
            operation_error(
                operation.name(),
                operation.version(),
                context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESOURCE_PLANNING_FAILED",
                    error_type,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                error.to_string(),
                model_details(),
            )
        }
        FeaRunError::AcceleratorRequired(error) => operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: ACCELERATOR_REQUIRED_ERROR_CODE,
                error_type: OperationErrorType::Backend,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            error.to_string(),
            model_details(),
        ),
        FeaRunError::MissingElectromagneticSource { frequency_hz } => operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.invalid_model_error_code(),
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("no nonzero electromagnetic source at {frequency_hz} Hz"),
            model_details(),
        ),
        FeaRunError::Assembly(message) | FeaRunError::Solve(message) => operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.assembly_error_code(),
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            message,
            model_details(),
        ),
        FeaRunError::FormulationAdmission(error) => operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.formulation_admission_error_code(),
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            error.to_string(),
            model_details(),
        ),
        FeaRunError::FormulationAvailability(error) => operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.formulation_admission_error_code(),
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            error.to_string(),
            model_details(),
        ),
        FeaRunError::SolverFormulationCompatibility(error) => operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.formulation_admission_error_code(),
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            error.to_string(),
            model_details(),
        ),
    }
}

pub(super) fn admit_runtime_proxy_for_operation(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    branch: RuntimeProxyFormulation,
    operation: AnalysisOperation,
    model: &AnalysisModel,
    context: &OperationContext,
) -> Result<AnalysisRunIdentity, OperationErrorEnvelope> {
    admit_runtime_proxy(family, procedure, branch).map_err(|error| {
        operation_error(
            operation.name(),
            operation.version(),
            context,
            OperationErrorSpec {
                error_code: operation.formulation_admission_error_code(),
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            error.to_string(),
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_ceiling_is_capacity_not_invalid_physics() {
        let model = runmat_analysis_fea::fixtures::fixture_model(
            runmat_analysis_fea::fixtures::FixtureId::StructuralTransientTet4Reference,
        );
        let context = OperationContext::new(None, None);
        for resource in [
            ResourcePlanningError::MemoryBudget {
                required_memory_bytes: 2,
                available_memory_bytes: 1,
                required_scratch_bytes: 0,
                available_scratch_bytes: 0,
            },
            ResourcePlanningError::CanonicalWireCapacity {
                required: 2,
                maximum: 1,
            },
            ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                required: 2,
                maximum: 1,
            },
        ] {
            let ceiling = map_fea_run_error(
                AnalysisOperation::Transient,
                &model,
                &context,
                FeaRunError::Resource(resource),
            );
            assert_eq!(ceiling.error_type, OperationErrorType::Capacity);
            assert_eq!(ceiling.error_code, "RM.FEA.RESOURCE_PLANNING_FAILED");
        }
        let invalid = map_fea_run_error(
            AnalysisOperation::Transient,
            &model,
            &context,
            FeaRunError::Resource(ResourcePlanningError::Invalid("bad facts".into())),
        );
        assert_eq!(invalid.error_type, OperationErrorType::Contract);
    }
}
