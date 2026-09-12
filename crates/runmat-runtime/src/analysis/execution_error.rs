use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisModel, AnalysisProcedure, FormulationRequirement, PhysicsFamily,
};
use runmat_analysis_fea::{AnalysisRunIdentity, FeaRunError};

use crate::operations::{
    operation_error, OperationContext, OperationErrorEnvelope, OperationErrorSeverity,
    OperationErrorSpec, OperationErrorType,
};

use super::execution_dispatch::AnalysisOperation;
use super::formulation_admission::{admit_runtime_proxy, RuntimeProxyFormulation};

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
    match err {
        FeaRunError::Cancelled => operation_error(
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
        ),
        FeaRunError::InvalidModel(message)
        | FeaRunError::Observation(message)
        | FeaRunError::InvalidFrequencyPlan(message) => operation_error(
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
        FeaRunError::Assembly(message) => operation_error(
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
    }
}

pub(super) fn admit_runtime_proxy_for_operation(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    requirement: FormulationRequirement,
    branch: RuntimeProxyFormulation,
    operation: AnalysisOperation,
    model: &AnalysisModel,
    context: &OperationContext,
) -> Result<AnalysisRunIdentity, OperationErrorEnvelope> {
    admit_runtime_proxy(family, procedure, requirement, branch).map_err(|error| {
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
