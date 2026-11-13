use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralRandomResponseAssemblyRequest,
    StructuralRandomResponseIntervalRequest, StructuralRandomResponseIntervalResult,
    StructuralRandomResponseNumericalReceipt, StructuralRandomResponseSolverResult,
    StructuralRandomResponseWorkload,
};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects,
};

macro_rules! whole_object_access {
    ($prepare:ident, $decode:ident, $ty:ty, $kind:path, $label:literal, $identity:expr) => {
        pub fn $prepare(
            value: &$ty,
            chunk_bytes: usize,
            namespace: ObjectNamespace,
            limits: ObjectInventoryLimits,
        ) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
            prepare_analysis_objects(
                $kind,
                $identity(value),
                &value.canonical_bytes()?,
                chunk_bytes,
                namespace,
                limits,
            )
        }

        pub fn $decode(prepared: &PreparedAnalysisObjects) -> AnalysisBridgeResult<$ty> {
            if prepared.manifest.kind != $kind {
                return Err(AnalysisBridgeError::Invalid(
                    concat!("expected ", $label).into(),
                ));
            }
            let value = <$ty>::from_canonical_bytes(&prepared.payload())?;
            if prepared.manifest.logical_identity != $identity(&value) {
                return Err(AnalysisBridgeError::Identity(concat!(
                    $label,
                    " has the wrong logical identity"
                )));
            }
            Ok(value)
        }
    };
}

whole_object_access!(
    prepare_structural_random_response_workload_objects,
    decode_prepared_structural_random_response_workload,
    StructuralRandomResponseWorkload,
    AnalysisNumericalObjectKind::StructuralRandomResponseWorkload,
    "a structural random-response workload object",
    |value: &StructuralRandomResponseWorkload| value.identity().digest()
);
whole_object_access!(
    prepare_structural_random_response_interval_request_objects,
    decode_prepared_structural_random_response_interval_request,
    StructuralRandomResponseIntervalRequest,
    AnalysisNumericalObjectKind::StructuralRandomResponseIntervalRequest,
    "a structural random-response interval request object",
    |value: &StructuralRandomResponseIntervalRequest| value.identity().digest()
);
whole_object_access!(
    prepare_structural_random_response_interval_result_objects,
    decode_prepared_structural_random_response_interval_result,
    StructuralRandomResponseIntervalResult,
    AnalysisNumericalObjectKind::StructuralRandomResponseIntervalResult,
    "a structural random-response interval result object",
    |value: &StructuralRandomResponseIntervalResult| value.identity().digest()
);
whole_object_access!(
    prepare_structural_random_response_assembly_request_objects,
    decode_prepared_structural_random_response_assembly_request,
    StructuralRandomResponseAssemblyRequest,
    AnalysisNumericalObjectKind::StructuralRandomResponseAssemblyRequest,
    "a structural random-response assembly request object",
    |value: &StructuralRandomResponseAssemblyRequest| value.identity().digest()
);
whole_object_access!(
    prepare_structural_random_response_numerical_receipt_objects,
    decode_prepared_structural_random_response_numerical_receipt,
    StructuralRandomResponseNumericalReceipt,
    AnalysisNumericalObjectKind::NumericalReceipt,
    "a structural random-response numerical receipt object",
    |value: &StructuralRandomResponseNumericalReceipt| value.identity().digest()
);

pub fn decode_prepared_structural_random_response_solver_result(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralRandomResponseSolverResult> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural random-response solver result object".into(),
        ));
    }
    StructuralRandomResponseSolverResult::from_canonical_bytes(&prepared.payload())
        .map_err(Into::into)
}
