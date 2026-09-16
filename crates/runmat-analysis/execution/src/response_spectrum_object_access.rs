use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralModalResponseSpectrumAssemblyRequest,
    StructuralModalResponseSpectrumDirectionRequest,
    StructuralModalResponseSpectrumDirectionResult,
    StructuralModalResponseSpectrumNumericalReceipt, StructuralModalResponseSpectrumSolverResult,
    StructuralModalResponseSpectrumWorkload,
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
    prepare_structural_modal_response_spectrum_workload_objects,
    decode_prepared_structural_modal_response_spectrum_workload,
    StructuralModalResponseSpectrumWorkload,
    AnalysisNumericalObjectKind::StructuralModalResponseSpectrumWorkload,
    "a structural modal response-spectrum workload object",
    |value: &StructuralModalResponseSpectrumWorkload| value.identity().digest()
);
whole_object_access!(
    prepare_structural_modal_response_spectrum_direction_request_objects,
    decode_prepared_structural_modal_response_spectrum_direction_request,
    StructuralModalResponseSpectrumDirectionRequest,
    AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionRequest,
    "a structural modal response-spectrum direction request object",
    |value: &StructuralModalResponseSpectrumDirectionRequest| value.identity().digest()
);
whole_object_access!(
    prepare_structural_modal_response_spectrum_direction_result_objects,
    decode_prepared_structural_modal_response_spectrum_direction_result,
    StructuralModalResponseSpectrumDirectionResult,
    AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult,
    "a structural modal response-spectrum direction result object",
    |value: &StructuralModalResponseSpectrumDirectionResult| value.identity().digest()
);
whole_object_access!(
    prepare_structural_modal_response_spectrum_assembly_request_objects,
    decode_prepared_structural_modal_response_spectrum_assembly_request,
    StructuralModalResponseSpectrumAssemblyRequest,
    AnalysisNumericalObjectKind::StructuralModalResponseSpectrumAssemblyRequest,
    "a structural modal response-spectrum assembly request object",
    |value: &StructuralModalResponseSpectrumAssemblyRequest| value.identity().digest()
);
whole_object_access!(
    prepare_structural_modal_response_spectrum_numerical_receipt_objects,
    decode_prepared_structural_modal_response_spectrum_numerical_receipt,
    StructuralModalResponseSpectrumNumericalReceipt,
    AnalysisNumericalObjectKind::NumericalReceipt,
    "a structural modal response-spectrum numerical receipt object",
    |value: &StructuralModalResponseSpectrumNumericalReceipt| value.identity().digest()
);

pub fn decode_prepared_structural_modal_response_spectrum_solver_result(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralModalResponseSpectrumSolverResult> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural modal response-spectrum solver result object".into(),
        ));
    }
    StructuralModalResponseSpectrumSolverResult::from_canonical_bytes(&prepared.payload())
        .map_err(Into::into)
}
