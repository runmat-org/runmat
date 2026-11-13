use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralFrequencyEntryRequest, StructuralFrequencyEntryResult,
    StructuralFrequencyNumericalReceipt, StructuralFrequencyPhysicalSample,
    StructuralFrequencyResultAssemblyRequest, StructuralFrequencySolverResult,
    StructuralFrequencySolverResultManifest, StructuralFrequencyWorkload,
};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ObjectNamespace;

use crate::{
    prepare_analysis_objects, prepare_partitioned_analysis_objects, AnalysisBridgeError,
    AnalysisBridgeResult, PreparedAnalysisObjects,
};

macro_rules! whole_object_access {
    ($prepare:ident, $decode:ident, $ty:ty, $kind:path, $label:literal, $identity:expr) => {
        pub fn $prepare(
            value: &$ty,
            chunk_bytes: usize,
            namespace: ObjectNamespace,
            limits: ObjectInventoryLimits,
        ) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
            let bytes = value.canonical_bytes()?;
            prepare_analysis_objects(
                $kind,
                $identity(value),
                &bytes,
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
    prepare_structural_frequency_workload_objects,
    decode_prepared_structural_frequency_workload,
    StructuralFrequencyWorkload,
    AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
    "a structural frequency workload object",
    |value: &StructuralFrequencyWorkload| value.identity().digest()
);
whole_object_access!(
    prepare_structural_frequency_entry_request_objects,
    decode_prepared_structural_frequency_entry_request,
    StructuralFrequencyEntryRequest,
    AnalysisNumericalObjectKind::StructuralFrequencyEntryRequest,
    "a structural frequency entry request object",
    |value: &StructuralFrequencyEntryRequest| value.identity().digest()
);
whole_object_access!(
    prepare_structural_frequency_entry_result_objects,
    decode_prepared_structural_frequency_entry_result,
    StructuralFrequencyEntryResult,
    AnalysisNumericalObjectKind::StructuralFrequencyEntryResult,
    "a structural frequency entry result object",
    |value: &StructuralFrequencyEntryResult| value.identity().digest()
);
whole_object_access!(
    prepare_structural_frequency_result_assembly_request_objects,
    decode_prepared_structural_frequency_result_assembly_request,
    StructuralFrequencyResultAssemblyRequest,
    AnalysisNumericalObjectKind::StructuralFrequencyResultAssemblyRequest,
    "a structural frequency result assembly request object",
    |value: &StructuralFrequencyResultAssemblyRequest| value.identity().digest()
);
whole_object_access!(
    prepare_structural_frequency_numerical_receipt_objects,
    decode_prepared_structural_frequency_numerical_receipt,
    StructuralFrequencyNumericalReceipt,
    AnalysisNumericalObjectKind::NumericalReceipt,
    "a structural frequency numerical receipt object",
    |value: &StructuralFrequencyNumericalReceipt| value.identity().digest()
);

pub fn prepare_structural_frequency_solver_result_objects(
    result: &StructuralFrequencySolverResult,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    result.validate()?;
    let manifest_bytes = result.manifest().canonical_bytes()?;
    let sample_bytes = result
        .samples()
        .iter()
        .map(StructuralFrequencyPhysicalSample::canonical_bytes)
        .collect::<Result<Vec<_>, _>>()?;
    let mut partitions = Vec::with_capacity(result.samples().len() + 1);
    partitions.push(runmat_analysis_fea::execution::AnalysisPayloadPartition {
        logical_identity: result.manifest().identity().digest(),
        payload: &manifest_bytes,
    });
    partitions.extend(
        result
            .samples()
            .iter()
            .zip(&sample_bytes)
            .map(
                |(sample, payload)| runmat_analysis_fea::execution::AnalysisPayloadPartition {
                    logical_identity: sample.identity().digest(),
                    payload,
                },
            ),
    );
    prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        result.manifest().identity().digest(),
        &partitions,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_structural_frequency_solver_result(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralFrequencySolverResult> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural frequency solver result object".into(),
        ));
    }
    let manifest = StructuralFrequencySolverResultManifest::from_canonical_bytes(
        &prepared.partition_payload(prepared.manifest.logical_identity)?,
    )?;
    let samples = manifest
        .samples()
        .iter()
        .map(|entry| {
            StructuralFrequencyPhysicalSample::from_canonical_bytes(
                &prepared.partition_payload(entry.sample().digest())?,
            )
            .map_err(AnalysisBridgeError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = StructuralFrequencySolverResult::from_manifest_and_samples(manifest, samples)?;
    if prepared.manifest.logical_identity != result.manifest().identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency solver result has the wrong logical identity",
        ));
    }
    Ok(result)
}
