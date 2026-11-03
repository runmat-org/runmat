use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::value::{PublishedObjectCatalog, PublishedObjectSource};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::analysis::contracts::AnalysisArtifactRecord;
use crate::analysis::FrequencySampleArtifactRootInventory;
use crate::analysis::{
    result_policy, run_operation_for_identity, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisStudyStepNumericalResultData, AnalysisStudyStepRunData, QualityGate,
};

use super::{validate_run_id_token, AnalysisFieldPageLookup};

pub(super) const ACOUSTIC_FREQUENCY_ARTIFACT_SCHEMA_VERSION: &str =
    "analysis_acoustic_frequency_artifact/2026-10-001";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PersistedAcousticFrequencyArtifact {
    pub schema_version: String,
    pub created_at: String,
    pub op_version: String,
    pub step: AnalysisStudyStepRunData,
    pub object_catalog: PublishedObjectCatalog,
    pub artifact_roots: FrequencySampleArtifactRootInventory,
}

pub(super) fn prepare_artifact(
    step: &AnalysisStudyStepRunData,
    object_catalog: &PublishedObjectCatalog,
    artifact_roots: &FrequencySampleArtifactRootInventory,
    created_at: String,
) -> Result<PersistedAcousticFrequencyArtifact, String> {
    let artifact = PersistedAcousticFrequencyArtifact {
        schema_version: ACOUSTIC_FREQUENCY_ARTIFACT_SCHEMA_VERSION.to_string(),
        created_at,
        op_version: step.run_op_version.clone(),
        step: step.clone(),
        object_catalog: object_catalog.clone(),
        artifact_roots: artifact_roots.clone(),
    };
    validate_artifact(&artifact, &step.run_id)?;
    Ok(artifact)
}

pub(super) fn decode_artifact(
    bytes: &[u8],
    expected_run_id: &str,
) -> Result<PersistedAcousticFrequencyArtifact, String> {
    let artifact = serde_json::from_slice::<PersistedAcousticFrequencyArtifact>(bytes)
        .map_err(|error| format!("failed to parse acoustic frequency artifact: {error}"))?;
    validate_artifact(&artifact, expected_run_id)?;
    Ok(artifact)
}

pub(super) fn validate_artifact(
    artifact: &PersistedAcousticFrequencyArtifact,
    expected_run_id: &str,
) -> Result<(), String> {
    validate_run_id_token(expected_run_id)?;
    if artifact.schema_version != ACOUSTIC_FREQUENCY_ARTIFACT_SCHEMA_VERSION
        || artifact.step.run_id != expected_run_id
        || artifact.op_version != artifact.step.run_op_version
    {
        return Err(
            "acoustic frequency artifact schema, key, or operation version is invalid".into(),
        );
    }
    let AnalysisStudyStepNumericalResultData::AcousticFrequency(result) =
        &artifact.step.numerical_result
    else {
        return Err("acoustic frequency artifact contains another numerical result kind".into());
    };
    // The manifest's authority was proven against the complete numerical result at admission.
    let authority = result.publication.authority();
    let workload_identity = result
        .workload
        .identity()
        .map_err(|error| error.to_string())?;
    let runmat_analysis_fea::execution::FieldTopologyIdentity::VolumeMesh { mesh_digest } =
        authority.topology()
    else {
        return Err("acoustic frequency authority has a non-volume topology".into());
    };
    if authority.workload() != workload_identity
        || authority.plan() != result.workload.plan()
        || authority.entries().len() != result.entries.len()
    {
        return Err(
            "acoustic frequency artifact has inconsistent complete numerical evidence".into(),
        );
    }
    result
        .resource_usage
        .validate_identity()
        .map_err(|error| error.to_string())?;
    let mut solver_convergence = QualityGate::Pass;
    let mut result_quality = QualityGate::Pass;
    let mut quality_reasons = Vec::new();
    for (numerical, completed) in authority.entries().iter().zip(&result.entries) {
        let expected = result
            .workload
            .entry(numerical.entry)
            .map_err(|error| error.to_string())?;
        let summary = &completed.result;
        let request = runmat_analysis_fea::execution::AccuracyEvaluationRequest::new(
            summary.producer,
            runmat_analysis_fea::execution::AccuracySourceKind::AcousticHarmonic,
            summary.source_result,
            completed.observation_catalog.as_ref(),
            expected
                .model
                .sequence
                .steps
                .iter()
                .find(|step| step.step_id == expected.step_id)
                .ok_or("acoustic frequency workload omits its authored step")?
                .accuracy
                .clone(),
        )
        .map_err(|error| error.to_string())?;
        let source_receipt =
            runmat_analysis_fea::execution::AccuracySourceReceipt::AcousticHarmonic(
                completed.numerical_receipt.clone(),
            );
        let expected_accuracy = runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(
            &request,
            source_receipt
                .evidence(&request)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if completed.accuracy != expected_accuracy {
            return Err(
                "acoustic frequency accuracy differs from authored criteria and exact receipt"
                    .into(),
            );
        }
        completed
            .numerical_receipt
            .validate()
            .map_err(|error| error.to_string())?;
        completed
            .accuracy
            .validate()
            .map_err(|error| error.to_string())?;
        completed
            .baseline_usage
            .validate_identity()
            .map_err(|error| error.to_string())?;
        completed
            .terminal_usage
            .validate_identity()
            .map_err(|error| error.to_string())?;
        completed
            .publication
            .validate_sources(&completed.accuracy, &completed.baseline_usage)
            .map_err(|error| error.to_string())?;
        crate::analysis::structural_study::lifecycle::accuracy_completion::validate_publication_chain(
            completed.host_workload_digest,
            &runmat_analysis_fea::planning::resource::FeaResourcePhase::AcousticHarmonic(numerical.workload),
            &completed.accuracy, &completed.baseline_usage, &completed.terminal_usage,
            &completed.publication,
        )?;
        let accuracy =
            crate::analysis::structural_study::lifecycle::accuracy_completion::classify_accuracy(
                &completed.accuracy,
            );
        solver_convergence = worst_gate(solver_convergence, accuracy.solver_convergence);
        result_quality = worst_gate(result_quality, accuracy.result_quality);
        quality_reasons.extend(accuracy.quality_reasons);
        let receipt = completed.numerical_receipt.receipt();
        let expected_workload = expected.identity().map_err(|error| error.to_string())?;
        if completed.entry != numerical.entry
            || numerical.workload != expected_workload
            || numerical.source
                != runmat_analysis_fea::execution::FieldRecoverySource::SolverResult(
                    summary.source_result,
                )
            || summary.source_workload != expected_workload
            || summary.producer != authority.producer()
            || summary.mesh_digest != mesh_digest
            || summary.frequency != expected.frequency
            || summary.formulation != artifact.step.identity
            || completed.numerical_receipt.source_workload() != numerical.workload
            || completed.numerical_receipt.source_result() != summary.source_result
            || completed.accuracy.source_result != summary.source_result
            || completed.accuracy.producer != summary.producer
            || receipt.producer != summary.producer
            || receipt.topology_digest != Digest::from_bytes(*summary.mesh_digest.bytes())
            || receipt.frequency_hz != expected.frequency.canonical_value()
            || receipt.relative_residual != summary.relative_residual
            || receipt.power_relative_error != summary.power_relative_error
            || receipt.solver_relative_tolerance != expected.solver.relative_tolerance
            || receipt.power_balance_relative_tolerance != expected.power_balance_relative_tolerance
        {
            return Err(
                "acoustic frequency artifact entry differs from its original numerical evidence"
                    .into(),
            );
        }
        let state_requests = expected
            .state_publication_requests(
                Digest::from_bytes(*summary.mesh_digest.bytes()),
                summary.source_result,
            )
            .map_err(|error| error.to_string())?;
        if state_requests.len() != completed.states.len() {
            return Err("acoustic frequency state inventory differs from original result".into());
        }
        for (request, state) in state_requests.iter().zip(&completed.states) {
            state.verify_identity()?;
            if state.producer != request.producer
                || state.source_result != request.source_result
                || state.topology_digest != request.topology_digest
                || state.state_id != request.declaration.state_id
                || state.kind != request.declaration.kind
                || state.completeness != request.declaration.completeness
                || state.persistence != request.declaration.persistence
            {
                return Err("acoustic frequency state differs from its exact entry source".into());
            }
        }
    }
    if artifact.step.solver_convergence != solver_convergence
        || artifact.step.result_quality != result_quality
        || artifact.step.quality_reasons != quality_reasons
    {
        return Err("acoustic frequency artifact quality differs from all-entry evidence".into());
    }
    artifact
        .artifact_roots
        .validate(&crate::analysis::FrequencySamplePublicationAccess {
            publication: &result.publication,
            authority,
            object_catalog: &artifact.object_catalog,
        })?;
    let operation = run_operation_for_identity(&artifact.step.identity)
        .ok_or_else(|| "acoustic frequency result identity has no runtime operation".to_string())?;
    if operation.name() != artifact.step.run_operation
        || operation.version() != artifact.step.run_op_version
    {
        return Err("acoustic frequency artifact operation differs from its run identity".into());
    }
    if artifact.step.numerically_acceptable
        != (artifact.step.solver_convergence == QualityGate::Pass
            && artifact.step.result_quality == QualityGate::Pass)
        || artifact.step.run_status
            != result_policy::classify_run_status(
                artifact.step.result_quality,
                artifact.step.numerically_acceptable,
            )
    {
        return Err("acoustic frequency artifact quality classification is inconsistent".into());
    }
    Ok(())
}

fn worst_gate(left: QualityGate, right: QualityGate) -> QualityGate {
    match (left, right) {
        (QualityGate::Fail, _) | (_, QualityGate::Fail) => QualityGate::Fail,
        (QualityGate::Warn, _) | (_, QualityGate::Warn) => QualityGate::Warn,
        _ => QualityGate::Pass,
    }
}

/// Compare original accepted arrays in canonical component order without recovery or allocation.
#[cfg(test)]
fn native_values_match(
    field: &runmat_analysis_core::AnalysisField,
    expected: impl Iterator<Item = f64>,
) -> bool {
    match &field.values {
        runmat_analysis_core::AnalysisFieldValues::HostF64(values) => values
            .iter()
            .map(|value| value.to_bits())
            .eq(expected.map(f64::to_bits)),
        runmat_analysis_core::AnalysisFieldValues::DeviceRef(_) => false,
    }
}

pub(super) fn ingest_objects(
    artifact: &PersistedAcousticFrequencyArtifact,
    source: &dyn PublishedObjectSource,
) -> Result<BTreeMap<Digest, Vec<u8>>, String> {
    validate_artifact(artifact, &artifact.step.run_id)?;
    let mut objects = BTreeMap::new();
    for reference in &artifact.object_catalog.objects {
        let bytes = source.read_verified(reference)?;
        if bytes.len() as u64 != reference.encoded_length
            || Digest::sha256(&bytes) != reference.logical_digest
            || objects.insert(reference.logical_digest, bytes).is_some()
        {
            return Err(
                "acoustic frequency source differs from its declared object catalog".into(),
            );
        }
    }
    validate_closure(artifact, |digest| objects.get(&digest).cloned())?;
    Ok(objects)
}

pub(super) fn validate_closure(
    artifact: &PersistedAcousticFrequencyArtifact,
    mut read: impl FnMut(Digest) -> Option<Vec<u8>>,
) -> Result<(), String> {
    let AnalysisStudyStepNumericalResultData::AcousticFrequency(result) =
        &artifact.step.numerical_result
    else {
        return Err("acoustic frequency artifact contains another result kind".into());
    };
    let authority = result.publication.authority();
    artifact.artifact_roots.admit(
        crate::analysis::FrequencySamplePublicationAccess {
            publication: &result.publication,
            authority,
            object_catalog: &artifact.object_catalog,
        },
        |digest| Ok(read(digest)),
    )?;
    Ok(())
}

pub(super) fn same_content(
    left: &PersistedAcousticFrequencyArtifact,
    right: &PersistedAcousticFrequencyArtifact,
) -> bool {
    left.schema_version == right.schema_version
        && left.op_version == right.op_version
        && left.step == right.step
        && left.object_catalog == right.object_catalog
        && left.artifact_roots == right.artifact_roots
}

pub(super) fn field_page(
    artifact: &PersistedAcousticFrequencyArtifact,
    selection: &AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
    read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisFieldPageLookup, String> {
    validate_artifact(artifact, &artifact.step.run_id)?;
    super::validate_field_id(&selection.field_id)?;
    let AnalysisStudyStepNumericalResultData::AcousticFrequency(result) =
        &artifact.step.numerical_result
    else {
        return Err("acoustic frequency artifact contains another result kind".into());
    };
    let authority = result.publication.authority();
    let mut read = read;
    let admitted = artifact.artifact_roots.admit(
        crate::analysis::FrequencySamplePublicationAccess {
            publication: &result.publication,
            authority,
            object_catalog: &artifact.object_catalog,
        },
        &mut read,
    )?;
    admitted.load_field_page(&artifact.step.run_id, selection, options, read)
}

pub(super) fn artifact_record(
    artifact: &PersistedAcousticFrequencyArtifact,
) -> AnalysisArtifactRecord {
    let field_ids = match &artifact.step.numerical_result {
        AnalysisStudyStepNumericalResultData::AcousticFrequency(result) => result
            .publication
            .samples()
            .iter()
            .flat_map(|sample| sample.fields.fields.iter())
            .map(|entry| entry.definition.field_id().to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        AnalysisStudyStepNumericalResultData::StructuralFrequency(_)
        | AnalysisStudyStepNumericalResultData::Standard
        | AnalysisStudyStepNumericalResultData::StructuralResponseSpectrum(_)
        | AnalysisStudyStepNumericalResultData::StructuralRandomResponse(_) => Vec::new(),
    };
    AnalysisArtifactRecord {
        run_id: artifact.step.run_id.clone(),
        created_at: artifact.created_at.clone(),
        identity: artifact.step.identity.clone(),
        op_version: artifact.op_version.clone(),
        field_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::storage::{
        AnalysisArtifactStore, FilesystemAnalysisArtifactStore, InMemoryAnalysisArtifactStore,
    };

    #[test]
    fn rehashed_native_fields_cannot_substitute_original_accepted_values() {
        use runmat_analysis_core::{
            AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
        };
        use runmat_analysis_fea::results::field::{
            RecoveredField, RecoveredFieldDefinition, RecoveredFieldEntityDomain,
            RecoveredFieldNumericRepresentation, RecoveredFieldSample,
        };
        for (quantity, location, domain, value_shape, shape, accepted) in [
            (
                ObservationQuantity::AcousticPressure,
                FieldLocation::Node,
                RecoveredFieldEntityDomain::Nodes,
                ObservationValueShape::Scalar,
                vec![2, 2],
                vec![1.0, -2.0, 3.0, -4.0],
            ),
            (
                ObservationQuantity::AcousticParticleVelocity,
                FieldLocation::Element,
                RecoveredFieldEntityDomain::VolumeElements,
                ObservationValueShape::Vector3,
                vec![1, 3, 2],
                vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
            ),
        ] {
            let definition = RecoveredFieldDefinition::new(
                "native",
                quantity,
                location,
                domain,
                value_shape,
                RecoveredFieldNumericRepresentation::ComplexInterleaved,
            )
            .unwrap();
            let sample = RecoveredFieldSample::FrequencyHertz(2.0);
            let original = RecoveredField::new(
                definition.clone(),
                sample,
                AnalysisField::host_f64("native", definition.layout(), shape.clone(), accepted.clone()),
            )
            .unwrap();
            assert!(native_values_match(
                original.storage(),
                accepted.iter().copied()
            ));
            for index in 0..accepted.len() {
                let mut altered = accepted.clone();
                altered[index] += 0.25;
                // Rebuild a valid content identity, preserving all semantic field metadata.
                let tampered = RecoveredField::new(
                    definition.clone(),
                    sample,
                    AnalysisField::host_f64("native", definition.layout(), shape.clone(), altered),
                )
                .unwrap();
                tampered.validate().unwrap();
                assert_ne!(tampered.identity(), original.identity());
                assert!(!native_values_match(
                    tampered.storage(),
                    accepted.iter().copied()
                ));
            }
            let mut reordered = accepted.clone();
            reordered.swap(0, 1);
            assert!(!native_values_match(
                &AnalysisField::host_f64("native", original.storage().layout.clone(), shape.clone(), reordered),
                accepted.iter().copied()
            ));
            assert!(!native_values_match(
                original.storage(),
                accepted.iter().copied().take(1)
            ));
            assert!(!native_values_match(
                original.storage(),
                accepted.iter().copied().chain([0.0])
            ));
        }
    }

    #[test]
    fn native_value_comparison_preserves_signed_zero_bits() {
        let field = runmat_analysis_core::AnalysisField::host_f64("native", runmat_analysis_core::AnalysisFieldLayout::scalar(runmat_analysis_core::ObservationQuantity::Residual, runmat_analysis_core::FieldLocation::Global), vec![1], vec![0.0]);
        assert!(native_values_match(&field, [0.0].into_iter()));
        assert!(!native_values_match(&field, [-0.0].into_iter()));
    }

    #[test]
    fn acoustic_schema_is_dispatched_strictly_without_structural_reinterpretation() {
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema_version": ACOUSTIC_FREQUENCY_ARTIFACT_SCHEMA_VERSION
        }))
        .unwrap();
        let error = match super::super::decode_persisted_artifact(&bytes, "frequency-run") {
            Ok(_) => panic!("incomplete acoustic artifact was accepted"),
            Err(error) => error,
        };
        assert!(
            error.contains("failed to parse acoustic frequency artifact"),
            "{error}"
        );
        let unknown = serde_json::to_vec(&serde_json::json!({
            "schema_version": "analysis_acoustic_frequency_artifact/retired"
        }))
        .unwrap();
        assert!(
            matches!(super::super::decode_persisted_artifact(&unknown, "frequency-run"),
            Err(error) if error.contains("unsupported analysis run artifact schema"))
        );
    }

    #[test]
    fn acoustic_lookup_rejects_invalid_keys_before_backend_access() {
        let memory = InMemoryAnalysisArtifactStore::new();
        let filesystem = FilesystemAnalysisArtifactStore::new(std::path::PathBuf::from(
            "unused-acoustic-key-test-store",
        ));
        for key in ["", "../escape", "parent/child", "."] {
            assert!(memory.load_acoustic_frequency(key).is_err(), "{key}");
            assert!(filesystem.load_acoustic_frequency(key).is_err(), "{key}");
        }
        assert!(memory
            .load_acoustic_frequency("absent-frequency-run")
            .unwrap()
            .is_none());
    }
}
