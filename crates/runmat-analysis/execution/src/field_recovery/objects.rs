//! Canonical recovered-field transport shared by producers and original-source inputs.
use super::*;
use runmat_execution::Digest;

/// Bound source, partition encoding and shared transport construction before copying values.
pub fn recovered_field_transport_reservation_bytes(
    artifact: &runmat_analysis_fea::results::field::RecoveredFieldArtifact,
    chunk_bytes: usize,
) -> AnalysisBridgeResult<u64> {
    use runmat_analysis_fea::{
        execution::AnalysisTransportProfiler,
        results::field::{RecoveredFieldPartition, RECOVERED_FIELD_PARTITION_ENTITY_COUNT},
    };
    use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
    let limits = CanonicalLimits::platform_content(4_000_000, 64);
    let source = profile_serialized(artifact, limits).map_err(invalid)?;
    let manifest = artifact.manifest().map_err(invalid)?;
    let catalog = profile_serialized(&manifest, limits).map_err(invalid)?;
    let mut profiler = AnalysisTransportProfiler::new(chunk_bytes)?;
    profiler.push(catalog.encoded_bytes_upper_bound)?;
    let mut workspace = catalog
        .encoded_bytes_upper_bound
        .checked_add(catalog.materialization_resident_bytes_estimate)
        .ok_or_else(|| invalid("field source workspace overflows"))?;
    for field in artifact.fields() {
        let mut remaining = field.definition().stored_entities(artifact.entity_counts);
        while remaining > 0 {
            let count = remaining.min(RECOVERED_FIELD_PARTITION_ENTITY_COUNT);
            let profile =
                RecoveredFieldPartition::profile_shape(field.definition(), field.sample(), count)
                    .map_err(invalid)?;
            profiler.push(profile.encoded_bytes_upper_bound)?;
            workspace = workspace
                .checked_add(profile.encoded_bytes_upper_bound)
                .and_then(|bytes| {
                    bytes.checked_add(profile.materialization_resident_bytes_estimate)
                })
                .ok_or_else(|| invalid("field partition workspace overflows"))?;
            remaining -= count;
        }
    }
    let transport = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    let conversion = crate::projected_transport_output_resource_facts(&transport)?;
    source
        .materialization_resident_bytes_estimate
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(workspace))
        .and_then(|bytes| {
            bytes.checked_add(conversion.construction_resident_bytes_admission_budget.0)
        })
        .and_then(|bytes| {
            conversion
                .transport_closure_bytes_upper_bound
                .0
                .checked_mul(2)
                .and_then(|closure| bytes.checked_add(closure))
        })
        .ok_or_else(|| invalid("field source construction residency overflows"))
}

pub fn prepare_recovered_field_objects(
    artifact: &runmat_analysis_fea::results::field::RecoveredFieldArtifact,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    use runmat_analysis_fea::execution::AnalysisPayloadPartition;
    let partitions = recovered_field_payload_partitions(artifact)?;
    let borrowed = partitions
        .iter()
        .map(|(logical_identity, payload)| AnalysisPayloadPartition {
            logical_identity: *logical_identity,
            payload,
        })
        .collect::<Vec<_>>();
    crate::prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::Field,
        artifact.identity().digest(),
        &borrowed,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub(crate) fn recovered_field_payload_partitions(
    artifact: &runmat_analysis_fea::results::field::RecoveredFieldArtifact,
) -> AnalysisBridgeResult<Vec<(Digest, Vec<u8>)>> {
    use runmat_analysis_fea::results::field::canonical_field_partitions;
    let mut partitions = vec![(
        artifact.identity().digest(),
        artifact
            .manifest()
            .map_err(invalid)?
            .canonical_bytes()
            .map_err(invalid)?,
    )];
    for field in artifact.fields() {
        for partition in canonical_field_partitions(field).map_err(invalid)? {
            partitions.push((
                partition.identity.digest(),
                partition.canonical_bytes().map_err(invalid)?,
            ));
        }
    }
    Ok(partitions)
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
