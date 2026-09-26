//! Thermal result transport through shared numerical objects and recovered-field partitions.
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisObjectManifest, AnalysisPayloadPartition,
    CanonicalSemanticState, SemanticStatePublicationRequest, ThermalSolverResultManifest,
};
use runmat_analysis_fea::results::field::{RecoveredFieldPartition, RecoveredFieldView};
use runmat_analysis_fea::FeaThermalRunResult;
use runmat_execution::{value::ValueRef, Digest};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace,
};

use crate::{
    import_analysis_partition, prepare_partitioned_analysis_objects, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisPartitionImport, PreparedAnalysisObjects,
};

pub fn prepare_thermal_solver_result_objects(
    result: &FeaThermalRunResult,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let manifest = ThermalSolverResultManifest::from_result(result)?;
    let encoding = manifest.encoding_profile()?;
    // Reject oversized canonical payloads before the final partition encoding pass.
    // Manifest construction above already encodes field and partition identities.
    // Root/chunk inventory validation remains authoritative after transport construction;
    // this early guard is not a complete process-residency or pre-solve admission claim.
    if encoding.encoded_bytes_upper_bound > limits.max_total_bytes
        || encoding.contract_count >= limits.max_objects as u64
    {
        return Err(AnalysisBridgeError::Invalid(
            "thermal canonical payload projection exceeds output inventory limits".into(),
        ));
    }
    let mut payloads = vec![(manifest.identity().digest(), manifest.canonical_bytes()?)];
    let histories = [
        &result.temperature_snapshots,
        &result.temperature_gradient_snapshots,
        &result.heat_flux_snapshots,
        &result.energy_storage_rate_snapshots,
    ];
    for (ordinal, entries) in manifest.fields.iter().enumerate() {
        for (entry, history) in entries.iter().zip(histories) {
            let field = RecoveredFieldView::new(&entry.definition, entry.sample, &history[ordinal])
                .map_err(field_error)?;
            if field.identity() != entry.field {
                return Err(AnalysisBridgeError::Identity(
                    "thermal field differs from result manifest",
                ));
            }
            for partition in field.canonical_partitions().map_err(field_error)? {
                let partition = partition.map_err(field_error)?;
                payloads.push((
                    partition.identity.digest(),
                    partition.canonical_bytes().map_err(field_error)?,
                ));
            }
        }
    }
    let partitions = payloads
        .iter()
        .map(|(logical_identity, payload)| AnalysisPayloadPartition {
            logical_identity: *logical_identity,
            payload,
        })
        .collect::<Vec<_>>();
    prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        manifest.identity().digest(),
        &partitions,
        chunk_bytes,
        namespace,
        limits,
    )
}

/// Decode only the terminal temperature from an already prepared closure.
pub fn decode_prepared_thermal_state(
    publication: &SemanticStatePublicationRequest,
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<CanonicalSemanticState> {
    publication.validate()?;
    let manifest = ThermalSolverResultManifest::from_canonical_bytes(
        &prepared.partition_payload(publication.source_result.digest())?,
    )?;
    validate_inventory(&prepared.manifest, &manifest)?;
    manifest.validate_publication_request(publication)?;
    let partitions = manifest
        .terminal_temperature()?
        .partitions
        .iter()
        .map(|descriptor| {
            decode_partition(
                descriptor.partition.digest(),
                &prepared.partition_payload(descriptor.partition.digest())?,
            )
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    Ok(manifest.publish_terminal(publication, partitions)?)
}

pub struct ThermalStateImport<'a> {
    pub root: &'a ValueRef,
    pub publication: &'a SemanticStatePublicationRequest,
    pub expected_payload_digest: Digest,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

/// Read only the result manifest and terminal-temperature storage blocks.
pub fn import_thermal_state(
    source: &impl CacheImport,
    request: ThermalStateImport<'_>,
) -> AnalysisBridgeResult<CanonicalSemanticState> {
    request.publication.validate()?;
    let load = |identity| {
        import_analysis_partition(
            source,
            AnalysisPartitionImport {
                root: request.root,
                expected_kind: AnalysisNumericalObjectKind::SolverResult,
                expected_logical_identity: request.publication.source_result.digest(),
                expected_payload_digest: request.expected_payload_digest,
                partition_identity: identity,
                access: request.access,
                limits: request.limits,
            },
        )
    };
    let header = load(request.publication.source_result.digest())?;
    let manifest = ThermalSolverResultManifest::from_canonical_bytes(&header.payload())?;
    validate_inventory(&header.manifest, &manifest)?;
    manifest.validate_publication_request(request.publication)?;
    // Bound the union of selected semantic payloads before fetching temperature blocks.
    let selected = std::iter::once(manifest.identity().digest()).chain(
        manifest
            .terminal_temperature()?
            .partitions
            .iter()
            .map(|part| part.partition.digest()),
    );
    let mut bytes = request.root.encoded_length;
    let mut objects = 1usize;
    for identity in selected {
        let part = header
            .manifest
            .partitions
            .iter()
            .find(|part| part.logical_identity == identity)
            .ok_or(AnalysisBridgeError::Identity(
                "thermal result omits a required partition",
            ))?;
        bytes = bytes.checked_add(part.encoded_length).ok_or_else(|| {
            AnalysisBridgeError::Invalid("thermal selected byte count overflow".into())
        })?;
        objects = objects
            .checked_add(part.chunk_count as usize)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid("thermal selected object count overflow".into())
            })?;
    }
    if bytes > request.limits.max_total_bytes || objects > request.limits.max_objects {
        return Err(AnalysisBridgeError::Invalid(
            "thermal selected closure exceeds inventory limits".into(),
        ));
    }
    let partitions = manifest
        .terminal_temperature()?
        .partitions
        .iter()
        .map(|descriptor| {
            let imported = load(descriptor.partition.digest())?;
            decode_partition(descriptor.partition.digest(), &imported.payload())
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    Ok(manifest.publish_terminal(request.publication, partitions)?)
}

fn validate_inventory(
    objects: &AnalysisObjectManifest,
    result: &ThermalSolverResultManifest,
) -> AnalysisBridgeResult<()> {
    objects.validate()?;
    result.validate()?;
    let expected = std::iter::once(result.identity().digest()).chain(
        result
            .fields
            .iter()
            .flat_map(|sample| sample.iter())
            .flat_map(|field| field.partitions.iter())
            .map(|partition| partition.partition.digest()),
    );
    if objects.kind != AnalysisNumericalObjectKind::SolverResult
        || objects.logical_identity != result.identity().digest()
        || !expected.eq(objects
            .partitions
            .iter()
            .map(|partition| partition.logical_identity))
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal numerical object has an unrelated or incomplete field inventory",
        ));
    }
    Ok(())
}

fn decode_partition(
    expected: Digest,
    bytes: &[u8],
) -> AnalysisBridgeResult<RecoveredFieldPartition> {
    let partition = RecoveredFieldPartition::from_canonical_bytes(bytes).map_err(field_error)?;
    if partition.identity.digest() != expected {
        return Err(AnalysisBridgeError::Identity(
            "thermal partition identity differs from its descriptor",
        ));
    }
    Ok(partition)
}

fn field_error(
    error: runmat_analysis_fea::results::field::RecoveredFieldError,
) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
