//! Actual canonical shape admission over provider-enforced bounded immutable reads.
use super::*;
use runmat_analysis_fea::execution::{
    AnalysisObjectManifest, ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_MANIFEST_MEDIA_TYPE,
};
#[cfg(test)]
use runmat_execution_artifact::ArtifactResult;

fn limit(message: &'static str) -> AnalysisBridgeError {
    runmat_analysis_fea::execution::AnalysisExecutionError::Limit(message).into()
}
fn admit(bytes: u64, maximum: u64) -> AnalysisBridgeResult<()> {
    if bytes > maximum {
        return Err(
            runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "original phase import workspace",
                required: bytes,
                maximum,
            }
            .into(),
        );
    }
    Ok(())
}

pub(crate) fn profile_import_workspace_phases(
    source: &impl CacheImport,
    phases: &[&[ValueRef]],
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
    maximum: u64,
) -> AnalysisBridgeResult<u64> {
    let mut peak = 0;
    for references in phases {
        let references = unique(references.to_vec())?;
        let bounded = CallerBoundedImport::new(source, limits)?.close(&references, access)?;
        let encoded = references.iter().try_fold(0_u64, |total, r| {
            total
                .checked_add(r.encoded_length)
                .ok_or_else(|| limit("profile transport inventory overflow"))
        })?;
        // Original transport, the assembled payload and a verification/read copy can coexist.
        // Shared fixed codec allowance covers inventory/manifest/collection metadata per object.
        let fixed =
            runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
                0,
                references.len() as u64,
            )
            .ok_or_else(|| limit("profile metadata inventory overflow"))?;
        let mut bytes = encoded
            .checked_mul(3)
            .and_then(|n| n.checked_add(fixed))
            .ok_or_else(|| limit("profile raw inventory overflow"))?;
        admit(bytes, maximum)?; // before any payload read or typed materialization
        let references = bounded.references();
        for reference in references
            .iter()
            .filter(|r| r.media_type != ANALYSIS_CHUNK_MEDIA_TYPE)
        {
            let raw = read(&bounded, reference, limits)?;
            bytes = bytes
                .checked_add(
                    runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&raw)?,
                )
                .ok_or_else(|| limit("profile contract inventory overflow"))?;
            admit(bytes, maximum)?;
            if reference.media_type == ANALYSIS_MANIFEST_MEDIA_TYPE {
                // Only the original manifest is decoded here; never the numerical payload.
                let manifest = AnalysisObjectManifest::from_canonical_bytes(&raw)?;
                let declared = crate::analysis_object_references(
                    &manifest,
                    reference,
                    access,
                    ValueRefKind::ResultObject,
                )?;
                if declared.iter().any(|r| !references.contains(r)) {
                    return Err(invalid(
                        "profile manifest reads outside original phase inventory",
                    ));
                }
                let objects = crate::import_analysis_objects(
                    &bounded,
                    reference,
                    manifest.kind,
                    manifest.logical_identity,
                    manifest.payload_digest,
                    access,
                    limits,
                )?;
                for partition in &objects.manifest.partitions {
                    let payload = objects.partition_payload(partition.logical_identity)?;
                    bytes = bytes.checked_add(runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&payload)?)
                        .ok_or_else(|| limit("profile semantic inventory overflow"))?;
                    admit(bytes, maximum)?;
                }
            } else if reference.media_type != crate::response::ANALYSIS_OUTPUT_SET_MEDIA_TYPE {
                return Err(invalid(
                    "profile contract is not an original analysis object",
                ));
            }
        }
        peak = peak.max(bytes);
    }
    Ok(peak)
}

#[cfg(test)]
mod tests;
