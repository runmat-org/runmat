//! Shared closed study inventory checks; no formulation or numerical authority.
use crate::{AnalysisBridgeError, AnalysisBridgeResult};
use runmat_execution::{
    value::{ValueRef, ValueRefKind},
    Digest,
};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess,
};
use std::collections::BTreeMap;

mod profiling;
mod import;
pub(crate) use import::CallerBoundedImport;
pub(crate) use profiling::profile_import_workspace_phases;

/// Simultaneous original transport, assembled payloads and hostile codec materialization.
/// This is import workspace, not a substitute reservation for executing source programs.
pub(crate) fn import_workspace(references: &[ValueRef]) -> AnalysisBridgeResult<u64> {
    let encoded = references.iter().try_fold(0_u64, |bytes, object| {
        bytes.checked_add(object.encoded_length).ok_or({
            runmat_analysis_fea::execution::AnalysisExecutionError::Limit(
                "study import inventory overflow",
            )
        })
    })?;
    let decoded = runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
        encoded,
        u64::try_from(references.len()).map_err(|_| {
            runmat_analysis_fea::execution::AnalysisExecutionError::Limit(
                "study import count overflow",
            )
        })?,
    )
    .ok_or(
        runmat_analysis_fea::execution::AnalysisExecutionError::Limit(
            "study decode inventory overflow",
        ),
    )?;
    // Keep every object in this phase covered, including partial chunks assembled
    // into logical payloads and simultaneous output/receipt decode work.
    encoded
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(decoded))
        .ok_or_else(|| {
            runmat_analysis_fea::execution::AnalysisExecutionError::Limit(
                "study import workspace overflow",
            )
            .into()
        })
}
/// Peak of sequential phase imports. Original decoded state and reference inventories
/// remain charged by the coordinator; temporary transport/decode workspace is reusable.
/// Cross-phase identity conflicts are still rejected before replay.
pub(crate) fn import_workspace_phases(phases: &[&[ValueRef]]) -> AnalysisBridgeResult<u64> {
    let mut references = Vec::new();
    let mut maximum = 0;
    for phase in phases {
        references.extend(phase.iter().cloned());
        let phase = unique(phase.to_vec())?;
        maximum = maximum.max(import_workspace(&phase)?);
    }
    unique(references)?;
    Ok(maximum)
}

pub(crate) fn unique(references: Vec<ValueRef>) -> AnalysisBridgeResult<Vec<ValueRef>> {
    let mut by_id = BTreeMap::new();
    for reference in references {
        if let Some(previous) = by_id.insert(reference.id, reference.clone()) {
            if previous != reference {
                return Err(invalid("study closure reference identity conflict"));
            }
        }
    }
    Ok(by_id.into_values().collect())
}

pub(crate) fn check_inventory(
    references: &[ValueRef],
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    if references.len() > limits.max_objects {
        return Err(invalid("study closure object limit exceeded"));
    }
    let mut total = 0_u64;
    for reference in references {
        access.validate_reference(
            reference,
            ValueRefKind::ResultObject,
            &reference.media_type,
            &reference.value_schema,
        )?;
        total = total
            .checked_add(reference.encoded_length)
            .ok_or_else(|| invalid("study closure byte count overflow"))?;
        if reference.encoded_length > limits.max_object_bytes || total > limits.max_total_bytes {
            return Err(invalid("study closure byte limit exceeded"));
        }
    }
    Ok(())
}

pub(crate) fn read(
    source: &impl CacheImport,
    reference: &ValueRef,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<Vec<u8>> {
    if reference.encoded_length > limits.max_object_bytes {
        return Err(invalid("study object byte limit exceeded"));
    }
    let bytes = source
        .read_verified_bounded(reference.logical_digest, reference.encoded_length)?
        .ok_or_else(|| invalid("study closure child is missing"))?;
    if bytes.len() as u64 != reference.encoded_length
        || Digest::sha256(&bytes) != reference.logical_digest
    {
        return Err(invalid("study closure child bytes differ from reference"));
    }
    Ok(bytes)
}

fn invalid(message: &str) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}

#[cfg(test)]
mod tests;
