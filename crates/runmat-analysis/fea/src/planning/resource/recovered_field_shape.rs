//! Shared canonical catalog/field-partition shape, independent of the physics family.
use super::*;
use crate::execution::FieldRecoveryRequest;
use crate::results::field::{
    RecoveredField, RecoveredFieldArtifactManifest, RecoveredFieldDefinition,
    RecoveredFieldPartition, RecoveredFieldSample, RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};
use runmat_canonical_codec::SerializedCanonicalProfile;
pub(super) fn shape(
    request: &FieldRecoveryRequest,
    definitions: Vec<RecoveredFieldDefinition>,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let counts = request
        .plan
        .requirements
        .iter()
        .map(|r| {
            usize::try_from(r.cost.output_entity_count)
                .map_err(|_| ResourcePlanningError::Overflow("native field entities"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    shape_with_counts(request, definitions, counts)
}
pub(super) fn shape_with_counts(
    request: &FieldRecoveryRequest,
    definitions: Vec<RecoveredFieldDefinition>,
    counts: Vec<usize>,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    shape_with_counts_at(request, definitions, counts, RecoveredFieldSample::Final)
}

/// Shape of single-sample native fields at an explicit sample coordinate.
pub(super) fn shape_at(
    request: &FieldRecoveryRequest,
    definitions: Vec<RecoveredFieldDefinition>,
    sample: RecoveredFieldSample,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let counts = request
        .plan
        .requirements
        .iter()
        .map(|r| {
            usize::try_from(r.cost.output_entity_count)
                .map_err(|_| ResourcePlanningError::Overflow("native field entities"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    shape_with_counts_at(request, definitions, counts, sample)
}

fn shape_with_counts_at(
    request: &FieldRecoveryRequest,
    definitions: Vec<RecoveredFieldDefinition>,
    counts: Vec<usize>,
    sample: RecoveredFieldSample,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let invalid = |e: &dyn std::fmt::Display| ResourcePlanningError::Invalid(e.to_string());
    let catalog = if sample == RecoveredFieldSample::Final {
        RecoveredFieldArtifactManifest::profile_final_shape(request, &definitions, &counts)
    } else {
        RecoveredFieldArtifactManifest::profile_sampled_shape(
            request,
            &definitions,
            &counts,
            sample,
        )
    }
    .map_err(|e| invalid(&e))?;
    let mut partitions = vec![catalog.encoded_bytes_upper_bound];
    let mut workspace = catalog.materialization_resident_bytes_estimate;
    let add_profile = |total: &mut u64,
                       profile: SerializedCanonicalProfile|
     -> Result<(), ResourcePlanningError> {
        *total = total
            .checked_add(profile.materialization_resident_bytes_estimate)
            .and_then(|n| n.checked_add(profile.encoded_bytes_upper_bound))
            .ok_or(ResourcePlanningError::Overflow("native field workspace"))?;
        Ok(())
    };
    for (definition, count) in definitions.iter().zip(counts) {
        add_profile(
            &mut workspace,
            RecoveredField::identity_profile_shape(definition, sample, count)
                .map_err(|e| invalid(&e))?,
        )?;
        let mut remaining = count;
        while remaining > 0 {
            let count = remaining.min(RECOVERED_FIELD_PARTITION_ENTITY_COUNT);
            let profile = RecoveredFieldPartition::profile_shape(definition, sample, count)
                .map_err(|e| invalid(&e))?;
            partitions.push(profile.encoded_bytes_upper_bound);
            add_profile(&mut workspace, profile)?;
            remaining -= count;
        }
    }
    Ok((partitions, ByteCount(workspace)))
}
