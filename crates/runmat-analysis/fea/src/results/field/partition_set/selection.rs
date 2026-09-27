//! Metadata-only canonical storage-block selection shared by imports and publication checks.
use super::*;
use crate::results::field::RecoveredFieldArtifactEntry;

#[derive(Debug)]
pub struct RecoveredFieldPartitionSelection<'a> {
    pub descriptors: Vec<&'a RecoveredFieldPartitionDescriptor>,
    pub identity: RecoveredFieldPartitionSetIdentity,
}

/// Validates canonical contiguous block order and unique identities; returns covered entity count.
/// The artifact manifest additionally checks this count against its authoritative entity domain.
pub fn validate_canonical_field_partition_descriptors(
    descriptors: &[RecoveredFieldPartitionDescriptor],
) -> Result<usize, RecoveredFieldError> {
    if descriptors.is_empty() {
        return Err(invalid("field storage descriptor inventory is empty"));
    }
    let mut next = 0usize;
    for (ordinal, descriptor) in descriptors.iter().enumerate() {
        if descriptor.first_source_index != next
            || descriptor.entity_count == 0
            || descriptor.entity_count > RECOVERED_FIELD_PARTITION_ENTITY_COUNT
            || (ordinal + 1 < descriptors.len()
                && descriptor.entity_count != RECOVERED_FIELD_PARTITION_ENTITY_COUNT)
        {
            return Err(invalid(
                "field storage descriptors are not a canonical contiguous inventory",
            ));
        }
        next = next
            .checked_add(descriptor.entity_count)
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
    }
    // Exact pointer-vector scratch; no allocator-dependent tree nodes in metadata admission.
    let mut identities = Vec::new();
    identities
        .try_reserve_exact(descriptors.len())
        .map_err(|_| invalid("field storage descriptor validation allocation failed"))?;
    identities.extend(descriptors.iter());
    identities.sort_unstable_by_key(|descriptor| descriptor.partition);
    if identities
        .windows(2)
        .any(|pair| pair[0].partition == pair[1].partition)
    {
        return Err(invalid(
            "field storage descriptors repeat a partition identity",
        ));
    }
    Ok(next)
}

/// Selects the exact canonical storage blocks needed for the sorted source indices. This does
/// not read numerical values or attest descriptor hashes against payloads: artifact/source
/// validation and the importing worker remain responsible for those independent boundaries.
pub fn select_canonical_field_partitions<'a>(
    entry: &'a RecoveredFieldArtifactEntry,
    required_indices: &[usize],
) -> Result<RecoveredFieldPartitionSelection<'a>, RecoveredFieldError> {
    entry.definition.validate()?;
    entry.sample.validate()?;
    let entity_count = validate_canonical_field_partition_descriptors(&entry.partitions)?;
    if required_indices.is_empty()
        || required_indices.windows(2).any(|pair| pair[0] >= pair[1])
        || required_indices
            .last()
            .is_some_and(|index| *index >= entity_count)
    {
        return Err(invalid(
            "field source indices must be nonempty, increasing and covered by the inventory",
        ));
    }
    let mut descriptors = Vec::new();
    descriptors
        .try_reserve_exact(entry.partitions.len().min(required_indices.len()))
        .map_err(|_| invalid("field storage descriptor selection allocation failed"))?;
    let mut next_required = 0;
    for descriptor in &entry.partitions {
        let start = next_required;
        while required_indices
            .get(next_required)
            .is_some_and(|index| descriptor.contains(*index))
        {
            next_required += 1;
        }
        if next_required > start {
            descriptors.push(descriptor);
        }
    }
    if next_required != required_indices.len() {
        return Err(invalid(
            "field source index is absent from the selected storage blocks",
        ));
    }
    let identity = derive_partition_set_identity(
        1,
        entry.field,
        &entry.definition,
        entry.sample,
        PartitionIdentities::Descriptors(&descriptors),
    )?;
    Ok(RecoveredFieldPartitionSelection {
        descriptors,
        identity,
    })
}

fn invalid(message: &str) -> RecoveredFieldError {
    RecoveredFieldError::InvalidArtifact(message.into())
}

#[cfg(test)]
mod tests;
