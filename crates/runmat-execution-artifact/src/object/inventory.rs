use super::{LogicalObject, ObjectNamespace};
use crate::{ArtifactError, ArtifactResult};

type InventoryName<'a> = (ObjectNamespace, &'a str);

/// Requested Rust heap slots for borrowed inventory-name validation. Names are not cloned;
/// allocator overhead, native allocations, and process RSS are outside this bound.
pub fn inventory_validation_scratch_bytes(object_count: usize) -> ArtifactResult<u64> {
    object_count
        .checked_mul(std::mem::size_of::<InventoryName<'_>>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| ArtifactError::Limit("bundle name validation scratch overflow".into()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectInventoryLimits {
    pub max_objects: usize,
    pub max_object_bytes: u64,
    pub max_total_bytes: u64,
}

impl Default for ObjectInventoryLimits {
    fn default() -> Self {
        Self {
            max_objects: 100_000,
            max_object_bytes: 512 * 1024 * 1024,
            max_total_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

pub fn validate_inventory(
    objects: &[LogicalObject],
    limits: ObjectInventoryLimits,
) -> ArtifactResult<()> {
    if objects.len() > limits.max_objects {
        return Err(ArtifactError::Limit("too many bundle objects".into()));
    }
    inventory_validation_scratch_bytes(objects.len())?;
    let mut names = Vec::new();
    names.try_reserve_exact(objects.len()).map_err(|_| {
        ArtifactError::Limit("cannot reserve bundle name validation scratch".into())
    })?;
    let mut total = 0_u64;
    // Save the first non-name error, then inspect only its preceding valid prefix. A duplicate
    // in that prefix used to fail first, so deferred sorting preserves the previous precedence.
    let validation = (|| {
        for object in objects {
            object.validate()?;
            if object.descriptor.encoded_length > limits.max_object_bytes {
                return Err(ArtifactError::Limit(format!(
                    "object {} is too large",
                    object.descriptor.logical_name
                )));
            }
            total = total
                .checked_add(object.descriptor.encoded_length)
                .ok_or_else(|| ArtifactError::Limit("bundle size overflow".into()))?;
            if total > limits.max_total_bytes {
                return Err(ArtifactError::Limit("bundle is too large".into()));
            }
            names.push((
                object.descriptor.namespace,
                object.descriptor.logical_name.as_str(),
            ));
        }
        Ok(())
    })();
    names.sort_unstable();
    if names.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ArtifactError::Invalid(
            "duplicate object logical name in one namespace".into(),
        ));
    }
    validation
}

#[cfg(test)]
mod tests;
