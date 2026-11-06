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

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectInventoryLimits {
    pub max_objects: usize,
    pub max_object_bytes: u64,
    pub max_total_bytes: u64,
}

/// Largest single object one process can materialize: an object is read into one Rust
/// allocation, bounded by `isize::MAX` bytes (2 GiB - 1 on wasm32).
pub const PLATFORM_MAXIMUM_OBJECT_BYTES: u64 = isize::MAX as u64;
/// Bytes one process can address: 4 GiB on wasm32; 64-bit address space binds no machine.
pub const PLATFORM_ADDRESS_SPACE_BYTES: u64 = if usize::BITS < 64 {
    1_u64 << usize::BITS
} else {
    u64::MAX
};
/// Object slots are addressed by u32 indices in transport manifests.
const PLATFORM_MAXIMUM_OBJECTS: u64 = u32::MAX as u64;

/// A run-data inventory needs more than its admitted budget allows. Names the bound and both
/// byte or object counts so the run can be resized.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{subject} requires {required} {resource}, the admitted inventory allows {maximum}")]
pub struct InventoryCapacityError {
    pub subject: &'static str,
    pub resource: &'static str,
    pub required: u64,
    pub maximum: u64,
}

impl ObjectInventoryLimits {
    /// The single law for run-data inventories: a run may hold the bytes its admitted budget
    /// allows, within the platform ceilings. Objects occupy at least one byte, so the budget
    /// also bounds the object count.
    pub fn admitted(budget_bytes: u64) -> Self {
        let total = budget_bytes.clamp(1, PLATFORM_ADDRESS_SPACE_BYTES);
        Self {
            max_objects: usize::try_from(total.min(PLATFORM_MAXIMUM_OBJECTS)).unwrap_or(usize::MAX),
            max_object_bytes: total.min(PLATFORM_MAXIMUM_OBJECT_BYTES),
            max_total_bytes: total,
        }
    }

    /// Admits a projected producer inventory before its construction. `objects` counts every
    /// object, root included.
    pub fn require_projection(
        self,
        subject: &'static str,
        objects: u64,
        largest_object_bytes: u64,
        total_bytes: u64,
    ) -> Result<(), InventoryCapacityError> {
        let check = |resource, required: u64, maximum: u64| {
            if required > maximum {
                Err(InventoryCapacityError {
                    subject,
                    resource,
                    required,
                    maximum,
                })
            } else {
                Ok(())
            }
        };
        check("output bytes", total_bytes, self.max_total_bytes)?;
        check(
            "bytes in one object",
            largest_object_bytes,
            self.max_object_bytes,
        )?;
        check("objects", objects, self.max_objects as u64)
    }
}

/// Default for hostile imports and tests that name no budget. Run producers derive their limits
/// with [`ObjectInventoryLimits::admitted`].
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
            if object.descriptor.namespace == ObjectNamespace::ProjectRuntimePolicy {
                super::validate_project_runtime_policy_object(object)?;
            }
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
