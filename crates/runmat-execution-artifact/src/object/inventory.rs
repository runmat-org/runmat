use std::collections::BTreeSet;

use super::LogicalObject;
use crate::{ArtifactError, ArtifactResult};

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
    let mut names = BTreeSet::new();
    let mut total = 0_u64;
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
        if !names.insert((
            object.descriptor.namespace,
            object.descriptor.logical_name.as_str(),
        )) {
            return Err(ArtifactError::Invalid(
                "duplicate object logical name in one namespace".into(),
            ));
        }
    }
    Ok(())
}
