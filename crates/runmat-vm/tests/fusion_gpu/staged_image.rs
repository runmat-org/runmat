//! Numeric software fixture for the selected image stage provider contracts.
//! Each child models one ordinary recipe, not a monolithic image shortcut.

use super::*;

// This numerical fixture has no managed host allocator. Finite metadata
// lifecycle qualification belongs to the actual provider fixture instead.
pub(super) fn require_unmanaged_metadata(
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> runmat_accelerate_api::ProviderExecutionResult<()> {
    if admission.is_some() {
        return Err(runmat_accelerate_api::ProviderExecutionError::from(
            anyhow::anyhow!("software image fixture has no finite metadata owner"),
        ));
    }
    Ok(())
}

#[path = "staged_image/real_binary.rs"]
pub(super) mod real_binary;
#[path = "staged_image/real_elementwise.rs"]
mod real_elementwise;
#[path = "staged_image/ordered_mean.rs"]
mod ordered_mean;
#[path = "staged_image/power_pair.rs"]
mod power_pair;
