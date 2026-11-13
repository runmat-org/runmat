use runmat_types::LabRank;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::indexing::selectors::index_scalar_from_value;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

/// Read one retained lab value through the selected runtime's Composite lease.
pub async fn read_composite_entry(
    runtime: &RuntimeContext,
    handle: runmat_value::SharedCompositeHandle,
    raw_indices: &[Value],
) -> Result<crate::execution::RootedValue, RuntimeError> {
    super::lease::validate_composite(runtime, &handle)?;
    if raw_indices.len() != 1 {
        return Err(semantic_error(
            "CompositeIndexArity",
            "Composite brace indexing requires one lab index",
        ));
    }
    let rank = index_scalar_from_value(&raw_indices[0])
        .await?
        .and_then(|index| index.positive_usize())
        .and_then(|index| u32::try_from(index).ok())
        .filter(|rank| *rank <= handle.gang.labs.0)
        .map(LabRank)
        .ok_or_else(|| {
            semantic_error(
                "CompositeIndexOutOfBounds",
                "Composite lab index must be a positive integer within the SPMD gang",
            )
        })?;
    runtime
        .service_ports()
        .require_distributed("Composite brace indexing")
        .map_err(|error| semantic_error("RunMat:RuntimeCapabilityUnavailable", error.to_string()))?
        .composite_entry(handle, rank)
        .await?
        .ok_or_else(|| {
            semantic_error(
                "CompositeEntryUnavailable",
                "the selected lab did not assign this Composite value",
            )
        })
}
