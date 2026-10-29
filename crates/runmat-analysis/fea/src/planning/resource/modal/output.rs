//! Numeric modal basis storage and bounded per-mode descriptive records.
use super::{checked_add, checked_mul, ResourcePlanningError};
use crate::planning::resource::{
    numeric_profile::numeric_inventory_profile,
    static_profiles::profiled_canonical_output_resource_facts, PreparedOutputResourceFacts,
};

pub(in crate::planning::resource) fn result_output(
    values: u64,
    modes: u64,
    residual_vectors: u64,
    full_dofs: u64,
    layout_nodes: u64,
) -> Result<PreparedOutputResourceFacts, ResourcePlanningError> {
    let numeric = numeric_inventory_profile(values)?;
    // Per mode: fixed basis/directional/eigenspace map keys, scalar field
    // descriptors and enum spellings. Numeric values and array/container nodes
    // are counted separately. Residual direction metadata has its own allowance.
    // Fixed provenance, diagnostics and run/basis envelopes retain the existing
    // 64 KiB overhead. No recovered mesh or repeated geometry IDs live here.
    let metadata = checked_add(
        checked_add(
            64 * 1024,
            checked_mul(modes, 4096, "modal mode metadata")?,
            "modal metadata",
        )?,
        checked_mul(residual_vectors, 16 * 1024, "modal residual metadata")?,
        "modal metadata",
    )?;
    // The basis retains one full physical DOF layout, shared by projected systems.
    // Include node-set arrays, ordered active-kind enum spellings, row maps, row
    // indices and the checked mass-normalization header. These are not repeated
    // for every mode, but cannot disappear from admission just because Arc shares them.
    let metadata = checked_add(
        metadata,
        checked_add(
            checked_mul(full_dofs, 96, "modal physical DOF row metadata")?,
            checked_mul(layout_nodes, 96, "modal physical node-set metadata")?,
            "modal physical coordinate metadata",
        )?,
        "modal metadata and coordinate layout",
    )?;
    let metadata_resident =
        runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(metadata, 1)
            .ok_or(ResourcePlanningError::Overflow(
            "modal metadata materialization",
        ))?;
    profiled_canonical_output_resource_facts(runmat_canonical_codec::SerializedCanonicalProfile {
        encoded_bytes_upper_bound: checked_add(
            numeric.encoded_bytes_upper_bound,
            metadata,
            "modal numeric and metadata wire",
        )?,
        materialization_resident_bytes_estimate: checked_add(
            numeric.materialization_resident_bytes_estimate,
            metadata_resident,
            "modal numeric and metadata construction",
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: A60000DOF/256vector/3residual numeric output shape retains the complete
    //   search-capacity bound and conservative metadata while avoiding the unknown-shape byte
    //   multiplier. This count-only planning witness does not establish an eigensolve, physical
    //   convergence or whole-task capacity.
    #[test]
    fn full_search_capacity_does_not_treat_every_numeric_byte_as_a_map_node() {
        let values = super::super::modal_output_value_count(60_000, 256, 3).unwrap();
        let facts = result_output(values, 256, 3, 60_000, 20_000).unwrap();
        let generic = crate::planning::resource::projected_canonical_output_resource_facts(
            facts.encoded_payload_bytes_upper_bound.0,
        )
        .unwrap();
        // The generic direct-encoding projection is bounded by three encoded copies.
        assert!(
            generic.construction_resident_bytes_admission_budget.0
                >= 3 * facts.encoded_payload_bytes_upper_bound.0
        );
        assert!(facts.construction_resident_bytes_admission_budget.0 < 2 * 1024 * 1024 * 1024);
        assert!(
            result_output(values, 257, 3, 60_000, 20_000)
                .unwrap()
                .construction_resident_bytes_admission_budget
                > facts.construction_resident_bytes_admission_budget
        );
    }
}
