//! Checked count-only numerical preflight. Additive requested-heap reservations,
//! not measured RSS, scheduler allocation, native admission or a release gate.
//! The borrowed model/mesh and canonical result encoding are outside this view.

use super::MagnetostaticError;
use crate::topology::AnalysisProjection;
use runmat_analysis_core::AnalysisModel;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_meshing_core::SolverMeshArtifact;

const LIMITS: CanonicalLimits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MagneticNumericalReservation {
    pub edge_count_upper_bound: u64,
    pub saddle_dimension_upper_bound: u64,
    pub quadrature_sample_count: u64,
    pub support_preparation_bytes: u64,
    pub projection_bytes: u64,
    pub property_and_source_bytes: u64,
    pub assembly_and_solver_bytes: u64,
    pub retained_response_bytes: u64,
    pub total_bytes: u64,
}

/// Check before canonical validation, projection/property construction or numerical
/// assembly. Shared projection profiling counts malformed raw connectivity and
/// arbitrary provenance strings without allocating the projected graph.
pub(crate) fn require_canonical_ceiling(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    maximum: u64,
) -> Result<MagneticNumericalReservation, MagnetostaticError> {
    let (nodes, elements, faces, projection, metadata, selected) =
        canonical_validation_facts(model, mesh)?;
    require_fits(
        from_counts(nodes, elements, faces, projection, metadata, selected)?,
        maximum,
    )
}

fn canonical_validation_facts(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<(u64, u64, u64, u64, u64, u64), MagnetostaticError> {
    let projection = AnalysisProjection::derivation_resident_bytes_estimate(mesh)
        .map_err(|error| resource(error.to_string()))?;
    let metadata = profile_serialized(model, LIMITS)
        .map_err(|error| resource(error.to_string()))?
        .materialization_resident_bytes_estimate;
    // Property compilation clones one material and audit per selected volume.
    // Reserve the largest complete authored material and assignment for every
    // element, rather than assuming identifiers/constitutive laws have fixed size.
    let mut material = 0;
    for value in &model.materials {
        material = material.max(
            profile_serialized(value, LIMITS)
                .map_err(|error| resource(error.to_string()))?
                .materialization_resident_bytes_estimate,
        );
    }
    let mut assignment = 0;
    for value in &model.material_assignments {
        assignment = assignment.max(
            profile_serialized(value, LIMITS)
                .map_err(|error| resource(error.to_string()))?
                .materialization_resident_bytes_estimate,
        );
    }
    Ok((
        count(mesh.topology.nodes.len())?,
        count(mesh.topology.volume_elements.len())?,
        count(mesh.topology.boundary_faces.len())?,
        projection,
        metadata,
        add(material, assignment)?,
    ))
}

/// Source-bound validation only: exact raw topology/provenance, support maps and
/// replicated selected properties. No matrix, Krylov storage or retained response.
pub(crate) fn canonical_validation_workspace(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<u64, MagnetostaticError> {
    let (nodes, elements, faces, projection, metadata, selected) =
        canonical_validation_facts(model, mesh)?;
    let (support, properties) = validation_components(nodes, elements, faces, metadata, selected)?;
    // validate_for_mesh independently regenerates the sampled layout: node and element
    // identities, element corner indices and the derived oriented edges (sample identities
    // and geometry are derived on demand, never stored). Four slots cover vector growth and
    // concurrent owned/validated copies; uniqueness trees are already included in support.
    // Canonical identity materialization is charged separately by the retained-copy phase's
    // authenticated full source codec profile.
    use crate::results::field::SampledOrientedEdge;
    use runmat_meshing_core::StableDigest;
    let sampling = sum(&[
        mul(mul(nodes, 4)?, std::mem::size_of::<StableDigest>() as u64)?,
        mul(
            mul(elements, 4)?,
            std::mem::size_of::<(StableDigest, [u32; 4])>() as u64,
        )?,
        mul(
            mul(mul(elements, 6)?, 4)?,
            std::mem::size_of::<SampledOrientedEdge>() as u64,
        )?,
    ])?;
    sum(&[support, mul(projection, 3)?, properties, sampling])
}

fn validation_components(
    nodes: u64,
    elements: u64,
    boundary_faces: u64,
    model_metadata_bytes: u64,
    selected_metadata_bytes: u64,
) -> Result<(u64, u64), MagnetostaticError> {
    let edges = mul(elements, 6)?;
    let support = sum(&[
        16_384,
        mul(nodes, 2048)?,
        mul(edges, 2048)?,
        mul(elements, 8192)?,
        mul(boundary_faces, 2048)?,
    ])?;
    // Four overlapping owned/audit/identity representations per selected volume.
    let properties = sum(&[
        mul(model_metadata_bytes, 4)?,
        mul(mul(elements, selected_metadata_bytes)?, 4)?,
        mul(elements, 1024)?,
    ])?;
    Ok((support, properties))
}

fn from_counts(
    nodes: u64,
    elements: u64,
    boundary_faces: u64,
    projection_bytes: u64,
    model_metadata_bytes: u64,
    selected_metadata_bytes: u64,
) -> Result<MagneticNumericalReservation, MagnetostaticError> {
    // Sharing/grounding only reduce these bounds. Tet10 geometry still has six
    // lowest-order corner-edge unknowns, not ten-node vector unknowns.
    let edges = mul(elements, 6)?;
    let dimension = add(edges, nodes)?;
    let samples = mul(elements, 15)?;
    usize::try_from(dimension).map_err(|_| resource("host dimension overflow"))?;
    usize::try_from(samples).map_err(|_| resource("host sample count overflow"))?;
    // Shared GMRES reports a u32 counter; do not authorize a saturating policy
    // whose nominal iteration count cannot be represented by that counter.
    let host_dimension =
        usize::try_from(dimension).map_err(|_| resource("host dimension overflow"))?;

    // All sequential support phases are added: seen-node identities, edge
    // inventory/key/index maps and retained oriented-edge/element records;
    // face owners/adjacency/exterior sets/queues; boundary vertex links and
    // neighbor sets; free-edge/gauge-row maps. Partial BTree nodes, headers and
    // vector growth receive deliberately loose per-reference allowances.
    // Shared selected-property owned payload plus identity/audit materialization,
    // source realization metadata and simultaneous input/assembled volume arrays.
    let (support, properties) = validation_components(
        nodes,
        elements,
        boundary_faces,
        model_metadata_bytes,
        selected_metadata_bytes,
    )?;

    // Element contribution slots, not observed nonzeros: physical K has at most
    // 36 per tetrahedron; saddle has 36 + 2*24. No gauge-gauge entries. BTree
    // support preparation, retained plan, validation rederivation, cloned planned
    // assemblers and finished CSR are additive, never presumed aliased.
    let support_entries = mul(elements, 120)?;
    let matrices = sum(&[
        mul(support_entries, 512)?,
        mul(add(add(edges, dimension)?, 2)?, 256)?,
    ])?;
    // The saddle is factored directly by the shared sparse layer (symmetric indefinite,
    // nested dissection); reserve its count-based factor model. RHS/solution and post-solve
    // physical/gauge check buffers are reserved separately.
    let workspace = if host_dimension == 0 {
        0
    } else {
        runmat_sparse_solvers::estimate::direct_reservation_bytes(
            host_dimension,
            usize::try_from(support_entries)
                .map_err(|_| resource("saddle entry count overflow"))?,
            false,
        )
    };
    let krylov = sum(&[workspace, mul(mul(dimension, 10)?, 16)?, 4096])?;
    let numerical = add(matrices, krylov)?;

    // Retained response plus its sampled layout: per edge slot a circulation (8) and a derived
    // oriented edge (96); per sample A and B (48); per element mean B and permeability (32)
    // plus identity and corner indices (48); per node an identity (32). Four slots cover
    // growth and a concurrent retained-result clone, never presumed to share arrays. This
    // does not reserve encoded result payloads or native field publication.
    let retained = sum(&[
        4096,
        mul(edges, 4 * (8 + 96))?,
        mul(samples, 4 * 48)?,
        mul(nodes, 4 * 32)?,
        mul(elements, 4 * (32 + 48))?,
    ])?;
    let total = sum(&[
        support,
        mul(projection_bytes, 3)?,
        properties,
        numerical,
        retained,
    ])?;
    Ok(MagneticNumericalReservation {
        edge_count_upper_bound: edges,
        saddle_dimension_upper_bound: dimension,
        quadrature_sample_count: samples,
        support_preparation_bytes: support,
        projection_bytes,
        property_and_source_bytes: properties,
        assembly_and_solver_bytes: numerical,
        retained_response_bytes: retained,
        total_bytes: total,
    })
}

fn require_fits(
    reservation: MagneticNumericalReservation,
    maximum: u64,
) -> Result<MagneticNumericalReservation, MagnetostaticError> {
    if reservation.total_bytes > maximum {
        return Err(resource(format!(
            "requested numerical heap {} exceeds authored maximum_resident_bytes {maximum}",
            reservation.total_bytes
        )));
    }
    Ok(reservation)
}
fn count(value: usize) -> Result<u64, MagnetostaticError> {
    u64::try_from(value).map_err(|_| resource("cardinality overflow"))
}
fn add(left: u64, right: u64) -> Result<u64, MagnetostaticError> {
    left.checked_add(right)
        .ok_or_else(|| resource("byte/count sum overflow"))
}
fn mul(left: u64, right: u64) -> Result<u64, MagnetostaticError> {
    left.checked_mul(right)
        .ok_or_else(|| resource("byte/count product overflow"))
}
fn sum(values: &[u64]) -> Result<u64, MagnetostaticError> {
    values.iter().try_fold(0, |total, &value| add(total, value))
}
fn resource(detail: impl Into<String>) -> MagnetostaticError {
    MagnetostaticError::Unsupported(format!(
        "magnetic numerical resource preflight: {}",
        detail.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnetic_count_envelope_is_checked_additive_and_has_exact_ceiling() {
        let reservation = from_counts(27, 48, 48, 1000, 500, 200).unwrap();
        assert_eq!(reservation.edge_count_upper_bound, 288);
        assert_eq!(reservation.saddle_dimension_upper_bound, 315);
        assert_eq!(reservation.quadrature_sample_count, 720);
        assert_eq!(
            reservation.total_bytes,
            reservation.support_preparation_bytes
                + 3 * reservation.projection_bytes
                + reservation.property_and_source_bytes
                + reservation.assembly_and_solver_bytes
                + reservation.retained_response_bytes
        );
        assert_eq!(
            require_fits(reservation, reservation.total_bytes).unwrap(),
            reservation
        );
        assert!(require_fits(reservation, reservation.total_bytes - 1).is_err());
        let larger = from_counts(27, 48, 48, 1001, 501, 201).unwrap();
        assert!(larger.total_bytes > reservation.total_bytes);
    }

    #[test]
    fn magnetic_count_and_iteration_overflow_fail_before_materialization() {
        assert!(from_counts(1, u64::MAX, 1, 0, 0, 0).is_err());
        assert!(from_counts(u64::MAX, 1, 1, 0, 0, 0).is_err());
        assert!(from_counts(1, 1, 1, u64::MAX, 0, 0).is_err());
        assert!(from_counts(1, 1, 1, 0, 0, u64::MAX).is_err());
        // The direct factor reservation grows with the unknown count; there is no
        // iteration-counter ceiling any more.
        let large = from_counts(u64::from(u32::MAX), 0, 0, 0, 0, 0).unwrap();
        assert!(large.total_bytes > 256 * (1u64 << 30));
    }

    #[test]
    fn canonical_preflight_counts_provenance_and_rejects_tiny_ceiling_without_projection() {
        use crate::fixtures::{fixture_model, FixtureId};
        use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mut mesh = canonical_tetrahedron_solver_mesh(order);
            let base = require_canonical_ceiling(&model, &mesh, u64::MAX).unwrap();
            assert!(require_canonical_ceiling(&model, &mesh, base.total_bytes - 1).is_err());
            assert!(require_canonical_ceiling(&model, &mesh, 0).is_err());
            mesh.topology.volume_elements[0]
                .region_id
                .assembly_path
                .push("instance".repeat(512));
            let grown = require_canonical_ceiling(&model, &mesh, u64::MAX).unwrap();
            assert!(grown.total_bytes > base.total_bytes);
            // No canonical validation/geometry constructor ran: a stale digest
            // is still profiled so its later validation scratch is reserved.
            mesh.topology.volume_elements[0].node_ids.extend(100..200);
            assert!(
                require_canonical_ceiling(&model, &mesh, u64::MAX)
                    .unwrap()
                    .total_bytes
                    > grown.total_bytes
            );
        }
    }

    #[test]
    fn magnetic_retained_validation_amplifies_selected_metadata_per_volume_and_checks_overflow() {
        use crate::physics::{electrical::tests::mesh, magnetic::test_support::authored_model};
        use runmat_meshing_core::ElementOrder;
        let base = authored_model(0.0);
        let mut large = base.clone();
        large.materials[0]
            .name
            .push_str(&"long magnetic material metadata ".repeat(2048));
        let one = mesh(ElementOrder::Tet4, 1);
        let eight = mesh(ElementOrder::Tet4, 2);
        let small_growth = canonical_validation_workspace(&large, &one).unwrap()
            - canonical_validation_workspace(&base, &one).unwrap();
        let large_growth = canonical_validation_workspace(&large, &eight).unwrap()
            - canonical_validation_workspace(&base, &eight).unwrap();
        assert!(small_growth > 0);
        assert!(large_growth > 4 * small_growth);
        assert!(validation_components(u64::MAX, 1, 1, 1, 1).is_err());
        assert!(validation_components(1, u64::MAX, 1, 1, 1).is_err());
        assert!(validation_components(1, 1, 1, 1, u64::MAX).is_err());
    }
}
