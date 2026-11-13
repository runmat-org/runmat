//! Checked count/profile-only local requested-heap admission before compilation.
//! Sequential phases are summed, not presumed aliased. This is not measured RSS,
//! allocator overhead, encoded native transport policy, or scheduler admission.
//! Borrowed original model/mesh storage is excluded; owned validation/identity,
//! compiled source, sparse solve and retained/cold-validation copies are included.
use super::{MaxwellCurrentSample, MaxwellError, MaxwellFieldSample, MaxwellVolumeData};
use crate::{results::field::SampledOrientedEdge, topology::AnalysisProjection};
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, ElectricalConstraintKind, ExcitationAction,
};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_meshing_core::SolverMeshArtifact;

const LIMITS: CanonicalLimits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaxwellNumericalReservation {
    pub edge_count_upper_bound: u64,
    pub real_block_dimension_upper_bound: u64,
    pub support_entries_upper_bound: u64,
    pub quadrature_sample_count: u64,
    pub topology_and_properties_bytes: u64,
    pub compiled_and_retained_bytes: u64,
    pub assembly_and_solver_bytes: u64,
    pub total_bytes: u64,
}

/// Select only the borrowed authored policy before any model validation or
/// projection. Exact singleton/family validation follows admission in compilation.
pub(crate) fn require_selected_ceiling(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
) -> Result<MaxwellNumericalReservation, MaxwellError> {
    let mut steps = model
        .sequence
        .steps
        .iter()
        .filter(|s| s.step_id.as_str() == step_id);
    let step = steps
        .next()
        .ok_or_else(|| resource("selected step absent"))?;
    if steps.next().is_some() {
        return Err(resource("selected step is ambiguous"));
    }
    require_canonical_ceiling(model, mesh, step.accuracy.resources.resident_byte_budget())
}

pub(crate) fn require_canonical_ceiling(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    maximum: u64,
) -> Result<MaxwellNumericalReservation, MaxwellError> {
    let projection = AnalysisProjection::derivation_resident_bytes_estimate(mesh)
        .map_err(|e| resource(e.to_string()))?;
    let metadata = profile_serialized(model, LIMITS).map_err(|e| resource(e.to_string()))?;
    let mut selected = 0_u64;
    for material in &model.materials {
        selected = selected.max(
            profile_serialized(material, LIMITS)
                .map_err(|e| resource(e.to_string()))?
                .materialization_resident_bytes_estimate,
        );
    }
    let mut assignment = 0_u64;
    for value in &model.material_assignments {
        assignment = assignment.max(
            profile_serialized(value, LIMITS)
                .map_err(|e| resource(e.to_string()))?
                .materialization_resident_bytes_estimate,
        );
    }
    // Every current terminal contributes at most one voltage unknown and 13
    // element triplets (six column, six row, one diagonal). Activation and
    // conductor grouping can only reduce this authored upper bound.
    let current_drives = count(
        model
            .constraints
            .iter()
            .filter(|constraint| {
                matches!(
                    constraint,
                    AnalysisConstraint::Electrical(value)
                        if matches!(&value.kind, ElectricalConstraintKind::IntegralCurrent { .. })
                )
            })
            .count(),
    )?;
    let current_drives = add(
        current_drives,
        count(
            model
                .excitations
                .iter()
                .filter(|action| matches!(action, ExcitationAction::ElectricCurrent(_)))
                .count(),
        )?,
    )?;
    let reservation = from_counts(
        count(mesh.topology.nodes.len())?,
        count(mesh.topology.volume_elements.len())?,
        count(mesh.topology.boundary_faces.len())?,
        projection,
        metadata.materialization_resident_bytes_estimate,
        add(selected, assignment)?,
        current_drives,
    )?;
    if reservation.total_bytes > maximum {
        return Err(resource(format!(
            "requested local numerical heap {} exceeds authored maximum_resident_bytes {maximum}",
            reservation.total_bytes
        )));
    }
    Ok(reservation)
}

fn from_counts(
    nodes: u64,
    elements: u64,
    faces: u64,
    projection: u64,
    metadata: u64,
    selected: u64,
    current_drives: u64,
) -> Result<MaxwellNumericalReservation, MaxwellError> {
    // Lowest-order corner-edge E, including affine Tet10. Grounding only reduces these.
    let edges = mul(elements, 6)?;
    let dimension = add(edges, current_drives)?;
    let samples = mul(elements, 15)?;
    let entries = mul(elements, add(36, mul(current_drives, 13)?)?)?;
    let topology = sum(&[
        16_384,
        mul(projection, 2)?,
        mul(metadata, 4)?,
        mul(mul(elements, selected)?, 2)?,
        mul(nodes, 512)?,
        mul(edges, 256)?,
        mul(elements, 1024)?,
        mul(faces, 256)?,
    ])?;
    // Input sampling, compiled sources/laws and the retained response, each kept twice
    // (solve and cold re-evaluation), plus canonical identity encodings.
    let payload = sum(&[
        8192,
        slots::<SampledOrientedEdge>(edges, 2)?,
        // The sampled layout stores identities and corner indices per element and node; sample
        // identities and geometry are derived on demand.
        slots::<(runmat_meshing_core::StableDigest, [u32; 4])>(elements, 2)?,
        slots::<runmat_meshing_core::StableDigest>(nodes, 2)?,
        slots::<MaxwellCurrentSample>(samples, 2)?,
        slots::<MaxwellFieldSample>(samples, 2)?,
        slots::<MaxwellVolumeData>(elements, 2)?,
        mul(edges, 64)?,
    ])?;
    let retained = mul(payload, 3)?;
    // Original triplets, builder triplets, coalescing scratch and compressed
    // columns (also reused by cold residual certification), then sparse LU factors.
    // Nested-dissection fill of
    // a 3D edge-element operator grows like n^(4/3); 16 complex entries per n^(4/3)
    // bounds L and U together for the meshes this kernel factors.
    let fill = (dimension as f64).powf(4.0 / 3.0) * 16.0 * 32.0;
    if !fill.is_finite() || fill > u64::MAX as f64 / 4.0 {
        return Err(resource("sparse factor envelope overflow"));
    }
    let numerical = sum(&[
        mul(entries, 4 * 32)?,
        fill as u64,
        mul(dimension, 16 * 8)?,
        mul(nodes, 256)?,
        65_536,
    ])?;
    Ok(MaxwellNumericalReservation {
        edge_count_upper_bound: edges,
        real_block_dimension_upper_bound: mul(dimension, 2)?,
        support_entries_upper_bound: entries,
        quadrature_sample_count: samples,
        topology_and_properties_bytes: topology,
        compiled_and_retained_bytes: retained,
        assembly_and_solver_bytes: numerical,
        total_bytes: sum(&[topology, retained, numerical])?,
    })
}
fn slots<T>(count: u64, copies: u64) -> Result<u64, MaxwellError> {
    mul(mul(count, copies)?, std::mem::size_of::<T>() as u64)
}
fn count(value: usize) -> Result<u64, MaxwellError> {
    u64::try_from(value).map_err(|_| resource("cardinality overflow"))
}
fn add(a: u64, b: u64) -> Result<u64, MaxwellError> {
    a.checked_add(b)
        .ok_or_else(|| resource("count/byte sum overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64, MaxwellError> {
    a.checked_mul(b)
        .ok_or_else(|| resource("count/byte product overflow"))
}
fn sum(values: &[u64]) -> Result<u64, MaxwellError> {
    values.iter().try_fold(0, |n, &v| add(n, v))
}
fn resource(detail: impl Into<String>) -> MaxwellError {
    MaxwellError::Unsupported(format!(
        "Maxwell local resource preflight: {}",
        detail.into()
    ))
}

#[cfg(test)]
mod tests;
