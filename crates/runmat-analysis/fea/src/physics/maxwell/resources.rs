//! Checked count/profile-only local requested-heap admission before compilation.
//! Sequential phases are summed, not presumed aliased. This is not measured RSS,
//! allocator overhead, encoded native transport policy, or scheduler admission.
//! Borrowed original model/mesh storage is excluded; owned validation/identity,
//! compiled source, sparse solve and retained/cold-validation copies are included.
use super::{MaxwellCurrentSample, MaxwellError, MaxwellFieldSample, MaxwellVolumeData};
use crate::{
    results::field::{SampledOrientedEdge, VolumeFieldIntegrationSample},
    solve::krylov::{GmresPolicy, GmresWorkEnvelope},
    topology::AnalysisProjection,
};
use runmat_analysis_core::AnalysisModel;
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
    require_canonical_ceiling(model, mesh, step.accuracy.resources.maximum_resident_bytes)
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
    let reservation = from_counts(
        count(mesh.topology.nodes.len())?,
        count(mesh.topology.volume_elements.len())?,
        count(mesh.topology.boundary_faces.len())?,
        projection,
        metadata.materialization_resident_bytes_estimate,
        add(selected, assignment)?,
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
) -> Result<MaxwellNumericalReservation, MaxwellError> {
    // Lowest-order corner-edge E, including affine Tet10. Sharing and PEC only
    // reduce these bounds; no magnetic gauge or saddle dimensions are imported.
    let edges = mul(elements, 6)?;
    let dimension = mul(edges, 2)?;
    let samples = mul(elements, 15)?;
    let entries = mul(elements, 4 * 36)?;
    let host_dimension =
        usize::try_from(dimension).map_err(|_| resource("host dimension overflow"))?;
    for value in [edges, samples, entries] {
        let bytes = mul(value, 1024)?;
        if bytes > isize::MAX as u64 {
            return Err(resource("host allocation envelope overflow"));
        }
    }
    // Raw metadata, validation, selected material clones/audits, exact exterior
    // face ownership, edge maps, row trees and vector growth. Sum every phase.
    let topology = sum(&[
        16_384,
        mul(projection, 8)?,
        mul(metadata, 8)?,
        mul(mul(elements, selected)?, 8)?,
        mul(nodes, 2048)?,
        mul(edges, 4096)?,
        mul(elements, 8192)?,
        mul(faces, 2048)?,
    ])?;
    // Eight slots cover growing compiled/retained copies and independently
    // regenerated cold validation sampling. Identity codecs are charged below.
    let payload = sum(&[
        8192,
        mul(mul(nodes, 8)?, 32)?,
        mul(mul(elements, 8)?, 160)?,
        slots::<SampledOrientedEdge>(edges, 8)?,
        slots::<VolumeFieldIntegrationSample>(samples, 8)?,
        slots::<MaxwellCurrentSample>(samples, 8)?,
        slots::<MaxwellFieldSample>(samples, 8)?,
        slots::<MaxwellVolumeData>(elements, 8)?,
        mul(edges, 8 * (64 + 16))?,
    ])?;
    // A deliberately loose byte-per-logical-byte encoding allowance for owned
    // input/response identities and canonical validation temporaries (numeric
    // JSON values/digest text/container structure), not a transport guarantee.
    let retained = mul(payload, 16)?;
    // All four real complex blocks per local 6x6 contribution. Charge support
    // BTree nodes, exact/rederived plans, planned assemblers and finished CSR.
    let matrices = sum(&[mul(entries, 1024)?, mul(add(dimension, 1)?, 512)?])?;
    // The complete real block is factored by sparse LU in the shared sparse layer.
    let recurrence = if host_dimension == 0 {
        0
    } else {
        runmat_sparse_solvers::estimate::direct_reservation_bytes(
            host_dimension,
            usize::try_from(entries).map_err(|_| resource("block entry count overflow"))?,
            true,
        )
    };
    // Element basis/moment quadrature, source sums, corner-gradient charge tests,
    // all original RHS/solution/SpMV/residual buffers outside shared recurrence.
    let numerical = sum(&[
        matrices,
        recurrence,
        mul(dimension, 16 * 16)?,
        mul(nodes, 1024)?,
        mul(elements, 16_384)?,
        65_536,
    ])?;
    Ok(MaxwellNumericalReservation {
        edge_count_upper_bound: edges,
        real_block_dimension_upper_bound: dimension,
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
