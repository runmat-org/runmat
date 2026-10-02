//! Native observations keep the original cell and ALL owner-oriented face entities.
use super::{
    topology::ObservationTopology, ObservationError, ObservationLimits, ObservationResult,
    ResolvedObservationPlan,
};
use crate::{
    results::field::{FiniteVolumeFieldTopology, RecoveredFieldCatalog, RecoveredFieldDefinition},
    topology::{AnalysisProjection, FiniteVolumeTopology},
};
use runmat_analysis_core::ProcedureStep;

/// Borrowed original-native discretization authority. Validation remains phase-local;
/// constructing this context never clones a topology or recovers numerical values.
#[derive(Clone, Copy)]
pub struct FiniteVolumeObservationSource<'a> {
    pub layout: &'a FiniteVolumeFieldTopology,
    pub projection: &'a AnalysisProjection,
    pub geometry: &'a FiniteVolumeTopology,
}

pub fn finite_volume_observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &super::ObservationPartitionPlan,
    partition: &super::ObservationPartition,
    source: FiniteVolumeObservationSource<'_>,
    limits: ObservationLimits,
) -> Result<Vec<usize>, ObservationError> {
    let FiniteVolumeObservationSource {
        layout,
        projection,
        geometry,
    } = source;
    remaining_limits(
        layout,
        projection,
        geometry,
        limits,
        metadata_resident_bytes(plan)?,
    )?;
    super::sampled::validate_plan(plan)?;
    let topology = ObservationTopology::from_finite_volume(layout, projection, geometry)?;
    super::distributed::source_selection::observation_partition_source_indices(
        plan,
        partition_plan,
        partition,
        &topology,
    )
}

pub fn prepare_finite_volume_observation_field_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &super::ObservationPartitionPlan,
    partition: &super::ObservationPartition,
    source: FiniteVolumeObservationSource<'_>,
    field: &crate::results::field::RecoveredField,
    limits: ObservationLimits,
) -> Result<crate::results::field::RecoveredFieldPartitionSet, ObservationError> {
    let FiniteVolumeObservationSource {
        layout,
        projection,
        geometry,
    } = source;
    remaining_limits(
        layout,
        projection,
        geometry,
        limits,
        metadata_resident_bytes(plan)?,
    )?;
    super::sampled::validate_plan(plan)?;
    let topology = ObservationTopology::from_finite_volume(layout, projection, geometry)?;
    super::distributed::source_selection::prepare_observation_field_partition(
        plan,
        partition_plan,
        partition,
        field,
        &topology,
    )
}

pub fn evaluate_finite_volume_observation_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &super::ObservationPartitionPlan,
    partition: &super::ObservationPartition,
    source: FiniteVolumeObservationSource<'_>,
    fields: &crate::results::field::RecoveredFieldPartitionSet,
    limits: ObservationLimits,
) -> Result<super::ObservationPartitionContribution, ObservationError> {
    let FiniteVolumeObservationSource {
        layout,
        projection,
        geometry,
    } = source;
    remaining_limits(
        layout,
        projection,
        geometry,
        limits,
        metadata_resident_bytes(plan)?,
    )?;
    super::sampled::validate_plan(plan)?;
    let topology = ObservationTopology::from_finite_volume(layout, projection, geometry)?;
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "finite-volume partition topology differs from original plan",
        ));
    }
    super::distributed::evaluate_observation_partition_with_attribution(
        plan,
        partition_plan,
        partition,
        fields,
        &topology,
        None,
    )
}

/// Additional requested heap before any owned topology clone or source
/// revalidation. Original inputs are borrowed, not charged as new storage.
pub fn finite_volume_observation_topology_resident_bytes(
    layout: &FiniteVolumeFieldTopology,
    projection: &AnalysisProjection,
    geometry: &FiniteVolumeTopology,
) -> Result<u64, ObservationError> {
    let overflow = || ObservationError::Limit("finite-volume observation topology bytes");
    let layout_bytes = runmat_canonical_codec::serialized_materialization_resident_bytes_estimate(
        layout,
        super::OBSERVATION_LIMITS,
    )
    .map_err(invalid)?;
    let mut bytes = projection
        .clone_resident_bytes_estimate()
        .map_err(invalid)?
        .checked_add(
            FiniteVolumeTopology::reserve(projection, geometry.policy)
                .map_err(invalid)?
                .scratch_bytes as u64,
        )
        .and_then(|v| v.checked_add(layout_bytes.checked_mul(2)?))
        .ok_or_else(overflow)?;
    for (count, width) in [
        (
            projection.nodes.len(),
            2 * std::mem::size_of::<super::topology::ObservationNode>()
                + 3 * (std::mem::size_of::<(u64, Vec<usize>)>() + 4 * std::mem::size_of::<usize>()),
        ),
        (
            geometry.cells.len(),
            8 * std::mem::size_of::<super::topology::ObservationElement>()
                + 3 * (std::mem::size_of::<(runmat_meshing_core::StableDigest, usize)>()
                    + 4 * std::mem::size_of::<usize>())
                + 12 * std::mem::size_of::<usize>(),
        ),
        (
            geometry.faces.len(),
            2 * std::mem::size_of::<super::topology::ObservationElement>()
                + 2 * std::mem::size_of::<Option<usize>>()
                + 6 * (std::mem::size_of::<usize>() + 4 * std::mem::size_of::<usize>()),
        ),
    ] {
        bytes = bytes
            .checked_add(
                (count as u64)
                    .checked_mul(width as u64)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
    }
    // Four transient quadrature labels plus final canonical cell label coexist.
    for cell in &geometry.cells {
        bytes = bytes
            .checked_add(
                (cell.region.source_topology_id.len() as u64)
                    .checked_mul(10)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
    }
    if bytes > isize::MAX as u64 {
        return Err(overflow());
    }
    Ok(bytes)
}

fn remaining_limits(
    layout: &FiniteVolumeFieldTopology,
    projection: &AnalysisProjection,
    geometry: &FiniteVolumeTopology,
    mut limits: ObservationLimits,
    metadata_bytes: u64,
) -> Result<ObservationLimits, ObservationError> {
    let bytes = finite_volume_observation_topology_resident_bytes(layout, projection, geometry)?
        .checked_add(metadata_bytes)
        .ok_or(ObservationError::Limit(
            "finite-volume observation metadata residency",
        ))?;
    if bytes > limits.maximum_peak_resident_bytes {
        return Err(ObservationError::Limit(
            "finite-volume observation topology residency",
        ));
    }
    limits.maximum_peak_resident_bytes -= bytes;
    Ok(limits)
}

fn metadata_resident_bytes<T: serde::Serialize>(value: &T) -> Result<u64, ObservationError> {
    // Borrowed profiling precedes identity encoding and spec/topology clones.
    // Reserve canonical materialization and its encoded identity buffer together.
    let profile = runmat_canonical_codec::profile_serialized(value, super::OBSERVATION_LIMITS)
        .map_err(invalid)?;
    profile
        .materialization_resident_bytes_estimate
        .checked_add(profile.encoded_bytes_upper_bound)
        .ok_or(ObservationError::Limit(
            "finite-volume observation metadata residency",
        ))
}

impl ObservationTopology {
    pub(super) fn from_finite_volume(
        layout: &FiniteVolumeFieldTopology,
        projection: &AnalysisProjection,
        geometry: &FiniteVolumeTopology,
    ) -> Result<Self, ObservationError> {
        layout.validate_for(projection, geometry).map_err(invalid)?;
        let mut result = Self::from_projection(projection);
        result.identity = None;
        result.set_finite_volume_digest(layout.identity().discretization_digest());
        result.volume_integration_points.clear();
        result.volume_elements = geometry
            .cells
            .iter()
            .map(|cell| super::topology::ObservationElement {
                identity: cell.element_identity,
                region_id: cell.region.source_topology_id.clone(),
                centroid_m: cell.centroid_m,
            })
            .collect();
        result.oriented_faces = layout
            .oriented_faces()
            .iter()
            .zip(&geometry.faces)
            .map(|(face, geo)| super::topology::ObservationElement {
                identity: face.identity,
                region_id: String::new(),
                centroid_m: geo.centroid_m,
            })
            .collect();
        result.face_boundary_sources = geometry
            .faces
            .iter()
            .map(|f| f.boundary_source.map(|b| b.projection_boundary_index))
            .collect();
        result.finite_volume = Some(layout.clone());
        Ok(result)
    }
}
fn invalid(e: impl std::fmt::Display) -> ObservationError {
    ObservationError::Invalid(e.to_string())
}

pub fn resolve_finite_volume_observation_plan(
    step: &ProcedureStep,
    layout: &FiniteVolumeFieldTopology,
    projection: &AnalysisProjection,
    geometry: &FiniteVolumeTopology,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    super::resolve::validate_request_counts(step, limits)?;
    // Shared discrete-native envelope: final fields/extrema, never interpolated flux.
    super::sampled::validate_sampled_requests(step)?;
    let limits = remaining_limits(
        layout,
        projection,
        geometry,
        limits,
        metadata_resident_bytes(&(&step.observations, fields))?,
    )?;
    let topology = ObservationTopology::from_finite_volume(layout, projection, geometry)?;
    super::resolve::resolve_observation_plan_for_topology(
        step, &topology, None, None, fields, limits,
    )
}

pub fn evaluate_finite_volume_observations(
    plan: &ResolvedObservationPlan,
    layout: &FiniteVolumeFieldTopology,
    projection: &AnalysisProjection,
    geometry: &FiniteVolumeTopology,
    fields: &RecoveredFieldCatalog,
    limits: ObservationLimits,
) -> Result<Vec<ObservationResult>, ObservationError> {
    let remaining = remaining_limits(
        layout,
        projection,
        geometry,
        limits,
        metadata_resident_bytes(plan)?,
    )?;
    super::sampled::validate_plan(plan)?;
    let requested = plan.observations.iter().try_fold(0u64, |sum, o| {
        sum.checked_add(o.resolved_cost.peak_resident_bytes)
            .ok_or(ObservationError::Limit(
                "finite-volume observation coexisting output bytes",
            ))
    })?;
    if requested > remaining.maximum_peak_resident_bytes {
        return Err(ObservationError::Limit(
            "finite-volume observation coexisting output residency",
        ));
    }
    let topology = ObservationTopology::from_finite_volume(layout, projection, geometry)?;
    super::evaluate::evaluate_observations_for_topology(plan, &topology, fields)
}

/// Verify original authored coverage and independently resolve every retained
/// target against exact original cell/all-face layout; self-hashing is insufficient.
pub fn verify_finite_volume_observation_plan(
    step: &ProcedureStep,
    layout: &FiniteVolumeFieldTopology,
    projection: &AnalysisProjection,
    geometry: &FiniteVolumeTopology,
    fields: &[RecoveredFieldDefinition],
    plan: &ResolvedObservationPlan,
    limits: ObservationLimits,
) -> Result<(), ObservationError> {
    super::resolve::validate_request_counts(step, limits)?;
    let metadata = metadata_resident_bytes(plan)?
        .checked_add(metadata_resident_bytes(&(&step.observations, fields))?)
        .ok_or(ObservationError::Limit(
            "finite-volume observation metadata residency",
        ))?;
    let remaining = remaining_limits(layout, projection, geometry, limits, metadata)?;
    super::sampled::validate_sampled_requests(step)?;
    super::resolve::verify_authored_observations(step, plan)?;
    if plan
        .observations
        .iter()
        .any(|o| o.resolved_cost.peak_resident_bytes > remaining.maximum_peak_resident_bytes)
    {
        return Err(ObservationError::Limit(
            "finite-volume observation verification residency",
        ));
    }
    let topology = ObservationTopology::from_finite_volume(layout, projection, geometry)?;
    super::resolve::verify_finite_volume_observation_plan_for_topology(
        step, &topology, fields, plan,
    )
}

#[cfg(test)]
mod tests;
