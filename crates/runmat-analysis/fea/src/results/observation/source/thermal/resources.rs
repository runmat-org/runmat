//! Original manifest-validation facts plus counted projection/schedule phases.
use super::*;
use crate::results::observation::{ObservationFieldCatalogSource, ObservationPublicationCatalog};

pub(super) fn require(required: u64, maximum: u64) -> AnalysisExecutionResult<()> {
    if required > maximum {
        return Err(AnalysisExecutionError::Limit(
            "thermal source construction workspace",
        ));
    }
    Ok(())
}

impl ObservationPublicationCatalog {
    /// Additional workspace for sequential original manifest validation and projection/schedule
    /// construction and dependency indexes. Borrowed catalog/workload/mesh, other model validation, observation
    /// topology/target reconstruction, allocator overhead and RSS are excluded.
    /// Raw mesh inventories supply conservative projection facts; the original schedule owner
    /// derives its admitted inventory. This does not certify source identity or geometry;
    /// original-source validation remains mandatory after resource admission.
    pub fn thermal_source_construction_resident_bytes_admitted(
        &self,
        workload: &ThermalWorkload,
        mesh: &SolverMeshArtifact,
        maximum_resident_bytes: u64,
    ) -> AnalysisExecutionResult<u64> {
        let ObservationFieldCatalogSource::Thermal { manifest } = self.source().source() else {
            return Err(invalid("thermal construction source family differs"));
        };
        construction_resident_bytes_admitted(workload, manifest, mesh, maximum_resident_bytes)
    }

    /// Workspace of `validate_for` over a thermal source, beside the borrowed catalog,
    /// workload and mesh. Validation runs two stages in sequence:
    /// - the source authority (`thermal_source_construction_resident_bytes_admitted`), which
    ///   is dropped before the next stage;
    /// - re-derivation of the execution grouping: one projection, then the observation
    ///   topology over it.
    ///
    /// The peak is the larger stage. Copies of the catalog's own plan and selected values are
    /// bounded by the catalog's decoded size, which the caller accounts.
    pub fn thermal_source_validation_resident_bytes_admitted(
        &self,
        workload: &ThermalWorkload,
        mesh: &SolverMeshArtifact,
        maximum_resident_bytes: u64,
    ) -> AnalysisExecutionResult<u64> {
        let ObservationFieldCatalogSource::Thermal { manifest } = self.source().source() else {
            return Err(invalid("thermal validation source family differs"));
        };
        thermal_source_validation_resident_bytes(
            workload,
            manifest,
            self.plan(),
            mesh,
            maximum_resident_bytes,
        )
    }
}

/// Workspace of validating a thermal observation source against `plan` and re-deriving its
/// execution grouping, beside the borrowed workload, manifest and mesh. The stages run in
/// sequence and each drops its storage before the next:
/// - the source authority (projection, schedule, manifest and identities);
/// - one projection derivation;
/// - the retained projection with the observation topology over it.
///
/// The peak is the larger stage. Publication-catalog validation and observation-partition
/// evaluation both run these stages, so both read this law.
pub fn thermal_source_validation_resident_bytes(
    workload: &ThermalWorkload,
    manifest: &ThermalSolverResultManifest,
    plan: &crate::results::observation::ResolvedObservationPlan,
    mesh: &SolverMeshArtifact,
    maximum_resident_bytes: u64,
) -> AnalysisExecutionResult<u64> {
    // Thermal sources publish node and element fields; a face topology is never derived.
    if crate::results::observation::distributed::source_selection::observation_topology_faces(plan)
    {
        return Err(invalid("thermal observations select no faces or cells"));
    }
    let construction =
        construction_resident_bytes_admitted(workload, manifest, mesh, maximum_resident_bytes)?;
    let derivation =
        AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
    let retained =
        AnalysisProjection::retained_resident_bytes_upper_bound(mesh).map_err(invalid)?;
    let topology =
        crate::results::observation::solid_observation_topology_resident_bytes_for_mesh(mesh)
            .map_err(|error| invalid(error.to_string()))?;
    let grouping = retained
        .checked_add(topology)
        .ok_or(AnalysisExecutionError::Limit(
            "thermal source validation workspace",
        ))?
        .max(derivation);
    let bytes = construction.max(grouping);
    require(bytes, maximum_resident_bytes)?;
    Ok(bytes)
}

pub(in crate::results::observation) fn construction_resident_bytes_admitted(
    workload: &ThermalWorkload,
    manifest: &ThermalSolverResultManifest,
    mesh: &SolverMeshArtifact,
    maximum_resident_bytes: u64,
) -> AnalysisExecutionResult<u64> {
    let step = workload.validated_step_with_dependency_graph_budget(maximum_resident_bytes)?;
    let maximum_resident_bytes =
        maximum_resident_bytes.min(step.accuracy.resources.resident_byte_budget());
    let identities = workload.source_identity_resident_bytes_upper_bound()?;
    require(identities, maximum_resident_bytes)?;
    let validation = manifest.validation_resident_bytes_upper_bound()?;
    require(validation, maximum_resident_bytes)?;
    let projected =
        AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
    require(projected, maximum_resident_bytes)?;
    let retained =
        AnalysisProjection::retained_resident_bytes_upper_bound(mesh).map_err(invalid)?;
    require(retained, maximum_resident_bytes)?;
    let schedule = crate::execution::ThermalSolveSchedule::derive_admitted(
        &workload.model,
        step,
        workload.procedure,
        maximum_resident_bytes - retained,
    )
    .map_err(invalid)?;
    let bytes = retained
        .checked_add(schedule.construction_resident_bytes())
        .ok_or(AnalysisExecutionError::Limit(
            "thermal source construction workspace",
        ))?
        .max(projected)
        .max(validation)
        .max(identities);
    require(bytes, maximum_resident_bytes)?;
    Ok(bytes)
}
