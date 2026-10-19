//! Shared bounded canonical fluid source compilation. No procedure is rewritten.
use super::{invalid, AnalysisExecutionResult, FluidSteadyWorkload, LIMITS};
use crate::{
    physics::cfd::canonical::{
        CanonicalFluidInput, CanonicalFluidPolicy, CanonicalFluidProblem, CanonicalFluidReservation,
    },
    properties::{CompiledPropertyMapIdentity, PreparedProperties},
    topology::{AnalysisProjection, FiniteVolumeProjectionPolicy, FiniteVolumeTopology},
};
use runmat_analysis_core::{
    validate_model, AnalysisConstraint, AnalysisModel, CfdDomain, FluidConstraint,
};
use runmat_canonical_codec::serialized_materialization_resident_bytes_estimate;
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::Serialize;
pub(crate) struct PreparedFluidSource {
    pub source: AnalysisProjection,
    pub topology: FiniteVolumeTopology,
    pub material: CfdDomain,
    pub property_identity: CompiledPropertyMapIdentity,
    pub property_audit_digest: Digest,
    pub constraints: Vec<FluidConstraint>,
    pub preparation_resident_bytes: u64,
}

impl PreparedFluidSource {
    pub(super) fn prepare<T: Serialize>(
        authority: &T,
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        geometry_policy: FiniteVolumeProjectionPolicy,
        policy: CanonicalFluidPolicy,
        ceiling: u64,
    ) -> AnalysisExecutionResult<Self> {
        // Count/profile only over borrowed inputs before validation/projection/compiler expansion.
        // These are requested-heap estimates, not allocator/RSS or scheduler accounting.
        if mesh
            .topology
            .volume_elements
            .iter()
            .any(|e| e.order != runmat_meshing_core::ElementOrder::Tet4)
        {
            return Err(invalid(
                "portable canonical fluid currently admits Tet4 only",
            ));
        }
        let geometry =
            FiniteVolumeTopology::reserve_mesh(mesh, geometry_policy).map_err(invalid)?;
        let numerical =
            CanonicalFluidReservation::for_counts(&geometry, geometry.face_incidence_count, policy)
                .map_err(invalid)?;
        let projection =
            AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
        // Two whole authored metadata profiles conservatively cover cloned constraints/material,
        // shared identity serialization and validation scratch coexisting with derived records.
        let metadata = serialized_materialization_resident_bytes_estimate(authority, LIMITS)
            .map_err(invalid)?;
        let base = add_bytes(
            add_bytes(projection, geometry.scratch_bytes as u64)?,
            add_bytes(metadata, metadata)?,
        )?;
        let preliminary = add_bytes(base, numerical.resident_bytes_upper_bound)?;
        require_bytes(preliminary, ceiling)?;
        let prepared = PreparedProperties::prepare(model, Some(mesh), ceiling - preliminary)
            .map_err(invalid)?;
        // Material source authentication independently recompiles: both maps can coexist.
        let preparation_resident_bytes = add_bytes(
            base,
            add_bytes(
                prepared.estimated_resident_bytes,
                prepared.estimated_resident_bytes,
            )?,
        )?;
        require_bytes(
            add_bytes(
                preparation_resident_bytes,
                numerical.resident_bytes_upper_bound,
            )?,
            ceiling,
        )?;
        validate_model(model).map_err(invalid)?;
        let source = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let topology = FiniteVolumeTopology::derive(&source, geometry_policy).map_err(invalid)?;
        let properties = prepared.materialize().map_err(invalid)?;
        let material = crate::study::derive_canonical_fluid_material(model, mesh, &properties)
            .map_err(invalid)?;
        let constraints = model
            .constraints
            .iter()
            .map(|c| match c {
                AnalysisConstraint::Fluid(c) => Ok(c.clone()),
                _ => Err(invalid("nonfluid constraint")),
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        Ok(PreparedFluidSource {
            source,
            topology,
            material,
            property_identity: properties.identity,
            property_audit_digest: properties.audit_digest,
            constraints,
            preparation_resident_bytes,
        })
    }
    pub(super) fn input<'a>(
        &'a self,
        model: &'a AnalysisModel,
        policy: CanonicalFluidPolicy,
    ) -> CanonicalFluidInput<'a> {
        CanonicalFluidInput {
            source: &self.source,
            topology: &self.topology,
            material: &self.material,
            constraints: &self.constraints,
            boundary_operators: &model.boundaries,
            signals: &model.signals,
            excitations: &model.excitations,
            policy,
        }
    }
}

impl PreparedFluidSource {
    pub fn problem<'a>(
        &'a self,
        workload: &'a FluidSteadyWorkload,
    ) -> AnalysisExecutionResult<CanonicalFluidProblem<'a>> {
        let problem = CanonicalFluidProblem::prepare(self.input(&workload.model, workload.policy))
            .map_err(invalid)?;
        require_bytes(
            add_bytes(
                self.preparation_resident_bytes,
                problem.reservation().resident_bytes_upper_bound,
            )?,
            workload.resident_ceiling()?,
        )?;
        Ok(problem)
    }
}

pub(super) fn add_bytes(a: u64, b: u64) -> AnalysisExecutionResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("canonical fluid preparation byte estimate overflow"))
}
pub(super) fn require_bytes(required: u64, maximum: u64) -> AnalysisExecutionResult<()> {
    if required > maximum || required > isize::MAX as u64 {
        return Err(invalid(format!(
            "canonical fluid preparation requested heap {required} exceeds ceiling {maximum}"
        )));
    }
    Ok(())
}
