//! Authored singleton harmonic intent compiled to frozen canonical physical input.
//! The legacy-named MagneticVolumeCurrentDensity action is the shared impressed
//! volume J contract; boundary CurrentDensity is never reinterpreted as volume J.
mod source;
mod validation;

use super::{invalid, topology, MaxwellError, MaxwellInput, MaxwellResponse, MaxwellVolumeData};
use crate::results::field::{SampledVolumeFieldTopology, VolumeFieldSamplingRule};
use crate::topology::AnalysisProjection;
use runmat_analysis_core::*;
use runmat_meshing_core::{BoundaryFaceRole, SolverMeshArtifact};
use std::collections::BTreeSet;
pub use validation::validate_canonical_input;

pub fn compile_canonical_input(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    requirement: FormulationRequirement,
    residual_target: f64,
) -> Result<MaxwellInput, MaxwellError> {
    super::resources::require_selected_ceiling(model, mesh, step_id)?;
    let step = validate_canonical_input(model, step_id, requirement, residual_target)?;
    let IndependentDomain::FrequencyHertz(frequencies) = &step.independent_domain else {
        return Err(invalid("missing singleton harmonic frequency"));
    };
    let frequency_hz = frequencies[0];
    let projection = AnalysisProjection::derive(mesh).map_err(topology)?;
    super::geometry::spaces(&projection)?;
    let mut covered = BTreeSet::new();
    for constraint in &model.constraints {
        let AnalysisConstraint::Electrical(constraint) = constraint else {
            return Err(MaxwellError::Unsupported(
                "non-electric Maxwell boundary".into(),
            ));
        };
        let ElectricalConstraintKind::ElectricPerfectConductor { target } = &constraint.kind else {
            return Err(MaxwellError::Unsupported(
                "only homogeneous electric PEC is admitted".into(),
            ));
        };
        target.validate_resolution().map_err(invalid)?;
        if target.entity_dimension != EntityDimension::Face {
            return Err(invalid("PEC requires resolved face targets"));
        }
        let mut matched = false;
        for face in projection.boundary_faces_for_target(target) {
            if face.role != BoundaryFaceRole::Exterior {
                return Err(MaxwellError::Unsupported("internal PEC face".into()));
            }
            // Overlapping homogeneous PEC declarations have identical intent.
            covered.insert(face.stable_identity);
            matched = true;
        }
        if !matched {
            return Err(invalid("PEC target matches no canonical exterior face"));
        }
    }
    let exterior: BTreeSet<_> = projection
        .boundary_faces
        .iter()
        .filter(|face| face.role == BoundaryFaceRole::Exterior)
        .map(|face| face.stable_identity)
        .collect();
    if exterior.is_empty() || covered != exterior {
        return Err(MaxwellError::Unsupported(
            "explicit exact full-exterior PEC coverage required".into(),
        ));
    }
    let properties = crate::properties::compile_solid_property_map(model, mesh).map_err(invalid)?;
    let mut volumes = Vec::with_capacity(projection.elements.len());
    for element in &projection.elements {
        let material = &properties
            .solid_material(element.stable_identity)
            .ok_or_else(|| invalid("unassigned Maxwell material"))?
            .material;
        let law = material
            .electrical
            .as_ref()
            .ok_or_else(|| invalid("missing Maxwell material law"))?;
        if !law.conductivity_frequency_response.is_empty()
            || law.resistive_heating_coefficient != 0.0
        {
            return Err(MaxwellError::Unsupported(
                "dispersive or temperature-dependent Maxwell material".into(),
            ));
        }
        volumes.push(MaxwellVolumeData {
            element_identity: element.stable_identity,
            permeability_h_per_m: super::super::magnetic::VACUUM_PERMEABILITY_H_PER_M
                * law.relative_permeability,
            permittivity_f_per_m: super::super::electrical::VACUUM_PERMITTIVITY_F_PER_M
                * law.relative_permittivity,
            conductivity_s_per_m: law.conductivity_s_per_m,
        });
    }
    let sampling =
        SampledVolumeFieldTopology::derive(mesh, VolumeFieldSamplingRule::TetrahedronKeastFifteen)
            .map_err(invalid)?;
    let source_current = source::compile(model, mesh, step, frequency_hz, &sampling)?;
    let input = MaxwellInput {
        angular_frequency_rad_per_s: std::f64::consts::TAU * frequency_hz,
        sampling,
        volumes,
        source_current,
    };
    input.validate_for_mesh(mesh)?;
    Ok(input)
}

/// Compile original authored intent, then delegate to the retained physical kernel.
/// No independent solver, source clipping, gauge, or artificial damping is added.
pub fn solve_canonical_harmonic(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    requirement: FormulationRequirement,
    residual_target: f64,
) -> Result<MaxwellResponse, MaxwellError> {
    let input = compile_canonical_input(model, mesh, step_id, requirement, residual_target)?;
    super::solve_full_pec_volume(mesh, &input, residual_target)
}
