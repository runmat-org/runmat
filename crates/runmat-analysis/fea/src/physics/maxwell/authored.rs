//! Authored harmonic intent compiled to frozen canonical physical input, one per frequency.
//! The legacy-named MagneticVolumeCurrentDensity action is the shared impressed
//! volume J contract; boundary CurrentDensity is never reinterpreted as volume J.
//! Terminal currents and potentials drive solid conductors (see `drive`).
mod source;
mod validation;

use super::{invalid, topology, MaxwellError, MaxwellInput, MaxwellResponse, MaxwellVolumeData};
use crate::results::field::{SampledVolumeFieldTopology, VolumeFieldSamplingRule};
use crate::topology::AnalysisProjection;
use runmat_analysis_core::*;
use runmat_meshing_core::{BoundaryFaceRole, SolverMeshArtifact, StableDigest};
use std::collections::BTreeSet;
pub use validation::validate_canonical_input;

/// Authored analysis frequencies of the harmonic step, in authored order.
pub fn harmonic_frequencies(
    model: &AnalysisModel,
    step_id: &str,
    requirement: FormulationRequirement,
    residual_target: f64,
) -> Result<Vec<f64>, MaxwellError> {
    let step = validate_canonical_input(model, step_id, requirement, residual_target)?;
    let IndependentDomain::FrequencyHertz(frequencies) = &step.independent_domain else {
        return Err(invalid("missing harmonic frequencies"));
    };
    Ok(frequencies.clone())
}

/// Exterior faces with tangential E = 0, from `magnetic.vector_potential_ground`
/// (n x A = 0, hence n x E = 0 for E = -i omega A) and electric perfect conductors.
pub fn grounded_face_identities(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<Vec<StableDigest>, MaxwellError> {
    let projection = AnalysisProjection::derive(mesh).map_err(topology)?;
    grounded_faces(model, &projection)
}

fn grounded_faces(
    model: &AnalysisModel,
    projection: &AnalysisProjection,
) -> Result<Vec<StableDigest>, MaxwellError> {
    let mut covered = BTreeSet::new();
    for constraint in &model.constraints {
        let target = match constraint {
            AnalysisConstraint::Electrical(constraint) => match &constraint.kind {
                ElectricalConstraintKind::ElectricPerfectConductor { target } => target,
                _ => continue,
            },
            AnalysisConstraint::Magnetic(constraint) => match &constraint.kind {
                MagneticConstraintKind::VectorPotentialGround { target } => target,
                _ => {
                    return Err(MaxwellError::Unsupported(
                        "harmonic analysis admits magnetic.vector_potential_ground; conductivity and permittivity fix the gauge, so magnetic.coulomb_gauge, prescribed or periodic vector potentials are not admitted".into(),
                    ))
                }
            },
            _ => {
                return Err(MaxwellError::Unsupported(
                    "non-electromagnetic constraint in a harmonic analysis".into(),
                ))
            }
        };
        target.validate_resolution().map_err(invalid)?;
        if target.entity_dimension != EntityDimension::Face {
            return Err(invalid("vector-potential ground requires face targets"));
        }
        let mut matched = false;
        for face in projection.boundary_faces_for_target(target) {
            if face.role != BoundaryFaceRole::Exterior {
                return Err(MaxwellError::Unsupported(
                    "vector-potential ground applies to exterior faces".into(),
                ));
            }
            covered.insert(face.stable_identity);
            matched = true;
        }
        if !matched {
            return Err(invalid(format!(
                "ground {} matches no exterior face",
                constraint.constraint_id()
            )));
        }
    }
    Ok(covered.into_iter().collect())
}

pub fn compile_canonical_input(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    requirement: FormulationRequirement,
    residual_target: f64,
    frequency_hz: f64,
) -> Result<MaxwellInput, MaxwellError> {
    super::resources::require_selected_ceiling(model, mesh, step_id)?;
    let step = validate_canonical_input(model, step_id, requirement, residual_target)?;
    let IndependentDomain::FrequencyHertz(frequencies) = &step.independent_domain else {
        return Err(invalid("missing harmonic frequencies"));
    };
    if !frequencies
        .iter()
        .any(|value| value.to_bits() == frequency_hz.to_bits())
    {
        return Err(invalid(
            "frequency is not one of the step's authored frequencies",
        ));
    }
    let projection = AnalysisProjection::derive(mesh).map_err(topology)?;
    super::geometry::spaces(&projection)?;
    let grounded_faces = grounded_faces(model, &projection)?;
    if grounded_faces.is_empty() {
        return Err(MaxwellError::Unsupported(
            "a harmonic analysis needs a far field: ground the outer boundary with magnetic.vector_potential_ground".into(),
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
        if !law.conductivity_frequency_response.is_empty() {
            return Err(MaxwellError::Unsupported(
                "frequency-dependent conductivity in a harmonic analysis".into(),
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
    let omega = std::f64::consts::TAU * frequency_hz;
    let corner = super::drive::corner_elements(&projection)?;
    let grounded_nodes = super::geometry::grounded_nodes(&projection, &grounded_faces);
    let drives = super::drive::compile(&super::drive::DriveContext {
        model,
        step,
        projection: &projection,
        volumes: &volumes,
        corner: &corner,
        grounded_nodes: &grounded_nodes,
        omega,
        frequency_hz,
    })?;
    let input = MaxwellInput {
        angular_frequency_rad_per_s: omega,
        sampling,
        volumes,
        source_current,
        grounded_faces,
        drives,
    };
    input.validate_for_mesh(mesh)?;
    Ok(input)
}

/// Compile original authored intent at one frequency, then delegate to the physical kernel.
/// No independent solver, source clipping, gauge, or artificial damping is added.
pub fn solve_canonical_harmonic(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    requirement: FormulationRequirement,
    residual_target: f64,
    frequency_hz: f64,
) -> Result<MaxwellResponse, MaxwellError> {
    let input = compile_canonical_input(
        model,
        mesh,
        step_id,
        requirement,
        residual_target,
        frequency_hz,
    )?;
    super::solve(mesh, &input, residual_target)
}
