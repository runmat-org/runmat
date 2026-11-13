//! Exact authored magnetic materials, constant volume current and boundary/gauge
//! intent compiled before the physical numerical kernel. No native admission.
use super::{
    MagneticVolumeData, MagnetostaticError, MagnetostaticResponse, VACUUM_PERMEABILITY_H_PER_M,
};
use crate::{
    excitation::{
        realize_step_excitations, ExcitationSamplePoint, RealizedExcitationKind,
        RealizedExcitationTarget,
    },
    topology::AnalysisProjection,
};
use runmat_analysis_core::*;
use runmat_meshing_core::{BoundaryFaceRole, SolverMeshArtifact, StableDigest};
use std::collections::{BTreeMap, BTreeSet};

pub fn validate_canonical_input<'a>(
    model: &'a AnalysisModel,
    step_id: &str,
    residual_target: f64,
) -> Result<&'a ProcedureStep, MagnetostaticError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(MagnetostaticError::InvalidInput(
            "relative residual target".into(),
        ));
    }
    validate_model(model).map_err(|e| MagnetostaticError::InvalidInput(e.to_string()))?;
    if !matches!(
        model.electromagnetic,
        Some(ElectromagneticDomain::Magnetostatic {
            formulation: MagnetostaticFormulation::MagneticVectorPotential
        })
    ) {
        return Err(MagnetostaticError::InvalidInput(
            "distinct static magnetic vector-potential intent required".into(),
        ));
    }
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == step_id)
        .ok_or_else(|| MagnetostaticError::InvalidInput("magnetic step not found".into()))?;
    crate::results::observation::validate_sampled_requests(step)
        .map_err(|error| MagnetostaticError::Unsupported(error.to_string()))?;
    if step.family != PhysicsFamily::Electromagnetic
        || step.procedure != AnalysisProcedure::Static
        || step.solution_space != SolutionSpace::Direct
        || step.independent_domain != IndependentDomain::Static
        || step.settings != ProcedureSettings::None
        || step.restart != RestartPolicy::Fresh
        || !step.state_inputs.is_empty()
        || step.state_outputs.iter().any(|output| {
            output.kind != StateKind::ElectromagneticStatic
                || output.completeness != StateCompleteness::Complete
        })
        || !step.activations.is_empty()
        || model.sequence.steps.len() != 1
        || !model.sequence.dependencies.is_empty()
        || !model.sequence.transitions.is_empty()
        || !model.sequence.branches.is_empty()
        || model.structural.is_some()
        || model.thermal.is_some()
        || model.thermo_mechanical.is_some()
        || model.electro_thermal.is_some()
        || model.cfd.is_some()
        || !model.connectors.is_empty()
        || !model.interfaces.is_empty()
        || !model.initial_conditions.is_empty()
        || !model.boundaries.is_empty()
        || !model.response_spectra.is_empty()
        || !model.random_excitations.is_empty()
    {
        return Err(MagnetostaticError::Unsupported(
            "standalone magnetic step or unsupported lifecycle/coupling/boundary".into(),
        ));
    }
    if model.excitations.iter().any(|action| {
        !matches!(
            action,
            ExcitationAction::MagneticVolumeCurrentDensity(_)
                | ExcitationAction::MagneticVectorPotential(_)
        )
    }) {
        return Err(MagnetostaticError::Unsupported(
            "magnetostatics takes magnetic_volume_current_density and magnetic_vector_potential loads".into(),
        ));
    }
    Ok(step)
}

pub(crate) struct CompiledMagneticInput {
    pub projection: AnalysisProjection,
    /// Exterior faces with tangential A = 0, sorted; every other exterior face is insulated.
    pub grounded_faces: Vec<StableDigest>,
    /// Grounded faces held at a nonzero tangential vector potential (Wb/m), sorted by face.
    pub prescribed_potential: Vec<super::PrescribedVectorPotential>,
    pub volumes: Vec<MagneticVolumeData>,
}

pub(crate) fn compile_canonical_input(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    residual_target: f64,
) -> Result<CompiledMagneticInput, MagnetostaticError> {
    let step = validate_canonical_input(model, step_id, residual_target)?;
    super::resources::require_canonical_ceiling(
        model,
        mesh,
        step.accuracy.resources.resident_byte_budget(),
    )?;
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
    let raw: BTreeMap<_, _> = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| (element.stable_identity, element))
        .collect();
    let matches_volume = |target: &RegionTarget, identity| {
        let original = raw[&identity];
        target.entity_dimension == EntityDimension::Volume
            && (target.matches_persistent_entity(&original.region_id)
                || original
                    .provenance
                    .iter()
                    .any(|id| target.matches_persistent_entity(id)))
    };
    let mut ground_faces = BTreeSet::new();
    let mut gauged = BTreeSet::new();
    for constraint in &model.constraints {
        if !step.object_is_active(&StepObjectRef::Constraint(
            constraint.constraint_id().into(),
        )) {
            continue;
        }
        let AnalysisConstraint::Magnetic(constraint) = constraint else {
            return Err(MagnetostaticError::Unsupported(
                "nonmagnetic constraint".into(),
            ));
        };
        match &constraint.kind {
            MagneticConstraintKind::VectorPotentialGround { target } => {
                target
                    .validate_resolution()
                    .map_err(|e| MagnetostaticError::InvalidInput(e.to_string()))?;
                if target.entity_dimension != EntityDimension::Face {
                    return Err(MagnetostaticError::Unsupported(
                        "tangential ground needs exact exterior faces".into(),
                    ));
                }
                let mut matched = 0;
                for face in projection.boundary_faces_for_target(target) {
                    if face.role != BoundaryFaceRole::Exterior {
                        return Err(MagnetostaticError::Unsupported(
                            "internal magnetic grounding".into(),
                        ));
                    }
                    ground_faces.insert(face.stable_identity);
                    matched += 1;
                }
                if matched == 0 {
                    return Err(MagnetostaticError::InvalidInput(
                        "unmatched magnetic ground".into(),
                    ));
                }
            }
            MagneticConstraintKind::CoulombGauge { target } => {
                target
                    .validate_resolution()
                    .map_err(|e| MagnetostaticError::InvalidInput(e.to_string()))?;
                let mut matched = 0;
                for element in &projection.elements {
                    if matches_volume(target, element.stable_identity) {
                        gauged.insert(element.stable_identity);
                        matched += 1;
                    }
                }
                if matched == 0 {
                    return Err(MagnetostaticError::InvalidInput(
                        "unmatched magnetic gauge volume".into(),
                    ));
                }
            }
            _ => {
                return Err(MagnetostaticError::Unsupported(
                    crate::admission::MAGNETIC_STATIC_CONSTRAINT.message.into(),
                ))
            }
        }
    }
    // The weak Coulomb gauge is a property of the whole field: one authored gauge fixes the
    // gradient part of A everywhere, in air, iron and conductors alike.
    if gauged.is_empty() {
        return Err(MagnetostaticError::Unsupported(
            crate::admission::MAGNETIC_STATIC_GAUGE.message.into(),
        ));
    }
    let properties = crate::properties::compile_solid_property_map(model, mesh)
        .map_err(|e| MagnetostaticError::InvalidInput(e.to_string()))?;
    let mut volumes = Vec::with_capacity(projection.elements.len());
    for element in &projection.elements {
        let material = &properties
            .solid_material(element.stable_identity)
            .ok_or_else(|| MagnetostaticError::InvalidInput("unassigned magnetic material".into()))?
            .material;
        let law = material
            .electrical
            .as_ref()
            .ok_or_else(|| MagnetostaticError::InvalidInput("missing permeability law".into()))?;
        // Magnetostatics uses only the permeability; a material's conductivity laws
        // (temperature or frequency dependence) belong to the analyses that use conductivity.
        volumes.push(MagneticVolumeData {
            element_identity: element.stable_identity,
            permeability_h_per_m: runmat_types::physical::MagneticPermeability::new(
                VACUUM_PERMEABILITY_H_PER_M.canonical_value()
                    * law.relative_permeability.canonical_value(),
            ),
            current_density_a_per_m2: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::ElectricCurrentDensity,
                _,
            >::new([0.0; 3]),
        });
    }
    let loads = realize_step_excitations(model, step, ExcitationSamplePoint::Static)
        .map_err(|e| MagnetostaticError::InvalidInput(e.to_string()))?;
    let mut prescribed_potential = Vec::new();
    for load in loads.iter() {
        if let RealizedExcitationKind::MagneticVectorPotential(value) = load.kind {
            // A known tangential vector potential on exterior faces: those faces are grounded
            // at that value rather than at zero.
            if value.imaginary != [0.0; 3] || value.real.iter().any(|v| !v.is_finite()) {
                return Err(MagnetostaticError::Unsupported(
                    "a static vector potential must be finite and real".into(),
                ));
            }
            let RealizedExcitationTarget::Region(target) = &load.target else {
                return Err(MagnetostaticError::Unsupported(
                    "a vector potential load needs a face region".into(),
                ));
            };
            let mut matched = 0;
            for face in projection.boundary_faces_for_target(target) {
                if face.role != BoundaryFaceRole::Exterior {
                    return Err(MagnetostaticError::Unsupported(
                        "a vector potential load applies to exterior faces".into(),
                    ));
                }
                ground_faces.insert(face.stable_identity);
                prescribed_potential.push((
                    face.stable_identity,
                    runmat_types::physical::Quantity::new(value.real),
                ));
                matched += 1;
            }
            if matched == 0 {
                return Err(MagnetostaticError::InvalidInput(
                    "unmatched vector potential load".into(),
                ));
            }
            continue;
        }
        let RealizedExcitationKind::MagneticVolumeCurrentDensity(value) = load.kind else {
            return Err(MagnetostaticError::Unsupported(
                "nonmagnetic current realization".into(),
            ));
        };
        if value.imaginary != [0.0; 3]
            || value.real.iter().any(|v| !v.is_finite())
            || load.policy.frame != CoordinateFrame::Global
            || load.policy.kinematics != ExcitationKinematics::FixedFrame
            || load.policy.distribution != ExcitationDistribution::IntensityPerVolume
            || load.policy.projection.preserve_moment
            || load.policy.projection.preserve_resultant
        {
            return Err(MagnetostaticError::Unsupported(
                "finite real fixed global volume current required".into(),
            ));
        }
        let signal = model
            .signals
            .iter()
            .find(|signal| signal.signal_id() == &load.signal_id);
        if !matches!(signal, Some(PhysicalSignal::ElectricCurrentDensity(signal))
            if signal.frame == Some(CoordinateFrame::Global)
                && signal.shape == SignalShape::Vector3
                && signal.precision == SignalPrecision::Real64
                && matches!(signal.representation, SignalRepresentation::Constant(_)))
        {
            return Err(MagnetostaticError::Unsupported(
                "constant real global-frame vector current signal required".into(),
            ));
        }
        let mut matched = 0;
        for volume in &mut volumes {
            let selected = match &load.target {
                RealizedExcitationTarget::WholeModel => true,
                RealizedExcitationTarget::Region(target) => {
                    matches_volume(target, volume.element_identity)
                }
                _ => false,
            };
            if selected {
                matched += 1;
                volume.current_density_a_per_m2 = volume
                    .current_density_a_per_m2
                    .try_map_canonical_storage(|mut components| {
                        for (component, contribution) in components.iter_mut().zip(value.real) {
                            *component += contribution;
                        }
                        components
                    })
                    .map_err(|error| MagnetostaticError::InvalidInput(error.to_string()))?;
            }
        }
        if matched == 0 {
            return Err(MagnetostaticError::InvalidInput(
                "unmatched magnetic current volume".into(),
            ));
        }
    }
    if ground_faces.is_empty() {
        return Err(MagnetostaticError::Unsupported(
            "magnetostatics needs magnetic.vector_potential_ground on at least one exterior face (the far field); faces without a ground are magnetically insulated".into(),
        ));
    }
    prescribed_potential.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(CompiledMagneticInput {
        projection,
        grounded_faces: ground_faces.into_iter().collect(),
        prescribed_potential,
        volumes,
    })
}

pub fn solve_canonical_static(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    residual_target: f64,
) -> Result<MagnetostaticResponse, MagnetostaticError> {
    let compiled = compile_canonical_input(model, mesh, step_id, residual_target)?;
    super::solve_grounded_volume(
        &compiled.projection,
        &compiled.grounded_faces,
        &compiled.prescribed_potential,
        &compiled.volumes,
        residual_target,
    )
}
