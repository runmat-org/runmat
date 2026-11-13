use super::{assembly::ElectricalAssembly, ElectricalStaticError};
use crate::excitation::{RealizedExcitationKind, RealizedExcitationTarget};
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, ElectricalConstraintKind, EntityDimension, ExcitationAction,
    ProcedureStep, StepObjectRef,
};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeMap;

impl ElectricalAssembly {
    pub(super) fn with_constraints(
        mut self,
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        step: &ProcedureStep,
        sample: crate::excitation::ExcitationSamplePoint<'_>,
    ) -> Result<Self, ElectricalStaticError> {
        let indices = self
            .projection
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.stable_identity, index))
            .collect::<BTreeMap<_, _>>();
        let mut owners = vec![false; self.projection.nodes.len()];
        let mut ids = std::collections::BTreeSet::new();
        let mut pairings = Vec::new();
        for constraint in &model.constraints {
            if !step.object_is_active(&StepObjectRef::Constraint(
                constraint.constraint_id().into(),
            )) {
                continue;
            }
            let AnalysisConstraint::Electrical(constraint) = constraint else {
                return Err(ElectricalStaticError::Unsupported(
                    "non-electrical constraint".into(),
                ));
            };
            if constraint.constraint_id.is_empty() || !ids.insert(&constraint.constraint_id) {
                return Err(ElectricalStaticError::Boundary(
                    "empty or duplicate constraint identity".into(),
                ));
            }
            let (target, potential) = match &constraint.kind {
                ElectricalConstraintKind::PrescribedPotential { target, value } => {
                    let sampled = crate::excitation::sample_prescribed_potential(
                        value,
                        &model.signals,
                        sample,
                    )
                    .map_err(|error| ElectricalStaticError::Boundary(error.to_string()))?;
                    if sampled.imaginary != 0.0 || !sampled.real.is_finite() {
                        return Err(ElectricalStaticError::Boundary(
                            "static potential must be finite real".into(),
                        ));
                    }
                    (target, sampled.real)
                }
                ElectricalConstraintKind::PotentialReference { target } => (target, 0.0),
                ElectricalConstraintKind::Periodic(periodic) => {
                    pairings.push(crate::physics::scalar_constraints::ScalarPairing {
                        constraint_id: &constraint.constraint_id,
                        pair: &periodic.pair,
                        transform: &periodic.transform,
                        offset: 0.0,
                    });
                    continue;
                }
                _ => {
                    return Err(ElectricalStaticError::Unsupported(
                        "equipotential, floating, integral current or periodic constraint".into(),
                    ))
                }
            };
            target
                .validate_resolution()
                .map_err(|error| ElectricalStaticError::Boundary(error.to_string()))?;
            let resolved =
                crate::constraints::resolve_nodal_target(mesh, target, &constraint.constraint_id)
                    .map_err(|error| ElectricalStaticError::Boundary(error.to_string()))?;
            let mut nodes = Vec::with_capacity(resolved.matched_node_identities.len());
            for identity in resolved.matched_node_identities {
                let node = indices[&identity];
                if self.prescribed[node].is_some_and(|old| old != potential) {
                    return Err(ElectricalStaticError::Boundary(
                        "conflicting prescribed potentials".into(),
                    ));
                }
                if owners[node] {
                    return Err(ElectricalStaticError::Boundary(
                        "overlapping terminal constraint ownership".into(),
                    ));
                }
                owners[node] = true;
                self.prescribed[node] = Some(potential);
                nodes.push(node);
            }
            self.terminal_nodes
                .push((constraint.constraint_id.clone(), nodes));
        }
        self.relations =
            crate::physics::scalar_constraints::relations(mesh, &self.projection, &pairings)
                .map_err(ElectricalStaticError::Boundary)?;
        // `terminal_voltage` loads hold a terminal at a sampled potential (a source that may
        // vary in time), exactly as a prescribed potential does.
        if model
            .excitations
            .iter()
            .any(|action| matches!(action, ExcitationAction::TerminalVoltage(_)))
        {
            let loads = crate::excitation::realize_step_excitations(model, step, sample)
                .map_err(|error| ElectricalStaticError::InvalidInput(error.to_string()))?;
            for load in loads.iter() {
                let RealizedExcitationKind::TerminalVoltage(value) = load.kind else {
                    continue;
                };
                if value.imaginary != 0.0 || !value.real.is_finite() {
                    return Err(ElectricalStaticError::Boundary(
                        "terminal voltage must be finite real".into(),
                    ));
                }
                let RealizedExcitationTarget::Region(target) = &load.target else {
                    return Err(ElectricalStaticError::Boundary(
                        "terminal voltage requires a face region".into(),
                    ));
                };
                if target.entity_dimension != EntityDimension::Face {
                    return Err(ElectricalStaticError::Boundary(
                        "terminal voltage requires a face region".into(),
                    ));
                }
                let id = load.excitation_id.0.as_str();
                let mut nodes = std::collections::BTreeSet::new();
                for face in self.projection.boundary_faces_for_target(target) {
                    nodes.extend(face.node_indices.iter().copied());
                }
                if nodes.is_empty() {
                    return Err(ElectricalStaticError::Boundary(format!(
                        "terminal voltage {id} matches no boundary face"
                    )));
                }
                for &node in &nodes {
                    if owners[node] {
                        return Err(ElectricalStaticError::Boundary(
                            "overlapping terminal ownership".into(),
                        ));
                    }
                    owners[node] = true;
                    self.prescribed[node] = Some(value.real);
                }
                self.terminal_nodes
                    .push((id.to_string(), nodes.into_iter().collect()));
            }
        }
        Ok(self)
    }
}
