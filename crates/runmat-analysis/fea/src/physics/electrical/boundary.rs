use super::{assembly::ElectricalAssembly, ElectricalStaticError};
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, ElectricalConstraintKind, ProcedureStep, StepObjectRef,
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
        let mut owners = vec![None; self.projection.nodes.len()];
        let mut ids = std::collections::BTreeSet::new();
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
                if owners[node].is_some() {
                    return Err(ElectricalStaticError::Boundary(
                        "overlapping terminal constraint ownership".into(),
                    ));
                }
                owners[node] = Some(constraint.constraint_id.as_str());
                self.prescribed[node] = Some(potential);
                nodes.push(node);
            }
            self.terminal_nodes
                .push((constraint.constraint_id.clone(), nodes));
        }
        Ok(self)
    }
}
