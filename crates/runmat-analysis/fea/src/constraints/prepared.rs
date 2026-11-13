//! Resolve each authored target occurrence once, then move its selection into lowering.
//! Routing consumes ownership evidence from the same selection; it does not rerun resolution.
use super::{
    target::resolve_nodal_target,
    workspace::{check_cancelled, reserve},
    ConstraintPlanError, ResolvedConstraintTarget,
};
use crate::physics::structural::topology::{
    explicit_target_node_indices,
    node_space::CanonicalStructuralNodeSpace,
    target::{resolve_mixed_structural_target, MixedStructuralTargetOwnership},
};
use runmat_analysis_core::{
    RegionTarget, StructuralConstraint, StructuralConstraintKind, StructuralModel,
    TargetCardinality,
};
use runmat_meshing_core::SolverMeshArtifact;

struct PreparedTarget {
    resolved: ResolvedConstraintTarget,
    nodes: Vec<usize>,
    requires_mixed: bool,
}

pub(super) struct PreparedTargets<'a> {
    selections: std::vec::IntoIter<(&'a RegionTarget, PreparedTarget)>,
    requires_mixed: bool,
}

impl<'a> PreparedTargets<'a> {
    pub(super) fn derive(
        constraint: &'a StructuralConstraint,
        mesh: Option<&SolverMeshArtifact>,
        mixed: Option<(&StructuralModel, &CanonicalStructuralNodeSpace)>,
    ) -> Result<Self, ConstraintPlanError> {
        check_cancelled()?;
        let profile = profile::derive(constraint, mesh, mixed.map(|(model, _)| model))?;
        let mut selections = reserve(profile.target_occurrences)?;
        let mut requires_mixed = mixed.is_some() && mesh.is_none();
        visit_targets(&constraint.kind, |target| {
            check_cancelled()?;
            let selection = if let Some((structural, topology)) = mixed {
                resolve_target(mesh, structural, topology, constraint, target)?
            } else {
                PreparedTarget {
                    resolved: resolve_nodal_target(
                        mesh.ok_or_else(|| {
                            ConstraintPlanError::Invalid(
                                "solid target preparation requires a mesh".into(),
                            )
                        })?,
                        target,
                        &constraint.constraint_id,
                    )?,
                    nodes: Vec::new(),
                    requires_mixed: false,
                }
            };
            requires_mixed |= selection.requires_mixed;
            selections.push((target, selection));
            Ok(())
        })?;
        Ok(Self {
            selections: selections.into_iter(),
            requires_mixed,
        })
    }

    /// Borrow original selections before expansion; this does not consume or resolve them.
    pub(super) fn selections(
        &self,
    ) -> impl ExactSizeIterator<Item = (&ResolvedConstraintTarget, &[usize])> {
        self.selections
            .as_slice()
            .iter()
            .map(|(_, selection)| (&selection.resolved, selection.nodes.as_slice()))
    }

    pub(super) fn requires_mixed(&self) -> bool {
        self.requires_mixed
    }

    /// Count authored term occurrences before equation-term expansion. Duplicates are
    /// deliberately retained here; the row owner combines them in original order.
    pub(super) fn remaining_node_count(&self) -> Result<usize, ConstraintPlanError> {
        self.selections
            .as_slice()
            .iter()
            .try_fold(0, |count, (_, selection)| {
                check_cancelled()?;
                super::workspace::count_add(
                    count,
                    if self.requires_mixed {
                        selection.nodes.len()
                    } else {
                        selection.resolved.matched_node_identities.len()
                    },
                )
            })
    }

    pub(super) fn take(
        &mut self,
        target: &RegionTarget,
    ) -> Result<(ResolvedConstraintTarget, Vec<usize>), ConstraintPlanError> {
        check_cancelled()?;
        let (authored, selection) = self.selections.next().ok_or_else(|| {
            ConstraintPlanError::Invalid("constraint lowering exhausted prepared targets".into())
        })?;
        if !std::ptr::eq(authored, target) {
            return Err(ConstraintPlanError::Invalid(
                "constraint lowering changed prepared target order".into(),
            ));
        }
        Ok((selection.resolved, selection.nodes))
    }

    pub(super) fn finish(self) -> Result<(), ConstraintPlanError> {
        if self.selections.len() != 0 {
            return Err(ConstraintPlanError::Invalid(
                "constraint lowering left unused prepared targets".into(),
            ));
        }
        Ok(())
    }
}

fn resolve_target(
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
) -> Result<PreparedTarget, ConstraintPlanError> {
    let ownership = if let Some(mesh) = mesh {
        resolve_mixed_structural_target(target, mesh, structural, topology)
            .map_err(|error| target_error(error, constraint))?
    } else {
        let mut canonical_nodes = explicit_target_node_indices(structural, &target.region_id)?;
        for node in &mut canonical_nodes {
            *node = topology.explicit_node_indices[*node];
        }
        if canonical_nodes.is_empty() {
            return Err(ConstraintPlanError::EmptyTarget {
                constraint_id: constraint.constraint_id.clone(),
                region_id: target.region_id.clone(),
            });
        }
        MixedStructuralTargetOwnership::Explicit { canonical_nodes }
    };
    let explicit_ownership = matches!(ownership, MixedStructuralTargetOwnership::Explicit { .. });
    let solid_ownership = matches!(ownership, MixedStructuralTargetOwnership::Solid { .. });
    let nodes = ownership.into_canonical_nodes();
    // A mesh node that carries realized beam or shell rotations is constrained in the mixed
    // node space, so a fixed support also fixes its rotations.
    let requires_mixed = !solid_ownership
        || nodes
            .iter()
            .any(|node| topology.nodes[*node].owns_explicit_rotation);
    let resolved = if !explicit_ownership {
        resolve_nodal_target(
            mesh.expect("solid target ownership requires a solver mesh"),
            target,
            &constraint.constraint_id,
        )?
    } else {
        let mut identities = super::workspace::reserve(nodes.len())?;
        identities.extend(nodes.iter().map(|node| topology.nodes[*node].identity));
        identities.sort_unstable();
        identities.dedup();
        if target.cardinality == TargetCardinality::One && identities.len() != 1 {
            return Err(ConstraintPlanError::Cardinality {
                constraint_id: constraint.constraint_id.clone(),
                actual: identities.len(),
            });
        }
        let selection_digest = super::target::identity::digest(
            target,
            &[] as &[runmat_meshing_core::PersistentEntityId],
            &identities,
        )?;
        ResolvedConstraintTarget {
            authored: target.clone(),
            matched_entities: Vec::new(),
            matched_node_identities: identities,
            selection_digest,
        }
    };
    Ok(PreparedTarget {
        resolved,
        nodes,
        requires_mixed,
    })
}

fn target_error(
    error: crate::physics::structural::topology::target::MixedStructuralTargetError,
    constraint: &StructuralConstraint,
) -> ConstraintPlanError {
    match error {
        crate::physics::structural::topology::target::MixedStructuralTargetError::Storage(
            error,
        ) => error.into(),
        other => ConstraintPlanError::Invalid(format!(
            "constraint `{}` target resolution failed: {other}",
            constraint.constraint_id
        )),
    }
}

fn visit_targets<'a>(
    kind: &'a StructuralConstraintKind,
    mut visit: impl FnMut(&'a runmat_analysis_core::RegionTarget) -> Result<(), ConstraintPlanError>,
) -> Result<(), ConstraintPlanError> {
    match kind {
        StructuralConstraintKind::Fixed { target }
        | StructuralConstraintKind::NormalDisplacement { target, .. }
        | StructuralConstraintKind::Symmetry { target }
        | StructuralConstraintKind::Antisymmetry { target } => visit(target),
        StructuralConstraintKind::Prescribed(values) => visit(&values.target),
        StructuralConstraintKind::PrescribedMotion(values) => visit(&values.target),
        StructuralConstraintKind::RigidCoupling(coupling)
        | StructuralConstraintKind::DistributingCoupling(coupling) => visit(&coupling.target),
        StructuralConstraintKind::Tie(pair) => {
            visit(&pair.primary)?;
            visit(&pair.secondary)
        }
        StructuralConstraintKind::Periodic(periodic) => {
            visit(&periodic.pair.primary)?;
            visit(&periodic.pair.secondary)
        }
        StructuralConstraintKind::Equation(equation) => {
            for term in &equation.terms {
                visit(&term.target)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;

pub(in crate::constraints) mod profile;
