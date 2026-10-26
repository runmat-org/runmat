//! Constraint admissibility over shared explicit-shell geometry.

use crate::topology::storage::{self, SortedValues};

use runmat_analysis_core::StructuralModel;
use runmat_meshing_core::StableDigest;

use crate::physics::structural::topology::{
    explicit_surface::{
        explicit_shell_surface_geometry, ExplicitShellSurfaceError, ExplicitShellSurfaceNode,
    },
    node_space::CanonicalStructuralNodeSpace,
};

use super::{ConstraintCouplingWeight, ConstraintPlanError, ResolvedConstraintTarget};

pub(super) struct ExplicitSurfaceGeometry {
    pub nodes: SortedValues<StableDigest, ExplicitShellSurfaceNode>,
    pub weights: Vec<ConstraintCouplingWeight>,
}

pub(super) fn resolve_explicit_surface(
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    target: &ResolvedConstraintTarget,
    constraint_id: &str,
) -> Result<ExplicitSurfaceGeometry, ConstraintPlanError> {
    let surface = explicit_shell_surface_geometry(structural, topology, &target.authored.region_id)
        .map_err(|error| map_surface_error(error, constraint_id))?;
    if !surface
        .nodes
        .keys()
        .eq(target.matched_node_identities.iter())
    {
        return Err(ConstraintPlanError::UnsupportedTarget {
            constraint_id: constraint_id.into(),
            reason: "constraint target is not one complete explicit shell region",
        });
    }
    let total_measure = surface.area_m2.canonical_value();
    if !total_measure.is_finite() || total_measure <= 0.0 {
        return Err(ConstraintPlanError::NormalUnavailable {
            constraint_id: constraint_id.into(),
        });
    }
    let mut weights = storage::reserve(surface.nodes.len())?;
    weights.extend(
        surface
            .nodes
            .entries()
            .map(|(node_identity, node)| ConstraintCouplingWeight {
                node_identity: *node_identity,
                weight: node.nodal_measure_m2 / total_measure,
            }),
    );
    Ok(ExplicitSurfaceGeometry {
        nodes: surface.nodes,
        weights,
    })
}

fn map_surface_error(error: ExplicitShellSurfaceError, constraint_id: &str) -> ConstraintPlanError {
    match error {
        ExplicitShellSurfaceError::Storage(error) => error.into(),
        ExplicitShellSurfaceError::NonShellElement { .. }
        | ExplicitShellSurfaceError::EmptyRegion { .. } => ConstraintPlanError::UnsupportedTarget {
            constraint_id: constraint_id.into(),
            reason:
                "normal and reference-point surface constraints require an explicit shell region",
        },
        ExplicitShellSurfaceError::MissingNode {
            element_id,
            node_id,
        } => ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` shell `{element_id}` references absent node {node_id}"
        )),
        ExplicitShellSurfaceError::DegenerateTriangle { element_id } => {
            ConstraintPlanError::Invalid(format!(
                "constraint `{constraint_id}` shell `{element_id}` has zero area"
            ))
        }
        ExplicitShellSurfaceError::UndefinedNodeNormal { .. } => {
            ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            }
        }
    }
}
