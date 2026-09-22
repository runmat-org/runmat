//! Constraint admissibility over shared explicit-shell geometry.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::StructuralModel;
use runmat_meshing_core::StableDigest;

use crate::physics::structural::topology::{
    explicit_surface::{explicit_shell_surface_geometry, ExplicitShellSurfaceError},
    node_space::CanonicalStructuralNodeSpace,
};

use super::{ConstraintCouplingWeight, ConstraintPlanError, ResolvedConstraintTarget};

pub(super) struct ExplicitSurfaceGeometry {
    pub coordinates: BTreeMap<StableDigest, [f64; 3]>,
    pub weights: Vec<ConstraintCouplingWeight>,
    pub normals: BTreeMap<StableDigest, [f64; 3]>,
}

pub(super) fn resolve_explicit_surface(
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    target: &ResolvedConstraintTarget,
    constraint_id: &str,
) -> Result<ExplicitSurfaceGeometry, ConstraintPlanError> {
    let surface = explicit_shell_surface_geometry(structural, topology, &target.authored.region_id)
        .map_err(|error| map_surface_error(error, constraint_id))?;
    let selected = surface
        .coordinates_m
        .keys()
        .copied()
        .collect::<BTreeSet<_>>();
    if selected != target.matched_node_identities.iter().copied().collect() {
        return Err(ConstraintPlanError::UnsupportedTarget {
            constraint_id: constraint_id.into(),
            reason: "constraint target is not one complete explicit shell region",
        });
    }
    let total_measure = surface.area_m2;
    if !total_measure.is_finite() || total_measure <= 0.0 {
        return Err(ConstraintPlanError::NormalUnavailable {
            constraint_id: constraint_id.into(),
        });
    }
    let weights = surface
        .nodal_measures_m2
        .into_iter()
        .map(|(node_identity, measure)| ConstraintCouplingWeight {
            node_identity,
            weight: measure / total_measure,
        })
        .collect();
    Ok(ExplicitSurfaceGeometry {
        coordinates: surface.coordinates_m,
        weights,
        normals: surface.oriented_node_normals,
    })
}

fn map_surface_error(error: ExplicitShellSurfaceError, constraint_id: &str) -> ConstraintPlanError {
    match error {
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
