//! Geometry owned by explicit structural shell regions.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{StructuralElementKind, StructuralModel};
use runmat_meshing_core::StableDigest;

use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;

use super::{
    vector::{add, cross, magnitude, scale, subtract},
    ConstraintCouplingWeight, ConstraintPlanError, ResolvedConstraintTarget,
};

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
    let authored_nodes = structural
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    let coordinates = topology
        .nodes
        .iter()
        .map(|node| (node.identity, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut measures = BTreeMap::<StableDigest, f64>::new();
    let mut normal_sums = BTreeMap::<StableDigest, [f64; 3]>::new();
    let mut selected = BTreeSet::new();
    let mut shell_count = 0_usize;

    for element in structural
        .elements
        .iter()
        .filter(|element| element.region_id == target.authored.region_id)
    {
        let StructuralElementKind::Shell(shell) = &element.kind else {
            return Err(ConstraintPlanError::UnsupportedTarget {
                constraint_id: constraint_id.into(),
                reason: "normal and reference-point surface constraints require an explicit shell region",
            });
        };
        shell_count += 1;
        let identities = shell.node_ids.map(|node_id| {
            authored_nodes
                .get(&node_id)
                .and_then(|index| topology.explicit_node_indices.get(*index))
                .and_then(|index| topology.nodes.get(*index))
                .map(|node| node.identity)
                .ok_or_else(|| {
                    ConstraintPlanError::Invalid(format!(
                        "constraint `{constraint_id}` shell `{}` references absent node {node_id}",
                        element.element_id
                    ))
                })
        });
        let [a, b, c] = identities;
        let [a, b, c] = [a?, b?, c?];
        let points = [coordinates[&a], coordinates[&b], coordinates[&c]];
        let area_vector = cross(
            subtract(points[1], points[0]),
            subtract(points[2], points[0]),
        );
        let double_area = magnitude(area_vector);
        if !double_area.is_finite() || double_area <= 1.0e-18 {
            return Err(ConstraintPlanError::Invalid(format!(
                "constraint `{constraint_id}` shell `{}` has zero area",
                element.element_id
            )));
        }
        let nodal_measure = double_area / 6.0;
        for identity in [a, b, c] {
            selected.insert(identity);
            *measures.entry(identity).or_insert(0.0) += nodal_measure;
            normal_sums
                .entry(identity)
                .and_modify(|normal| *normal = add(*normal, area_vector))
                .or_insert(area_vector);
        }
    }

    if shell_count == 0 || selected != target.matched_node_identities.iter().copied().collect() {
        return Err(ConstraintPlanError::UnsupportedTarget {
            constraint_id: constraint_id.into(),
            reason: "constraint target is not one complete explicit shell region",
        });
    }
    let total_measure = measures.values().sum::<f64>();
    if !total_measure.is_finite() || total_measure <= 0.0 {
        return Err(ConstraintPlanError::NormalUnavailable {
            constraint_id: constraint_id.into(),
        });
    }
    let weights = measures
        .into_iter()
        .map(|(node_identity, measure)| ConstraintCouplingWeight {
            node_identity,
            weight: measure / total_measure,
        })
        .collect();
    let normals = normal_sums
        .into_iter()
        .map(|(identity, normal)| {
            let length = magnitude(normal);
            if !length.is_finite() || length <= 1.0e-12 {
                return Err(ConstraintPlanError::NormalUnavailable {
                    constraint_id: constraint_id.into(),
                });
            }
            Ok((identity, scale(normal, length.recip())))
        })
        .collect::<Result<_, _>>()?;
    Ok(ExplicitSurfaceGeometry {
        coordinates,
        weights,
        normals,
    })
}
