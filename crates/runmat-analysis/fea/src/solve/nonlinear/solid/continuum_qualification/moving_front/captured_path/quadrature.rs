//! Reintegrate frozen accepted displacement trajectories; NEVER rerun equilibrium.
//! Stream one point's complete history so refinement does not retain all point states.
use super::*;
use crate::topology::quadrature::{tensor_tetrahedron_rule, visit_volume_rule};

pub(super) struct Snapshot {
    pub factor: f64,
    pub displacement: Vec<f64>,
    pub reduced_force: Vec<f64>,
}

pub(super) fn reintegrate(
    mesh: &SolverMeshArtifact,
    summary: &AssemblySummary,
    rule: &[(f64, f64)],
    snapshots: &[Snapshot],
    material: &IsotropicConstitutiveModel,
) -> Vec<Vec<f64>> {
    let projection = AnalysisProjection::derive(mesh).unwrap();
    let mut forces = vec![vec![0.0; summary.dof_count]; snapshots.len()];
    let mut volume = 0.0;
    for element in &projection.elements {
        let coordinates = element
            .node_indices
            .iter()
            .map(|&i| projection.nodes[i].coordinates_m.canonical_value())
            .collect::<Vec<_>>();
        let rows = element
            .node_indices
            .iter()
            .map(|&node| {
                [
                    StructuralDofKind::Ux,
                    StructuralDofKind::Uy,
                    StructuralDofKind::Uz,
                ]
                .map(|kind| summary.structural_dof_layout.index(node, kind).unwrap())
            })
            .collect::<Vec<_>>();
        visit_volume_rule(
            element.order,
            &coordinates,
            tensor_tetrahedron_rule(rule),
            |p| {
                assert!(p.volume_weight_m3 > 0.0);
                volume += p.volume_weight_m3;
                // A NEW point evolves from zero along EVERY captured accepted increment.
                // Never initialize it by interpolating old quadrature history.
                let mut state = PlasticityPointState::default();
                for (snapshot, force) in snapshots.iter().zip(&mut forces) {
                    let mut gradient = [[0.0; 3]; 3];
                    for (rows, shape_gradient) in rows.iter().zip(p.physical_shape_gradients()) {
                        for (component, row) in gradient.iter_mut().enumerate() {
                            for (axis, value) in row.iter_mut().enumerate() {
                                *value +=
                                    snapshot.displacement[rows[component]] * shape_gradient[axis];
                            }
                        }
                    }
                    let strain = [
                        gradient[0][0],
                        gradient[1][1],
                        gradient[2][2],
                        gradient[1][2] + gradient[2][1],
                        gradient[0][2] + gradient[2][0],
                        gradient[0][1] + gradient[1][0],
                    ];
                    let response = material.update(strain, &state).unwrap();
                    let [xx, yy, zz, yz, xz, xy] = response.stress_pa;
                    for (rows, g) in rows.iter().zip(p.physical_shape_gradients()) {
                        // Direct tensor virtual work, not production B-column assembly.
                        for (row, traction) in rows.iter().zip([
                            xx * g[0] + xy * g[1] + xz * g[2],
                            xy * g[0] + yy * g[1] + yz * g[2],
                            xz * g[0] + yz * g[1] + zz * g[2],
                        ]) {
                            force[*row] += p.volume_weight_m3 * traction;
                        }
                    }
                    state = response.state;
                }
            },
        )
        .unwrap();
    }
    assert!((volume - 1.0).abs() < 1e-10);
    forces
}

pub(super) fn compare(
    mesh: &SolverMeshArtifact,
    summary: &AssemblySummary,
    problem: &SolidNewtonProblem<'_>,
    snapshots: &[Snapshot],
    material: &IsotropicConstitutiveModel,
    cells: usize,
    quadratic: bool,
) {
    use crate::topology::quadrature::{
        GAUSS_LEGENDRE_EIGHT_UNIT, GAUSS_LEGENDRE_FIVE_UNIT, GAUSS_LEGENDRE_SIXTEEN_UNIT,
        GAUSS_LEGENDRE_TWELVE_UNIT,
    };
    let rules = [
        GAUSS_LEGENDRE_FIVE_UNIT.as_slice(),
        GAUSS_LEGENDRE_EIGHT_UNIT.as_slice(),
        GAUSS_LEGENDRE_TWELVE_UNIT.as_slice(),
        GAUSS_LEGENDRE_SIXTEEN_UNIT.as_slice(),
    ];
    let load_norm = norm(
        &problem
            .reduction
            .transpose_apply(&summary.physical_load)
            .unwrap(),
    );
    assert!(load_norm > 0.0);
    let mut previous: Option<Vec<Vec<f64>>> = None;
    for rule in rules {
        let forces = reintegrate(mesh, summary, rule, snapshots, material)
            .iter()
            .map(|force| problem.reduction.transpose_apply(force).unwrap())
            .collect::<Vec<_>>();
        for (index, (snapshot, force)) in snapshots.iter().zip(&forces).enumerate() {
            let scale = load_norm * snapshot.factor;
            let original_delta = norm(
                &force
                    .iter()
                    .zip(&snapshot.reduced_force)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>(),
            ) / scale;
            let refinement_delta = previous.as_ref().map(|coarse| {
                norm(
                    &force
                        .iter()
                        .zip(&coarse[index])
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                ) / scale
            });
            assert!(original_delta.is_finite());
            assert!(refinement_delta.is_none_or(|value| value.is_finite()));
            // Preserve the registered exact checks. No nonlinear threshold is weakened.
            if !quadratic || snapshot.factor <= 0.125 {
                assert!(original_delta < 1e-8,
                    "exact integration cells={cells} quadratic={quadratic} factor={} points={} delta={original_delta:e}",
                    snapshot.factor, rule.len().pow(3));
            }
            eprintln!("frozen-refinement cells={cells} quadratic={quadratic} factor={} points={} original_delta={original_delta:e} refinement_delta={refinement_delta:?}",
                snapshot.factor, rule.len().pow(3));
        }
        previous = Some(forces);
    }
}
