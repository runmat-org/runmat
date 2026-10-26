use super::*;
use crate::physics::structural::plasticity::PlasticityPointState;

pub(super) fn measure(
    mesh: &SolverMeshArtifact,
    summary: &AssemblySummary,
    u: &[f64],
    history: &[PlasticityPointState],
    dissipation: f64,
    evaluation: &NonlinearSolidEvaluation,
) -> Errors {
    let projection = AnalysisProjection::derive(mesh).unwrap();
    let mut u_error = 0.0;
    let mut u_norm = 0.0;
    for element in &projection.elements {
        projection
            .visit_element_quadrature(element, |point| {
                let actual: [f64; 3] = std::array::from_fn(|axis| {
                    element
                        .node_indices
                        .iter()
                        .enumerate()
                        .map(|(local, &node)| {
                            let kind = [
                                StructuralDofKind::Ux,
                                StructuralDofKind::Uy,
                                StructuralDofKind::Uz,
                            ][axis];
                            point.shape_values()[local]
                                * u[summary.structural_dof_layout.index(node, kind).unwrap()]
                        })
                        .sum()
                });
                let reference = displacement(point.position_m[0]);
                u_error += point.volume_weight_m3
                    * ((actual[0] - reference).powi(2) + actual[1].powi(2) + actual[2].powi(2));
                u_norm += point.volume_weight_m3 * reference * reference;
            })
            .unwrap();
    }
    assert_eq!(evaluation.point_evidence.len(), history.len());
    let mut stress_error = 0.0;
    let mut stress_norm = 0.0;
    let mut alpha_error = 0.0;
    let mut alpha_norm = 0.0;
    for (point, state) in evaluation.point_evidence.iter().zip(history) {
        let reference = stress(point.position_m[0]);
        let reference_alpha = alpha(point.position_m[0]);
        assert!(reference_alpha > 0.0 && state.equivalent_plastic_strain > 0.0);
        for (actual, expected) in point.stress_pa.iter().zip(reference) {
            stress_error += point.volume_weight_m3 * (actual - expected).powi(2);
            stress_norm += point.volume_weight_m3 * expected * expected;
        }
        alpha_error +=
            point.volume_weight_m3 * (state.equivalent_plastic_strain - reference_alpha).powi(2);
        alpha_norm += point.volume_weight_m3 * reference_alpha * reference_alpha;
    }
    let actual_energies = [
        evaluation
            .element_evidence
            .iter()
            .map(|e| e.elastic_energy_j.canonical_value())
            .sum::<f64>(),
        evaluation
            .element_evidence
            .iter()
            .map(|e| e.hardening_energy_j.canonical_value())
            .sum::<f64>(),
        dissipation,
    ];
    let energy_references = energy_references();
    let energies = std::array::from_fn(|i| (actual_energies[i] / energy_references[i] - 1.0).abs());
    let mut axial_reaction = 0.0;
    let mut lateral_moment = 0.0;
    for (i, node) in mesh.topology.nodes.iter().enumerate() {
        for (axis, kind) in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .into_iter()
        .enumerate()
        {
            let row = summary.structural_dof_layout.index(i, kind).unwrap();
            let reaction = evaluation.internal_force[row] - summary.physical_load[row];
            if axis == 0 && node.coordinates_m.canonical_value()[0] == 0.0 {
                axial_reaction += reaction;
            }
            if axis > 0 {
                lateral_moment += node.coordinates_m.canonical_value()[axis] * reaction;
            }
        }
    }
    assert!(
        (axial_reaction + stress(0.0)[0]).abs() / stress(0.0)[0] < 1e-8,
        "axialreaction={axial_reaction}"
    );
    let expected_moment = 2.0 * stress(0.5)[1];
    Errors {
        displacement: (u_error / u_norm).sqrt(),
        stress: (stress_error / stress_norm).sqrt(),
        alpha: (alpha_error / alpha_norm).sqrt(),
        energies,
        lateral_moment: (lateral_moment / expected_moment - 1.0).abs(),
    }
}
