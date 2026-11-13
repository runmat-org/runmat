//! Closed-form proportional plane-strain J2 reference, no return-map replay.
use super::*;

pub(crate) fn state(yield_stress: f64, factor: f64) -> (f64, f64, [f64; 6]) {
    // E=1000, nu=1/4 => mu=lambda=400, Cxxxx=1200.
    let stress = 5.6 * factor;
    let denominator = 3.0 * 400.0 + 30.0;
    let strain = if stress <= 1.5 * yield_stress {
        stress / 1200.0
    } else {
        (stress - 800.0 * yield_stress / denominator) / (1200.0 - 640_000.0 / denominator)
    };
    let alpha = ((800.0 * strain - yield_stress) / denominator).max(0.0);
    let lateral = 400.0 * strain + 400.0 * alpha;
    (strain, alpha, [stress, lateral, lateral, 0.0, 0.0, 0.0])
}

pub(crate) fn verify(mesh: &SolverMeshArtifact, block: &NonlinearSolidBlockArtifact, factor: f64) {
    assert!((block.result.accepted_state.load_factor() - factor).abs() < 1e-14);
    let references = [state(2.0, factor), state(3.5, factor)];
    let result = &block.result;
    let count = result.load_factors.len();
    assert_eq!(result.displacement_snapshots.len(), count);
    assert_eq!(result.stress_snapshots.len(), count);
    assert_eq!(result.equivalent_plastic_strain_snapshots.len(), count);
    assert_eq!(result.plastic_strain_snapshots.len(), count);
    // Every accepted output is checked, including the elastic/yield transition in each
    // layer. Terminal polynomial shape alone would not establish the accepted history.
    for sample in 0..count {
        verify_sample(mesh, result, sample);
    }
    assert!(references[0].1 > references[1].1);
    if factor == 0.9 {
        assert_eq!(references[1].1, 0.0);
    } else {
        assert!(references[1].1 > 0.0);
    }
    for (node, displacement) in mesh
        .topology
        .nodes
        .iter()
        .zip(block.result.accepted_state.displacement().chunks_exact(3))
    {
        let x = node.coordinates_m.canonical_value()[0];
        let expected = references[0].0 * x.min(0.5) + references[1].0 * (x - 0.5).max(0.0);
        assert!((displacement[0] - expected).abs() < 2e-9);
        assert!(displacement[1].abs() < 2e-9 && displacement[2].abs() < 2e-9);
    }
    for (element_id, _, point) in block.result.accepted_state.point_history() {
        let element = mesh
            .topology
            .volume_elements
            .iter()
            .find(|e| e.stable_identity == element_id)
            .unwrap();
        let layer = usize::from(element.region_id.source_topology_id == "layer_1");
        let alpha = references[layer].1;
        assert!((point.equivalent_plastic_strain - alpha).abs() < 2e-9);
        for (actual, exact) in point.engineering_plastic_strain.iter().zip([
            alpha,
            -alpha / 2.0,
            -alpha / 2.0,
            0.0,
            0.0,
            0.0,
        ]) {
            assert!((actual - exact).abs() < 2e-9);
        }
    }
    for (element, stress) in mesh.topology.volume_elements.iter().zip(
        block
            .result
            .stress_snapshots
            .last()
            .unwrap()
            .chunks_exact(6),
    ) {
        let layer = usize::from(element.region_id.source_topology_id == "layer_1");
        for (actual, exact) in stress.iter().zip(references[layer].2) {
            assert!((actual - exact).abs() < 2e-7, "stress {actual} != {exact}");
        }
    }
    let expected_dissipation = 0.5 * (2.0 * references[0].1 + 3.5 * references[1].1);
    assert!(
        (block
            .result
            .accepted_state
            .accumulated_plastic_dissipation_j()
            - expected_dissipation)
            .abs()
            < 2e-9
    );
}

fn verify_sample(
    mesh: &SolverMeshArtifact,
    result: &runmat_analysis_fea::solve::nonlinear::NonlinearSolveResult,
    sample: usize,
) {
    let references = [
        state(2.0, result.load_factors[sample]),
        state(3.5, result.load_factors[sample]),
    ];
    let displacement = &result.displacement_snapshots[sample];
    let stress = &result.stress_snapshots[sample];
    let alpha = &result.equivalent_plastic_strain_snapshots[sample];
    let plastic = &result.plastic_strain_snapshots[sample];
    let elements = mesh.topology.volume_elements.len();
    assert_eq!(displacement.len(), mesh.topology.nodes.len() * 3);
    assert_eq!(stress.len(), elements * 6);
    assert_eq!(alpha.len(), elements);
    assert_eq!(plastic.len(), elements * 6);
    for (node, values) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
        let x = node.coordinates_m.canonical_value()[0];
        let expected = references[0].0 * x.min(0.5) + references[1].0 * (x - 0.5).max(0.0);
        assert!((values[0] - expected).abs() < 2e-9);
        assert!(values[1].abs() < 2e-9 && values[2].abs() < 2e-9);
    }
    for (index, element) in mesh.topology.volume_elements.iter().enumerate() {
        let layer = usize::from(element.region_id.source_topology_id == "layer_1");
        let (_, expected_alpha, expected_stress) = references[layer];
        assert!((alpha[index] - expected_alpha).abs() < 2e-9);
        for (&actual, expected) in stress[6 * index..6 * (index + 1)]
            .iter()
            .zip(expected_stress)
        {
            assert!((actual - expected).abs() < 2e-7);
        }
        for (&actual, expected) in plastic[6 * index..6 * (index + 1)].iter().zip([
            expected_alpha,
            -expected_alpha / 2.0,
            -expected_alpha / 2.0,
            0.0,
            0.0,
            0.0,
        ]) {
            assert!((actual - expected).abs() < 2e-9);
        }
    }
}
