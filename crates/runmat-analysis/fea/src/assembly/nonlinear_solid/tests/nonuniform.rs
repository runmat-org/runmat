//! Point-kernel and tensor virtual-work references for nonproportional element paths.
//! These checks isolate assembly; they do not replace continuum convergence criteria.
use super::*;

fn displacement(mesh: &SolverMeshArtifact, initial: &AssemblySummary, stage: usize) -> Vec<f64> {
    let mut u = vec![0.0; initial.dof_count];
    for (node, point) in mesh.topology.nodes.iter().enumerate() {
        let [x, y, z] = point.coordinates_m.canonical_value();
        let values = match stage {
            0 => [0.024 * x + 0.016 * x * x, -0.006 * x * y, 0.004 * x * z],
            1 => [0.012 * x - 0.008 * x * x, 0.023 * x * y, -0.017 * x * z],
            _ => [0.03 * x + 0.025 * x * x, -0.009 * y * y, 0.014 * x * z],
        };
        for (kind, value) in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .into_iter()
        .zip(values)
        {
            u[initial.structural_dof_layout.index(node, kind).unwrap()] = value;
        }
    }
    u
}

// Closed derivatives of the nodal polynomial above. On the unit Tet4, products
// xy/xz vanish at every corner and squares interpolate to their linear coordinate.
// On the affine Tet10 all degree-two polynomials are represented exactly.
fn reference_strain(order: ElementOrder, stage: usize, [x, y, z]: [f64; 3]) -> [f64; 6] {
    match (order, stage) {
        (ElementOrder::Tet4, 0) => [0.04, 0.0, 0.0, 0.0, 0.0, 0.0],
        (ElementOrder::Tet4, 1) => [0.004, 0.0, 0.0, 0.0, 0.0, 0.0],
        (ElementOrder::Tet4, _) => [0.055, -0.009, 0.0, 0.0, 0.0, 0.0],
        (ElementOrder::Tet10, 0) => [
            0.024 + 0.032 * x,
            -0.006 * x,
            0.004 * x,
            0.0,
            0.004 * z,
            -0.006 * y,
        ],
        (ElementOrder::Tet10, 1) => [
            0.012 - 0.016 * x,
            0.023 * x,
            -0.017 * x,
            0.0,
            -0.017 * z,
            0.023 * y,
        ],
        (ElementOrder::Tet10, _) => [0.03 + 0.05 * x, -0.018 * y, 0.014 * x, 0.0, 0.014 * z, 0.0],
    }
}

#[test]
fn nonproportional_element_history_preserves_tensor_work_and_consistent_tangent() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = fixture(order, true);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let mut committed = basis.zero_state();
        let material = IsotropicConstitutiveModel::from_material(&model.materials[0]).unwrap();
        let mut reference_history = vec![PlasticityPointState::default(); basis.point_count()];
        for stage in 0..3 {
            let u = displacement(&mesh, &initial, stage);
            let before = committed.clone();
            let evaluation = basis.evaluate(&u, &committed).unwrap();
            let mut reference_stresses = Vec::with_capacity(basis.point_count());
            for ((point, actual), reference_state) in evaluation
                .point_evidence
                .iter()
                .zip(&evaluation.trial_state.points)
                .zip(&mut reference_history)
            {
                let strain = reference_strain(order, stage, point.position_m);
                for (actual, expected) in point.total_engineering_strain.iter().zip(strain) {
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "polynomial strain {order:?} stage={stage}"
                    );
                }
                // The independently qualified point law receives closed polynomial strain,
                // never assembly strain or an interpolated quadrature history.
                let reference = material.update(strain, reference_state).unwrap();
                for (actual, expected) in point.stress_pa.iter().zip(reference.stress_pa) {
                    assert!(
                        (actual - expected).abs() < 1e-10,
                        "polynomial stress {order:?} stage={stage}"
                    );
                }
                for (actual, expected) in actual
                    .engineering_plastic_strain
                    .iter()
                    .zip(reference.state.engineering_plastic_strain)
                {
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "original plastic tensor {order:?} stage={stage}"
                    );
                }
                assert!(
                    (actual.equivalent_plastic_strain - reference.state.equivalent_plastic_strain)
                        .abs()
                        < 1e-12
                );
                reference_stresses.push(reference.stress_pa);
                *reference_state = reference.state;
            }
            let direction = (0..u.len())
                .map(|i| (i as f64 + 0.3).sin())
                .collect::<Vec<_>>();
            let other = (0..u.len())
                .map(|i| (i as f64 + 0.7).cos())
                .collect::<Vec<_>>();
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            let mut work = 0.0;
            let mut ordinal = 0;
            for element in &projection.elements {
                projection
                    .visit_element_quadrature(element, |point| {
                        let mut gradient = [[0.0; 3]; 3];
                        for (&node, shape) in element
                            .node_indices
                            .iter()
                            .zip(point.physical_shape_gradients())
                        {
                            for (axis, kind) in [
                                StructuralDofKind::Ux,
                                StructuralDofKind::Uy,
                                StructuralDofKind::Uz,
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                let value = direction
                                    [initial.structural_dof_layout.index(node, kind).unwrap()];
                                for (entry, derivative) in gradient[axis].iter_mut().zip(shape) {
                                    *entry += value * derivative;
                                }
                            }
                        }
                        // Tensor double contraction counts each physical shear component twice.
                        let stress = reference_stresses[ordinal];
                        work += point.volume_weight_m3
                            * (stress[0] * gradient[0][0]
                                + stress[1] * gradient[1][1]
                                + stress[2] * gradient[2][2]
                                + stress[3] * (gradient[1][2] + gradient[2][1])
                                + stress[4] * (gradient[0][2] + gradient[2][0])
                                + stress[5] * (gradient[0][1] + gradient[1][0]));
                        ordinal += 1;
                    })
                    .unwrap();
            }
            assert_eq!(ordinal, basis.point_count());
            let assembled = evaluation
                .internal_force
                .iter()
                .zip(&direction)
                .map(|(f, d)| f * d)
                .sum::<f64>();
            assert!(
                (assembled - work).abs() < 1e-11 * work.abs().max(1.0),
                "{order:?} stage={stage}"
            );
            let action = spmv(&evaluation.tangent, &direction).unwrap();
            let reverse = spmv(&evaluation.tangent, &other).unwrap();
            let left = action.iter().zip(&other).map(|(a, b)| a * b).sum::<f64>();
            let right = reverse
                .iter()
                .zip(&direction)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            assert!((left - right).abs() < 1e-11 * left.abs().max(1.0));
            for h in [1e-6, 1e-7] {
                let perturbed = |sign: f64| {
                    u.iter()
                        .zip(&direction)
                        .map(|(u, d)| u + sign * h * d)
                        .collect::<Vec<_>>()
                };
                let plus = basis.evaluate(&perturbed(1.0), &committed).unwrap();
                let minus = basis.evaluate(&perturbed(-1.0), &committed).unwrap();
                let error = action
                    .iter()
                    .zip(plus.internal_force.iter().zip(&minus.internal_force))
                    .map(|(a, (p, m))| ((p - m) / (2.0 * h) - a).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm = action.iter().map(|a| a * a).sum::<f64>().sqrt();
                assert!(
                    error / norm < 1e-7,
                    "{order:?} stage={stage} h={h} tangent error={}",
                    error / norm
                );
            }
            assert_eq!(committed, before);
            assert_eq!(
                basis.evaluate(&u, &committed).unwrap().trial_state,
                evaluation.trial_state
            );
            if order == ElementOrder::Tet10 {
                let mut values = evaluation
                    .trial_state
                    .points
                    .iter()
                    .map(|p| p.equivalent_plastic_strain);
                let first = values.next().unwrap();
                assert!(
                    values.any(|value| (value - first).abs() > 1e-6),
                    "Tet10 path must exercise nonuniform point history"
                );
            }
            committed = evaluation.trial_state;
        }
    }
}

#[test]
fn cancelled_evaluation_allocates_no_trial_and_preserves_original_history() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = fixture(order, true);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let u = displacement(&mesh, &initial, 0);
        let history = basis.evaluate(&u, &basis.zero_state()).unwrap().trial_state;
        let before = history.clone();
        let context = crate::progress::FeaProgressContext::default();
        context.cancellation.cancel();
        let guard = crate::progress::replace_fea_progress_context(context);
        let (result, retained, peak) =
            crate::topology::measure_test_allocations(|| basis.evaluate(&u, &history));
        assert!(matches!(result, Err(NonlinearSolidError::Cancelled)));
        assert_eq!((retained, peak), (0, 0));
        assert_eq!(history, before);
        drop(guard);
        assert!(basis.evaluate(&u, &history).is_ok());
    }
}
