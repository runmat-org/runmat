use super::super::mesh::Basis;
use super::*;
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use num_traits::ToPrimitive;
use std::rc::Rc;

#[tokio::test]
async fn lattice_candidates_preserve_original_rows_and_binding_radius() {
    use super::super::{
        linear::{Linear, ProposalStop},
        scaling::Scaling,
    };
    use clarabel::algebra::CscMatrix;
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
        Some(&runtime),
    );
    let geometry = Linear::new(
        CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]),
        vec![0.0],
        0,
        &mut cooperation,
    )
    .await
    .unwrap();
    let scaling = Scaling::unit(2);
    let (_, active) = geometry
        .binding_cone(&[-0.05, -0.05], 1.0, 0.01, &scaling, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert!(!active);
    let (_, active) = geometry
        .binding_cone(&[-0.05, -0.05], 1.0, 0.2, &scaling, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert!(active);
    assert!(matches!(
        geometry
            .mesh_candidate(&[0.0, 0.0], &[1.0, 0.0], 0.25, 1e-9, &mut cooperation)
            .await
            .unwrap(),
        Err(ProposalStop::Blocked)
    ));
    let point = geometry
        .mesh_candidate(&[0.0, 0.0], &[1.0, -1.0], 0.25, 1e-9, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(point.length, 0.25);
    assert_eq!(point.point, [0.25, -0.25]);
}

#[tokio::test]
async fn frozen_triangular_frames_keep_level_identity_span_and_wide_completion() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
        Some(&runtime),
    );
    for (n, basis, mesh) in [
        (1, Basis::Minimal, 0.25),
        (2, Basis::Maximal, 0.25),
        (3, Basis::Minimal, 0.25),
        (2, Basis::Minimal, 2f64.powi(-120)),
    ] {
        let mut refinements = Refinements::new(n).unwrap();
        let mut model = frame(&mut refinements, mesh, basis, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        let expected = if basis == Basis::Minimal {
            n + 1
        } else {
            2 * n
        };
        assert_eq!(model.count().unwrap(), expected);
        let original = model.seed.clone();
        let next = frame(&mut refinements, mesh, basis, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        assert!(
            Rc::ptr_eq(&original, &next.seed),
            "the distinguished direction is fixed at this level"
        );
        let mut directions = Vec::new();
        for index in 0..expected {
            let ray = model
                .direction(index, &mut cooperation)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(ray.len(), n);
            assert!(ray.iter().any(|&value| value != 0.0));
            assert!(ray
                .iter()
                .all(|value| value.is_finite() && value.fract() == 0.0));
            directions.push(ray);
        }
        if basis == Basis::Maximal {
            for column in 0..n {
                for row in 0..n {
                    assert_eq!(directions[column][row], -directions[column + n][row]);
                }
            }
        } else {
            assert!(
                certificate::positive(&model, &mut cooperation)
                    .await
                    .unwrap(),
                "actual binary64 completion remains positive"
            );
            if mesh == 0.25 {
                for row in 0..n {
                    assert_eq!(
                        directions[n][row],
                        -directions[..n].iter().map(|ray| ray[row]).sum::<f64>()
                    );
                }
            }
            let pivot = model.seed.pivot;
            model.negative.as_mut().unwrap()[pivot] = &model.seed.values[pivot] * 3;
            assert!(
                !certificate::positive(&model, &mut cooperation)
                    .await
                    .unwrap(),
                "a wrong-sign pivot cannot certify a positive completion"
            );
        }
        let q = level::Level::for_mesh(mesh)
            .unwrap()
            .integer()
            .to_f64()
            .unwrap();
        assert!(directions
            .iter()
            .flat_map(|ray| ray.iter())
            .all(|value| value.abs()
                <= q * if basis == Basis::Minimal {
                    n as f64
                } else {
                    1.0
                }));
        let turn = random::try_acquire_turn(Some(&runtime), "objective").unwrap();
        drop(turn);
    }
}
