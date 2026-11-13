use super::{
    augmented, cone::Cone, evaluation::Point, linear::Linear, mesh::Basis, scaling::Scaling,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use clarabel::algebra::CscMatrix;
mod public;

#[tokio::test]
async fn visited_cache_retains_the_original_numeric_argument_and_evicts_without_payload_copies() {
    use super::cache::{Cache, Parameters};
    use runmat_value::{Tensor, Value};
    let mut cooperation = cooperation();
    let mut original = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let pointer = original.as_f64_slice().unwrap().as_ptr();
    let mut cache = Cache::new(Parameters {
        capacity: 1,
        tolerance: 1e-6,
    })
    .unwrap();
    cache.record(&Value::Tensor(original.clone())).unwrap();
    assert_eq!(cache.first_values().unwrap().as_ptr(), pointer);
    assert!(original.try_as_f64_slice_mut_unique().is_none());
    assert!(cache
        .contains(&[1.0, 2.0 + 1e-7], &mut cooperation)
        .await
        .unwrap());
    assert!(!cache
        .contains(&[1.0, 2.01], &mut cooperation)
        .await
        .unwrap());
    cache
        .record(&Value::Tensor(
            Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap(),
        ))
        .unwrap();
    assert!(
        original.try_as_f64_slice_mut_unique().is_some(),
        "eviction releases the sole history reference"
    );
    assert!(!cache.contains(&[1.0, 2.0], &mut cooperation).await.unwrap());
    cache.clear();
    assert!(!cache.contains(&[3.0, 4.0], &mut cooperation).await.unwrap());
    assert!(Cache::new(Parameters {
        capacity: 0,
        tolerance: 1e-6
    })
    .is_err());
    // Both point orientations reference the original actual matrix, including
    // noncontiguous column-major row views; neither creates coordinate payloads.
    for columns in [false, true] {
        let data = if columns {
            vec![1.0, 2.0, 3.0, 4.0]
        } else {
            vec![1.0, 3.0, 2.0, 4.0]
        };
        let mut backing = Tensor::new(data, vec![2, 2]).unwrap();
        cache.record_matrix(&backing, 0, 2, 2, columns).unwrap();
        assert!(backing.try_as_f64_slice_mut_unique().is_none());
        assert!(cache.contains(&[1.0, 2.0], &mut cooperation).await.unwrap());
        assert!(!cache.contains(&[1.0, 3.0], &mut cooperation).await.unwrap());
        cache.clear();
        assert!(backing.try_as_f64_slice_mut_unique().is_some());
    }
}

fn cooperation() -> ScalarCooperation {
    ScalarCooperation::for_builtin(runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity, None)
}

#[tokio::test]
async fn feasible_generators_keep_oblique_and_nearly_dependent_faces() {
    let mut cooperation = cooperation();
    let free = Cone::full(2, &mut cooperation).await.unwrap();
    assert_eq!(
        free.direction(2, Basis::Minimal, &mut cooperation)
            .await
            .unwrap()
            .unwrap(),
        [-1.0, -1.0]
    );
    assert_eq!(
        free.oriented_direction(2, Basis::Minimal, true, &mut cooperation)
            .await
            .unwrap()
            .unwrap(),
        [1.0, 1.0]
    );
    let mut halfspace = Cone::full(2, &mut cooperation).await.unwrap();
    halfspace
        .intersect(&[(0, 1.0)], &mut cooperation)
        .await
        .unwrap();
    let inward = halfspace
        .direction(2, Basis::Minimal, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(inward, [-1.0, 0.0]);
    assert_eq!(
        halfspace
            .oriented_direction(2, Basis::Minimal, true, &mut cooperation)
            .await
            .unwrap()
            .unwrap(),
        inward
    );
    let mut cone = Cone::full(2, &mut cooperation).await.unwrap();
    // Coordinate rejection would find no direction along this feasible line.
    cone.intersect(&[(0, 1.0), (1, -1.0)], &mut cooperation)
        .await
        .unwrap();
    cone.intersect(&[(0, -1.0), (1, 1.0)], &mut cooperation)
        .await
        .unwrap();
    cone.intersect(&[(0, -1.0)], &mut cooperation)
        .await
        .unwrap();
    assert_eq!(cone.len(Basis::Maximal).unwrap(), 1);
    assert_eq!(
        cone.direction(0, Basis::Maximal, &mut cooperation)
            .await
            .unwrap()
            .unwrap(),
        [1.0, 1.0]
    );

    let mut cone = Cone::full(2, &mut cooperation).await.unwrap();
    cone.intersect(&[(0, 1.0), (1, -1e-20)], &mut cooperation)
        .await
        .unwrap();
    cone.intersect(&[(0, -1.0), (1, 1e-20)], &mut cooperation)
        .await
        .unwrap();
    cone.intersect(&[(1, -1.0)], &mut cooperation)
        .await
        .unwrap();
    assert_eq!(cone.len(Basis::Maximal).unwrap(), 1);
    let ray = cone
        .direction(0, Basis::Maximal, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(ray, [1e-20, 1.0]);

    let mut cone = Cone::full(64, &mut cooperation).await.unwrap();
    let normal: Vec<_> = (0..64).map(|coordinate| (coordinate, 1.0)).collect();
    cone.intersect(&normal, &mut cooperation).await.unwrap();
    assert_eq!(cone.len(Basis::Maximal).unwrap(), 127);
    assert_eq!(cone.len(Basis::Minimal).unwrap(), 65);
    assert!(cone.stored_nonzeros() <= 2 * 64);
    // A pointed cut exercises incidence adjacency after lineality vanishes.
    let mut cone = Cone::full(2, &mut cooperation).await.unwrap();
    cone.intersect(&[(0, -1.0)], &mut cooperation)
        .await
        .unwrap();
    cone.intersect(&[(1, -1.0)], &mut cooperation)
        .await
        .unwrap();
    cone.intersect(&[(0, 1.0), (1, -1.0)], &mut cooperation)
        .await
        .unwrap();
    assert_eq!(cone.len(Basis::Maximal).unwrap(), 2);
    for index in 0..cone.len(Basis::Maximal).unwrap() {
        let ray = cone
            .direction(index, Basis::Maximal, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        assert!(ray[0] >= 0.0 && ray[1] >= 0.0 && ray[0] <= ray[1]);
    }
}

#[tokio::test]
async fn feasible_prefix_respects_narrow_box_and_original_equality() {
    let mut cooperation = cooperation();
    let matrix = CscMatrix::new(2, 1, vec![0, 2], vec![0, 1], vec![-1.0, 1.0]);
    let geometry = Linear::new(matrix, vec![5e-7, 5e-7], 0, &mut cooperation)
        .await
        .unwrap();
    // Both nearby rows can be binding on a coarse mesh. Refinement restores
    // their feasible cone; the interval is not declared inconsistent.
    let coarse = geometry
        .cone(&[0.0], 1.0, 1e-3, &Scaling::unit(1), &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(coarse.len(Basis::Maximal).unwrap(), 0);
    let fine = geometry
        .cone(&[0.0], 1e-7, 1e-3, &Scaling::unit(1), &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(fine.len(Basis::Maximal).unwrap(), 2);
    let candidate = geometry
        .candidate_with_step(&[0.0], &[1.0], 1.0, 1e-12, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(candidate.point, [5e-7]);
    assert_eq!(candidate.length, 5e-7);
    assert!(geometry
        .feasible(&candidate.point, 1e-12, &mut cooperation)
        .await
        .unwrap());

    let matrix = CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, -1.0]);
    let geometry = Linear::new(matrix, vec![0.0], 1, &mut cooperation)
        .await
        .unwrap();
    let cone = geometry
        .cone(&[0.0, 0.0], 1.0, 1e-3, &Scaling::unit(2), &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cone.len(Basis::Maximal).unwrap(), 2);
    assert!(geometry
        .candidate_with_step(&[0.0, 0.0], &[1.0, 0.0], 1.0, 1e-12, &mut cooperation)
        .await
        .unwrap()
        .is_err());
}

#[tokio::test]
async fn augmented_state_changes_real_multipliers_and_penalty_without_callbacks() {
    let mut cooperation = cooperation();
    let parameters = augmented::Parameters {
        initial_penalty: 10.0,
        penalty_factor: 100.0,
        mesh_tolerance: 1e-6,
        constraint_tolerance: 1e-6,
    };
    let point = Point {
        x: vec![1.2],
        f: 0.64,
        inequality: vec![0.2].into(),
        equality: Vec::new().into(),
    };
    let mut state = augmented::Augmented::new(&point, parameters, &mut cooperation)
        .await
        .unwrap();
    let first = state
        .merit(&point, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert!((first - 0.84).abs() < 1e-14);
    let tolerance = state.inner_tolerance();
    assert_eq!(
        state
            .completed_inner(&point, &mut cooperation)
            .await
            .unwrap(),
        Some(augmented::Update::Multipliers)
    );
    assert!(state.inner_tolerance() < tolerance);
    assert!(
        state
            .merit(&point, &mut cooperation)
            .await
            .unwrap()
            .unwrap()
            > first
    );
    let violated = Point {
        x: vec![11.0],
        f: 0.0,
        inequality: vec![10.0].into(),
        equality: Vec::new().into(),
    };
    assert_eq!(
        state
            .completed_inner(&violated, &mut cooperation)
            .await
            .unwrap(),
        Some(augmented::Update::Penalty)
    );
    assert!(state.inner_tolerance() < tolerance);
}
