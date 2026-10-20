use super::*;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{AccelProvider, HostLogicalTensorView, ThreadProviderGuard};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dense_rank_zero_and_rank_one_points_keep_triangulation_geometry() {
    let point = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2]).expect("rank-one point"));
    let triangulation = object(
        block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![point]),
        ))
        .expect("one point"),
    );
    let Value::Tensor(stored) = triangulation.properties().get("X").expect("stored points") else {
        panic!("point matrix")
    };
    assert_eq!(stored.shape(), [1, 2]);
    assert_eq!(stored.materialize_f64(), vec![0.0, 1.0]);

    let scalar = Value::Tensor(Tensor::new(vec![0.0], vec![]).expect("rank-zero tensor"));
    let error = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![scalar]),
    ))
    .expect_err("rank-zero tensor has no coordinate columns");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn one_row_point_matrix_with_empty_constraints_is_not_a_coordinate_pair() {
    let points = tensor(&[0.0, 0.0], 1, 2);
    let empty = tensor(&[], 0, 2);
    let dt = object(
        block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![points.clone(), empty]),
        ))
        .unwrap(),
    );
    let Value::Tensor(stored) = dt.properties().get("X").unwrap() else {
        panic!("point matrix")
    };
    assert_eq!(stored.shape(), [1, 2]);
    let vectors = object(
        block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                points,
                tensor(&[1.0, 1.0], 1, 2),
            ]),
        ))
        .unwrap(),
    );
    let Value::Tensor(stored) = vectors.properties().get("X").unwrap() else {
        panic!("paired coordinates")
    };
    assert_eq!(stored.shape(), [2, 2]);

    let singleton_tail = Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2, 1]).unwrap());
    assert!(block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![singleton_tail])
    ))
    .is_ok());
    let genuine_nd = Value::Tensor(Tensor::new(vec![0.0; 4], vec![1, 2, 2]).unwrap());
    assert!(block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![genuine_nd])
    ))
    .is_err());
}

#[test]
fn invalid_resident_geometry_is_rejected_before_any_provider_download() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _owner = ThreadProviderGuard::set(Some(provider));
    let invalid_matrix = Tensor::new(vec![0.0; 9], vec![3, 3]).unwrap();
    let resident =
        crate::builtins::common::gpu_helpers::upload_tensor(provider, &invalid_matrix).unwrap();
    provider.reset_telemetry();
    let error = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            resident.clone(),
        )]),
    ))
    .expect_err("3-D geometry is unsupported before readback");
    assert_eq!(error.identifier(), ERROR_UNSUPPORTED.identifier);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

    let nd = Tensor::new(vec![0.0; 4], vec![1, 2, 2]).unwrap();
    let resident_nd = crate::builtins::common::gpu_helpers::upload_tensor(provider, &nd).unwrap();
    provider.reset_telemetry();
    let error = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            resident_nd.clone(),
        )]),
    ))
    .expect_err("genuine N-D point shape is rejected before readback");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(resident_nd.allocation_lease().validate_live().is_ok());

    let dt = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            3,
            2,
        )]),
    ))
    .unwrap();
    let error = block_on(nearest_neighbor_builtin(
        dt,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            resident.clone(),
        )]),
    ))
    .expect_err("invalid resident query shape is rejected before readback");
    assert_eq!(error.identifier(), ERROR_UNSUPPORTED.identifier);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

    let logical = provider
        .upload_logical_raw(&HostLogicalTensorView {
            data: &[1, 0],
            shape: &[1, 2],
        })
        .unwrap();
    provider.reset_telemetry();
    let error = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(logical)]),
    ))
    .expect_err("logical resident points are not numeric coordinates");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

    let valid_vector = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
    let resident_vector =
        crate::builtins::common::gpu_helpers::upload_tensor(provider, &valid_vector).unwrap();
    provider.reset_telemetry();
    let error = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(resident_vector.clone()),
            tensor(&[f64::NAN, 1.0], 1, 2),
        ]),
    ))
    .expect_err("known nonfinite host coordinate precedes resident gather");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(resident_vector.allocation_lease().validate_live().is_ok());
    assert!(resident.allocation_lease().validate_live().is_ok());
}

#[test]
fn admitted_resident_single_point_constraint_form_downloads_once() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _owner = ThreadProviderGuard::set(Some(provider));
    let input = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
    let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input).unwrap();
    provider.reset_telemetry();
    let dt = object(
        block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident.clone()),
                tensor(&[], 0, 2),
            ]),
        ))
        .expect("admitted resident point matrix"),
    );
    let Value::Tensor(points) = dt.properties().get("X").unwrap() else {
        panic!("point matrix")
    };
    assert_eq!(points.shape(), [1, 2]);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 16);
    assert!(resident.allocation_lease().validate_live().is_ok());
}
