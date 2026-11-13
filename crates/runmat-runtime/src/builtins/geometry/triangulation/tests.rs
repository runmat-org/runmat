use super::*;
use futures::executor::block_on;

#[path = "tests/preflight.rs"]
mod preflight;

#[test]
fn free_boundary_aliases_reject_excess_outputs_before_receiver_access() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    for error in [
        block_on(free_boundary_builtin(Value::Num(0.0))).expect_err("fixed boundary output bound"),
        block_on(delaunay_tri_free_boundary_builtin(Value::Num(0.0)))
            .expect_err("fixed method output bound"),
    ] {
        assert_eq!(
            error.identifier(),
            runmat_builtins::FREE_BOUNDARY_ERROR_TOO_MANY_OUTPUTS.identifier,
        );
    }
}

#[test]
fn constructor_and_nearest_aliases_reject_excess_outputs_before_geometry() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let constructor = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(f64::NAN)]),
    ))
    .expect_err("constructor output bound precedes input inspection");
    assert_eq!(
        constructor.identifier(),
        runmat_builtins::DELAUNAY_TRI_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    for error in [
        block_on(nearest_neighbor_builtin(
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("function output bound"),
        block_on(delaunay_tri_nearest_neighbor_builtin(
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("method output bound"),
    ] {
        assert_eq!(
            error.identifier(),
            runmat_builtins::NEAREST_NEIGHBOR_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }
}

#[test]
fn point_location_aliases_bound_outputs_before_receiver_and_return_barycentric_rows() {
    {
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        for error in [
            block_on(point_location_builtin(
                Value::Num(0.0),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect_err("function output bound"),
            block_on(delaunay_tri_point_location_builtin(
                Value::Num(0.0),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect_err("method output bound"),
        ] {
            assert_eq!(
                error.identifier(),
                runmat_builtins::POINT_LOCATION_ERROR_TOO_MANY_OUTPUTS.identifier
            );
        }
    }
    let dt = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            3,
            2,
        )]),
    ))
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    for result in [
        block_on(point_location_builtin(
            dt.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                &[0.25, 0.25],
                1,
                2,
            )]),
        ))
        .unwrap(),
        block_on(delaunay_tri_point_location_builtin(
            dt.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                &[0.25, 0.25],
                1,
                2,
            )]),
        ))
        .unwrap(),
    ] {
        let Value::OutputList(outputs) = result else {
            panic!("two selected outputs")
        };
        assert_eq!(outputs.len(), 2);
        let Value::Tensor(ti) = &outputs[0] else {
            panic!("triangle indices")
        };
        let Value::Tensor(bc) = &outputs[1] else {
            panic!("barycentric coordinates")
        };
        assert_eq!(ti.materialize_f64(), vec![1.0]);
        assert_eq!(bc.shape(), [1, 3]);
        assert!((bc.materialize_f64().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}

fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new_2d(data.to_vec(), rows, cols).expect("test tensor"))
}

fn integer_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor"))
}

fn object(value: Value) -> ObjectInstance {
    let Value::Object(object) = value else {
        panic!("expected object")
    };
    object
}

#[test]
fn delaunaytri_builds_legacy_and_modern_properties() {
    let dt = object(
        block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                4,
                2,
            )]),
        ))
        .unwrap(),
    );
    assert!(dt.class_name().is(DELAUNAY_TRI_CLASS));
    let Value::Tensor(points) = dt.properties().get("X").unwrap() else {
        panic!("expected points")
    };
    assert_eq!(points.shape(), [4, 2]);
    let Value::Tensor(tri) = dt.properties().get("Triangulation").unwrap() else {
        panic!("expected triangulation")
    };
    assert_eq!(tri.shape(), [2, 3]);
    assert!(dt.properties().contains_key("ConnectivityList"));
    assert!(dt.properties().contains_key("Points"));
}

#[test]
fn delaunaytri_rejects_point_matrices_that_are_not_two_dimensional() {
    let err = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            &[0.0, 1.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0],
            2,
            4,
        )]),
    ))
    .expect_err("4-column point matrix should fail");
    assert!(err.message.contains("N-by-2"));
}

#[test]
fn delaunaytri_accepts_coordinate_vectors() {
    let dt = object(
        block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(&[0.0, 1.0, 0.0], 1, 3),
                tensor(&[0.0, 0.0, 1.0], 1, 3),
            ]),
        ))
        .unwrap(),
    );
    let Value::Tensor(points) = dt.properties().get("Points").unwrap() else {
        panic!("expected points")
    };
    assert_eq!(points.shape(), [3, 2]);
}

#[test]
fn point_parsers_read_typed_integer_storage_exactly() {
    let points = matrix_points(
        &integer_tensor(IntegerStorage::U16(vec![0, 1, 0, 0, 0, 1]), 3, 2),
        BUILTIN_NAME,
    )
    .expect("matrix points");
    assert_eq!(points, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    let vector_points = vector_points(
        &integer_tensor(IntegerStorage::I16(vec![0, 1, 0]), 1, 3),
        &integer_tensor(IntegerStorage::I16(vec![0, 0, 1]), 1, 3),
        BUILTIN_NAME,
    )
    .expect("vector points");
    assert_eq!(vector_points, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
}

#[test]
fn topology_parsers_read_typed_integer_storage_exactly() {
    let constraints =
        constraints_from_value(&integer_tensor(IntegerStorage::U16(vec![1, 2, 2, 3]), 2, 2))
            .expect("constraints");
    assert_eq!(constraints, vec![[1, 2], [2, 3]]);

    let connectivity = connectivity_from_value(
        &integer_tensor(IntegerStorage::U16(vec![1, 2, 3]), 1, 3),
        Some(3),
    )
    .expect("connectivity");
    assert_eq!(connectivity, vec![[0, 1, 2]]);
}

#[test]
fn delaunaytri_integer_extensions_follow_compatibility_mode() {
    let points = || integer_tensor(IntegerStorage::U16(vec![0, 1, 0, 0, 0, 1]), 3, 2);
    let empty_topology = || integer_tensor(IntegerStorage::U16(Vec::new()), 0, 2);
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![points()]),
        ))
        .expect_err("MATLAB mode rejects typed-integer constructor coordinates");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension")
        );

        let error = block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
                empty_topology(),
            ]),
        ))
        .expect_err("MATLAB mode rejects typed-integer constructor topology");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DelaunayTriIntegerTopologyExtension")
        );

        let resident_points = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::U16(vec![0, 1, 0, 0, 0, 1]),
                shape: vec![3, 2],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap();
        let error = block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                resident_points.handle.clone(),
            )]),
        ))
        .expect_err("MATLAB mode rejects resident integer coordinates before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension")
        );

        let resident_topology = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![0, 2],
                runmat_accelerate_api::NumericElementType::U16,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let error = block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
                Value::GpuTensor(resident_topology.handle.clone()),
            ]),
        ))
        .expect_err("MATLAB mode rejects resident integer topology before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DelaunayTriIntegerTopologyExtension")
        );
        assert_eq!(resident_points.probe.retirements(), 0);
        assert_eq!(resident_topology.probe.retirements(), 0);
        drop((resident_points.handle, resident_topology.handle));
        assert_eq!(resident_points.probe.retirements(), 1);
        assert_eq!(resident_topology.probe.retirements(), 1);
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![points()])
        ))
        .is_ok());
        assert!(block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
                empty_topology(),
            ])
        ))
        .is_ok());
    }
}

#[test]
fn delaunaytri_object_and_query_extensions_follow_compatibility_mode() {
    let make_object = || {
        object(
            block_on(delaunay_tri_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                    &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    3,
                    2,
                )]),
            ))
            .expect("double constructor"),
        )
    };
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

        let mut integer_points = make_object();
        integer_points.replace_property_unbounded(
            POINTS_PROPERTY.to_string(),
            integer_tensor(IntegerStorage::U16(vec![0, 1, 0, 0, 0, 1]), 3, 2),
        );
        let error = block_on(nearest_neighbor_builtin(
            Value::Object(integer_points),
            crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                &[0.1, 0.1],
                1,
                2,
            )]),
        ))
        .expect_err("MATLAB mode rejects integer object points");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension")
        );

        let object = Value::Object(make_object());
        let error = block_on(nearest_neighbor_builtin(
            object,
            crate::builtins::common::test_support::builtin_arguments(vec![integer_tensor(
                IntegerStorage::U16(vec![0, 0]),
                1,
                2,
            )]),
        ))
        .expect_err("MATLAB mode rejects integer query points");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DelaunayTriIntegerCoordinatesExtension")
        );

        let mut integer_topology = make_object();
        integer_topology.replace_property_unbounded(
            CONNECTIVITY_PROPERTY.to_string(),
            integer_tensor(IntegerStorage::U16(vec![1, 2, 3]), 1, 3),
        );
        let error = block_on(free_boundary_builtin(Value::Object(integer_topology)))
            .expect_err("MATLAB mode rejects integer object topology");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DelaunayTriIntegerTopologyExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let object = Value::Object(make_object());
        assert!(block_on(nearest_neighbor_builtin(
            object,
            crate::builtins::common::test_support::builtin_arguments(vec![integer_tensor(
                IntegerStorage::U16(vec![0, 0]),
                1,
                2
            )]),
        ))
        .is_ok());

        let mut integer_topology = make_object();
        integer_topology.replace_property_unbounded(
            CONNECTIVITY_PROPERTY.to_string(),
            integer_tensor(IntegerStorage::U16(vec![1, 2, 3]), 1, 3),
        );
        assert!(block_on(free_boundary_builtin(Value::Object(integer_topology))).is_ok());
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn topology_index_parser_preserves_wide_integer_values_exactly() {
    let wide = (1_u64 << 53) + 1;
    let tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide index");
    assert_eq!(
        tensor_positive_index(&tensor, 0, usize::MAX).expect("exact wide index"),
        wide as usize
    );
    assert!(tensor_positive_index(&tensor, 0, (wide - 1) as usize).is_err());
}

#[test]
fn delaunaytri_empty_constructor_returns_empty_topology() {
    let dt = object(
        block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
    );
    let Value::Tensor(points) = dt.properties().get("X").unwrap() else {
        panic!("expected points")
    };
    let Value::Tensor(tri) = dt.properties().get("Triangulation").unwrap() else {
        panic!("expected triangulation")
    };
    assert_eq!(points.shape(), [0, 2]);
    assert_eq!(tri.shape(), [0, 3]);
}

#[test]
fn delaunaytri_rejects_3d_and_nonempty_constraints_deterministically() {
    let err = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
            3,
            3,
        )]),
    ))
    .expect_err("3-D should be explicit");
    assert_eq!(err.identifier(), ERROR_UNSUPPORTED.identifier);

    let err = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 2),
            tensor(&[1.0, 2.0], 1, 2),
        ]),
    ))
    .expect_err("constraints should be explicit");
    assert_eq!(err.identifier(), ERROR_UNSUPPORTED.identifier);
}

#[test]
fn delaunaytri_query_helpers_return_expected_shapes() {
    let dt = block_on(delaunay_tri_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            3,
            2,
        )]),
    ))
    .unwrap();
    let Value::Tensor(edges) = block_on(free_boundary_builtin(dt.clone())).unwrap() else {
        panic!("expected boundary tensor")
    };
    assert_eq!(edges.shape(), [3, 2]);

    let Value::Tensor(nn) = block_on(nearest_neighbor_builtin(
        dt.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(&[0.1, 0.8], 1, 2)]),
    ))
    .unwrap() else {
        panic!("expected nearest tensor")
    };
    assert_eq!(nn.materialize_f64(), vec![3.0]);

    let location = block_on(point_location_builtin(
        dt,
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(&[0.25, 0.25], 1, 2)]),
    ))
    .unwrap();
    let Value::Tensor(ti) = location else {
        panic!("expected location tensor")
    };
    assert_eq!(ti.shape(), [1, 1]);
    assert_eq!(ti.materialize_f64(), vec![1.0]);
}

#[test]
fn point_location_rejects_inconsistent_object_connectivity() {
    let mut object = object(
        block_on(delaunay_tri_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                3,
                2,
            )]),
        ))
        .unwrap(),
    );
    object.replace_property_unbounded(
        CONNECTIVITY_PROPERTY.to_string(),
        tensor(&[1.0, 2.0, 9.0], 1, 3),
    );
    let err = block_on(point_location_builtin(
        Value::Object(object),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(&[0.25, 0.25], 1, 2)]),
    ))
    .expect_err("bad object connectivity should be rejected");
    assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
}
