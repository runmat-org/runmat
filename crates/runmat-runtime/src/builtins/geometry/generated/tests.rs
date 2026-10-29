use futures::executor::block_on;
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_geometry_core::{
    GeometryFeatureOperation, GeometryRegionRole, GeometryRegionSelector, GeometryScalarExpression,
};
use runmat_value::{IntValue, IntegerStorage, Tensor, Value};

use super::codec::{feature_from_value, generated_from_value, region_output_from_value};
use super::combine::{
    geometry_finish_builtin, geometry_intersect_builtin, geometry_subtract_builtin,
    geometry_union_builtin,
};
use super::features::{
    geometry_chamfer_builtin, geometry_extrude_builtin, geometry_fillet_builtin,
    geometry_hole_builtin, geometry_pocket_builtin, geometry_revolve_builtin,
};
use super::primitives::{
    geometry_box_builtin, geometry_cylinder_builtin, geometry_instance_builtin,
    geometry_sphere_builtin, geometry_transform_builtin,
};
use super::regions::{
    geometry_edges_builtin, geometry_edges_introduced_by_builtin, geometry_faces_builtin,
    geometry_faces_introduced_by_builtin, geometry_solid_builtin, geometry_with_region_builtin,
};

fn num(value: f64) -> Value {
    Value::Num(value)
}

fn vector(values: [f64; 3]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![3, 1]).unwrap())
}

/// MATLAB 4-by-4 matrix given row by row; stored column-major like a script value.
fn matrix(rows: [[f64; 4]; 4]) -> Value {
    let mut data = Vec::with_capacity(16);
    for column in 0..4 {
        for row in rows {
            data.push(row[column]);
        }
    }
    Value::Tensor(Tensor::new(data, vec![4, 4]).unwrap())
}

fn translation(dx: f64, dy: f64, dz: f64) -> Value {
    matrix([
        [1.0, 0.0, 0.0, dx],
        [0.0, 1.0, 0.0, dy],
        [0.0, 0.0, 1.0, dz],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn unit_box() -> Value {
    block_on(geometry_box_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(1.0),
            num(1.0),
            num(1.0),
        ]),
    ))
    .unwrap()
}

fn last_operation(value: &Value) -> GeometryFeatureOperation {
    feature_from_value(value, "test")
        .unwrap()
        .features
        .last()
        .unwrap()
        .operation
        .clone()
}

fn identifier(result: crate::BuiltinResult<Value>) -> String {
    result
        .expect_err("expected an error")
        .identifier()
        .unwrap_or_default()
        .to_string()
}

/// Canonical values of an admitted expression list; the translation column of a
/// transform is a typed length, every other coefficient a dimensionless literal.
fn literals(values: &[GeometryScalarExpression]) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GeometryScalarExpression::Literal { value } => {
                assert!(
                    !matches!(index, 3 | 7 | 11),
                    "translation coefficient {index} must be a typed length"
                );
                *value
            }
            GeometryScalarExpression::Canonical { value } => {
                assert!(
                    matches!(index, 3 | 7 | 11),
                    "coefficient {index} is dimensionless and must stay a literal"
                );
                assert_eq!(
                    value.semantic_kind(),
                    Some(runmat_types::physical::QuantityKind::Length),
                    "coefficient {index} is not a length"
                );
                value.canonical_value()
            }
            other => panic!("expected a canonical coefficient, got {other:?}"),
        })
        .collect()
}

#[test]
fn typed_builder_composes_regions_and_crosses_the_portable_value_boundary() {
    let stock = block_on(geometry_box_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(0.08),
            num(0.008),
            num(0.05),
        ]),
    ))
    .unwrap();
    let hole = block_on(geometry_cylinder_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(0.005),
            num(0.008),
            vector([0.0, 1.0, 0.0]),
            vector([0.015, 0.0, 0.015]),
        ]),
    ))
    .unwrap();
    let hole_region = block_on(geometry_faces_introduced_by_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![hole.clone()]),
    ))
    .unwrap();
    let finished = block_on(geometry_subtract_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![stock, hole]),
    ))
    .unwrap();
    let body_region = block_on(geometry_solid_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![finished.clone()]),
    ))
    .unwrap();
    let generated = block_on(geometry_finish_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![finished]),
    ))
    .unwrap();
    let generated = block_on(geometry_with_region_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            generated,
            Value::String("body".into()),
            body_region,
        ]),
    ))
    .unwrap();
    let generated = block_on(geometry_with_region_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            generated,
            Value::String("hole_wall".into()),
            hole_region,
        ]),
    ))
    .unwrap();

    let product = generated_from_value(&generated, "test").unwrap();
    assert_eq!(product.graph.features.len(), 3);
    assert!(matches!(
        product.graph.features.last().unwrap().operation,
        GeometryFeatureOperation::Subtract { .. }
    ));
    assert!(matches!(
        product.region_outputs["hole_wall"].selector,
        GeometryRegionSelector::BoundaryIntroducedBy(_)
    ));

    let payload = crate::execution::value_codec::encode_inline_value(&generated).unwrap();
    let ValuePayload::Inline(value) = payload else {
        panic!("generated geometry must remain inline");
    };
    let InlineValue::ImmutableValueClass(data) = value.as_ref() else {
        panic!("generated geometry must use the immutable class codec");
    };
    assert_eq!(data.type_identity, super::GENERATED_TYPE_IDENTITY);
}

#[test]
fn duplicate_region_names_and_untyped_values_fail_closed() {
    let feature = unit_box();
    let region = block_on(geometry_solid_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![feature.clone()]),
    ))
    .unwrap();
    let generated = block_on(geometry_finish_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![feature]),
    ))
    .unwrap();
    let generated = block_on(geometry_with_region_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            generated,
            Value::String("body".into()),
            region.clone(),
        ]),
    ))
    .unwrap();
    assert_eq!(
        identifier(block_on(geometry_with_region_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                generated,
                Value::String("body".into()),
                region,
            ])
        ))),
        "RunMat:geometry:withRegion:InvalidInput"
    );
    assert_eq!(
        identifier(block_on(geometry_finish_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![num(1.0)])
        ))),
        "RunMat:geometry:finish:InvalidInput"
    );
}

#[test]
fn instances_require_explicit_affine_occurrences() {
    let feature = unit_box();
    assert_eq!(
        identifier(block_on(geometry_instance_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![feature.clone()])
        ))),
        "RunMat:geometry:instance:InvalidInput"
    );
    let instance = block_on(geometry_instance_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            feature,
            translation(0.0, 0.0, 0.0),
            translation(2.0, 0.0, 0.0),
        ]),
    ))
    .unwrap();
    let GeometryFeatureOperation::Instance { transforms, .. } = last_operation(&instance) else {
        panic!("expected instance");
    };
    assert_eq!(transforms.len(), 2);
}

#[test]
fn transform_converts_matlab_matrices_to_the_row_major_contract() {
    // Translation in T(1:3,4) and a +90 degree rotation about z.
    let rotate_and_move = matrix([
        [0.0, -1.0, 0.0, 0.1],
        [1.0, 0.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    let moved = block_on(geometry_transform_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![unit_box(), rotate_and_move]),
    ))
    .unwrap();
    let GeometryFeatureOperation::Transform { matrix, .. } = last_operation(&moved) else {
        panic!("expected transform");
    };
    assert_eq!(
        literals(&matrix),
        vec![
            0.0, -1.0, 0.0, 0.1, //
            1.0, 0.0, 0.0, 0.2, //
            0.0, 0.0, 1.0, 0.3, //
            0.0, 0.0, 0.0, 1.0,
        ]
    );
}

#[test]
fn transform_rejects_non_affine_and_wrongly_shaped_matrices() {
    let projective = matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0, 1.0],
    ]);
    assert_eq!(
        identifier(block_on(geometry_transform_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![unit_box(), projective])
        ))),
        "RunMat:geometry:transform:InvalidInput"
    );
    let flat = Value::Tensor(Tensor::new(vec![0.0; 16], vec![1, 16]).unwrap());
    assert_eq!(
        identifier(block_on(geometry_transform_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![unit_box(), flat])
        ))),
        "RunMat:geometry:transform:InvalidInput"
    );
    let not_affine_instance = matrix([[0.0; 4]; 4]);
    assert_eq!(
        identifier(block_on(geometry_instance_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                unit_box(),
                not_affine_instance
            ])
        ))),
        "RunMat:geometry:instance:InvalidInput"
    );
}

#[test]
fn errors_report_the_calling_builtin_identifier() {
    // Feature-graph contract violation.
    assert_eq!(
        identifier(block_on(geometry_box_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                num(1.0),
                num(0.0),
                num(1.0)
            ])
        ))),
        "RunMat:geometry:box:InvalidInput"
    );
    assert_eq!(
        identifier(block_on(geometry_cylinder_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                num(0.01),
                num(0.02),
                vector([0.0, 0.0, 0.0]),
                vector([0.0, 0.0, 0.0]),
            ])
        ))),
        "RunMat:geometry:cylinder:InvalidInput"
    );
    // Arity.
    assert_eq!(
        identifier(block_on(geometry_box_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![num(1.0), num(1.0)])
        ))),
        "RunMat:geometry:box:InvalidInput"
    );
    assert_eq!(
        identifier(block_on(geometry_solid_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))),
        "RunMat:geometry:solid:InvalidInput"
    );
    // Scalar conversion.
    assert_eq!(
        identifier(block_on(geometry_box_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Bool(true),
                num(1.0),
                num(1.0)
            ])
        ))),
        "RunMat:geometry:box:InvalidInput"
    );
    assert_eq!(
        identifier(block_on(geometry_sphere_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("r".into()),
                vector([0.0, 0.0, 0.0]),
            ])
        ))),
        "RunMat:geometry:sphere:InvalidInput"
    );
    // Edge selection that names a feature outside the target graph.
    let other = block_on(geometry_box_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(2.0),
            num(2.0),
            num(2.0),
        ]),
    ))
    .unwrap();
    let edges = block_on(geometry_edges_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![other]),
    ))
    .unwrap();
    assert_eq!(
        identifier(block_on(geometry_fillet_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                unit_box(),
                edges,
                num(0.1)
            ])
        ))),
        "RunMat:geometry:fillet:InvalidInput"
    );
    // Region output that names a feature outside the finished part.
    let part = block_on(geometry_finish_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![unit_box()]),
    ))
    .unwrap();
    let foreign = block_on(geometry_box_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(3.0),
            num(3.0),
            num(3.0),
        ]),
    ))
    .unwrap();
    let foreign = block_on(geometry_solid_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![foreign]),
    ))
    .unwrap();
    assert_eq!(
        identifier(block_on(geometry_with_region_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                part,
                Value::String("x".into()),
                foreign,
            ])
        ))),
        "RunMat:geometry:withRegion:InvalidInput"
    );
}

#[test]
fn cylinder_types_lengths_and_positions_and_keeps_the_axis_dimensionless() {
    let cylinder = block_on(geometry_cylinder_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(0.01),
            num(0.02),
            vector([0.0, 0.0, 1.0]),
            vector([0.1, 0.2, 0.3]),
        ]),
    ))
    .unwrap();
    let GeometryFeatureOperation::Cylinder {
        radius_m,
        height_m,
        axis,
        origin_m,
    } = last_operation(&cylinder)
    else {
        panic!("expected cylinder");
    };
    let length = |expression: &GeometryScalarExpression| match expression {
        GeometryScalarExpression::Canonical { value } => {
            assert_eq!(
                value.semantic_kind(),
                Some(runmat_types::physical::QuantityKind::Length)
            );
            value.canonical_value()
        }
        other => panic!("expected a typed length, got {other:?}"),
    };
    assert_eq!(length(&radius_m), 0.01);
    assert_eq!(length(&height_m), 0.02);
    assert_eq!(
        origin_m.iter().map(length).collect::<Vec<_>>(),
        vec![0.1, 0.2, 0.3]
    );
    for (component, expected) in axis.iter().zip([0.0, 0.0, 1.0]) {
        assert!(
            matches!(component, GeometryScalarExpression::Literal { value } if *value == expected),
            "axis component must stay a dimensionless literal, got {component:?}"
        );
    }
}

#[test]
fn integer_scalars_are_admitted_and_integer_vectors_rejected() {
    let integer = |value: i32| Value::Int(IntValue::I32(value));
    assert!(block_on(geometry_box_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            integer(1),
            integer(2),
            integer(3)
        ])
    ))
    .is_ok());
    let integer_axis =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I32(vec![0, 0, 1]), vec![1, 3]).unwrap());
    assert_eq!(
        identifier(block_on(geometry_cylinder_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                num(0.01),
                num(0.02),
                integer_axis,
                vector([0.0, 0.0, 0.0]),
            ])
        ))),
        "RunMat:geometry:cylinder:InvalidInput"
    );
}

#[test]
fn sphere_and_booleans_record_typed_operations() {
    let ball = block_on(geometry_sphere_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(0.02),
            vector([0.0, 0.0, 0.0]),
        ]),
    ))
    .unwrap();
    assert!(matches!(
        last_operation(&ball),
        GeometryFeatureOperation::Sphere { .. }
    ));
    let slab = block_on(geometry_cylinder_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(0.02),
            num(0.02),
            vector([0.0, 0.0, 1.0]),
            vector([0.0, 0.0, 0.0]),
        ]),
    ))
    .unwrap();
    let dome = block_on(geometry_intersect_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![ball.clone(), slab.clone()]),
    ))
    .unwrap();
    let GeometryFeatureOperation::Intersect { operands } = last_operation(&dome) else {
        panic!("expected intersect");
    };
    assert_eq!(operands.len(), 2);
    let fused = block_on(geometry_union_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![ball.clone(), slab, ball]),
    ))
    .unwrap();
    let GeometryFeatureOperation::Union { operands } = last_operation(&fused) else {
        panic!("expected union");
    };
    assert_eq!(operands.len(), 3);
    // Shared features are merged once.
    assert_eq!(
        feature_from_value(&fused, "test").unwrap().features.len(),
        3
    );
    assert_eq!(
        identifier(block_on(geometry_union_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![unit_box()])
        ))),
        "RunMat:geometry:union:InvalidInput"
    );
}

#[test]
fn face_and_edge_selections_carry_role_cardinality_and_lineage() {
    let stock = unit_box();
    for (value, role, introduced) in [
        (
            block_on(geometry_faces_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![stock.clone()]),
            ))
            .unwrap(),
            GeometryRegionRole::Face,
            false,
        ),
        (
            block_on(geometry_edges_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![stock.clone()]),
            ))
            .unwrap(),
            GeometryRegionRole::Edge,
            false,
        ),
        (
            block_on(geometry_edges_introduced_by_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![stock.clone()]),
            ))
            .unwrap(),
            GeometryRegionRole::Edge,
            true,
        ),
    ] {
        let output = region_output_from_value(&value, "test").unwrap();
        assert_eq!(output.role, role);
        assert_eq!(
            matches!(
                output.selector,
                GeometryRegionSelector::BoundaryIntroducedBy(_)
            ),
            introduced
        );
    }
}

#[test]
fn full_feature_builder_surface_preserves_typed_dependencies_and_selections() {
    let stock = unit_box();
    let profile = block_on(geometry_box_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            num(0.25),
            num(0.25),
            num(0.1),
        ]),
    ))
    .unwrap();
    let extruded = block_on(geometry_extrude_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            profile.clone(),
            num(0.5),
            vector([0.0, 0.0, 1.0]),
        ]),
    ))
    .unwrap();
    let revolved = block_on(geometry_revolve_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            profile.clone(),
            vector([0.0, 1.0, 0.0]),
            vector([-1.0, 0.0, 0.0]),
            num(std::f64::consts::PI),
        ]),
    ))
    .unwrap();
    let holed = block_on(geometry_hole_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            stock.clone(),
            num(0.1),
            num(1.0),
            vector([0.0, 0.0, 1.0]),
            vector([0.5, 0.5, 0.0]),
        ]),
    ))
    .unwrap();
    let pocketed = block_on(geometry_pocket_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            stock.clone(),
            profile,
            num(0.1),
            vector([0.0, 0.0, 1.0]),
        ]),
    ))
    .unwrap();
    let edges = block_on(geometry_edges_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![stock.clone()]),
    ))
    .unwrap();
    let filleted = block_on(geometry_fillet_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            stock.clone(),
            edges.clone(),
            num(0.05),
        ]),
    ))
    .unwrap();
    let chamfered = block_on(geometry_chamfer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![stock, edges, num(0.05)]),
    ))
    .unwrap();

    for (value, expected) in [
        (extruded, "extrude"),
        (revolved, "revolve"),
        (holed, "hole"),
        (pocketed, "pocket"),
        (filleted, "fillet"),
        (chamfered, "chamfer"),
    ] {
        let actual = match last_operation(&value) {
            GeometryFeatureOperation::Extrude { .. } => "extrude",
            GeometryFeatureOperation::Revolve { .. } => "revolve",
            GeometryFeatureOperation::Hole { .. } => "hole",
            GeometryFeatureOperation::Pocket { .. } => "pocket",
            GeometryFeatureOperation::Fillet { .. } => "fillet",
            GeometryFeatureOperation::Chamfer { .. } => "chamfer",
            _ => panic!("unexpected generated feature operation"),
        };
        assert_eq!(actual, expected);
    }
}
