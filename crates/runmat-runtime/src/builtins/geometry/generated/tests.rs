use futures::executor::block_on;
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_geometry_core::{GeometryFeatureOperation, GeometryRegionSelector};
use runmat_value::{Tensor, Value};

use super::codec::generated_from_value;
use super::combine::{geometry_finish_builtin, geometry_subtract_builtin};
use super::primitives::{
    geometry_box_builtin, geometry_cylinder_builtin, geometry_instance_builtin,
};
use super::regions::{
    geometry_faces_introduced_by_builtin, geometry_solid_builtin, geometry_with_region_builtin,
};

#[test]
fn typed_builder_composes_regions_and_crosses_the_portable_value_boundary() {
    let stock = block_on(geometry_box_builtin(0.08, 0.008, 0.05)).unwrap();
    let hole = block_on(geometry_cylinder_builtin(
        0.005,
        0.008,
        vector([0.0, 1.0, 0.0]),
        vector([0.015, 0.0, 0.015]),
    ))
    .unwrap();
    let hole_region = block_on(geometry_faces_introduced_by_builtin(hole.clone())).unwrap();
    let finished = block_on(geometry_subtract_builtin(vec![stock, hole])).unwrap();
    let body_region = block_on(geometry_solid_builtin(finished.clone())).unwrap();
    let generated = block_on(geometry_finish_builtin(finished)).unwrap();
    let generated = block_on(geometry_with_region_builtin(
        generated,
        "body".into(),
        body_region,
    ))
    .unwrap();
    let generated = block_on(geometry_with_region_builtin(
        generated,
        "hole_wall".into(),
        hole_region,
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
    let feature = block_on(geometry_box_builtin(1.0, 1.0, 1.0)).unwrap();
    let region = block_on(geometry_solid_builtin(feature.clone())).unwrap();
    let generated = block_on(geometry_finish_builtin(feature)).unwrap();
    let generated = block_on(geometry_with_region_builtin(
        generated,
        "body".into(),
        region.clone(),
    ))
    .unwrap();
    assert!(block_on(geometry_with_region_builtin(
        generated,
        "body".into(),
        region,
    ))
    .is_err());
    assert!(block_on(geometry_finish_builtin(Value::Num(1.0))).is_err());
}

#[test]
fn instances_require_explicit_affine_occurrences() {
    let feature = block_on(geometry_box_builtin(1.0, 1.0, 1.0)).unwrap();
    assert!(block_on(geometry_instance_builtin(feature.clone(), Vec::new())).is_err());
    let identity = Value::Tensor(
        Tensor::new(
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            vec![4, 4],
        )
        .unwrap(),
    );
    let instance = block_on(geometry_instance_builtin(feature, vec![identity])).unwrap();
    let product = super::codec::feature_from_value(&instance, "test").unwrap();
    assert!(matches!(
        product.features.last().unwrap().operation,
        GeometryFeatureOperation::Instance { .. }
    ));
}

fn vector(values: [f64; 3]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![3, 1]).unwrap())
}
