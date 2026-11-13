use runmat_geometry_core::{
    GeneratedGeometryProduct, GeneratedGeometryRegionOutput, GeometryFeatureGraph,
};
use runmat_value::{IntegerStorage, ObjectInstance, Tensor, Value};

use super::{
    ensure_classes_registered, invalid, CONTRACT_PROPERTY, GENERATED_GEOMETRY_CLASS,
    GEOMETRY_FEATURE_CLASS, GEOMETRY_REGION_OUTPUT_CLASS,
};
use crate::BuiltinResult;

pub(super) fn feature_value(
    graph: GeometryFeatureGraph,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let product = GeneratedGeometryProduct {
        graph,
        region_outputs: Default::default(),
    };
    product
        .validate()
        .map_err(|error| invalid(builtin, error.to_string()))?;
    object_value(
        GEOMETRY_FEATURE_CLASS,
        product
            .canonical_bytes()
            .map_err(|error| invalid(builtin, error.to_string()))?,
        builtin,
    )
}

pub(super) fn generated_value(
    product: GeneratedGeometryProduct,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    product
        .validate()
        .map_err(|error| invalid(builtin, error.to_string()))?;
    object_value(
        GENERATED_GEOMETRY_CLASS,
        product
            .canonical_bytes()
            .map_err(|error| invalid(builtin, error.to_string()))?,
        builtin,
    )
}

pub(super) fn region_output_value(
    output: &GeneratedGeometryRegionOutput,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let bytes = output
        .canonical_bytes()
        .map_err(|error| invalid(builtin, error.to_string()))?;
    object_value(GEOMETRY_REGION_OUTPUT_CLASS, bytes, builtin)
}

pub(super) fn feature_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<GeometryFeatureGraph> {
    let bytes = object_bytes(value, GEOMETRY_FEATURE_CLASS, builtin)?;
    GeneratedGeometryProduct::from_canonical_bytes(&bytes)
        .map(|product| product.graph)
        .map_err(|error| invalid(builtin, error.to_string()))
}

pub(super) fn generated_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<GeneratedGeometryProduct> {
    let bytes = object_bytes(value, GENERATED_GEOMETRY_CLASS, builtin)?;
    GeneratedGeometryProduct::from_canonical_bytes(&bytes)
        .map_err(|error| invalid(builtin, error.to_string()))
}

pub(super) fn region_output_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<GeneratedGeometryRegionOutput> {
    let bytes = object_bytes(value, GEOMETRY_REGION_OUTPUT_CLASS, builtin)?;
    GeneratedGeometryRegionOutput::from_canonical_bytes(&bytes)
        .map_err(|error| invalid(builtin, error.to_string()))
}

pub(super) fn validate_feature_object(object: &ObjectInstance) -> Result<(), String> {
    validate_product_object(object, GEOMETRY_FEATURE_CLASS).map(|_| ())
}

pub(super) fn validate_generated_object(object: &ObjectInstance) -> Result<(), String> {
    validate_product_object(object, GENERATED_GEOMETRY_CLASS).map(|_| ())
}

pub(super) fn validate_region_output_object(object: &ObjectInstance) -> Result<(), String> {
    let bytes = object_contract_bytes(object, GEOMETRY_REGION_OUTPUT_CLASS)?;
    GeneratedGeometryRegionOutput::from_canonical_bytes(&bytes)
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn validate_product_object(
    object: &ObjectInstance,
    class_name: &str,
) -> Result<GeneratedGeometryProduct, String> {
    let bytes = object_contract_bytes(object, class_name)?;
    GeneratedGeometryProduct::from_canonical_bytes(&bytes).map_err(|error| error.to_string())
}

fn object_value(class_name: &str, bytes: Vec<u8>, builtin: &'static str) -> BuiltinResult<Value> {
    ensure_classes_registered();
    let length = bytes.len();
    let fields = [(
        CONTRACT_PROPERTY,
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(bytes), vec![length, 1])
                .map_err(|error| invalid(builtin, error))?,
        ),
    )];
    // These legacy entrypoints have no selected contextual factory authority.
    // The shared constructor enforces ordinary policy rather than demoting an
    // admitted object to an unclaimed mutable generation.
    crate::execution::object::construct_object_from_borrowed_fields(
        None,
        class_name,
        fields.iter().map(|(name, value)| (*name, value)),
    )
    .map(|output| output.with_owned(|value| value))
}

fn object_bytes(value: &Value, class_name: &str, builtin: &'static str) -> BuiltinResult<Vec<u8>> {
    let Value::Object(object) = value else {
        return Err(invalid(builtin, format!("expected {class_name}")));
    };
    object_contract_bytes(object, class_name).map_err(|message| invalid(builtin, message))
}

fn object_contract_bytes(object: &ObjectInstance, class_name: &str) -> Result<Vec<u8>, String> {
    if object.class_name().display_name() != class_name || object.properties().len() != 1 {
        return Err(format!("expected one immutable {class_name} value"));
    }
    let Some(Value::Tensor(contract)) = object.properties().get(CONTRACT_PROPERTY) else {
        return Err(format!("{class_name} has no byte contract"));
    };
    let Some(IntegerStorage::U8(bytes)) = contract.integer_storage() else {
        return Err(format!("{class_name} contract must be uint8"));
    };
    if contract.shape() != [bytes.len(), 1] {
        return Err(format!("{class_name} contract has an invalid shape"));
    }
    Ok(bytes.clone())
}
