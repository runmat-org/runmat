use runmat_execution::value::{
    DenseValue, ElementType, InlineValue, RegisteredData, RegisteredField, ValueLimits,
    ValuePayload, ValueRef, ValueRefKind,
};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionResponse,
};
use runmat_geometry_core::GeneratedGeometryProduct;

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

use super::{
    AdmittedGeometryGenerator, GeneratedGeometryExecutionReceipt, GeometryMaterializationSource,
    GENERATED_GEOMETRY_PRODUCT_MEDIA_TYPE, GENERATED_GEOMETRY_PRODUCT_VALUE_SCHEMA,
};

pub const GENERATED_GEOMETRY_VALUE_TYPE: &str = "runmat.geometry.generated-product";
pub const GENERATED_GEOMETRY_VALUE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedGeneratedGeometryProduct {
    pub product: GeneratedGeometryProduct,
    pub receipt: GeneratedGeometryExecutionReceipt,
    pub object: LogicalObject,
    pub root: ValueRef,
    pub source: GeometryMaterializationSource,
}

impl PreparedGeneratedGeometryProduct {
    pub fn write_to(&self, store: &mut impl CacheExport) -> AnalysisBridgeResult<()> {
        store.write_verified(&self.object)?;
        Ok(())
    }

    pub fn root(&self) -> &ValueRef {
        &self.root
    }
}

pub fn encode_generated_geometry_value(
    product: &GeneratedGeometryProduct,
) -> AnalysisBridgeResult<ValuePayload> {
    let bytes = encode_generated_geometry_product(product)?;
    let payload =
        ValuePayload::Inline(Box::new(InlineValue::ImmutableValueClass(RegisteredData {
            type_identity: GENERATED_GEOMETRY_VALUE_TYPE.into(),
            schema_version: GENERATED_GEOMETRY_VALUE_SCHEMA_VERSION,
            fields: vec![RegisteredField {
                name: "contract".into(),
                value: ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
                    element_type: ElementType::U8,
                    shape: vec![bytes.len() as u64, 1],
                    little_endian_data: bytes,
                }))),
            }],
        })));
    payload.validate(ValueLimits::default())?;
    Ok(payload)
}

pub fn encode_generated_geometry_product(
    product: &GeneratedGeometryProduct,
) -> AnalysisBridgeResult<Vec<u8>> {
    product
        .validate()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    product
        .canonical_bytes()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
}

pub fn decode_generated_geometry_product(
    bytes: &[u8],
) -> AnalysisBridgeResult<GeneratedGeometryProduct> {
    GeneratedGeometryProduct::from_canonical_bytes(bytes)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
}

pub fn decode_generated_geometry_value(
    payload: &ValuePayload,
) -> AnalysisBridgeResult<GeneratedGeometryProduct> {
    payload.validate(ValueLimits::default())?;
    let ValuePayload::Inline(value) = payload else {
        return Err(AnalysisBridgeError::Invalid(
            "generated geometry value must be a closed inline value".into(),
        ));
    };
    let InlineValue::ImmutableValueClass(data) = value.as_ref() else {
        return Err(AnalysisBridgeError::Invalid(
            "geometry generator did not return a typed geometry value".into(),
        ));
    };
    let [field] = data.fields.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "generated geometry value has an invalid field set".into(),
        ));
    };
    let ValuePayload::Inline(contract) = &field.value else {
        return Err(AnalysisBridgeError::Invalid(
            "generated geometry contract must be inline".into(),
        ));
    };
    let InlineValue::Dense(contract) = contract.as_ref() else {
        return Err(AnalysisBridgeError::Invalid(
            "generated geometry contract must be an unsigned byte vector".into(),
        ));
    };
    if data.type_identity != GENERATED_GEOMETRY_VALUE_TYPE
        || data.schema_version != GENERATED_GEOMETRY_VALUE_SCHEMA_VERSION
        || field.name != "contract"
        || contract.element_type != ElementType::U8
        || contract.shape != [contract.little_endian_data.len() as u64, 1]
    {
        return Err(AnalysisBridgeError::Invalid(
            "generated geometry value has an incompatible schema".into(),
        ));
    }
    decode_generated_geometry_product(&contract.little_endian_data)
}

pub fn prepare_generated_geometry_product(
    admitted: &AdmittedGeometryGenerator,
    response: ProgramExecutionResponse,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<PreparedGeneratedGeometryProduct> {
    let ProgramExecutionResponse::Success { outputs } = response else {
        return Err(AnalysisBridgeError::Invalid(
            "geometry generator did not return one inline result".into(),
        ));
    };
    let [value] = <[_; 1]>::try_from(outputs).map_err(|_| {
        AnalysisBridgeError::Invalid("geometry generator did not return one inline result".into())
    })?;
    let product = decode_generated_geometry_value(&value)?;
    let bytes = encode_generated_geometry_product(&product)?;
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        format!("geometry/generated/{}", admitted.materialization_identity),
        GENERATED_GEOMETRY_PRODUCT_MEDIA_TYPE,
        bytes,
    )?;
    let root = access.reference(
        &object,
        ValueRefKind::ResultObject,
        GENERATED_GEOMETRY_PRODUCT_VALUE_SCHEMA,
    )?;
    let receipt = GeneratedGeometryExecutionReceipt {
        schema_version: 1,
        admission: admitted.clone(),
        product_digest: root.logical_digest,
        product_encoded_length: root.encoded_length,
    };
    receipt.validate()?;
    let source = GeometryMaterializationSource::Generated {
        receipt: Box::new(receipt.clone()),
    };
    Ok(PreparedGeneratedGeometryProduct {
        product,
        receipt,
        object,
        root,
        source,
    })
}
