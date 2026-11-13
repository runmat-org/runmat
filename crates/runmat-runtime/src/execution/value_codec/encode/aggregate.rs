//! Durable aggregate admission at the canonical Value encoder boundary.

use super::{
    encode, encode_complex_storage, encode_sparse, encode_tensor, field_path, index_path, shape,
    CodecPurpose,
};
use crate::execution::value_codec::ValueCodecError;
use runmat_execution::value::{
    DenseValue, DurableDecompositionPayload, DurableDenseOrthogonalFactor, DurableDynamicProperty,
    DurableEmptyFactorKind, DurableFactorPermutation, DurableNumericFactor, DurableObjectArray,
    DurableOrthogonalFactor, DurablePlainObject, DurableStructArray, DurableStructArrayField,
    DurableStructuralDecomposition, ElementType, StructField,
};
use runmat_value::{
    DenseOrthogonalFactorData, ExactHostSparseClass, FactorPermutation, HostNumericBuffer,
    NativeDenseFactorMatrix, NumericDType, NumericFactorCandidate, NumericFactorKind, ObjectArray,
    ObjectInstance, ObjectPersistencePayloadKind, OrthogonalFactorData, StructArray, Value,
};

#[path = "aggregate/numeric_factor.rs"]
mod numeric_factor;
#[cfg(test)]
pub(super) fn encode_factor_for_test(
    factor: &runmat_value::NumericFactorData,
    path: &str,
) -> Result<DurableNumericFactor, ValueCodecError> {
    numeric_factor::encode_factor(factor, path)
}

pub(super) fn struct_array(
    value: &StructArray,
    path: &str,
    purpose: CodecPurpose,
    runtime: Option<&dyn super::InlineSourceAuthority>,
    workspace: Option<&super::super::ValidationWorkspaceOwner>,
) -> Result<DurableStructArray, ValueCodecError> {
    let fields = value
        .field_names()
        .map(|name| {
            Ok(DurableStructArrayField {
                name: name.to_owned(),
                values: value
                    .field_values(name)
                    .expect("field name from same array")
                    .iter()
                    .enumerate()
                    .map(|(index, item)| {
                        encode(
                            item,
                            &format!("{}[{index}]", field_path(path, name)),
                            purpose,
                            runtime,
                            workspace,
                        )
                    })
                    .collect::<Result<_, _>>()?,
            })
        })
        .collect::<Result<Vec<_>, ValueCodecError>>()?;
    Ok(DurableStructArray {
        shape: shape(value.shape(), path)?,
        fields,
    })
}

pub(super) fn object_array(
    value: &ObjectArray,
    path: &str,
    purpose: CodecPurpose,
    runtime: Option<&dyn super::InlineSourceAuthority>,
    workspace: Option<&super::super::ValidationWorkspaceOwner>,
) -> Result<DurableObjectArray, ValueCodecError> {
    let elements = value
        .data()
        .iter()
        .enumerate()
        .map(|(index, item)| match item {
            Value::Object(object) => plain_object(
                object,
                &index_path(path, index),
                purpose,
                runtime,
                workspace,
            ),
            _ => Err(ValueCodecError::unsupported(
                &index_path(path, index),
                "handle objects are not durable",
            )),
        })
        .collect::<Result<_, _>>()?;
    Ok(DurableObjectArray {
        class_identity: value.class_name().clone(),
        shape: shape(value.shape(), path)?,
        elements,
    })
}

pub(super) fn plain_object(
    value: &ObjectInstance,
    path: &str,
    purpose: CodecPurpose,
    runtime: Option<&dyn super::InlineSourceAuthority>,
    workspace: Option<&super::super::ValidationWorkspaceOwner>,
) -> Result<DurablePlainObject, ValueCodecError> {
    let decomposition = match value.persistence_payload_kind() {
        ObjectPersistencePayloadKind::PlainValue | ObjectPersistencePayloadKind::NumericFactor
            if value.is_class(runmat_types::standard::DECOMPOSITION) =>
        {
            use crate::builtins::math::linalg::factor::decomposition::{
                decomposition_persistence_kind, DecompositionPersistenceKind,
            };
            let kind = decomposition_persistence_kind(value)
                .map_err(|error| ValueCodecError::runtime(path, error))?;
            Some(match kind {
                DecompositionPersistenceKind::Factor(_) => {
                    DurableDecompositionPayload::Data(Box::new(numeric_factor::encode_factor(
                        value.numeric_factor_data().ok_or_else(|| {
                            ValueCodecError::invalid(path, "numerical factor data is missing")
                        })?,
                        path,
                    )?))
                }
                DecompositionPersistenceKind::Diagonal => DurableDecompositionPayload::Structural(
                    DurableStructuralDecomposition::Diagonal,
                ),
                DecompositionPersistenceKind::Triangular => {
                    DurableDecompositionPayload::Structural(
                        DurableStructuralDecomposition::Triangular,
                    )
                }
            })
        }
        ObjectPersistencePayloadKind::PlainValue => None,
        ObjectPersistencePayloadKind::NumericFactor => {
            return Err(ValueCodecError::invalid(
                path,
                "factor data has a different object class",
            ))
        }
        _ => {
            return Err(ValueCodecError::unsupported(
                path,
                "native object authority is not durable",
            ))
        }
    };
    let mut properties = value
        .properties()
        .iter()
        .map(|(name, item)| {
            Ok(StructField {
                name: name.to_owned(),
                value: encode(item, &field_path(path, name), purpose, runtime, workspace)?,
            })
        })
        .collect::<Result<Vec<_>, ValueCodecError>>()?;
    properties.sort_by(|a, b| a.name.cmp(&b.name));
    let mut dynamic_properties = value
        .dynamic_property_definitions()
        .map(|(name, definition)| {
            if definition.metadata_handle.is_some() {
                return Err(ValueCodecError::unsupported(
                    &field_path(path, name),
                    "GC-owned dynamic property metadata is not durable",
                ));
            }
            Ok(DurableDynamicProperty {
                name: name.to_owned(),
                definition_name: definition.name.to_string(),
                defining_class: definition.defining_class.as_identity().clone(),
                get_access: definition.get_access,
                set_access: definition.set_access,
                dependent: definition.dependent,
                hidden: definition.hidden,
                transient: definition.transient,
                non_copyable: definition.non_copyable,
                abort_set: definition.abort_set,
                set_observable: definition.set_observable,
                get_observable: definition.get_observable,
                description: definition.description.to_string(),
            })
        })
        .collect::<Result<Vec<_>, ValueCodecError>>()?;
    dynamic_properties.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(DurablePlainObject {
        class_identity: value.class_name().clone(),
        properties,
        dynamic_properties,
        decomposition,
    })
}
