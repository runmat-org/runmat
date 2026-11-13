//! Runtime bindings for canonical underlying-type metadata queries.
use crate::builtins::introspection::class_identity::class_identity_for_value;
#[cfg(test)]
use crate::builtins::introspection::class_identity::class_name_for_value;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{handle_integer_type, handle_is_logical, handle_precision};
use runmat_builtins::{
    BuiltinErrorDescriptor, IS_UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD,
    IS_UNDERLYING_TYPE_ERROR_TYPE_NAME_INVALID, UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "underlyingType",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::underlying_type"
)]
fn underlying_type_builtin(value: Value) -> BuiltinResult<String> {
    underlying_type_for_value_checked(&value)
}

#[runtime_builtin(
    name = "isUnderlyingType",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::underlying_type"
)]
fn is_underlying_type_builtin(value: Value, typename: Value) -> BuiltinResult<Value> {
    let requested = parse_type_name(&typename)?;
    Ok(Value::Bool(underlying_type_matches_checked(
        &value, requested,
    )?))
}

/// Return the canonical underlying MATLAB data type for a runtime value.
#[cfg(test)]
pub(crate) fn underlying_type_for_value(value: &Value) -> String {
    underlying_type_for_value_checked(value).unwrap_or_else(|_| class_name_for_value(value))
}

fn underlying_type_for_value_checked(value: &Value) -> BuiltinResult<String> {
    underlying_class_identity_for_value_checked_for(value, UnderlyingQueryOwner::Type)
        .map(|identity| identity.display_name().to_owned())
}

#[derive(Clone, Copy)]
enum UnderlyingQueryOwner {
    Type,
    Predicate,
}

/// Shared typed underlying-class authority. Public text queries format this
/// result; validators compare admitted identities without reparsing class tables.
pub(crate) fn underlying_class_identity_for_value_checked(
    value: &Value,
) -> BuiltinResult<runmat_types::ClassIdentity> {
    underlying_class_identity_for_value_checked_for(value, UnderlyingQueryOwner::Predicate)
}

fn underlying_class_identity_for_value_checked_for(
    value: &Value,
    owner: UnderlyingQueryOwner,
) -> BuiltinResult<runmat_types::ClassIdentity> {
    match value {
        Value::GpuTensor(handle) => gpu_underlying_class_identity(handle, owner),
        _ => Ok(class_identity_for_value(value)),
    }
}

fn gpu_underlying_class_identity(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    owner: UnderlyingQueryOwner,
) -> BuiltinResult<runmat_types::ClassIdentity> {
    runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
        let mut error = metadata_error(owner, "no acceleration provider owns the input handle");
        error.source = Some(source.into_boxed_dyn_error());
        error
    })?;
    // Preferred execution precision is not the class of an owned allocation.
    // The owner has already admitted its immutable physical type and view.
    crate::builtins::common::gpu_helpers::expected_handle_numeric_element_type(handle)
        .map_err(|detail| metadata_error(owner, &detail))?;
    let integer = handle_integer_type(handle);
    let logical = handle_is_logical(handle);
    let precision = handle_precision(handle);
    if !crate::builtins::common::gpu_helpers::gpu_class_metadata_matches(
        handle, precision, integer, logical,
    ) {
        return Err(metadata_error(
            owner,
            "resident class metadata contradicts physical storage metadata",
        ));
    }
    crate::builtins::common::gpu_helpers::expected_gpu_class_identity(precision, integer, logical)
        .ok_or_else(|| metadata_error(owner, "physical numeric class is missing"))
}

fn metadata_error(owner: UnderlyingQueryOwner, detail: &str) -> RuntimeError {
    let (builtin, descriptor) = match owner {
        UnderlyingQueryOwner::Type => ("underlyingType", &UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD),
        UnderlyingQueryOwner::Predicate => (
            "isUnderlyingType",
            &IS_UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD,
        ),
    };
    build_runtime_error(format!("{builtin}: {detail}"))
        .with_builtin(builtin)
        .with_identifier(
            descriptor
                .identifier
                .expect("provider-payload descriptor identifier"),
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

#[cfg(test)]
pub(crate) fn underlying_type_matches(value: &Value, typename: &str) -> bool {
    underlying_type_matches_checked(
        value,
        runmat_types::ClassIdentity::new(typename.trim()).ok(),
    )
    .unwrap_or(false)
}

fn underlying_type_matches_checked(
    value: &Value,
    requested: Option<runmat_types::ClassIdentity>,
) -> BuiltinResult<bool> {
    let Some(requested) = requested else {
        return Ok(false);
    };
    let actual = underlying_class_identity_for_value_checked(value)?;
    Ok(requested == actual)
}

fn parse_type_name(value: &Value) -> BuiltinResult<Option<runmat_types::ClassIdentity>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => {
            Ok(runmat_types::ClassIdentity::new(s.trim()).ok())
        }
        Value::StringArray(sa)
            if sa.rows == 1
                && sa.cols == 1
                && !sa.data().is_empty()
                && !sa.data()[0].is_missing() =>
        {
            Ok(runmat_types::ClassIdentity::new(
                sa.data()[0]
                    .as_str()
                    .expect("checked nonmissing string")
                    .trim(),
            )
            .ok())
        }
        Value::CharArray(ca) if ca.rows == 1 => {
            let text = super::text::decode_utf16_with_error(ca.data(), || {
                is_underlying_type_error(&IS_UNDERLYING_TYPE_ERROR_TYPE_NAME_INVALID)
            })?;
            Ok(runmat_types::ClassIdentity::new(text.trim()).ok())
        }
        _ => Err(is_underlying_type_error(&IS_UNDERLYING_TYPE_ERROR_TYPE_NAME_INVALID).into()),
    }
}

fn is_underlying_type_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin("isUnderlyingType");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
        LogicalArray, NumericDType, ObjectInstance, StringArray, Tensor,
    };

    #[test]
    fn nested_underlying_type_query_uses_string_value_carrier() {
        let inner =
            underlying_type_builtin(Value::Int(IntValue::I16(7))).expect("inner underlying type");
        let value = Value::from(inner);
        assert!(matches!(value, Value::String(_)));
        assert_eq!(
            underlying_type_builtin(value).expect("outer type"),
            "string"
        );
    }

    #[test]
    fn provider_metadata_errors_keep_the_public_query_owner() {
        let type_error = metadata_error(UnderlyingQueryOwner::Type, "missing owner");
        assert_eq!(
            type_error.identifier(),
            UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD.identifier
        );
        assert!(type_error.message().starts_with("underlyingType:"));
        let predicate_error = metadata_error(UnderlyingQueryOwner::Predicate, "missing owner");
        assert_eq!(
            predicate_error.identifier(),
            IS_UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD.identifier
        );
        assert!(predicate_error.message().starts_with("isUnderlyingType:"));
    }

    #[test]
    fn underlying_type_reports_core_host_types() {
        assert_eq!(underlying_type_for_value(&Value::Num(1.0)), "double");
        assert_eq!(
            underlying_type_for_value(&Value::Tensor(
                Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], NumericDType::F32)
                    .expect("single tensor")
            )),
            "single"
        );
        assert_eq!(
            underlying_type_for_value(&Value::Int(IntValue::U16(7))),
            "uint16"
        );
        assert_eq!(underlying_type_for_value(&Value::Bool(true)), "logical");
        assert_eq!(
            underlying_type_for_value(&Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical array")
            )),
            "logical"
        );
        assert_eq!(
            underlying_type_for_value(&Value::CharArray(CharArray::new_row("abc"))),
            "char"
        );
        assert_eq!(
            underlying_type_for_value(&Value::String("abc".into())),
            "string"
        );
    }

    #[test]
    fn underlying_type_reports_typed_sparse_integer_class() {
        let sparse = runmat_value::SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::I64(vec![i64::MIN]),
        )
        .expect("int64 sparse");

        assert_eq!(
            underlying_type_for_value(&Value::SparseTensor(sparse)),
            "int64"
        );
    }

    #[test]
    fn underlying_type_reports_every_typed_complex_integer_class() {
        let cases = [
            (
                "int8",
                IntegerStorage::I8(vec![-1]),
                IntegerStorage::I8(vec![2]),
            ),
            (
                "int16",
                IntegerStorage::I16(vec![-3]),
                IntegerStorage::I16(vec![4]),
            ),
            (
                "int32",
                IntegerStorage::I32(vec![-5]),
                IntegerStorage::I32(vec![6]),
            ),
            (
                "int64",
                IntegerStorage::I64(vec![-7]),
                IntegerStorage::I64(vec![8]),
            ),
            (
                "uint8",
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U8(vec![2]),
            ),
            (
                "uint16",
                IntegerStorage::U16(vec![3]),
                IntegerStorage::U16(vec![4]),
            ),
            (
                "uint32",
                IntegerStorage::U32(vec![5]),
                IntegerStorage::U32(vec![6]),
            ),
            (
                "uint64",
                IntegerStorage::U64(vec![7]),
                IntegerStorage::U64(vec![8]),
            ),
        ];

        for (expected, real, imag) in cases {
            let storage = IntegerComplexStorage::new(real, imag).expect("matching components");
            let value = Value::ComplexTensor(
                ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex"),
            );
            assert_eq!(underlying_type_for_value(&value), expected);
            assert!(underlying_type_matches(&value, expected));
        }
    }

    #[test]
    fn underlying_type_matches_class_for_containers_and_objects() {
        let cell =
            Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap());
        assert_eq!(underlying_type_for_value(&cell), "cell");

        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert("x".into(), Value::Num(1.0));
        assert_eq!(
            underlying_type_for_value(&Value::Struct(st.finish())),
            "struct"
        );

        let object = ObjectInstance::new("pkg.Point");
        assert_eq!(
            underlying_type_for_value(&Value::Object(object)),
            "pkg.Point"
        );
    }

    #[test]
    fn is_underlying_type_accepts_string_and_char_names() {
        assert_eq!(
            is_underlying_type_builtin(Value::Int(IntValue::I32(3)), Value::from("int32"))
                .expect("string name"),
            Value::Bool(true)
        );
        assert_eq!(
            is_underlying_type_builtin(
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                Value::CharArray(CharArray::new_row("double")),
            )
            .expect("char name"),
            Value::Bool(true)
        );
        assert_eq!(
            is_underlying_type_builtin(Value::Bool(true), Value::from("double"))
                .expect("false predicate"),
            Value::Bool(false)
        );
        assert_eq!(
            is_underlying_type_builtin(Value::Num(1.0), Value::from("Double"))
                .expect("case-sensitive mismatch"),
            Value::Bool(false)
        );
    }

    #[test]
    fn is_underlying_type_accepts_scalar_string_array_only() {
        let name = Value::StringArray(
            StringArray::new(vec!["single".into()], vec![1, 1]).expect("string scalar"),
        );
        let value = Value::Tensor(
            Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::F32)
                .expect("single tensor"),
        );
        assert_eq!(
            is_underlying_type_builtin(value, name).expect("scalar string array"),
            Value::Bool(true)
        );

        let invalid = Value::StringArray(
            StringArray::new(vec!["double".into(), "single".into()], vec![1, 2])
                .expect("string row"),
        );
        let err = is_underlying_type_builtin(Value::Num(1.0), invalid)
            .expect_err("non-scalar string array");
        assert_eq!(
            err.identifier(),
            Some("RunMat:isUnderlyingType:TypeNameInvalid")
        );
    }

    #[test]
    fn underlying_type_uses_gpu_metadata_without_gather() {
        test_support::with_f32_test_provider(|provider| {
            let tensor =
                Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], NumericDType::F32).unwrap();
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload");
            let lease = handle.allocation_lease().downgrade();
            provider.reset_telemetry();

            assert_eq!(
                underlying_type_builtin(Value::GpuTensor(handle.clone())).expect("underlying"),
                "single"
            );
            assert_eq!(
                is_underlying_type_builtin(Value::GpuTensor(handle), Value::from("single"))
                    .expect("predicate"),
                Value::Bool(true)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert!(lease.acquire().is_none());
        });
    }

    #[test]
    fn underlying_type_reports_integer_gpu_class_from_native_provider_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|_| {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let tensor = Tensor::new(vec![1.2, -3.7, 123456.0], vec![3, 1]).unwrap();
            let output = crate::dispatcher::call_builtin(
                "gpuArray",
                &[Value::Tensor(tensor), Value::from("int32")],
            )
            .expect("gpuArray int32 native upload");
            let Value::GpuTensor(handle) = output.value() else {
                panic!("expected gpu tensor, got {:?}", output.value());
            };

            assert_eq!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(runmat_accelerate_api::IntegerElementType::I32)
            );
            assert_eq!(
                underlying_type_builtin(Value::GpuTensor(handle.clone())).expect("underlying"),
                "int32"
            );
            assert_eq!(
                is_underlying_type_builtin(Value::GpuTensor(handle.clone()), Value::from("int32"))
                    .expect("predicate"),
                Value::Bool(true)
            );
        });
    }

    #[test]
    fn underlying_type_uses_owner_storage_after_contradictory_wire_rejection() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let original = handle.wire_descriptor();
            let mut contradictory = original.clone();
            contradictory.physical = runmat_accelerate_api::GpuPhysicalDescriptor::new(
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
                1,
                8,
                8,
            )
            .unwrap();
            provider.reset_telemetry();
            assert_eq!(
                provider.acquire_live(&contradictory).unwrap_err(),
                runmat_accelerate_api::allocation::AllocationAcquisitionError::PhysicalMismatch
            );
            assert_eq!(handle.wire_descriptor(), original);
            let lease = handle.allocation_lease().downgrade();
            assert_eq!(
                underlying_type_builtin(Value::GpuTensor(handle)).unwrap(),
                "uint64"
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert!(lease.acquire().is_none());
        });
    }

    #[test]
    fn underlying_type_reports_logical_for_gpu_masks() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1, 0],
                    shape: &[1, 2],
                })
                .expect("native logical upload");
            let lease = handle.allocation_lease().downgrade();
            provider.reset_telemetry();

            assert_eq!(
                underlying_type_builtin(Value::GpuTensor(handle.clone())).expect("underlying"),
                "logical"
            );
            assert_eq!(
                is_underlying_type_builtin(Value::GpuTensor(handle), Value::from("logical"))
                    .expect("predicate"),
                Value::Bool(true)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert!(lease.acquire().is_none());
        });
    }
}
