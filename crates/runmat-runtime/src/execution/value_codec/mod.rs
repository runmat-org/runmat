mod decode;
mod encode;
mod error;
mod immutable_class;

pub use decode::decode_inline_value;
pub use encode::encode_inline_value;
pub use error::ValueCodecError;
pub use immutable_class::{
    register_immutable_value_class, ImmutableValueClassSchema, ImmutableValueClassValidator,
};

#[cfg(test)]
mod tests {
    use crate::execution::RuntimeExecutionServices;
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, DynamicPropertyDef, ForeignAffinity, ForeignLifetime,
        ForeignOwnership, ForeignRef, ForeignTypeIdentity, IntValue, IntegerComplexStorage,
        IntegerStorage, MException, ObjectArray, ObjectInstance, StringArray, StructValue, Tensor,
        Value,
    };
    use std::sync::Once;

    use runmat_execution::value::{InlineValue, RegisteredData, RegisteredField, ValuePayload};

    use super::{
        decode_inline_value, encode_inline_value, register_immutable_value_class,
        ImmutableValueClassSchema, ValueCodecError,
    };

    const TEST_CLASS: &str = "runmat.test.PortableValue";
    const TEST_TYPE: &str = "runmat.test.portable-value";
    const TEST_ALLOWED_FIELDS: &[&str] = &["count", "id", "nested"];
    const TEST_REQUIRED_FIELDS: &[&str] = &["id"];

    fn validate_test_object(object: &ObjectInstance) -> Result<(), String> {
        match object.properties.get("id") {
            Some(Value::String(id)) if !id.is_empty() => Ok(()),
            Some(_) => Err("portable test object's `id` must be a string".into()),
            None => Err("portable test object is missing `id`".into()),
        }
    }

    fn ensure_test_class_registered() {
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            register_immutable_value_class(ImmutableValueClassSchema {
                class_name: TEST_CLASS,
                type_identity: TEST_TYPE,
                schema_version: 3,
                allowed_fields: TEST_ALLOWED_FIELDS,
                required_fields: TEST_REQUIRED_FIELDS,
                validate: validate_test_object,
            })
            .unwrap();
        });
    }

    fn test_object(id: &str) -> ObjectInstance {
        ensure_test_class_registered();
        let mut object = ObjectInstance::new(TEST_CLASS.into());
        object
            .properties
            .insert("id".into(), Value::String(id.into()));
        object
    }

    #[test]
    fn portable_value_roundtrip_is_centralized_and_bit_exact() {
        let value = Value::Cell(
            CellArray::new(
                vec![
                    Value::Num(f64::from_bits(0x7ff8_0000_0000_0042)),
                    Value::Int(IntValue::U64(u64::MAX)),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let payload = encode_inline_value(&value).unwrap();
        let Value::Cell(decoded) = decode_inline_value(&payload).unwrap() else {
            panic!("expected decoded cell");
        };
        let Value::Num(number) = decoded.data[0] else {
            panic!("expected decoded number");
        };
        assert_eq!(number.to_bits(), 0x7ff8_0000_0000_0042);
        assert_eq!(decoded.data[1], Value::Int(IntValue::U64(u64::MAX)));
        assert_eq!(decoded.shape, vec![1, 2]);
    }

    #[test]
    fn stable_immutable_runtime_forms_round_trip_exactly() {
        let mut structure = StructValue::new();
        structure.insert(
            "tensor",
            Value::Tensor(
                Tensor::new(vec![f64::from_bits(0x8000_0000_0000_0000), 2.5], vec![2, 1]).unwrap(),
            ),
        );
        structure.insert(
            "strings",
            Value::StringArray(StringArray::new(vec!["α".into(), "β".into()], vec![1, 2]).unwrap()),
        );
        structure.insert(
            "chars",
            Value::CharArray(CharArray::new(vec!['a', 'β'], 2, 1).unwrap()),
        );
        structure.insert(
            "error",
            Value::MException(MException {
                identifier: "RunMat:test".into(),
                message: "failure".into(),
                stack: vec!["main:1".into()],
            }),
        );
        let value = Value::Struct(structure);
        let payload = encode_inline_value(&value).unwrap();
        let decoded = decode_inline_value(&payload).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn native_complex_classes_round_trip_without_widening() {
        let single = Value::ComplexTensor(
            ComplexTensor::from_f32(
                vec![
                    (f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_0042)),
                    (f32::MAX, f32::MIN_POSITIVE),
                ],
                vec![1, 2],
            )
            .unwrap(),
        );
        let Value::ComplexTensor(decoded_single) =
            decode_inline_value(&encode_inline_value(&single).unwrap()).unwrap()
        else {
            panic!("expected decoded single complex tensor");
        };
        let decoded_values = decoded_single.as_f32_slice().unwrap();
        assert_eq!(decoded_values[0].0.to_bits(), 0x8000_0000);
        assert_eq!(decoded_values[0].1.to_bits(), 0x7fc0_0042);
        assert_eq!(decoded_values[1], (f32::MAX, f32::MIN_POSITIVE));
        assert_eq!(decoded_single.shape, vec![1, 2]);

        let integer_components = [
            (
                IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
                IntegerStorage::I8(vec![i8::MAX, i8::MIN]),
            ),
            (
                IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
                IntegerStorage::I16(vec![i16::MAX, i16::MIN]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
                IntegerStorage::I32(vec![i32::MAX, i32::MIN]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::I64(vec![i64::MAX, i64::MIN]),
            ),
            (
                IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
                IntegerStorage::U8(vec![u8::MAX, u8::MIN]),
            ),
            (
                IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
                IntegerStorage::U16(vec![u16::MAX, u16::MIN]),
            ),
            (
                IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
                IntegerStorage::U32(vec![u32::MAX, u32::MIN]),
            ),
            (
                IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
                IntegerStorage::U64(vec![u64::MAX, u64::MIN]),
            ),
        ];
        for (real, imaginary) in integer_components {
            let value = Value::ComplexTensor(
                ComplexTensor::new_integer(
                    IntegerComplexStorage::new(real, imaginary).unwrap(),
                    vec![2, 1],
                )
                .unwrap(),
            );
            assert_eq!(
                decode_inline_value(&encode_inline_value(&value).unwrap()).unwrap(),
                value
            );
        }
    }

    #[test]
    fn callable_captures_round_trip_and_failures_name_the_local_path() {
        let closure = Value::Closure(runmat_value::Closure {
            function_name: "worker".into(),
            bound_function: Some(7),
            captures: vec![Value::Num(2.0)],
        });
        assert_eq!(
            decode_inline_value(&encode_inline_value(&closure).unwrap()).unwrap(),
            closure
        );
        let mut tampered = encode_inline_value(&closure).unwrap();
        let ValuePayload::Inline(ref mut inline) = tampered else {
            unreachable!("runtime encoding is inline");
        };
        let InlineValue::Callable(callable) = inline.as_mut() else {
            unreachable!("closure encoding is callable");
        };
        callable.qualified_name = "other_worker".into();
        assert!(decode_inline_value(&tampered).is_err());

        let mut structure = StructValue::new();
        let service = super::super::RuntimeExecutionService::new();
        structure.insert(
            "bad",
            Value::Future(runmat_execution::FutureHandle {
                id: runmat_execution::FutureId::derive(&[b"future"]),
                scope_id: service.scope_id(),
                outputs: runmat_execution::OutputContract {
                    requested_outputs: 1,
                },
            }),
        );
        let error = encode_inline_value(&Value::Cell(
            CellArray::new(vec![Value::Struct(structure)], 1, 1).unwrap(),
        ))
        .unwrap_err();
        assert!(error.to_string().contains("$[0].bad"));
    }

    #[test]
    fn live_execution_handles_are_never_boundary_payloads() {
        let service = super::super::RuntimeExecutionService::new();
        let future = runmat_execution::FutureHandle {
            id: runmat_execution::FutureId::derive(&[b"future"]),
            scope_id: service.scope_id(),
            outputs: runmat_execution::OutputContract {
                requested_outputs: 1,
            },
        };
        assert!(encode_inline_value(&Value::Future(future)).is_err());
    }

    #[test]
    fn live_foreign_references_require_a_manifest_adapter() {
        let reference = ForeignRef {
            host_identity: "host-a".into(),
            handle: 17,
            generation: 2,
            type_identity: ForeignTypeIdentity {
                family: "java".into(),
                name: "java.lang.Object".into(),
                version: 1,
            },
            ownership: ForeignOwnership::Shared,
            affinity: ForeignAffinity::OriginProcess,
            lifetime: ForeignLifetime::Session,
        };
        let error = encode_inline_value(&Value::Foreign(reference)).unwrap_err();
        assert!(matches!(
            error,
            ValueCodecError::Unsupported { ref path, rule }
                if path == "$" && rule.contains("interop-manifest adapter")
        ));
    }

    #[test]
    fn registered_immutable_value_class_round_trips_nested_values() {
        let mut child = test_object("child");
        child
            .properties
            .insert("count".into(), Value::Int(IntValue::U64(u64::MAX)));

        let mut details = StructValue::new();
        details.insert(
            "samples",
            Value::Tensor(Tensor::from_f32(vec![1.25, -2.5], vec![2, 1]).unwrap()),
        );
        details.insert("child", Value::Object(child));

        let mut object = test_object("root");
        object.properties.insert(
            "nested".into(),
            Value::Cell(CellArray::new(vec![Value::Struct(details)], 1, 1).unwrap()),
        );
        let value = Value::Object(object);
        let payload = encode_inline_value(&value).unwrap();
        let ValuePayload::Inline(inline) = &payload else {
            panic!("registered value class must encode inline");
        };
        let InlineValue::ImmutableValueClass(registered) = inline.as_ref() else {
            panic!("expected registered immutable value class");
        };
        assert_eq!(registered.type_identity, TEST_TYPE);
        assert_eq!(registered.schema_version, 3);
        assert!(registered
            .fields
            .windows(2)
            .all(|fields| fields[0].name < fields[1].name));
        assert_eq!(decode_inline_value(&payload).unwrap(), value);
    }

    #[test]
    fn immutable_class_encoding_and_identity_ignore_property_insertion_order() {
        ensure_test_class_registered();
        let mut first = ObjectInstance::new(TEST_CLASS.into());
        first
            .properties
            .insert("id".into(), Value::String("stable".into()));
        first
            .properties
            .insert("count".into(), Value::Int(IntValue::U32(7)));

        let mut second = ObjectInstance::new(TEST_CLASS.into());
        second
            .properties
            .insert("count".into(), Value::Int(IntValue::U32(7)));
        second
            .properties
            .insert("id".into(), Value::String("stable".into()));

        let first = encode_inline_value(&Value::Object(first)).unwrap();
        let second = encode_inline_value(&Value::Object(second)).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.logical_digest().unwrap(),
            second.logical_digest().unwrap()
        );
    }

    #[test]
    fn immutable_class_schema_and_object_shape_fail_closed() {
        ensure_test_class_registered();

        let unknown = Value::Object(ObjectInstance::new("runmat.test.Unknown".into()));
        assert!(matches!(
            encode_inline_value(&unknown),
            Err(ValueCodecError::Unsupported { .. })
        ));

        let missing = Value::Object(ObjectInstance::new(TEST_CLASS.into()));
        assert!(encode_inline_value(&missing)
            .unwrap_err()
            .to_string()
            .contains("missing required field `id`"));

        let mut unknown_field = test_object("unknown-field");
        unknown_field
            .properties
            .insert("other".into(), Value::Bool(true));
        assert!(encode_inline_value(&Value::Object(unknown_field))
            .unwrap_err()
            .to_string()
            .contains("unknown field `other`"));

        let mut invalid = test_object("invalid");
        invalid.properties.insert("id".into(), Value::Num(4.0));
        assert!(encode_inline_value(&Value::Object(invalid))
            .unwrap_err()
            .to_string()
            .contains("`id` must be a string"));

        let mut dynamic = test_object("dynamic");
        dynamic.insert_dynamic_property(
            "runtime".into(),
            DynamicPropertyDef::new("runtime".into(), TEST_CLASS.into()),
        );
        assert!(encode_inline_value(&Value::Object(dynamic))
            .unwrap_err()
            .to_string()
            .contains("cannot have dynamic properties"));
    }

    #[test]
    fn immutable_class_decode_rejects_unknown_identity_schema_and_fields() {
        ensure_test_class_registered();
        let payload = |type_identity: &str, schema_version, fields| {
            ValuePayload::Inline(Box::new(InlineValue::ImmutableValueClass(RegisteredData {
                type_identity: type_identity.into(),
                schema_version,
                fields,
            })))
        };
        let id = || RegisteredField {
            name: "id".into(),
            value: ValuePayload::Inline(Box::new(InlineValue::String("decoded".into()))),
        };

        assert!(
            decode_inline_value(&payload("runmat.test.unknown", 1, vec![id()]))
                .unwrap_err()
                .to_string()
                .contains("unknown immutable value-class type identity")
        );
        assert!(decode_inline_value(&payload(TEST_TYPE, 4, vec![id()]))
            .unwrap_err()
            .to_string()
            .contains("schema version 4, expected 3"));
        assert!(decode_inline_value(&payload(
            TEST_TYPE,
            3,
            vec![RegisteredField {
                name: "other".into(),
                value: ValuePayload::Inline(Box::new(InlineValue::Logical(true))),
            }],
        ))
        .unwrap_err()
        .to_string()
        .contains("unknown field `other`"));
        assert!(decode_inline_value(&payload(TEST_TYPE, 3, Vec::new()))
            .unwrap_err()
            .to_string()
            .contains("missing required field `id`"));
        assert!(decode_inline_value(&payload(
            TEST_TYPE,
            3,
            vec![RegisteredField {
                name: "id".into(),
                value: ValuePayload::Inline(Box::new(InlineValue::F64Bits(1.0_f64.to_bits()))),
            }],
        ))
        .unwrap_err()
        .to_string()
        .contains("`id` must be a string"));

        let duplicate = payload(TEST_TYPE, 3, vec![id(), id()]);
        assert!(decode_inline_value(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("field names must be unique and sorted"));
    }

    #[test]
    fn immutable_value_class_arrays_remain_explicitly_unsupported() {
        let array =
            ObjectArray::row(TEST_CLASS, vec![Value::Object(test_object("array"))]).unwrap();
        let error = encode_inline_value(&Value::ObjectArray(array)).unwrap_err();
        assert!(matches!(
            error,
            ValueCodecError::Unsupported { rule, .. }
                if rule.contains("shape-preserving registered array codec")
        ));
    }

    #[test]
    fn immutable_class_registry_rejects_invalid_and_conflicting_schemas() {
        fn accept(_: &ObjectInstance) -> Result<(), String> {
            Ok(())
        }
        assert!(register_immutable_value_class(ImmutableValueClassSchema {
            class_name: "runmat.test.ZeroSchema",
            type_identity: "runmat.test.zero-schema",
            schema_version: 0,
            allowed_fields: &[],
            required_fields: &[],
            validate: accept,
        })
        .is_err());
        assert!(register_immutable_value_class(ImmutableValueClassSchema {
            class_name: "runmat.test.RequiredOutsideAllowed",
            type_identity: "runmat.test.required-outside-allowed",
            schema_version: 1,
            allowed_fields: &[],
            required_fields: &["id"],
            validate: accept,
        })
        .is_err());
        register_immutable_value_class(ImmutableValueClassSchema {
            class_name: "runmat.test.UniqueRegistryClass",
            type_identity: "runmat.test.unique-registry-class",
            schema_version: 1,
            allowed_fields: &[],
            required_fields: &[],
            validate: accept,
        })
        .unwrap();
        assert!(register_immutable_value_class(ImmutableValueClassSchema {
            class_name: "runmat.test.UniqueRegistryClass",
            type_identity: "runmat.test.other-type",
            schema_version: 1,
            allowed_fields: &[],
            required_fields: &[],
            validate: accept,
        })
        .is_err());
        assert!(register_immutable_value_class(ImmutableValueClassSchema {
            class_name: "runmat.test.OtherRegistryClass",
            type_identity: "runmat.test.unique-registry-class",
            schema_version: 1,
            allowed_fields: &[],
            required_fields: &[],
            validate: accept,
        })
        .is_err());
    }
}
