mod decode;
mod durable;
mod encode;
mod error;
mod graph;
mod immutable_class;
pub(crate) mod selected_json;
mod validation_workspace;
pub use validation_workspace::ValidationWorkspaceOwner;
mod object_output;
mod outputs;
mod owned_prepare;
mod path;
mod shape_copy;
mod text_copy;

pub(crate) use decode::decode_durable_value_cooperatively_in_context;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use decode::decode_inline_value_rooted;
pub use decode::{
    decode_durable_values_cooperatively, decode_inline_value, decode_inline_value_owned,
    decode_inline_value_owned_with_limits, decode_inline_values_cooperatively,
    decode_inline_values_cooperatively_in_context, CooperativeValueCodecError,
};
pub use durable::{
    decode_durable_value, encode_durable_value, validate_durable_payload,
    validate_durable_preparation,
};
pub use encode::{encode_inline_value, encode_inline_value_in_context};
pub use error::ValueCodecError;
pub use graph::{
    decode_payload_graph_in_context, decode_payload_graph_with_selected_classes_in_context,
    decode_payload_graph_with_source_issuer_in_context, deferred_invocation_requires_graph,
    encode_payload_graph_in_context, encode_payload_graph_with_issuer, prepare_callable_targets,
    prepare_deferred_invocation_graph, prepare_payload_graph, GraphFrameSourceRequirement,
    GraphSourceIssuer, GraphSourceUnitFacts, PreparedPayloadGraph,
};
pub use graph::{GraphCallableProof, GraphCaptureProof, GraphNestedZeroCopy};
pub use immutable_class::{
    register_immutable_value_class, register_immutable_value_class_with_scratch,
    ImmutableValueClassSchema, ImmutableValueClassScratch, ImmutableValueClassValidator,
};
pub use outputs::{
    decode_inline_outputs, encode_inline_outputs, encode_inline_sequence, validate_output_arity,
};
pub use owned_prepare::OwnedInlineTransportPlan;

#[cfg(test)]
mod tests {
    use crate::execution::RuntimeExecutionServices;
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, DynamicPropertyDef, ForeignAffinity, ForeignLifetime,
        ForeignOwnership, ForeignRef, ForeignResourceKey, ForeignTypeIdentity, IntValue,
        IntegerComplexStorage, IntegerStorage, MException, ObjectArray, ObjectInstance,
        SparseTensor, StringArray, Tensor, Value,
    };
    use std::sync::Once;

    use runmat_execution::value::{InlineValue, RegisteredData, RegisteredField, ValuePayload};

    const TEST_CLASS: &str = "runmat.test.PortableValue";
    const TEST_TYPE: &str = "runmat.test.portable-value";
    const TEST_ALLOWED_FIELDS: &[&str] = &["count", "id", "nested"];
    const TEST_REQUIRED_FIELDS: &[&str] = &["id"];

    use super::{
        decode_durable_value, decode_inline_value, encode_durable_value, encode_inline_value,
        register_immutable_value_class, validate_durable_preparation, ImmutableValueClassSchema,
        ValueCodecError,
    };

    fn validate_test_object(object: &ObjectInstance) -> Result<(), String> {
        match object.properties().get("id") {
            Some(Value::String(runmat_value::StringElement::Text(id))) if !id.is_empty() => Ok(()),
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
        let mut object = ObjectInstance::new(TEST_CLASS);
        object.replace_property_unbounded("id".into(), Value::String(id.into()));
        object
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn dense_low_rank_values_round_trip_without_shape_or_class_conversion() {
        for dimensions in [Vec::new(), vec![2]] {
            let len = dimensions.iter().product::<usize>();
            let real = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX; len]), dimensions.clone())
                    .unwrap(),
            );
            let complex = Value::ComplexTensor(
                ComplexTensor::from_f32(vec![(0.25, -0.5); len], dimensions).unwrap(),
            );
            for source in [real, complex] {
                let encoded = encode_inline_value(&source).unwrap();
                let decoded = decode_inline_value(&encoded).unwrap();
                assert_eq!(decoded, source);
            }
        }
    }

    #[test]
    fn durable_nested_values_keep_character_pages_and_reject_context_bound_callables() {
        let chars = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 2, 1])
                .expect("character page"),
        );
        let value = Value::Cell(CellArray::new(vec![chars], 1, 1).unwrap());
        assert_eq!(
            decode_durable_value(&encode_durable_value(&value).unwrap()).unwrap(),
            value
        );

        let context_bound = Value::Cell(
            CellArray::new(vec![Value::FunctionHandle("local_function".into())], 1, 1).unwrap(),
        );
        let error = encode_durable_value(&context_bound).unwrap_err();
        assert!(matches!(error, ValueCodecError::Unsupported { ref path, .. } if path == "$[0]"));
    }

    #[test]
    fn character_wire_preserves_utf16_geometry_and_unpaired_units() {
        let source = Value::CharArray(
            CharArray::from_code_units_with_shape(
                vec![
                    runmat_value::Utf16CodeUnit::from_raw(0xD83D),
                    runmat_value::Utf16CodeUnit::from_raw(0xDE42),
                    runmat_value::Utf16CodeUnit::from_raw(0xD800),
                ],
                vec![1, 3],
            )
            .unwrap(),
        );
        let encoded = encode_inline_value(&source).unwrap();
        let ValuePayload::Inline(inline) = &encoded else {
            panic!("expected inline UTF-16 character payload");
        };
        let InlineValue::Char {
            encoding: runmat_execution::value::CharacterEncoding::Utf16CodeUnitsV1,
            shape,
            code_units,
        } = inline.as_ref()
        else {
            panic!("expected inline UTF-16 character payload");
        };
        assert_eq!(shape, &[1, 3]);
        assert_eq!(code_units, &[0xD83D, 0xDE42, 0xD800]);
        assert_eq!(decode_inline_value(&encoded).unwrap(), source);
    }

    #[test]
    fn resident_preparation_placeholder_does_not_certify_downloaded_payload_limits() {
        use runmat_accelerate_api::test_support::TestAllocationOwner;
        use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};

        const ELEMENTS: usize = (1024 * 1024 / 8) + 1;
        let owner = TestAllocationOwner::new();
        let resident = owner
            .allocate_zeroed(
                vec![1, ELEMENTS],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .expect("real resident backing");
        validate_durable_preparation(&Value::GpuTensor(resident.handle))
            .expect("placeholder checks grammar, not downloaded byte size");
        let host = Value::Tensor(
            Tensor::new(vec![0.0; ELEMENTS], vec![1, ELEMENTS]).expect("matching host payload"),
        );
        assert!(matches!(
            encode_durable_value(&host),
            Err(ValueCodecError::Invalid { .. })
        ));
    }

    #[test]
    fn preference_aggregates_are_durable_only_and_keep_plain_object_bits() {
        let mut object = runmat_value::ObjectInstance::new("PreferencePayload");
        object.replace_property_unbounded("revision".into(), Value::Int(IntValue::U64(u64::MAX)));
        object.replace_property_unbounded(
            "nan".into(),
            Value::Num(f64::from_bits(0x7ff8_0000_0000_0042)),
        );
        let value = Value::Object(object.clone());
        assert!(encode_inline_value(&value).is_err());
        let payload = encode_durable_value(&value).expect("plain object is durable");
        assert!(
            decode_inline_value(&payload).is_err(),
            "V2 decoder rejects durable tag"
        );
        assert!(payload
            .validate(runmat_execution::value::ValueLimits::default())
            .is_err());
        let Value::Object(decoded) = decode_durable_value(&payload).unwrap() else {
            panic!("plain object")
        };
        assert_eq!(decoded.class_name(), object.class_name());
        assert_eq!(
            decoded.properties()["revision"],
            Value::Int(IntValue::U64(u64::MAX))
        );
        let Value::Num(nan) = decoded.properties()["nan"] else {
            panic!("NaN")
        };
        assert_eq!(nan.to_bits(), 0x7ff8_0000_0000_0042);

        let array = Value::ObjectArray(
            runmat_value::ObjectArray::from_objects(
                "PreferencePayload",
                vec![object],
                vec![1, 1, 1],
            )
            .unwrap(),
        );
        let payload = encode_durable_value(&array).unwrap();
        let decoded = decode_durable_value(&payload).unwrap();
        let Value::ObjectArray(decoded_array) = &decoded else {
            panic!("object array")
        };
        assert_eq!(
            decoded_array.class_name(),
            &runmat_types::StaticClassIdentity::new("PreferencePayload").owned()
        );
        assert_eq!(decoded_array.shape(), &[1, 1, 1]);
        assert_eq!(decoded_array.data().len(), 1);
        let Value::Object(decoded_object) = &decoded_array.data()[0] else {
            panic!("plain object element")
        };
        assert_eq!(decoded_object.class_name(), decoded_array.class_name());
        assert_eq!(decoded_object.properties().len(), 2);
        assert_eq!(
            decoded_object.properties()["revision"],
            Value::Int(IntValue::U64(u64::MAX))
        );
        let Value::Num(nan) = decoded_object.properties()["nan"] else {
            panic!("NaN")
        };
        assert_eq!(nan.to_bits(), 0x7ff8_0000_0000_0042);
        assert_eq!(encode_durable_value(&decoded).unwrap(), payload);
        assert!(decode_inline_value(&payload).is_err());
    }

    #[test]
    fn durable_struct_array_preserves_observable_field_order() {
        let array = Value::StructArray(
            runmat_value::StructArray::empty(vec!["zeta".into(), "alpha".into()], vec![0, 2, 1])
                .unwrap(),
        );
        let payload = encode_durable_value(&array).unwrap();
        assert!(decode_inline_value(&payload).is_err());
        let Value::StructArray(decoded) = decode_durable_value(&payload).unwrap() else {
            panic!("structure array")
        };
        assert_eq!(
            decoded.field_names().collect::<Vec<_>>(),
            vec!["zeta", "alpha"]
        );
        assert_eq!(decoded.shape(), &[0, 2, 1]);
    }

    #[test]
    fn durable_nested_scalar_structure_keeps_insertion_order_without_changing_v2() {
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("zeta", Value::Int(IntValue::U64(u64::MAX)));
        structure.insert("alpha", Value::Num(1.0));
        let nested =
            Value::Cell(CellArray::new(vec![Value::Struct(structure.finish())], 1, 1).unwrap());
        let durable = encode_durable_value(&nested).unwrap();
        assert!(decode_inline_value(&durable).is_err());
        let Value::Cell(decoded) = decode_durable_value(&durable).unwrap() else {
            panic!("cell")
        };
        let Value::Struct(ordered) = &decoded.data()[0] else {
            panic!("structure")
        };
        assert_eq!(
            ordered.field_names().collect::<Vec<_>>(),
            vec!["zeta", "alpha"]
        );
        let v2 = encode_inline_value(&nested).unwrap();
        let Value::Cell(v2_decoded) = decode_inline_value(&v2).unwrap() else {
            panic!("V2 cell")
        };
        let Value::Struct(v2_struct) = &v2_decoded.data()[0] else {
            panic!("V2 structure")
        };
        assert_eq!(
            v2_struct.field_names().collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
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
        let Value::Num(number) = decoded.data()[0] else {
            panic!("expected decoded number");
        };
        assert_eq!(number.to_bits(), 0x7ff8_0000_0000_0042);
        assert_eq!(decoded.data()[1], Value::Int(IntValue::U64(u64::MAX)));
        assert_eq!(decoded.shape, vec![1, 2]);
    }

    #[test]
    fn transient_output_sequences_are_rejected_at_every_portable_value_depth() {
        let direct = encode_inline_value(&Value::output_list(vec![Value::Num(1.0)])).unwrap_err();
        assert!(matches!(
            direct,
            ValueCodecError::TransientSequenceNotPortable { ref path } if path == "$"
        ));

        let nested = Value::Cell(
            CellArray::new(vec![Value::output_list(vec![Value::Num(2.0)])], 1, 1).unwrap(),
        );
        let error = encode_inline_value(&nested).unwrap_err();
        assert!(matches!(
            error,
            ValueCodecError::TransientSequenceNotPortable { ref path } if path == "$[0]"
        ));

        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let text = "x = 1;";
        let source_map = runmat_execution::ExecutableSourceMapV2::new(vec![
            runmat_execution::ExecutableSourceRecordV2 {
                source_id: runmat_types::ProgramSourceId(0),
                owner_identity: "transient-codec-test".into(),
                relative_path: "transient.m".into(),
                content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
                display_name: "transient.m".into(),
                text: text.into(),
            },
        ])
        .unwrap();
        for value in [Value::output_list(vec![Value::Num(1.0)]), nested] {
            let error = super::encode_payload_graph_in_context(
                &[value],
                &runtime,
                &source_map,
                &Default::default(),
                &Default::default(),
                runmat_execution::value::ValueLimits::default(),
            )
            .unwrap_err();
            assert!(matches!(
                error,
                ValueCodecError::TransientSequenceNotPortable { .. }
            ));
        }
    }

    #[test]
    fn stable_immutable_runtime_forms_round_trip_exactly() {
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert(
            "tensor",
            Value::Tensor(
                Tensor::new(vec![f64::from_bits(0x8000_0000_0000_0000), 2.5], vec![2, 1]).unwrap(),
            ),
        );
        structure.insert(
            "strings",
            Value::StringArray(
                StringArray::from_elements_unbounded(
                    vec![
                        "α".into(),
                        "<missing>".into(),
                        "<MISSING>".into(),
                        runmat_value::StringElement::Missing,
                        "".into(),
                    ],
                    vec![1, 5],
                )
                .unwrap(),
            ),
        );
        structure.insert(
            "missing_scalar",
            Value::String(runmat_value::StringElement::Missing),
        );
        structure.insert(
            "chars",
            Value::CharArray(CharArray::from_unicode_scalars(vec!['a', 'β'], 2, 1).unwrap()),
        );
        structure.insert(
            "error",
            Value::MException({
                MException::from_owned_fields(
                    "RunMat:test".into(),
                    "failure".into(),
                    MException::stack_from_unbounded(vec!["main:1".into()]),
                    None,
                )
            }),
        );
        let value = Value::Struct(structure.finish());
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
        assert_eq!(
            decoded_values[1],
            runmat_value::ComplexElement(f32::MAX, f32::MIN_POSITIVE)
        );
        assert_eq!(decoded_single.shape(), &[1, 2]);

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
    fn sparse_complex_and_logical_values_round_trip_without_losing_storage_semantics() {
        let complex = Value::SparseTensor(
            SparseTensor::new_complex(
                3,
                2,
                vec![0, 2, 3],
                vec![0, 2, 1],
                vec![
                    (f64::from_bits(0x8000_0000_0000_0000), 2.5),
                    (3.0, f64::from_bits(0x7ff8_0000_0000_0042)),
                    (-4.0, 5.0),
                ],
            )
            .unwrap(),
        );
        let Value::SparseTensor(decoded) =
            decode_inline_value(&encode_inline_value(&complex).unwrap()).unwrap()
        else {
            panic!("expected sparse complex value");
        };
        assert!(decoded.is_complex());
        assert_eq!(decoded.col_ptrs, vec![0, 2, 3]);
        assert_eq!(decoded.row_indices, vec![0, 2, 1]);
        let values = decoded.as_complex_f64_slice().unwrap();
        assert_eq!(values[0].0.to_bits(), 0x8000_0000_0000_0000);
        assert_eq!(values[0].1, 2.5);
        assert_eq!(values[1].1.to_bits(), 0x7ff8_0000_0000_0042);
        assert_eq!(values[2], runmat_value::ComplexElement(-4.0, 5.0));

        let complex_single = Value::SparseTensor(
            SparseTensor::new_complex_f32(
                2,
                1,
                vec![0, 2],
                vec![0, 1],
                vec![(1.25, -2.5), (3.75, 4.5)],
            )
            .unwrap(),
        );
        let Value::SparseTensor(decoded) =
            decode_inline_value(&encode_inline_value(&complex_single).unwrap()).unwrap()
        else {
            panic!("expected sparse complex single value");
        };
        assert_eq!(
            decoded.numeric_dtype(),
            Some(runmat_value::NumericDType::F32)
        );
        assert_eq!(
            decoded.as_complex_f32_slice(),
            Some(
                [
                    runmat_value::ComplexElement(1.25, -2.5),
                    runmat_value::ComplexElement(3.75, 4.5),
                ]
                .as_slice()
            )
        );

        let logical = Value::SparseTensor(
            SparseTensor::new_logical(3, 2, vec![0, 2, 3], vec![0, 2, 1]).unwrap(),
        );
        assert_eq!(
            decode_inline_value(&encode_inline_value(&logical).unwrap()).unwrap(),
            logical
        );
    }

    #[test]
    fn callable_captures_round_trip_and_failures_name_the_local_path() {
        let origin = |function: usize| runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256("portable callable fixture"),
            function: runmat_types::ProgramFunctionId(function as u32),
        };
        let _catalog =
            crate::user_functions::install_source_function_catalog(Some(std::sync::Arc::new(
                [7usize, 9, 11]
                    .into_iter()
                    .map(|function| crate::user_functions::SourceFunctionInfo {
                        source_id: runmat_types::SourceId(function),
                        name: format!("function_{function}"),
                        function,
                        role: runmat_types::FunctionSourceRole::Unknown,
                        binding_origin: Some(origin(function)),
                        source_file: None,
                        definition_owners: None,
                    })
                    .collect(),
            )))
            .expect("call-port scope");
        let closure = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "worker".into(),
                binding: Some(runmat_value::BoundFunctionBinding::portable(7, origin(7))),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: vec![Value::Num(2.0)].into(),
            },
        ));
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

        let empty_anonymous = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "display_is_not_authority".into(),
                binding: Some(runmat_value::BoundFunctionBinding::portable(9, origin(9))),
                source_role: runmat_types::FunctionSourceRole::Anonymous,
                source_file: Some("/origin/worker.m".into()),
                captures: Vec::new().into(),
            },
        ));
        assert_eq!(
            decode_inline_value(&encode_inline_value(&empty_anonymous).unwrap()).unwrap(),
            empty_anonymous,
            "zero-capture closures must not become plain function handles"
        );
        let mut tampered_role = encode_inline_value(&empty_anonymous).unwrap();
        let ValuePayload::Inline(ref mut inline) = tampered_role else {
            unreachable!("runtime encoding is inline");
        };
        let InlineValue::Callable(callable) = inline.as_mut() else {
            unreachable!("closure encoding is callable");
        };
        callable.source_role = runmat_types::FunctionSourceRole::Local;
        assert!(decode_inline_value(&tampered_role).is_err());
        let mut tampered_file = encode_inline_value(&empty_anonymous).unwrap();
        let ValuePayload::Inline(ref mut inline) = tampered_file else {
            unreachable!("runtime encoding is inline");
        };
        let InlineValue::Callable(callable) = inline.as_mut() else {
            unreachable!("closure encoding is callable");
        };
        callable.source_file = Some("/destination/unrelated.m".into());
        assert!(decode_inline_value(&tampered_file).is_err());
        let mut tampered_form = encode_inline_value(&empty_anonymous).unwrap();
        let ValuePayload::Inline(ref mut inline) = tampered_form else {
            unreachable!("runtime encoding is inline");
        };
        let InlineValue::Callable(callable) = inline.as_mut() else {
            unreachable!("closure encoding is callable");
        };
        callable.form = runmat_execution::value::CallableForm::Handle;
        assert!(decode_inline_value(&tampered_form).is_err());
        let local_handle = Value::BoundFunctionHandle(
            runmat_value::BoundFunctionHandle::from_data(runmat_value::BoundFunctionHandleData {
                name: "local_display".into(),
                function: 11,
                authority: runmat_value::BoundFunctionAuthority::portable(origin(11)),
                source_role: runmat_types::FunctionSourceRole::Local,
                source_file: Some("/origin/local.m".into()),
            }),
        );
        assert_eq!(
            decode_inline_value(&encode_inline_value(&local_handle).unwrap()).unwrap(),
            local_handle,
            "bound-handle role must not be re-derived from the destination catalog"
        );
        let encoded = encode_inline_value(&local_handle).unwrap();
        let first = decode_inline_value(&encoded).unwrap();
        let second = decode_inline_value(&encoded).unwrap();
        assert_ne!(
            first.callable_instance_key(),
            second.callable_instance_key(),
            "independent inline reconstructions mint independent local instances"
        );

        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        let service = super::super::RuntimeExecutionService::new();
        structure.insert(
            "bad",
            Value::Future(
                runmat_execution::FutureHandle {
                    id: runmat_execution::FutureId::derive(&[b"future"]),
                    scope_id: service.scope_id(),
                    outputs: runmat_execution::OutputContract {
                        requested_outputs: 1,
                        owned_work: None,
                    },
                }
                .into(),
            ),
        );
        let error = encode_inline_value(&Value::Cell(
            CellArray::new(vec![Value::Struct(structure.finish())], 1, 1).unwrap(),
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
                owned_work: None,
            },
        };
        assert!(encode_inline_value(&Value::Future(future.into())).is_err());
    }

    #[test]
    fn distributed_and_composite_handles_round_trip_without_materializing_payloads() {
        use runmat_execution::{
            CompositeHandle, CompositeId, DistributedObjectId, DistributedValueHandle,
            ExecutionScopeId, GangHandle, GangId, PoolHandle, PoolId,
        };
        use runmat_types::{
            DistributedValueId, DistributionScheme, LabCount, NumericClass, NumericDomain,
            NumericFact, ParallelRegionId, ProgramFunctionId, RegionId, ValueFact, ValueKindFact,
        };

        let scope_id = ExecutionScopeId::derive(&[b"codec-distributed"]);
        let function = ProgramFunctionId(3);
        let owner_region = ParallelRegionId(RegionId {
            function,
            ordinal: 2,
        });
        let pool = PoolHandle {
            id: PoolId::derive(&[b"pool"]),
            scope_id,
            generation: 1,
        };
        let fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let distributed = DistributedValueHandle {
            id: DistributedObjectId::derive(&[b"value"]),
            contract: DistributedValueId {
                function,
                ordinal: 1,
            },
            owner: runmat_types::DistributedOwner::Region(owner_region),
            scope_id,
            generation: 1,
            pool: pool.clone(),
            partition_count: LabCount(2),
            value: fact.clone(),
            global_shape: vec![8, 2],
            scheme: DistributionScheme::Block { dimension: 1 },
            materializable: true,
        };
        let composite = CompositeHandle {
            id: CompositeId::derive(&[b"composite"]),
            owner_region,
            scope_id,
            generation: 1,
            gang: GangHandle {
                id: GangId::derive(&[b"gang"]),
                scope_id,
                generation: 1,
                pool,
                labs: LabCount(2),
            },
            value: fact,
        };
        for value in [
            Value::Distributed(Box::new(distributed).into()),
            Value::Composite(Box::new(composite).into()),
        ] {
            let payload = encode_inline_value(&value).unwrap();
            assert_eq!(decode_inline_value(&payload).unwrap(), value);
        }
    }

    #[test]
    fn immutable_codistributor_value_classes_round_trip_with_exact_properties() {
        let value = crate::parallel::codistributor::one_dimensional(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::Int(runmat_value::IntValue::U64(2)),
                Value::Tensor(
                    runmat_value::Tensor::new_integer(
                        runmat_value::IntegerStorage::U64(vec![1, 2, 3]),
                        vec![1, 3],
                    )
                    .unwrap(),
                ),
                Value::Tensor(
                    runmat_value::Tensor::new_integer(
                        runmat_value::IntegerStorage::U64(vec![4, 6]),
                        vec![1, 2],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
        let payload = encode_inline_value(&value).unwrap();
        assert_eq!(decode_inline_value(&payload).unwrap(), value);
    }

    #[test]
    fn unregistered_value_classes_do_not_cross_execution_boundaries() {
        let object = Value::Object(runmat_value::ObjectInstance::new(
            "project.LocalClass".to_owned(),
        ));
        let error = encode_inline_value(&object).unwrap_err();
        assert!(matches!(
            error,
            ValueCodecError::Unsupported { ref path, rule }
                if path == "$" && rule.contains("registered codec")
        ));
    }

    #[test]
    fn source_less_caught_exception_round_trips_typed_runtime_metadata() {
        let original = crate::build_runtime_error("failure")
            .with_identifier("RunMat:portable:Failure")
            .with_call_frames(vec![crate::CallFrame {
                function: "origin".into(),
                source_id: None,
                span: Some((2, 5)),
                source_owner: None,
                diagnostic_source: None,
            }])
            .with_phase("execution")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        let caught = Value::MException(crate::runtime_error::exception_from_error(original));
        let payload = encode_inline_value(&caught).expect("source-less exception is portable");
        let decoded = decode_inline_value(&payload).expect("portable exception decode");
        let Value::MException(decoded) = decoded else {
            panic!("expected exception");
        };
        let original = decoded.original().expect("typed runtime provenance");
        assert_eq!(original.context.call_frames[0].source_id, None);
        assert_eq!(original.context.phase.as_deref(), Some("execution"));
        assert_eq!(original.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            decoded
                .stack
                .iter()
                .map(|frame| frame.as_str())
                .collect::<Vec<_>>(),
            ["origin"]
        );

        let retained = decoded.clone();
        let reraised = crate::runtime_error::error_from_exception(decoded);
        let recaught = Value::MException(crate::runtime_error::exception_from_error(reraised));
        let repeated_payload = encode_inline_value(&recaught)
            .expect("shared source-less original remains portable after rethrow and recatch");
        let repeated = decode_inline_value(&repeated_payload).expect("repeated exception decode");
        let Value::MException(repeated) = repeated else {
            panic!("expected repeated exception");
        };
        let repeated_original = repeated.original().expect("typed repeated provenance");
        assert_eq!(repeated_original.context.call_frames[0].source_id, None);
        assert_eq!(
            repeated_original.context.phase.as_deref(),
            Some("execution")
        );
        assert_eq!(
            repeated_original.gpu_gather_retry,
            crate::GpuGatherRetry::Never
        );
        assert!(retained.original().is_some());
    }

    #[test]
    fn nested_caught_exception_projects_selected_source_without_importing_worker_id() {
        let base = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _unrelated = crate::source_context::lease_source_context_in_context(
            &base,
            "unrelated.m",
            None,
            "unrelated = 1;",
        )
        .unwrap();
        let text = "error('caught')";
        let map = runmat_execution::ExecutableSourceMapV2::new(vec![
            runmat_execution::ExecutableSourceRecordV2 {
                source_id: runmat_types::ProgramSourceId(0),
                owner_identity: "fixture".into(),
                relative_path: "remote_spmd.m".into(),
                display_name: "remote_spmd.m".into(),
                text: text.into(),
                content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
            },
        ])
        .unwrap();
        let runtime = crate::source_context::bind_portable_source_map_v2(base, &map).unwrap();
        let worker_id = runtime
            .program_source_id(runmat_types::ProgramSourceId(0))
            .unwrap();
        let original = crate::build_runtime_error("caught")
            .with_identifier("RunMat:test:Caught")
            .with_call_frames(vec![crate::CallFrame {
                function: "remote_spmd".into(),
                source_id: Some(worker_id.0),
                span: Some((0, 5)),
                source_owner: None,
                diagnostic_source: None,
            }])
            .build();
        let nested = Value::Cell(
            CellArray::new(
                vec![Value::MException(
                    crate::runtime_error::exception_from_error(original),
                )],
                1,
                1,
            )
            .unwrap(),
        );
        assert!(encode_inline_value(&nested).is_err());
        let payload = super::encode_inline_value_in_context(&nested, &runtime).unwrap();
        let ValuePayload::Inline(root) = &payload else {
            panic!("inline cell");
        };
        let InlineValue::Cell { values, .. } = root.as_ref() else {
            panic!("cell");
        };
        let ValuePayload::Inline(member) = &values[0] else {
            panic!("inline exception");
        };
        let InlineValue::Exception(exception) = member.as_ref() else {
            panic!("exception");
        };
        assert_eq!(
            exception.runtime.as_ref().unwrap().call_frames[0].program_source_id(),
            Some(runmat_types::ProgramSourceId(0))
        );
        assert_ne!(u64::try_from(worker_id.0).unwrap(), 0);
        let Value::Cell(decoded) = decode_inline_value(&payload).unwrap() else {
            panic!("cell");
        };
        let Value::MException(decoded) = &decoded.data()[0] else {
            panic!("exception");
        };
        assert_eq!(
            decoded.original().unwrap().context.call_frames[0].source_id,
            None
        );
        assert_eq!(
            decoded.original().unwrap().context.call_stack,
            ["remote_spmd @ remote_spmd.m:1:1"]
        );
        assert_eq!(
            decoded
                .stack
                .iter()
                .map(|frame| frame.as_str())
                .collect::<Vec<_>>(),
            ["remote_spmd"]
        );

        let selected = runtime
            .call_frame_source_owner_for_program_source(runmat_types::ProgramSourceId(0))
            .unwrap();
        for (owner, accepted) in [
            (selected, true),
            (crate::CallFrameSourceOwner::issue(), false),
        ] {
            let error = crate::build_runtime_error("caught")
                .with_call_frames(vec![crate::CallFrame {
                    function: "remote_spmd".into(),
                    source_id: Some(worker_id.0),
                    span: Some((0, 5)),
                    source_owner: Some(owner),
                    diagnostic_source: None,
                }])
                .build();
            let value = Value::MException(crate::runtime_error::exception_from_error(error));
            assert_eq!(
                super::encode_inline_value_in_context(&value, &runtime).is_ok(),
                accepted,
                "a frame token must match the source table that supplies its ordinal"
            );
        }
    }

    #[test]
    fn caught_exception_with_live_cause_is_not_silently_serialized() {
        let original = crate::build_runtime_error("failure")
            .with_source(std::io::Error::other("opaque cause"))
            .build();
        let caught = Value::MException(crate::runtime_error::exception_from_error(original));
        for result in [encode_inline_value(&caught), encode_durable_value(&caught)] {
            assert!(
                matches!(result, Err(ValueCodecError::Unsupported { rule, .. }) if rule.contains("live opaque cause"))
            );
        }
    }

    #[test]
    fn live_foreign_references_require_a_manifest_adapter() {
        let reference = ForeignRef::detached(
            ForeignResourceKey {
                host_identity: "host-a".into(),
                handle: 17,
                generation: 2,
            },
            ForeignTypeIdentity {
                family: "java".into(),
                name: "java.lang.Object".into(),
                version: 1,
            },
            ForeignOwnership::Shared,
            ForeignAffinity::OriginProcess,
            ForeignLifetime::Session,
        );
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
        child.replace_property_unbounded("count".into(), Value::Int(IntValue::U64(u64::MAX)));

        let mut details = runmat_value::UnboundedStructValueBuilder::new();
        details.insert(
            "samples",
            Value::Tensor(Tensor::from_f32(vec![1.25, -2.5], vec![2, 1]).unwrap()),
        );
        details.insert("child", Value::Object(child));

        let mut object = test_object("root");
        object.replace_property_unbounded(
            "nested".into(),
            Value::Cell(CellArray::new(vec![Value::Struct(details.finish())], 1, 1).unwrap()),
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
        let mut first = ObjectInstance::new(TEST_CLASS);
        first.replace_property_unbounded("id".into(), Value::String("stable".into()));
        first.replace_property_unbounded("count".into(), Value::Int(IntValue::U32(7)));

        let mut second = ObjectInstance::new(TEST_CLASS);
        second.replace_property_unbounded("count".into(), Value::Int(IntValue::U32(7)));
        second.replace_property_unbounded("id".into(), Value::String("stable".into()));

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

        let unknown = Value::Object(ObjectInstance::new("runmat.test.Unknown"));
        assert!(matches!(
            encode_inline_value(&unknown),
            Err(ValueCodecError::Unsupported { .. })
        ));

        let missing = Value::Object(ObjectInstance::new(TEST_CLASS));
        assert!(encode_inline_value(&missing)
            .unwrap_err()
            .to_string()
            .contains("missing required field `id`"));

        let mut unknown_field = test_object("unknown-field");
        unknown_field.replace_property_unbounded("other".into(), Value::Bool(true));
        assert!(encode_inline_value(&Value::Object(unknown_field))
            .unwrap_err()
            .to_string()
            .contains("unknown field `other`"));

        let mut invalid = test_object("invalid");
        invalid.replace_property_unbounded("id".into(), Value::Num(4.0));
        assert!(encode_inline_value(&Value::Object(invalid))
            .unwrap_err()
            .to_string()
            .contains("`id` must be a string"));

        let mut dynamic = test_object("dynamic")
            .into_unbounded_builder()
            .expect("ordinary fixture");
        dynamic.insert_dynamic(
            "runtime".into(),
            DynamicPropertyDef::new("runtime".into(), TEST_CLASS),
        );
        assert!(encode_inline_value(&Value::Object(dynamic.finish()))
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

#[cfg(test)]
#[path = "numeric_factor_tests.rs"]
mod numeric_factor_tests;
