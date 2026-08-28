mod decode;
mod encode;
mod error;

pub use decode::decode_inline_value;
pub use encode::encode_inline_value;
pub use error::ValueCodecError;

#[cfg(test)]
mod tests {
    use crate::execution::RuntimeExecutionServices;
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, ForeignAffinity, ForeignLifetime, ForeignOwnership,
        ForeignRef, ForeignResourceKey, ForeignTypeIdentity, IntValue, IntegerComplexStorage,
        IntegerStorage, MException, SparseTensor, StringArray, StructValue, Tensor, Value,
    };

    use runmat_execution::value::{InlineValue, ValuePayload};

    use super::{decode_inline_value, encode_inline_value, ValueCodecError};

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
        assert_eq!(
            decoded_values[1],
            runmat_value::ComplexElement(f32::MAX, f32::MIN_POSITIVE)
        );
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
            Value::Distributed(Box::new(distributed)),
            Value::Composite(Box::new(composite)),
        ] {
            let payload = encode_inline_value(&value).unwrap();
            assert_eq!(decode_inline_value(&payload).unwrap(), value);
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
}
