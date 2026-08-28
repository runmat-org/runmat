#[cfg(test)]
mod int_value_tests {
    use crate::{IntValue, Value};

    #[test]
    fn uint64_to_f64_does_not_clamp_through_int64() {
        let value = IntValue::U64(u64::MAX);
        assert_eq!(value.to_f64(), u64::MAX as f64);
        assert!(value.to_f64() > i64::MAX as f64);
    }

    #[test]
    fn decimal_string_preserves_full_signed_and_unsigned_range() {
        assert_eq!(
            IntValue::I64(i64::MIN).decimal_string(),
            "-9223372036854775808"
        );
        assert_eq!(
            IntValue::U64(u64::MAX).decimal_string(),
            "18446744073709551615"
        );
        assert_eq!(
            Value::Int(IntValue::U64(u64::MAX)).to_string(),
            "18446744073709551615"
        );
        assert_eq!(
            String::try_from(&Value::Int(IntValue::U64(u64::MAX))).expect("string conversion"),
            "18446744073709551615"
        );
    }

    #[test]
    fn checked_integer_conversions_do_not_saturate_or_change_sign() {
        assert_eq!(IntValue::I64(i64::MIN).try_to_i64(), Some(i64::MIN));
        assert_eq!(IntValue::U64(i64::MAX as u64).try_to_i64(), Some(i64::MAX));
        assert_eq!(IntValue::U64(u64::MAX).try_to_i64(), None);
        assert_eq!(IntValue::I32(i32::MIN).try_to_i32(), Some(i32::MIN));
        assert_eq!(IntValue::U64(i32::MAX as u64).try_to_i32(), Some(i32::MAX));
        assert_eq!(IntValue::U64(u64::MAX).try_to_i32(), None);
        assert_eq!(IntValue::I64(-1).try_to_u64(), None);
        assert_eq!(IntValue::U64(u64::MAX).try_to_u64(), Some(u64::MAX));
        assert_eq!(
            IntValue::U64(u64::MAX).try_to_usize(),
            usize::try_from(u64::MAX).ok()
        );
        assert_eq!(
            IntValue::I64(isize::MIN as i64).try_to_isize(),
            Some(isize::MIN)
        );
        assert_eq!(IntValue::U64(u64::MAX).try_to_isize(), None);
    }

    #[test]
    fn is_zero_checks_integer_storage_exactly() {
        let zeroes = [
            IntValue::I8(0),
            IntValue::I16(0),
            IntValue::I32(0),
            IntValue::I64(0),
            IntValue::U8(0),
            IntValue::U16(0),
            IntValue::U32(0),
            IntValue::U64(0),
        ];
        for value in zeroes {
            assert!(value.is_zero(), "{value:?} should be zero");
        }

        let nonzeroes = [
            IntValue::I8(-1),
            IntValue::I16(1),
            IntValue::I32(-1),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ];
        for value in nonzeroes {
            assert!(!value.is_zero(), "{value:?} should be nonzero");
        }
    }

    #[test]
    fn bool_conversion_uses_exact_integer_zero_test() {
        assert!(!bool::try_from(&Value::Int(IntValue::U64(0))).expect("zero bool"));
        assert!(bool::try_from(&Value::Int(IntValue::U64(u64::MAX))).expect("max bool"));
    }
}

#[cfg(test)]
mod integer_storage_tests {
    use crate::{
        ComplexStorage, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
        NumericDType, NumericScalar, NumericStorage, NumericStorageView, NumericStorageViewMut,
        Tensor,
    };

    #[test]
    fn uint64_tensor_keeps_exact_backing_values() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2])
            .expect("integer tensor");

        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![0, u64::MAX]))
        );
        assert_eq!(tensor.materialize_f64()[1], u64::MAX as f64);
    }

    #[test]
    fn integer_tensor_reports_its_exact_matlab_dtype() {
        let cases = [
            (IntegerStorage::I8(vec![0]), NumericDType::I8),
            (IntegerStorage::I16(vec![0]), NumericDType::I16),
            (IntegerStorage::I32(vec![0]), NumericDType::I32),
            (IntegerStorage::I64(vec![0]), NumericDType::I64),
            (IntegerStorage::U8(vec![0]), NumericDType::U8),
            (IntegerStorage::U16(vec![0]), NumericDType::U16),
            (IntegerStorage::U32(vec![0]), NumericDType::U32),
            (IntegerStorage::U64(vec![0]), NumericDType::U64),
        ];

        for (storage, dtype) in cases {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
            assert_eq!(tensor.numeric_dtype(), dtype);
            assert_eq!(
                tensor.numeric_dtype().class_name(),
                tensor.integer_storage().unwrap().class_name()
            );
        }
    }

    #[test]
    fn single_constructor_preserves_f32_values_and_dtype_across_storage_migration() {
        let source = vec![f32::MIN_POSITIVE, 1.0 / 10.0, f32::MAX];
        let expected: Vec<f64> = source.iter().map(|&value| f64::from(value)).collect();
        let tensor = Tensor::from_f32(source.clone(), vec![1, 3]).expect("single tensor");

        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.materialize_f64(), expected);
        assert!(tensor.integer_storage().is_none());
        assert_eq!(
            tensor.into_numeric_storage(),
            Ok(NumericStorage::F32(source))
        );
    }

    #[test]
    fn numeric_storage_derives_dtype_length_and_bytes_for_every_native_class() {
        let cases = [
            (NumericStorage::F64(vec![1.0, 2.0]), NumericDType::F64),
            (NumericStorage::F32(vec![1.0, 2.0]), NumericDType::F32),
            (NumericStorage::I8(vec![1, 2]), NumericDType::I8),
            (NumericStorage::I16(vec![1, 2]), NumericDType::I16),
            (NumericStorage::I32(vec![1, 2]), NumericDType::I32),
            (NumericStorage::I64(vec![1, 2]), NumericDType::I64),
            (NumericStorage::U8(vec![1, 2]), NumericDType::U8),
            (NumericStorage::U16(vec![1, 2]), NumericDType::U16),
            (NumericStorage::U32(vec![1, 2]), NumericDType::U32),
            (NumericStorage::U64(vec![1, 2]), NumericDType::U64),
        ];

        for (storage, dtype) in cases {
            assert_eq!(storage.numeric_dtype(), dtype);
            assert_eq!(storage.class_name(), dtype.class_name());
            assert_eq!(storage.len(), 2);
            assert!(!storage.is_empty());
            assert_eq!(storage.checked_byte_len(), Some(2 * dtype.byte_size()));
            assert_eq!(storage.view().numeric_dtype(), dtype);
            assert_eq!(storage.view().len(), 2);
            assert!(!storage.view().is_empty());
        }
    }

    #[test]
    fn numeric_storage_views_are_typed_and_mutate_without_coercion() {
        let mut cases = [
            NumericStorage::F64(vec![0.0]),
            NumericStorage::F32(vec![0.0]),
            NumericStorage::I8(vec![0]),
            NumericStorage::I16(vec![0]),
            NumericStorage::I32(vec![0]),
            NumericStorage::I64(vec![0]),
            NumericStorage::U8(vec![0]),
            NumericStorage::U16(vec![0]),
            NumericStorage::U32(vec![0]),
            NumericStorage::U64(vec![0]),
        ];

        for storage in &mut cases {
            let dtype = storage.numeric_dtype();
            let view = storage.view_mut();
            assert_eq!(view.numeric_dtype(), dtype);
            assert_eq!(view.len(), 1);
            assert!(!view.is_empty());
            match view {
                NumericStorageViewMut::F64(values) => values[0] = 64.0,
                NumericStorageViewMut::F32(values) => values[0] = 32.0,
                NumericStorageViewMut::I8(values) => values[0] = -8,
                NumericStorageViewMut::I16(values) => values[0] = -16,
                NumericStorageViewMut::I32(values) => values[0] = -32,
                NumericStorageViewMut::I64(values) => values[0] = -64,
                NumericStorageViewMut::U8(values) => values[0] = 8,
                NumericStorageViewMut::U16(values) => values[0] = 16,
                NumericStorageViewMut::U32(values) => values[0] = 32,
                NumericStorageViewMut::U64(values) => values[0] = 64,
            }
        }

        assert!(matches!(cases[0].view(), NumericStorageView::F64([64.0])));
        assert!(matches!(cases[1].view(), NumericStorageView::F32([32.0])));
        assert!(matches!(cases[2].view(), NumericStorageView::I8([-8])));
        assert!(matches!(cases[3].view(), NumericStorageView::I16([-16])));
        assert!(matches!(cases[4].view(), NumericStorageView::I32([-32])));
        assert!(matches!(cases[5].view(), NumericStorageView::I64([-64])));
        assert!(matches!(cases[6].view(), NumericStorageView::U8([8])));
        assert!(matches!(cases[7].view(), NumericStorageView::U16([16])));
        assert!(matches!(cases[8].view(), NumericStorageView::U32([32])));
        assert!(matches!(cases[9].view(), NumericStorageView::U64([64])));
    }

    #[test]
    fn numeric_storage_shape_validation_is_exact_and_overflow_safe() {
        let storage = NumericStorage::U64(vec![0, u64::MAX]);
        assert_eq!(storage.validate_shape(&[1, 2]), Ok(()));
        assert_eq!(storage.validate_shape(&[2]), Ok(()));
        assert!(storage.validate_shape(&[3]).is_err());
        assert!(storage.validate_shape(&[usize::MAX, 2]).is_err());

        let empty = NumericStorage::I16(Vec::new());
        assert_eq!(empty.validate_shape(&[0, 7, 3]), Ok(()));
        assert!(empty.is_empty());
    }

    #[test]
    fn numeric_storage_moves_integer_buffers_without_floating_conversion() {
        let exact = IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]);
        let storage = NumericStorage::from_integer_storage(exact.clone());

        assert_eq!(storage.numeric_dtype(), NumericDType::U64);
        assert!(storage.as_f64_slice().is_none());
        assert!(storage.as_f32_slice().is_none());
        assert_eq!(storage.into_integer_storage(), Ok(exact));

        let floating = NumericStorage::F32(vec![0.1]);
        assert_eq!(
            floating.clone().into_integer_storage(),
            Err(floating.clone())
        );
        assert_eq!(floating.as_f32_slice(), Some(&[0.1][..]));
        assert!(floating.as_f64_slice().is_none());
    }

    #[test]
    fn numeric_storage_reads_and_writes_exact_scalars_for_every_native_class() {
        let mut cases = [
            (NumericStorage::F64(vec![0.0]), NumericScalar::F64(64.25)),
            (NumericStorage::F32(vec![0.0]), NumericScalar::F32(32.25)),
            (NumericStorage::I8(vec![0]), NumericScalar::I8(i8::MIN)),
            (NumericStorage::I16(vec![0]), NumericScalar::I16(i16::MIN)),
            (NumericStorage::I32(vec![0]), NumericScalar::I32(i32::MIN)),
            (NumericStorage::I64(vec![0]), NumericScalar::I64(i64::MIN)),
            (NumericStorage::U8(vec![0]), NumericScalar::U8(u8::MAX)),
            (NumericStorage::U16(vec![0]), NumericScalar::U16(u16::MAX)),
            (NumericStorage::U32(vec![0]), NumericScalar::U32(u32::MAX)),
            (NumericStorage::U64(vec![0]), NumericScalar::U64(u64::MAX)),
        ];

        for (storage, value) in &mut cases {
            assert_eq!(value.numeric_dtype(), storage.numeric_dtype());
            assert_eq!(value.class_name(), storage.class_name());
            storage.set_value(0, *value).expect("same-class write");
            assert_eq!(storage.value_at(0), Some(*value));
            assert_eq!(storage.value_at(1), None);
        }

        assert!(cases[9].0.set_value(0, NumericScalar::F64(1.0)).is_err());
        assert!(cases[9].0.set_value(1, NumericScalar::U64(1)).is_err());
    }

    #[test]
    fn numeric_scalar_zero_and_finite_predicates_cover_every_native_class() {
        let zeros = [
            NumericScalar::F64(-0.0),
            NumericScalar::F32(-0.0),
            NumericScalar::I8(0),
            NumericScalar::I16(0),
            NumericScalar::I32(0),
            NumericScalar::I64(0),
            NumericScalar::U8(0),
            NumericScalar::U16(0),
            NumericScalar::U32(0),
            NumericScalar::U64(0),
        ];
        assert!(zeros.into_iter().all(NumericScalar::is_zero));
        assert!(zeros.into_iter().all(NumericScalar::is_finite));

        assert!(!NumericScalar::F64(f64::NAN).is_zero());
        assert!(!NumericScalar::F32(f32::INFINITY).is_zero());
        assert!(!NumericScalar::F64(f64::NAN).is_finite());
        assert!(!NumericScalar::F32(f32::INFINITY).is_finite());
        assert!(NumericScalar::I64(i64::MIN).is_finite());
        assert!(NumericScalar::U64(u64::MAX).is_finite());
    }

    #[test]
    fn numeric_scalar_f64_materialization_is_explicit_for_every_native_class() {
        let cases = [
            (NumericScalar::F64(64.25), 64.25),
            (NumericScalar::F32(32.25), 32.25),
            (NumericScalar::I8(-8), -8.0),
            (NumericScalar::I16(-16), -16.0),
            (NumericScalar::I32(-32), -32.0),
            (NumericScalar::I64(-64), -64.0),
            (NumericScalar::U8(8), 8.0),
            (NumericScalar::U16(16), 16.0),
            (NumericScalar::U32(32), 32.0),
            (NumericScalar::U64(64), 64.0),
        ];
        for (value, expected) in cases {
            assert_eq!(value.materialize_f64(), expected);
        }
        assert_eq!(
            NumericScalar::U64(9_007_199_254_740_993).materialize_f64(),
            9_007_199_254_740_992.0
        );
    }

    #[test]
    fn numeric_storage_allocates_zero_and_one_in_every_native_class() {
        let dtypes = [
            NumericDType::F64,
            NumericDType::F32,
            NumericDType::I8,
            NumericDType::I16,
            NumericDType::I32,
            NumericDType::I64,
            NumericDType::U8,
            NumericDType::U16,
            NumericDType::U32,
            NumericDType::U64,
        ];

        for dtype in dtypes {
            let zeros = NumericStorage::zeros(dtype, 3);
            let ones = NumericStorage::ones(dtype, 3);
            assert_eq!(zeros.numeric_dtype(), dtype);
            assert_eq!(ones.numeric_dtype(), dtype);
            assert_eq!(zeros.len(), 3);
            assert_eq!(ones.len(), 3);
            assert_eq!(zeros, ones.zeros_like(3));
            assert_eq!(ones, zeros.ones_like(3));
        }

        assert_eq!(
            NumericStorage::zeros(NumericDType::U64, 2),
            NumericStorage::U64(vec![0, 0])
        );
        assert_eq!(
            NumericStorage::ones(NumericDType::F32, 2),
            NumericStorage::F32(vec![1.0, 1.0])
        );
    }

    #[test]
    fn numeric_storage_gather_and_reorder_preserve_every_native_class() {
        let cases = [
            NumericStorage::F64(vec![1.0, 2.0, 3.0]),
            NumericStorage::F32(vec![1.0, 2.0, 3.0]),
            NumericStorage::I8(vec![1, 2, 3]),
            NumericStorage::I16(vec![1, 2, 3]),
            NumericStorage::I32(vec![1, 2, 3]),
            NumericStorage::I64(vec![1, 2, 3]),
            NumericStorage::U8(vec![1, 2, 3]),
            NumericStorage::U16(vec![1, 2, 3]),
            NumericStorage::U32(vec![1, 2, 3]),
            NumericStorage::U64(vec![1, 9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in cases {
            assert_eq!(storage.clone_for_shape(&[1, 3]), Ok(storage.clone()));
            assert!(storage.clone_for_shape(&[2, 2]).is_err());
            let gathered = storage.gather(&[2, 0, 2]).expect("exact gather");
            assert_eq!(gathered.numeric_dtype(), storage.numeric_dtype());
            assert_eq!(gathered.value_at(0), storage.value_at(2));
            assert_eq!(gathered.value_at(1), storage.value_at(0));
            assert_eq!(gathered.value_at(2), storage.value_at(2));
            assert_eq!(
                storage.reorder(&[2, 1, 0]).unwrap().value_at(0),
                storage.value_at(2)
            );
            assert!(storage.gather(&[3]).is_err());
            assert!(storage.reorder(&[0]).is_err());
        }
    }

    #[test]
    fn numeric_storage_resize_and_removal_preserve_every_native_class() {
        let cases = [
            NumericStorage::F64(vec![1.0, 2.0, 3.0]),
            NumericStorage::F32(vec![1.0, 2.0, 3.0]),
            NumericStorage::I8(vec![1, 2, 3]),
            NumericStorage::I16(vec![1, 2, 3]),
            NumericStorage::I32(vec![1, 2, 3]),
            NumericStorage::I64(vec![1, 2, 3]),
            NumericStorage::U8(vec![1, 2, 3]),
            NumericStorage::U16(vec![1, 2, 3]),
            NumericStorage::U32(vec![1, 2, 3]),
            NumericStorage::U64(vec![1, 9_007_199_254_740_993, u64::MAX]),
        ];

        for mut storage in cases {
            let dtype = storage.numeric_dtype();
            let first = storage.value_at(0);
            let third = storage.value_at(2);
            storage.resize_zeroed(5);
            assert_eq!(storage.numeric_dtype(), dtype);
            assert_eq!(storage.len(), 5);
            assert_eq!(storage.value_at(0), first);
            assert_eq!(storage.value_at(2), third);
            assert_eq!(
                storage.value_at(3).map(NumericScalar::materialize_f64),
                Some(0.0)
            );

            storage
                .remove_positions(&[3, 1, 3])
                .expect("in-class removal");
            assert_eq!(storage.numeric_dtype(), dtype);
            assert_eq!(storage.len(), 3);
            assert_eq!(storage.value_at(0), first);
            assert_eq!(storage.value_at(1), third);
            assert_eq!(
                storage.value_at(2).map(NumericScalar::materialize_f64),
                Some(0.0)
            );
            assert!(storage.remove_positions(&[3]).is_err());
        }
    }

    #[test]
    fn numeric_storage_floating_materialization_is_explicit_and_may_be_lossy() {
        let exact =
            NumericStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993, u64::MAX]);
        let as_f64 = exact.materialize_f64();
        let as_f32 = exact.materialize_f32();

        assert_eq!(
            exact.value_at(1),
            Some(NumericScalar::U64(9_007_199_254_740_993))
        );
        assert_eq!(as_f64[0], as_f64[1]);
        assert_eq!(as_f32.len(), 3);
        assert_eq!(
            NumericStorage::F32(vec![0.1, f32::MAX]).materialize_f64(),
            vec![f64::from(0.1_f32), f64::from(f32::MAX)]
        );
        assert_eq!(
            NumericStorage::F64(vec![0.1, f64::MAX]).materialize_f32(),
            vec![0.1_f32, f32::INFINITY]
        );
    }

    #[test]
    fn tensor_numeric_storage_bridge_round_trips_every_native_class() {
        let cases = [
            NumericStorage::F64(vec![f64::MIN_POSITIVE, f64::MAX]),
            NumericStorage::F32(vec![f32::MIN_POSITIVE, f32::MAX]),
            NumericStorage::I8(vec![i8::MIN, i8::MAX]),
            NumericStorage::I16(vec![i16::MIN, i16::MAX]),
            NumericStorage::I32(vec![i32::MIN, i32::MAX]),
            NumericStorage::I64(vec![i64::MIN, i64::MAX]),
            NumericStorage::U8(vec![u8::MIN, u8::MAX]),
            NumericStorage::U16(vec![u16::MIN, u16::MAX]),
            NumericStorage::U32(vec![u32::MIN, u32::MAX]),
            NumericStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in cases {
            let dtype = storage.numeric_dtype();
            let tensor =
                Tensor::from_numeric_storage(storage.clone(), vec![1, 2]).expect("tensor bridge");
            assert_eq!(tensor.numeric_dtype(), dtype);
            assert_eq!(tensor.numeric_value_at(0), storage.value_at(0));
            assert_eq!(tensor.numeric_value_at(1), storage.value_at(1));
            assert_eq!(tensor.numeric_value_at(2), None);
            assert_eq!(tensor.into_numeric_storage(), Ok(storage));
        }
    }

    #[test]
    fn typed_constructor_materializes_exact_integer_storage() {
        let tensor =
            Tensor::new_with_dtype(vec![-2.2, 12.8, 99_999.0], vec![1, 3], NumericDType::I16)
                .expect("typed tensor");

        assert_eq!(tensor.numeric_dtype(), NumericDType::I16);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I16(vec![-2, 13, i16::MAX]))
        );
    }

    #[test]
    fn integer_tensor_supports_empty_typed_arrays() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(Vec::new()), vec![0, 1])
            .expect("empty integer tensor");

        assert_eq!(
            tensor.integer_storage().map(IntegerStorage::class_name),
            Some("int64")
        );
        assert!(tensor.is_empty());
    }

    #[test]
    fn assignment_conversion_preserves_wide_exact_values_and_mutates_in_place() {
        let large = 9_007_199_254_740_993_u64;
        let mut storage = IntegerStorage::U64(vec![0, 1]);

        storage
            .set_exact_assignment(0, &IntValue::U64(large))
            .unwrap();
        storage.set_f64_assignment(1, -4.2).unwrap();

        assert_eq!(storage, IntegerStorage::U64(vec![large, 0]));
        assert_eq!(
            IntegerStorage::I8(vec![0]).cast_f64_assignment(200.6),
            IntValue::I8(i8::MAX)
        );
    }

    #[test]
    fn tensor_get2_reads_every_integer_class_from_exact_storage() {
        let cases = [
            IntegerStorage::I8(vec![-8, 7]),
            IntegerStorage::I16(vec![-16, 15]),
            IntegerStorage::I32(vec![-32, 31]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![8, u8::MAX]),
            IntegerStorage::U16(vec![16, u16::MAX]),
            IntegerStorage::U32(vec![32, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in cases {
            let expected = storage.to_f64_vec();
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");

            assert_eq!(tensor.get2(0, 0), Ok(expected[0]));
            assert_eq!(tensor.get2(0, 1), Ok(expected[1]));
        }
    }

    #[test]
    fn tensor_set2_updates_exact_integer_storage_without_a_mirror() {
        let cases = [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ];

        for storage in cases {
            let expected = storage.cast_f64_assignment(-2.6);
            let mut tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");

            tensor.set2(0, 0, -2.6).expect("integer assignment");

            assert_eq!(
                tensor
                    .integer_storage()
                    .and_then(|storage| storage.value_at(0)),
                Some(expected.clone())
            );
            assert_eq!(
                tensor.numeric_value_at(0),
                Some(NumericScalar::from(expected))
            );
        }
    }

    #[test]
    fn integer_tensor_rejects_shape_length_mismatches() {
        let err = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![3, 1])
            .expect_err("shape mismatch");
        assert!(err.contains("doesn't match shape"));
    }

    #[test]
    fn reshape_preserves_exact_integer_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![-1, i64::MAX]), vec![1, 2])
            .expect("integer tensor")
            .reshape(vec![2, 1])
            .expect("reshape");

        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I64(vec![-1, i64::MAX]))
        );
    }

    #[test]
    fn reshape_and_display_use_authoritative_integer_storage() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2],
        )
        .expect("integer tensor");

        let tensor = tensor.reshape(vec![1, 2]).expect("reshape");
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
        );

        let mut vector = tensor.clone();
        vector.shape = vec![2];
        vector.rows = 1;
        vector.cols = 2;
        assert_eq!(
            vector.to_string(),
            "[9007199254740993 18446744073709551615]"
        );
    }

    #[test]
    fn integer_complex_storage_preserves_paired_uint64_values() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_809]),
        )
        .expect("matching uint64 components");
        let tensor = ComplexTensor::new_integer(storage.clone(), vec![1, 2])
            .expect("integer complex tensor");

        assert_eq!(tensor.integer_storage(), Some(&storage));
        assert_eq!(
            tensor
                .integer_storage()
                .map(IntegerComplexStorage::class_name),
            Some("uint64")
        );
        assert_eq!(
            tensor.numeric_value_at(0),
            Some((
                NumericScalar::U64(9_223_372_036_854_775_809),
                NumericScalar::U64(u64::MAX)
            ))
        );
    }

    #[test]
    fn single_complex_tensor_keeps_one_native_f32_payload() {
        let values = vec![(1.25_f32, -2.5_f32), (f32::MAX, f32::MIN_POSITIVE)];
        let mut tensor =
            ComplexTensor::from_f32(values.clone(), vec![1, 2]).expect("single complex tensor");

        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(tensor.as_f32_slice(), Some(values.as_slice()));
        assert!(tensor.as_f64_slice().is_none());
        assert!(tensor.integer_storage().is_none());

        tensor
            .set_f64_assignment_at(0, 3.75, -4.5)
            .expect("single complex assignment");
        assert_eq!(
            tensor.complex_storage(),
            &ComplexStorage::F32(vec![(3.75_f32, -4.5_f32), values[1]])
        );
    }

    #[test]
    fn complex_storage_gather_preserves_native_component_class() {
        let single = ComplexStorage::F32(vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]);
        assert_eq!(
            single.gather(&[2, 0]),
            Ok(ComplexStorage::F32(vec![(3.0, -3.0), (1.0, -1.0)]))
        );

        let integer = ComplexStorage::Integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1, 9_007_199_254_740_993, u64::MAX]),
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 1]),
            )
            .expect("complex uint64 storage"),
        );
        assert_eq!(
            integer.gather(&[2, 1]),
            Ok(ComplexStorage::Integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
                    IntegerStorage::U64(vec![1, 9_007_199_254_740_993]),
                )
                .expect("gathered complex uint64 storage"),
            ))
        );
        assert!(integer.gather(&[3]).is_err());
    }

    #[test]
    fn complex_floating_reconstruction_preserves_requested_class() {
        let values = vec![(1.0, -2.0), (3.5, 4.25)];
        let single = ComplexTensor::from_f64_values_with_dtype(
            values.clone(),
            vec![2, 1],
            NumericDType::F32,
        )
        .expect("single reconstruction");
        let double = ComplexTensor::from_f64_values_with_dtype(
            values.clone(),
            vec![2, 1],
            NumericDType::F64,
        )
        .expect("double reconstruction");

        assert_eq!(single.numeric_dtype(), NumericDType::F32);
        assert_eq!(single.materialize_f64(), values);
        assert_eq!(double.numeric_dtype(), NumericDType::F64);
        assert_eq!(double.as_f64_slice(), Some(values.as_slice()));
    }

    #[test]
    fn integer_complex_display_uses_authoritative_storage() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
        )
        .expect("matching storage");
        let tensor = ComplexTensor::new_integer(storage, vec![2]).expect("complex tensor");

        assert_eq!(
            tensor.to_string(),
            "[9007199254740993+18446744073709551615i 18446744073709551615+9007199254740993i]"
        );
    }

    #[test]
    fn integer_complex_storage_rejects_mismatched_components() {
        let class_mismatch =
            IntegerComplexStorage::new(IntegerStorage::I64(vec![1]), IntegerStorage::U64(vec![1]))
                .expect_err("integer classes must match");
        assert!(class_mismatch.contains("matching class"));

        let length_mismatch = IntegerComplexStorage::new(
            IntegerStorage::I64(vec![1]),
            IntegerStorage::I64(vec![1, 2]),
        )
        .expect_err("component lengths must match");
        assert!(length_mismatch.contains("matching class and length"));
    }
}
