use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{HostNumericDataOwned, NumericElementType};

#[test]
fn conjugation_preserves_native_complex_bits_shape_and_independent_lifetimes() {
    let provider = InProcessProvider::new();
    let other = InProcessProvider::new();
    for single in [true, false] {
        // Include distinct NaN payloads, signed zeros, infinities and finite
        // values. Conjugation changes only each imaginary sign bit.
        let single_words = vec![
            1.25f32,
            -2.5,
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc01234),
            f32::from_bits(0xffc05678),
        ];
        let double_words = vec![
            1.25f64,
            -2.5,
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8000000001234),
            f64::from_bits(0xfff8000000005678),
        ];
        let data = if single {
            InProcessNumericData::F32(single_words.clone())
        } else {
            InProcessNumericData::F64(double_words.clone())
        };
        let input = provider
            .allocate_numeric_data(data, vec![1, 2, 2], GpuTensorStorage::ComplexInterleaved)
            .unwrap();
        let source_wire = input.wire_descriptor();
        let alias = input.clone();
        let error = block_on(other.unary_conj(&input)).unwrap_err();
        assert!(
            !error.is_unsupported(),
            "foreign ownership cannot turn into capability fallback"
        );
        drop(provider.unary_conj(&input));
        assert!(
            provider.acquire_live(&source_wire).is_ok(),
            "unpolled cancellation retains borrowed input"
        );
        let output = block_on(provider.unary_conj(&input))
            .unwrap()
            .expect("admitted conjugation");
        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(
            output.physical().storage(),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            output.physical().element_type(),
            if single {
                NumericElementType::F32
            } else {
                NumericElementType::F64
            }
        );
        assert!(!output.shares_allocation_with(&input));
        let output_wire = output.wire_descriptor();
        let check = |value: HostNumericDataOwned, conjugated: bool| match value {
            HostNumericDataOwned::F32(actual) if single => {
                for (index, (actual, original)) in actual.iter().zip(&single_words).enumerate() {
                    let expected = original.to_bits()
                        ^ if conjugated && index % 2 == 1 {
                            1 << 31
                        } else {
                            0
                        };
                    assert_eq!(actual.to_bits(), expected, "Single component {index}");
                }
                assert_eq!(actual.len(), single_words.len());
            }
            HostNumericDataOwned::F64(actual) if !single => {
                for (index, (actual, original)) in actual.iter().zip(&double_words).enumerate() {
                    let expected = original.to_bits()
                        ^ if conjugated && index % 2 == 1 {
                            1 << 63
                        } else {
                            0
                        };
                    assert_eq!(actual.to_bits(), expected, "Double component {index}");
                }
                assert_eq!(actual.len(), double_words.len());
            }
            other => panic!("native class changed: {other:?}"),
        };
        check(
            block_on(provider.download_numeric(&output)).unwrap().data,
            true,
        );
        check(
            block_on(provider.download_numeric(&alias)).unwrap().data,
            false,
        );
        drop(input);
        assert!(provider.acquire_live(&source_wire).is_ok());
        drop(alias);
        assert!(provider.acquire_live(&source_wire).is_err());
        check(
            block_on(provider.download_numeric(&output)).unwrap().data,
            true,
        );
        drop(output);
        assert!(provider.acquire_live(&output_wire).is_err());
    }
}

#[test]
fn conjugation_empty_complex_and_real_copy_preserve_class_and_shape() {
    let provider = InProcessProvider::new();
    for single in [true, false] {
        for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
            let data = if single {
                InProcessNumericData::F32(vec![])
            } else {
                InProcessNumericData::F64(vec![])
            };
            let input = provider
                .allocate_numeric_data(data, vec![2, 0, 3], storage)
                .unwrap();
            let output = block_on(provider.unary_conj(&input))
                .unwrap()
                .expect("admitted conjugation");
            assert_eq!(output.shape(), &[2, 0, 3]);
            assert_eq!(
                output.physical().element_type(),
                input.physical().element_type()
            );
            assert_eq!(output.physical().storage(), storage);
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(
                block_on(provider.download_numeric(&output))
                    .unwrap()
                    .data
                    .len(),
                0
            );
        }
    }
}

#[test]
fn malformed_complex_extent_is_rejected_by_owner_before_conjugation() {
    let provider = InProcessProvider::new();
    for length in [1, 3, 4] {
        for single in [true, false] {
            let data = if single {
                InProcessNumericData::F32(vec![1.0; length])
            } else {
                InProcessNumericData::F64(vec![1.0; length])
            };
            assert!(provider
                .allocate_numeric_data(data, vec![1, 1], GpuTensorStorage::ComplexInterleaved)
                .is_err());
        }
    }
}

#[test]
fn integer_complex_conjugation_is_explicitly_unsupported_without_touching_input() {
    use runmat_accelerate_api::HostIntegerDataOwned;
    let provider = InProcessProvider::new();
    let other = InProcessProvider::new();
    for data in [
        HostIntegerDataOwned::I8(vec![1, i8::MIN]),
        HostIntegerDataOwned::U64(vec![1, u64::MAX]),
    ] {
        let input = provider
            .allocate_numeric_data(
                InProcessNumericData::Integer(data),
                vec![1, 1],
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let wire = input.wire_descriptor();
        assert!(block_on(provider.unary_conj(&input)).unwrap().is_none());
        assert!(!block_on(other.unary_conj(&input))
            .unwrap_err()
            .is_unsupported());
        assert!(provider.acquire_live(&wire).is_ok());
        drop(input);
        assert!(provider.acquire_live(&wire).is_err());
    }
}
