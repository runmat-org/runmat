use super::*;
use runmat_value::{ComplexTensor, NumericDType};

fn sample(singleton: bool) -> (Vec<(f32, f32)>, Vec<usize>) {
    if singleton {
        (vec![(0.5, 0.25)], vec![1, 1, 1])
    } else {
        (vec![(0.0, 0.0), (0.5, 0.25)], vec![2, 1])
    }
}

fn assert_values(actual: &[(f64, f64)], singleton: bool) {
    let a = std::f64::consts::PI * 0.5;
    let b = std::f64::consts::PI * 0.25;
    let denominator = a * a + b * b;
    let nonzero = (
        (a.sin() * b.cosh() * a + a.cos() * b.sinh() * b) / denominator,
        (a.cos() * b.sinh() * a - a.sin() * b.cosh() * b) / denominator,
    );
    let expected = if singleton {
        vec![nonzero]
    } else {
        vec![(1.0, 0.0), nonzero]
    };
    assert_eq!(actual.len(), expected.len());
    for ((ar, ai), (er, ei)) in actual.iter().copied().zip(expected) {
        let tolerance = 4.0 * f64::from(f32::EPSILON);
        assert!((ar - er).abs() <= tolerance * er.abs().max(1.0));
        assert!((ai - ei).abs() <= tolerance * ei.abs().max(1.0));
    }
}

#[test]
fn host_complex_single_preserves_native_class_and_nd_singleton_shape() {
    for singleton in [false, true] {
        let (pairs, shape) = sample(singleton);
        let input = ComplexTensor::from_f32(pairs, shape.clone()).unwrap();
        let result = block_on(call_builtin_async("sinc", &[Value::ComplexTensor(input)])).unwrap();
        let Value::ComplexTensor(output) = result.value() else {
            panic!("single complex result may not collapse into Double scalar")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert!(output.as_f32_slice().is_some());
        assert_eq!(output.shape(), shape);
        assert_values(&output.materialize_f64(), singleton);
    }
}

#[test]
fn resident_complex_single_fallback_retains_typed_storage_intent_and_owner() {
    for singleton in [false, true] {
        for explicit in [false, true] {
            for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
                let provider = Box::leak(Box::new(Provider {
                    inner: Default::default(),
                    precision,
                    mode: Mode::Unsupported,
                    trace: Mutex::new(Vec::new()),
                }));
                let _selected = ThreadProviderGuard::set(Some(provider));
                let (pairs, shape) = sample(singleton);
                let words: Vec<f32> = pairs.iter().flat_map(|&(re, im)| [re, im]).collect();
                let input = provider
                    .inner
                    .upload_numeric_raw(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&words),
                        shape: &shape,
                        storage: GpuTensorStorage::ComplexInterleaved,
                    })
                    .unwrap()
                    .with_provenance(if explicit {
                        GpuHandleProvenance::Explicit
                    } else {
                        GpuHandleProvenance::Automatic
                    })
                    .unwrap();
                let args = [Value::GpuTensor(input.clone())];
                let result = block_on(call_builtin_async("sinc", &args)).unwrap();
                let Value::GpuTensor(output) = result.value() else {
                    panic!("supported native Single restoration must remain resident")
                };
                assert_eq!(output.shape(), shape);
                assert_eq!(handle_precision(output), Some(ProviderPrecision::F32));
                assert_eq!(handle_storage(output), GpuTensorStorage::ComplexInterleaved);
                assert_eq!(handle_provenance(output), handle_provenance(&input));
                assert!(!output.shares_allocation_with(&input));
                let expected_trace = if precision == ProviderPrecision::F32 {
                    vec!["sinc", "download", "upload"]
                } else {
                    vec!["download", "upload"]
                };
                assert_eq!(*provider.trace.lock().unwrap(), expected_trace);
                let downloaded = block_on(provider.inner.download_numeric(output)).unwrap();
                let HostNumericDataOwned::F32(data) = downloaded.data else {
                    panic!("actual native output storage must be F32")
                };
                let pairs: Vec<_> = data
                    .chunks_exact(2)
                    .map(|v| (f64::from(v[0]), f64::from(v[1])))
                    .collect();
                assert_values(&pairs, singleton);
                drop(result);
                drop(args);
                input.allocation_lease().validate_live().unwrap();
                let original = block_on(provider.inner.download_numeric(&input)).unwrap();
                assert_eq!(original.data, HostNumericDataOwned::F32(words));
                input.release().unwrap();
            }
        }
    }
}
