use super::*;

#[test]
#[ignore = "requires an actual SHADER_F64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn actual_one_element_nan_and_signed_zero_use_the_ordered_numeric_route() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires a real WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F64);
    for (values, mode, expected) in [
        (
            [-0.0_f32, -0.0],
            ProviderRealMeanNaNMode::Include,
            Some(0.0_f32),
        ),
        ([f32::NAN, 2.0], ProviderRealMeanNaNMode::Include, None),
        (
            [f32::NAN, 2.0],
            ProviderRealMeanNaNMode::Omit,
            Some(2.0_f32),
        ),
        ([f32::NAN, f32::NAN], ProviderRealMeanNaNMode::Omit, None),
    ] {
        let input = provider
            .upload_numeric_exec(&HostNumericTensorView {
                data: HostNumericDataView::F32(&values),
                shape: &[1, 1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let selected = provider
            .prepare_ordered_real_mean(&input, &[2], ProviderPrecision::F32, mode)
            .unwrap()
            .expect("one in-bounds nonempty axis is admissible");
        let output = provider
            .execute_ordered_real_mean(selected, None)
            .unwrap()
            .output;
        assert_eq!(output.shape(), &[1, 1, 1]);
        let observed = block_on(provider.download_numeric_exec(&output)).unwrap();
        let HostNumericDataOwned::F32(data) = observed.data else {
            panic!("selected output must retain Single physical class");
        };
        let value = data[0];
        match expected {
            Some(expected) => assert_eq!(value.to_bits(), expected.to_bits()),
            None => assert!(value.is_nan()),
        }
        input.allocation_lease().validate_live().unwrap();
    }
    let empty = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[]),
            shape: &[1, 0, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(provider
        .prepare_ordered_real_mean(
            &empty,
            &[1],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .is_none());
    empty.allocation_lease().validate_live().unwrap();
}
