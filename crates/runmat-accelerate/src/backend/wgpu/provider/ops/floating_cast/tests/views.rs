use super::*;

fn cast_view(provider: &WgpuProvider, source: NumericPrecision, target: NumericPrecision) {
    for shape in [vec![2, 2], vec![2, 3], vec![0, 3]] {
        for release_source_first in [false, true] {
            let count: usize = shape.iter().product();
            let values: Vec<f64> = (0..count).map(|i| 16_777_215.0 + i as f64).collect();
            let singles: Vec<f32> = values.iter().map(|v| *v as f32).collect();
            let data = match source {
                NumericPrecision::F32 => HostNumericDataView::F32(&singles),
                NumericPrecision::F64 => HostNumericDataView::F64(&values),
            };
            let input = provider
                .upload_numeric(&HostNumericTensorView {
                    data,
                    shape: &shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            let view = provider.transpose_exec(&input).unwrap();
            let metadata = runmat_accelerate_api::handle_transpose_info(&view).unwrap();
            let output =
                futures::executor::block_on(provider.floating_cast_exec(&view, target)).unwrap();
            assert_eq!(output.shape(), [shape[1], shape[0]]);
            assert_eq!(output.device_id(), view.device_id());
            assert_eq!(output.shares_allocation_with(&view), source == target);
            assert_eq!(
                runmat_accelerate_api::handle_transpose_info(&output),
                Some(metadata)
            );
            assert_eq!(
                runmat_accelerate_api::handle_transpose_info(&view),
                Some(metadata)
            );
            assert_eq!(
                output.descriptor().element_type,
                Some(match target {
                    NumericPrecision::F32 => NumericElementType::F32,
                    NumericPrecision::F64 => NumericElementType::F64,
                })
            );
            assert_eq!(output.descriptor().storage, Some(GpuTensorStorage::Real));
            let source_entry = provider.get_entry_raw(&view).unwrap();
            let output_entry = provider.get_entry_raw(&output).unwrap();
            assert_eq!(
                runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
                    &source_entry.buffer,
                    &output_entry.buffer
                ),
                source == target,
                "same-type conversion shares the physical lease without copying"
            );
            drop(source_entry);
            drop(output_entry);
            let mut expected = Vec::with_capacity(count);
            for row in 0..shape[0] {
                for col in 0..shape[1] {
                    let index = row + col * shape[0];
                    let value = match source {
                        NumericPrecision::F32 => f64::from(singles[index]),
                        NumericPrecision::F64 => values[index],
                    };
                    expected.push(match target {
                        NumericPrecision::F32 => f64::from(value as f32),
                        NumericPrecision::F64 => value,
                    });
                }
            }
            input.release().unwrap();
            let mut source_view = Some(view);
            if release_source_first {
                source_view.take().unwrap().release().unwrap();
            }
            let downloaded = block_on(provider.download_numeric(&output)).unwrap();
            assert_eq!(downloaded.shape, vec![shape[1], shape[0]]);
            let actual: Vec<f64> = match downloaded.data {
                HostNumericDataOwned::F32(data) => data.into_iter().map(f64::from).collect(),
                HostNumericDataOwned::F64(data) => data,
                other => panic!("floating cast returned {other:?}"),
            };
            assert_eq!(actual, expected);
            output.release().unwrap();
            if let Some(view) = source_view {
                assert!(block_on(provider.download_numeric(&view)).is_ok());
                assert_eq!(
                    runmat_accelerate_api::handle_transpose_info(&view),
                    Some(metadata)
                );
                view.release().unwrap();
            }
        }
    }
}

#[test]
fn actual_wgpu_single_identity_preserves_asymmetric_transposed_views_without_copy() {
    cast_view(provider(), NumericPrecision::F32, NumericPrecision::F32);
}

#[test]
#[ignore = "requires an actual shader-f64 adapter; run explicitly in the F64 qualification gate"]
fn actual_wgpu_floating_casts_preserve_square_rectangular_and_empty_transposed_values() {
    let provider = provider();
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F64,
        "actual shader-f64 required"
    );
    for source in [NumericPrecision::F32, NumericPrecision::F64] {
        for target in [NumericPrecision::F32, NumericPrecision::F64] {
            cast_view(provider, source, target);
        }
    }
}

#[test]
#[ignore = "requires an actual shader-f64 adapter; run explicitly in the F64 qualification gate"]
fn actual_wgpu_complex_double_to_single_cast_preserves_components_and_owner() {
    let provider = provider();
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, -2.0, 16_777_217.0, -16_777_217.0]),
            shape: &[2, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .expect("complex double upload");
    let output = block_on(provider.floating_cast_exec(&input, NumericPrecision::F32))
        .expect("componentwise complex cast");
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(output.device_id(), input.device_id());
    assert_eq!(
        output.descriptor().storage,
        Some(GpuTensorStorage::ComplexInterleaved)
    );
    assert_eq!(
        output.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert!(!output.shares_allocation_with(&input));
    input
        .release()
        .expect("release input without retiring output");
    let actual = block_on(provider.download_numeric(&output)).expect("complex single readback");
    assert_eq!(actual.shape, vec![2, 1]);
    assert_eq!(actual.storage, GpuTensorStorage::ComplexInterleaved);
    assert!(matches!(
        actual.data,
        HostNumericDataOwned::F32(ref data)
            if data == &[1.0, -2.0, 16_777_216.0, -16_777_216.0]
    ));
    output.release().expect("retire complex cast output");
}

#[test]
fn actual_wgpu_cast_rejects_incoherent_view_before_allocating_result() {
    // A private instance makes the existing allocation-table counter independent
    // of concurrent hardware tests.
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let allocations = provider.allocations.record_count().unwrap();
    let pipelines = provider.fused_pipeline_cache.lock().unwrap().len();
    let mut wire = input.wire_descriptor();
    wire.view = wire
        .view
        .with_layout(runmat_accelerate_api::GpuTensorLayout::Transpose(
            runmat_accelerate_api::TransposeInfo {
                base_rows: 1,
                base_cols: 4,
            },
        ));
    assert!(provider.acquire_live(&wire).is_err());
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    assert_eq!(
        provider.fused_pipeline_cache.lock().unwrap().len(),
        pipelines
    );
    assert_eq!(
        block_on(provider.download_numeric(&input)).unwrap().data,
        HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0])
    );
    input.release().unwrap();
}

#[test]
fn actual_wgpu_single_view_preserves_signed_zero_and_subnormal_bits() {
    let provider = provider();
    let values = [
        0.0_f32,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let view = provider.transpose_exec(&input).unwrap();
    let output = block_on(provider.floating_cast_exec(&view, NumericPrecision::F32)).unwrap();
    let HostNumericDataOwned::F32(actual) =
        block_on(provider.download_numeric(&output)).unwrap().data
    else {
        panic!("physical F32 required");
    };
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [0, 2, 4, 1, 3, 5].map(|i| values[i].to_bits())
    );
    input.release().unwrap();
    view.release().unwrap();
    output.release().unwrap();
}

#[test]
#[ignore = "requires an actual shader-f64 adapter; run explicitly in the F64 qualification gate"]
fn actual_wgpu_cross_precision_view_preserves_signed_zero_and_tiny_conversion() {
    let provider = provider();
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    let values = [
        0.0_f64,
        -0.0,
        f64::from(f32::from_bits(1)),
        -f64::from(f32::from_bits(1)),
        f64::from_bits(1),
        -f64::from_bits(1),
    ];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&values),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let view = provider.transpose_exec(&input).unwrap();
    let output = block_on(provider.floating_cast_exec(&view, NumericPrecision::F32)).unwrap();
    let HostNumericDataOwned::F32(actual) =
        block_on(provider.download_numeric(&output)).unwrap().data
    else {
        panic!("physical F32 required");
    };
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [0, 2, 4, 1, 3, 5].map(|i| (values[i] as f32).to_bits())
    );
    input.release().unwrap();
    view.release().unwrap();
    output.release().unwrap();
}
