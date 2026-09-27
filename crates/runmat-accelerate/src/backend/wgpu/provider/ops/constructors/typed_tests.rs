use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, HostNumericDataOwned, HostNumericTensorView};

#[test]
fn actual_wgpu_typed_zeros_and_logical_fill_preserve_producer_facts() {
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("typed constructors require an actual WGPU adapter");
    let fixtures = [
        (
            HostNumericDataOwned::F32(vec![7.0, 8.0]),
            GpuTensorStorage::Real,
        ),
        (
            HostNumericDataOwned::F32(vec![7.0, 8.0, 9.0, 10.0]),
            GpuTensorStorage::ComplexInterleaved,
        ),
        (HostNumericDataOwned::U8(vec![7, 8]), GpuTensorStorage::Real),
        (
            HostNumericDataOwned::U64(vec![7, u64::MAX]),
            GpuTensorStorage::Real,
        ),
    ];
    for (data, storage) in fixtures {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: data.as_view(),
                shape: &[1, 2],
                storage,
            })
            .unwrap();
        for shape in [vec![2, 1, 3], vec![usize::MAX, usize::MAX, 0]] {
            let output = provider.zeros_like(&input, &shape).unwrap();
            assert_eq!(output.shape(), shape);
            assert_eq!(
                output.physical().element_type(),
                input.physical().element_type()
            );
            assert_eq!(output.physical().storage(), storage);
            assert!(!output.shares_allocation_with(&input));
            let actual = block_on(provider.download_numeric(&output)).unwrap();
            let count = if shape.contains(&0) {
                0
            } else if storage == GpuTensorStorage::Real {
                6
            } else {
                12
            };
            let expected = match &data {
                HostNumericDataOwned::F32(_) => HostNumericDataOwned::F32(vec![0.0; count]),
                HostNumericDataOwned::U8(_) => HostNumericDataOwned::U8(vec![0; count]),
                HostNumericDataOwned::U64(_) => HostNumericDataOwned::U64(vec![0; count]),
                _ => unreachable!("fixture types are exhaustive above"),
            };
            assert_eq!(actual.data, expected);
        }
    }
    for value in [false, true] {
        let input = provider.logical_fill(&[1, 3], value).unwrap();
        assert!(input.view().is_logical());
        assert_eq!(
            block_on(provider.download(&input)).unwrap().data,
            vec![if value { 1.0 } else { 0.0 }; 3]
        );
        let output = provider.zeros_like(&input, &[2, 2]).unwrap();
        assert!(output.view().is_logical());
        assert_eq!(
            block_on(provider.download(&output)).unwrap().data,
            vec![0.0; 4]
        );
        let empty = provider
            .logical_fill(&[usize::MAX, usize::MAX, 0], value)
            .unwrap();
        assert!(empty.view().is_logical());
        assert_eq!(
            block_on(provider.download(&empty)).unwrap().data,
            Vec::<f64>::new()
        );
    }
}

#[test]
#[ignore = "requires an actual shader-f64 adapter; run in the F64 qualification gate"]
fn actual_wgpu_f64_zero_prototype_preserves_complex_storage() {
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("F64 constructor gate requires an actual adapter");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::F64(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let output = provider.zeros_like(&input, &[3, 1]).unwrap();
    assert_eq!(
        output.physical().element_type(),
        runmat_accelerate_api::NumericElementType::F64
    );
    assert_eq!(
        output.physical().storage(),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(
        block_on(provider.download_numeric(&output)).unwrap().data,
        HostNumericDataOwned::F64(vec![0.0; 6])
    );
}

#[test]
fn constructor_shape_counts_are_zero_first_and_checked() {
    assert_eq!(constructor_elements(&[usize::MAX, usize::MAX, 0]), Some(0));
    assert_eq!(constructor_elements(&[usize::MAX, 2]), None);
    assert_eq!(constructor_elements(&[2, 3, 4]), Some(24));
}
