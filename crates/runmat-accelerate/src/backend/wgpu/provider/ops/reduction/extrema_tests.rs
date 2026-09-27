use super::*;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorLayout, HostNumericDataView, HostNumericTensorView,
};

#[tokio::test]
#[ignore = "requires an actual WGPU device; must not silently skip"]
async fn dimensional_extrema_include_nan_layout_empty_and_workgroup_boundary() {
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(Default::default())
        .expect("actual WGPU qualification requires an adapter");
    if let Ok(requested) = std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        let actual = match provider.precision {
            crate::backend::wgpu::types::NumericPrecision::F32 => "f32",
            crate::backend::wgpu::types::NumericPrecision::F64 => "f64",
        };
        assert_eq!(
            requested.to_ascii_lowercase(),
            actual,
            "required precision unavailable; fallback is not hardware qualification"
        );
    }
    let upload = |values: &[f64], shape: &[usize]| {
        if provider.precision == crate::backend::wgpu::types::NumericPrecision::F32 {
            let values = values.iter().map(|&v| v as f32).collect::<Vec<_>>();
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&values),
                    shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        } else {
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(values),
                    shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        }
    };
    let input = upload(&[3.0, 1.0, f64::NAN, f64::INFINITY, -0.0, 0.0], &[2, 3]);
    let transposed = provider.transpose(&input).unwrap();
    assert!(matches!(
        transposed.view().layout(),
        GpuTensorLayout::Transpose(_)
    ));
    // The same logical slices are columns of the lazy transpose and rows of
    // the original allocation. Both dispatch axes must retain NaNs and origins.
    for minimum in [true, false] {
        let result = if minimum {
            provider.reduce_min_dim(&input, 1).await
        } else {
            provider.reduce_max_dim(&input, 1).await
        }
        .unwrap();
        assert_eq!(result.values.shape(), &[2, 1]);
        let values = provider.download(&result.values).await.unwrap().data;
        let indices = provider.download(&result.indices).await.unwrap().data;
        assert!(values[0].is_nan());
        assert_eq!(indices[0], 2.0);
        assert_eq!(values[1], if minimum { 0.0 } else { f64::INFINITY });
        assert_eq!(indices[1], if minimum { 3.0 } else { 2.0 });
    }
    for minimum in [true, false] {
        let result = if minimum {
            provider.reduce_min_dim(&transposed, 0).await
        } else {
            provider.reduce_max_dim(&transposed, 0).await
        }
        .unwrap();
        assert_eq!(result.values.shape(), &[1, 2]);
        let values = provider.download(&result.values).await.unwrap().data;
        let indices = provider.download(&result.indices).await.unwrap().data;
        assert!(values[0].is_nan());
        assert_eq!(indices[0], 2.0);
        assert_eq!(values[1], if minimum { 0.0 } else { f64::INFINITY });
        assert_eq!(indices[1], if minimum { 3.0 } else { 2.0 });
        assert!(!result.values.shares_allocation_with(&input));
        assert!(!result.indices.shares_allocation_with(&result.values));
    }
    let ties = upload(&[-0.0, 0.0, f64::NAN, f64::NAN], &[2, 2]);
    let result = provider.reduce_min_dim(&ties, 0).await.unwrap();
    let values = provider.download(&result.values).await.unwrap().data;
    assert_eq!(values[0].to_bits(), (-0.0f64).to_bits());
    assert!(values[1].is_nan());
    assert_eq!(
        provider.download(&result.indices).await.unwrap().data,
        vec![1.0, 1.0]
    );
    let maximum = provider.reduce_max_dim(&ties, 0).await.unwrap();
    assert_eq!(
        provider.download(&maximum.values).await.unwrap().data[0].to_bits(),
        0.0f64.to_bits()
    );
    assert_eq!(
        provider.download(&maximum.indices).await.unwrap().data,
        vec![2.0, 1.0]
    );
    let length = provider.workgroup_config.scalar as usize + 1;
    let boundary = upload(&vec![7.0; length], &[1, length]);
    let result = provider.reduce_max_dim(&boundary, 0).await.unwrap();
    assert_eq!(
        provider.download(&result.values).await.unwrap().data,
        vec![7.0; length]
    );
    assert_eq!(
        provider.download(&result.indices).await.unwrap().data,
        vec![1.0; length]
    );
    for shape in [[0, 3], [3, 0]] {
        let empty = upload(&[], &shape);
        for dim in [0, 1] {
            for minimum in [true, false] {
                let result = if minimum {
                    provider.reduce_min_dim(&empty, dim).await
                } else {
                    provider.reduce_max_dim(&empty, dim).await
                }
                .unwrap();
                assert_eq!(result.values.shape(), shape);
                assert_eq!(result.indices.shape(), shape);
                assert!(provider
                    .download(&result.values)
                    .await
                    .unwrap()
                    .data
                    .is_empty());
                assert!(provider
                    .download(&result.indices)
                    .await
                    .unwrap()
                    .data
                    .is_empty());
            }
        }
    }
}
