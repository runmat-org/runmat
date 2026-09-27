use super::*;
use runmat_accelerate_api::{AccelProvider, HostLogicalTensorView, HostTensorView};

fn values(provider: &WgpuProvider, input: &GpuTensorHandle) -> Vec<f64> {
    futures::executor::block_on(provider.download(input))
        .unwrap()
        .data
}

#[test]
#[ignore = "requires actual WGPU device; run explicitly for F32 and F64"]
fn transposed_diag_gather_and_in_place_scatter_use_logical_order() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let source = provider
        .upload(&HostTensorView {
            data: &[1., 2., 3., 4., 5., 6.],
            shape: &[2, 3],
        })
        .unwrap();
    let transposed = provider.transpose_exec(&source).unwrap();
    assert!(transposed.shares_allocation_with(&source));
    let diag = provider.diag_extract_exec(&transposed, 0).unwrap();
    assert_eq!(values(&provider, &diag), vec![1., 4.]);
    let gathered = provider
        .gather_linear_exec(&transposed, &[1, 4], &[2, 1])
        .unwrap();
    assert_eq!(values(&provider, &gathered), vec![3., 4.]);
    let replacement = provider
        .upload(&HostTensorView {
            data: &[20., 30.],
            shape: &[2, 1],
        })
        .unwrap();
    provider
        .scatter_linear_exec(&transposed, &[1, 4], &replacement)
        .unwrap();
    assert_eq!(values(&provider, &source), vec![1., 2., 20., 30., 5., 6.]);
    assert_eq!(
        values(&provider, &transposed),
        vec![1., 20., 5., 2., 30., 6.]
    );
    drop(diag);
    drop(gathered);
    assert!(source.allocation_lease().validate_live().is_ok());
    // Exact physical aliases retain snapshot assignment semantics, including
    // when the read and write logical orders differ.
    provider
        .scatter_linear_exec(&transposed, &[0, 1, 2, 3, 4, 5], &source)
        .unwrap();
    assert_eq!(
        values(&provider, &transposed),
        vec![1., 2., 20., 30., 5., 6.]
    );
}

#[test]
#[ignore = "requires actual WGPU F32 device; RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn logical_transposed_diag_and_functional_scatter_preserve_producer_and_source() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F32
    );
    let source = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1, 0, 1, 0, 0],
            shape: &[2, 3],
        })
        .unwrap();
    let transposed = provider.transpose_exec(&source).unwrap();
    let diag = provider.diag_extract_exec(&transposed, 0).unwrap();
    assert!(diag.view().is_logical());
    assert_eq!(values(&provider, &diag), vec![0., 1.]);
    let column = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 1, 1],
            shape: &[3, 1],
        })
        .unwrap();
    let replaced = provider
        .scatter_column_exec(&transposed, 0, &column)
        .unwrap();
    assert!(replaced.view().is_logical());
    assert!(!replaced.shares_allocation_with(&source));
    assert_eq!(values(&provider, &replaced), vec![1., 1., 1., 1., 1., 0.]);
    let row = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0],
            shape: &[1, 2],
        })
        .unwrap();
    let replaced = provider.scatter_row_exec(&transposed, 1, &row).unwrap();
    assert!(replaced.view().is_logical());
    assert_eq!(values(&provider, &replaced), vec![0., 1., 0., 1., 0., 0.]);
    assert_eq!(values(&provider, &source), vec![0., 1., 0., 1., 0., 0.]);
}

#[test]
#[ignore = "requires actual WGPU device and RUNMAT_WG=64"]
fn diagonal_constructor_covers_nondefault_workgroup_boundary() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.workgroup_config.scalar, 64,
        "run with RUNMAT_WG=64"
    );
    let data: Vec<f64> = (0..65).map(|value| value as f64 + 1.).collect();
    let source = provider
        .upload(&HostTensorView {
            data: &data,
            shape: &[65, 1],
        })
        .unwrap();
    let output = provider.diag_from_vector_exec(&source, 0).unwrap();
    let actual = values(&provider, &output);
    for (index, expected) in data.iter().enumerate() {
        assert_eq!(actual[index * 66], *expected);
    }
    assert_eq!(output.shape(), &[65, 65]);
}
