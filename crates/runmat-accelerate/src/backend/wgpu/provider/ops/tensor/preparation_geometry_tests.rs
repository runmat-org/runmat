use super::*;
use runmat_accelerate_api::{HostNumericDataOwned, HostNumericDataView, HostNumericTensorView};

#[test]
#[ignore = "requires actual WGPU device and RUNMAT_WG=64; execute explicitly"]
fn non512_lazy_transpose_materialization_covers_chunk_boundary() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.workgroup_config.scalar, 64,
        "run with RUNMAT_WG=64"
    );
    let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(64).unwrap() as usize;
    let columns = capacity / 2 + 1;
    let values: Vec<f32> = (0..2 * columns).map(|i| (i % 1024) as f32).collect();
    let source = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, columns],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let transposed = provider.transpose_exec(&source).unwrap();
    assert!(runmat_accelerate_api::handle_is_transposed(&transposed));
    let output = provider.reshape_exec(&transposed, &[columns, 2]).unwrap();
    assert!(!runmat_accelerate_api::handle_is_transposed(&output));
    assert_eq!(output.shape, vec![columns, 2]);
    let downloaded = futures::executor::block_on(provider.download_numeric(&output)).unwrap();
    let HostNumericDataOwned::F32(actual) = downloaded.data else {
        panic!("materialization changed physical storage");
    };
    assert_eq!(actual.len(), values.len());
    for (index, actual) in actual.iter().enumerate() {
        assert_eq!(
            *actual,
            values[index / columns + 2 * (index % columns)],
            "logical index {index}"
        );
    }
    provider.free(&output).unwrap();
    if transposed.buffer_id != source.buffer_id {
        provider.free(&transposed).unwrap();
    }
    provider.free(&source).unwrap();
}
