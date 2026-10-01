use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostNumericDataOwned,
};

#[test]
fn ordinary_int8_sum_does_not_saturate_before_double_result() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I8(&[100, 100, 100, 100]),
            shape: &[2, 2],
        })
        .unwrap();
    let columns = block_on(provider.reduce_sum_dim(&input, 0)).unwrap();
    let global = block_on(provider.reduce_sum(&input)).unwrap();
    let native = block_on(provider.reduce_integer_sum_native_dim(&input, 0)).unwrap();
    assert_eq!(columns.shape(), &[1, 2]);
    assert_eq!(global.shape(), &[1, 1]);
    assert_eq!(columns.physical().element_type(), NumericElementType::F64);
    assert_eq!(
        block_on(provider.download_numeric(&columns)).unwrap().data,
        HostNumericDataOwned::F64(vec![200.0, 200.0])
    );
    assert_eq!(
        block_on(provider.download_numeric(&global)).unwrap().data,
        HostNumericDataOwned::F64(vec![400.0])
    );
    assert_eq!(
        block_on(provider.download_integer(&native)).unwrap().data,
        HostIntegerDataOwned::I8(vec![127, 127])
    );
}

#[test]
fn ordinary_integer_sum_preserves_nd_geometry_empty_identity_and_wide_conversion() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[(1_u64 << 53) + 1, 1, 2, 3, 4, 5, 6, 7]),
            shape: &[2, 2, 2],
        })
        .unwrap();
    let along_first = block_on(provider.reduce_sum_dim(&input, 0)).unwrap();
    assert_eq!(along_first.shape(), &[1, 2, 2]);
    assert_eq!(
        block_on(provider.download_numeric(&along_first))
            .unwrap()
            .data,
        HostNumericDataOwned::F64(vec![(1_u64 << 53) as f64, 5.0, 9.0, 13.0])
    );
    let along_second = block_on(provider.reduce_sum_dim(&along_first, 1)).unwrap();
    assert_eq!(along_second.shape(), &[1, 1, 2]);
    assert_eq!(
        block_on(provider.download_numeric(&along_second))
            .unwrap()
            .data,
        HostNumericDataOwned::F64(vec![((1_u64 << 53) as f64) + 4.0, 22.0])
    );
    let empty = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(&[]),
            shape: &[0, 3],
        })
        .unwrap();
    let reduced_empty = block_on(provider.reduce_sum_dim(&empty, 0)).unwrap();
    assert_eq!(reduced_empty.shape(), &[1, 3]);
    assert_eq!(
        block_on(provider.download_numeric(&reduced_empty))
            .unwrap()
            .data,
        HostNumericDataOwned::F64(vec![0.0, 0.0, 0.0])
    );
}
