use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::{GpuHandleProvenance, HostTensorView};

#[test]
fn mapped_log_respects_zero_boundary_provenance_and_extension_policy() {
    test_support::with_test_provider(|provider| {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (data, expected) in [(vec![0.0, 2.0], true), (vec![-1.0, 2.0], false)] {
            let handle = provider
                .upload(&HostTensorView {
                    data: &data,
                    shape: &[2, 1],
                })
                .unwrap()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            assert_eq!(
                futures::executor::block_on(mapped_batch_equivalent(&Value::GpuTensor(
                    handle.clone()
                ),))
                .unwrap(),
                expected
            );
            let automatic = handle
                .with_provenance(GpuHandleProvenance::Automatic)
                .unwrap();
            assert!(
                futures::executor::block_on(mapped_batch_equivalent(&Value::GpuTensor(automatic),))
                    .unwrap()
            );
            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(
                futures::executor::block_on(mapped_batch_equivalent(&Value::GpuTensor(
                    handle.clone()
                ),))
                .unwrap()
            );
            handle.allocation_lease().validate_live().unwrap();
        }
        let logical = runmat_accelerate_api::upload_logical_protecting(
            provider,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1],
                shape: &[2, 1],
            },
            &[],
        )
        .unwrap();
        assert!(
            !futures::executor::block_on(mapped_batch_equivalent(&Value::GpuTensor(logical),))
                .unwrap()
        );
        let host = Value::Tensor(runmat_value::Tensor::new(vec![-1.0], vec![1, 1]).unwrap());
        assert!(futures::executor::block_on(mapped_batch_equivalent(&host)).unwrap());
    });
}
