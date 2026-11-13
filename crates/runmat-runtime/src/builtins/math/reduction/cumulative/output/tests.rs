use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::HostTensorView;

#[test]
fn requested_output_prefix_retires_every_discarded_resident_result() {
    test_support::with_test_provider(|provider| {
        for count in 0..=2 {
            let _outputs = crate::output_count::push_output_count(Some(count)).unwrap();
            let values = provider
                .upload(&HostTensorView {
                    data: &[2.0, 1.0],
                    shape: &[2, 1],
                })
                .unwrap();
            let indices = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[2, 1],
                })
                .unwrap();
            let value_owner = values.allocation_lease().downgrade();
            let index_owner = indices.allocation_lease().downgrade();
            let result = Publication::admit(CumulativeKind::Minimum, None)
                .unwrap()
                .finish(Value::GpuTensor(values), Value::GpuTensor(indices))
                .unwrap();
            let Value::OutputList(outputs) = &result else {
                panic!("requested output list")
            };
            assert_eq!(outputs.as_slice().len(), count);
            assert_eq!(value_owner.acquire().is_some(), count > 0);
            assert_eq!(index_owner.acquire().is_some(), count > 1);
            drop(result);
            assert!(value_owner.acquire().is_none());
            assert!(index_owner.acquire().is_none());
        }
    });
}
