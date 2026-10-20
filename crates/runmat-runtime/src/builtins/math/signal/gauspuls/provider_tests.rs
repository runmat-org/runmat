use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;

#[test]
fn resident_samples_gather_to_host_for_each_requested_component() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = provider
            .upload(&HostTensorView {
                data: &[0.0, 0.00025],
                shape: &[1, 2],
            })
            .unwrap();
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = block_on(gauspuls_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let Value::OutputList(outputs) = result else {
            panic!("components")
        };
        assert_eq!(outputs.len(), 3);
        for output in outputs.try_into_owned().expect("unique output list") {
            let Value::Tensor(tensor) = output else {
                panic!("host result")
            };
            assert_eq!(tensor.shape(), vec![1, 2]);
        }
        assert!(
            block_on(provider.download(&handle)).is_ok(),
            "borrowed source remains live"
        );
    });
}
