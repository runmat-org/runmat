use crate::builtins::common::{gpu_helpers, test_support};
use runmat_accelerate_api::{GpuHandleProvenance, HostLogicalTensorView, ProviderPrecision};
use runmat_value::Value;

#[test]
fn logical_trigonometry_uses_admitted_double_host_semantics_and_exact_owner() {
    test_support::with_test_provider(|provider| {
        for name in ["sin", "cos", "tan"] {
            let input = runmat_accelerate_api::upload_logical_protecting(
                provider,
                &HostLogicalTensorView {
                    data: &[0, 1],
                    shape: &[2, 1],
                },
                &[],
            )
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
            for extensions in [false, true] {
                let _policy = crate::compatibility::push_runmat_extensions_enabled(extensions);
                let result = futures::executor::block_on(crate::call_builtin_async(
                    name,
                    &[Value::GpuTensor(input.clone())],
                ));
                if !extensions {
                    assert!(result.unwrap_err().message().contains("compatibility"));
                    continue;
                }
                let result = result.unwrap();
                let Value::GpuTensor(output) = result.value() else {
                    panic!("resident Double expected");
                };
                assert_eq!(output.device_id(), input.device_id());
                assert_eq!(
                    runmat_accelerate_api::handle_precision(output),
                    Some(ProviderPrecision::F64)
                );
                assert!(!runmat_accelerate_api::handle_is_logical(output));
                let Value::Tensor(host) =
                    futures::executor::block_on(gpu_helpers::gather_value_async(result.value()))
                        .unwrap()
                else {
                    panic!("Double tensor expected");
                };
                assert_eq!(host.shape(), vec![2, 1]);
                let expected = match name {
                    "sin" => vec![0.0, 1.0f64.sin()],
                    "cos" => vec![1.0, 1.0f64.cos()],
                    "tan" => vec![0.0, 1.0f64.tan()],
                    _ => unreachable!(),
                };
                assert_eq!(host.materialize_f64(), expected);
            }
            input.allocation_lease().validate_live().unwrap();
        }
    });
}
