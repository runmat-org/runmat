use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

pub(in crate::builtins::acceleration::gpu::arrayfun) fn maybe_upload_uniform(
    value: Value,
    resident_inputs: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let Some(prototype) = resident_inputs.first() else {
        return Ok(value);
    };
    match value {
        value @ (Value::Tensor(_) | Value::LogicalArray(_) | Value::ComplexTensor(_)) => {
            let protected = resident_inputs.iter().collect::<Vec<_>>();
            gpu_helpers::restore_class_preserving_value_protecting(
                prototype, value, "arrayfun", &protected,
            )
        }
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use runmat_value::{LogicalArray, Tensor};

    #[test]
    fn logical_restoration_uses_a_logical_producer_and_preserves_input() {
        test_support::with_test_provider(|provider| {
            let source = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
            )
            .unwrap();
            for (data, shape) in [(vec![1, 0], vec![2, 1]), (vec![], vec![0, 3])] {
                let logical = LogicalArray::new(data, shape.clone()).unwrap();
                let output = maybe_upload_uniform(
                    Value::LogicalArray(logical),
                    std::slice::from_ref(&source),
                )
                .unwrap();
                let Value::GpuTensor(output) = output else {
                    unreachable!()
                };
                assert!(runmat_accelerate_api::handle_is_logical(&output));
                assert!(
                    gpu_helpers::exact_provider_for_handle(&output).is_some_and(|owner| {
                        runmat_accelerate_api::same_provider_instance(owner, provider)
                    })
                );
                assert_eq!(output.shape(), shape);
                assert!(!output.shares_allocation_with(&source));
                drop(output);
                source.allocation_lease().validate_live().unwrap();
            }
        });
    }
}
