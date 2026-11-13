use super::plan::{request_plan, tensor};
use super::provider::Provider;
use crate::accel::fusion::{execute_fusion_reduction, StackSliceGuard};
use futures::executor::block_on;
use runmat_accelerate::fusion_exec::{FusionExecutionOutcome, FusionExecutionRequest};
use runmat_accelerate::ReductionAxes;
use runmat_accelerate_api::{AccelProvider, HostTensorOwned, ThreadProviderGuard};
use runmat_value::Value;
use std::sync::atomic::Ordering;

#[test]
fn forced_reduction_respects_provider_geometry_and_keeps_cached_plan_unchanged() {
    runmat_gc::gc_test_context(|| {
        let provider = Provider::fixture();
        let _scope = ThreadProviderGuard::set(Some(provider));
        for (shape, axes, add, expected_shape, expected) in [
            (
                vec![2, 3],
                ReductionAxes::Explicit(vec![1]),
                false,
                vec![1, 3],
                vec![3., 7., 11.],
            ),
            (
                vec![2, 3],
                ReductionAxes::Explicit(vec![2]),
                false,
                vec![2, 1],
                vec![9., 12.],
            ),
            (
                vec![1, 3],
                ReductionAxes::Unspecified,
                false,
                vec![1, 1],
                vec![6.],
            ),
            (
                vec![1, 3],
                ReductionAxes::Explicit(vec![2]),
                false,
                vec![1, 1],
                vec![6.],
            ),
            (
                vec![2, 2, 2],
                ReductionAxes::All,
                false,
                vec![1, 1],
                vec![36.],
            ),
            (
                vec![2, 3],
                ReductionAxes::Explicit(vec![1]),
                true,
                vec![1, 3],
                vec![6., 14., 22.],
            ),
        ] {
            let plan = request_plan(&shape, axes, add);
            let original_dimension = plan.reduction_dim;
            let original_constants = plan.const_values.clone();
            let input = tensor(&shape);
            let inputs = if add {
                vec![input.clone(), input]
            } else {
                vec![input]
            };
            let mut stack =
                crate::interpreter::stack::InterpreterStack::from_unbounded(inputs.clone());
            let guard = StackSliceGuard::new(&mut stack, 0)
                .expect("the unbounded test stack has a unique operand slice");
            let before = provider.reductions.load(Ordering::SeqCst);
            let output = block_on(execute_fusion_reduction(
                &plan,
                FusionExecutionRequest {
                    plan: &plan,
                    inputs: inputs.clone(),
                    placement: None,
                    runtime: None,
                },
                guard,
                None,
            ))
            .unwrap();
            // This provider exposes only the matrix-shaped legacy hook, not a
            // checked typed recipe. N-D inputs must retain their operands for
            // canonical runtime execution rather than enter that raw hook.
            if shape.len() > 2 {
                assert!(matches!(output, FusionExecutionOutcome::SharedRuntime(_)));
                assert_eq!(stack.as_slice(), inputs.as_slice());
                assert_eq!(provider.reductions.load(Ordering::SeqCst), before);
                assert_eq!(plan.reduction_dim, original_dimension);
                assert_eq!(plan.const_values, original_constants);
                continue;
            }
            let FusionExecutionOutcome::Executed(result) = output else {
                panic!("supported direct route must execute")
            };
            assert!(result.ordered_stores.is_empty());
            assert!(result.completion.is_none());
            let Value::GpuTensor(output) = result.final_value else {
                panic!("supported direct route must return a resident result")
            };
            let HostTensorOwned { shape, data, .. } = block_on(provider.download(&output)).unwrap();
            assert_eq!(shape, expected_shape);
            assert_eq!(data, expected);
            assert_eq!(provider.reductions.load(Ordering::SeqCst), before + 1);
            assert!(stack.is_empty());
            assert_eq!(plan.reduction_dim, original_dimension);
            assert_eq!(plan.const_values, original_constants);
        }
    });
}

#[test]
fn forced_invalid_geometry_restores_stack_without_provider_queries_or_uploads() {
    runmat_gc::gc_test_context(|| {
        let provider = Provider::fixture();
        let _scope = ThreadProviderGuard::set(Some(provider));
        for (shape, axes) in [
            (vec![0, 3], ReductionAxes::All),
            (vec![3, 0], ReductionAxes::Explicit(vec![2])),
            (vec![2, 2, 2], ReductionAxes::Explicit(vec![1, 2])),
            (vec![2, 2, 2], ReductionAxes::Explicit(vec![4])),
            (vec![2, 3], ReductionAxes::Explicit(vec![0])),
            (vec![2, 3], ReductionAxes::Explicit(vec![3])),
        ] {
            let plan = request_plan(&shape, axes, false);
            let inputs = vec![tensor(&shape)];
            let mut stack =
                crate::interpreter::stack::InterpreterStack::from_unbounded(inputs.clone());
            let guard = StackSliceGuard::new(&mut stack, 0)
                .expect("the unbounded test stack has a unique operand slice");
            assert!(block_on(execute_fusion_reduction(
                &plan,
                FusionExecutionRequest {
                    plan: &plan,
                    inputs: inputs.clone(),
                    placement: None,
                    runtime: None
                },
                guard,
                None,
            ))
            .is_err());
            assert_eq!(stack.as_slice(), inputs.as_slice());
        }
        let plan = request_plan(&[2, 3], ReductionAxes::All, true);
        let inputs = vec![tensor(&[2, 1]), tensor(&[1, 3])];
        let mut stack = crate::interpreter::stack::InterpreterStack::from_unbounded(inputs.clone());
        let guard = StackSliceGuard::new(&mut stack, 0)
            .expect("the unbounded test stack has a unique operand slice");
        assert!(block_on(execute_fusion_reduction(
            &plan,
            FusionExecutionRequest {
                plan: &plan,
                inputs: inputs.clone(),
                placement: None,
                runtime: None
            },
            guard,
            None,
        ))
        .is_err());
        assert_eq!(stack.as_slice(), inputs.as_slice());
        for dimension in [1.5, f64::NAN, f64::INFINITY] {
            let mut plan = request_plan(&[2, 3], ReductionAxes::Unspecified, false);
            let id = plan.inputs.iter().copied().max().unwrap() + 100;
            plan.reduction_dim = Some(id);
            plan.const_values.insert(id, Value::Num(dimension));
            let inputs = vec![tensor(&[2, 3])];
            let mut stack =
                crate::interpreter::stack::InterpreterStack::from_unbounded(inputs.clone());
            let guard = StackSliceGuard::new(&mut stack, 0)
                .expect("the unbounded test stack has a unique operand slice");
            assert!(block_on(execute_fusion_reduction(
                &plan,
                FusionExecutionRequest {
                    plan: &plan,
                    inputs: inputs.clone(),
                    placement: None,
                    runtime: None,
                },
                guard,
                None,
            ))
            .is_err());
            assert_eq!(stack.as_slice(), inputs.as_slice());
        }
        assert_eq!(provider.queries.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.reductions.load(Ordering::SeqCst), 0);
    });
}
