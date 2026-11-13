use super::*;
use crate::graph::PrimitiveOp;
use runmat_accelerate_api::ProviderPrecision;

pub(super) fn power_plan() -> FusionGroupPlan {
    let mut plan = sum_plan();
    plan.inputs.push(3);
    plan.operations.insert(
        0,
        FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 3],
            output: Some(4),
        },
    );
    let FusionOp::Builtin { inputs, .. } = &mut plan.operations[1] else {
        unreachable!()
    };
    inputs[0] = 4;
    plan.reduction_data = Some(4);
    plan.group.span.end = plan.group.span.start + plan.operations.len();
    plan
}

#[test]
fn wider_mixed_power_reduction_declines_before_physical_queries_or_effects() {
    runmat_gc::gc_test_context(|| {
        for exponent in [
            Value::Num(3.0),
            Value::Num(f64::from_bits(2.0_f64.to_bits() + 1)),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![2, 1]).unwrap()),
        ] {
            let provider = TypedCallerProvider::registered(1 << 20);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let source = power_plan();
            let invocation = FusionExecutionRequest {
                plan: &source,
                inputs: vec![
                    Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                    exponent,
                ],
                placement: None,
                runtime: None,
            };
            let result = futures::executor::block_on(execute_reduction_with_shape_placed_async(
                invocation,
                2,
                1,
                &[1, 1],
                8,
            ))
            .unwrap();
            assert!(matches!(result, FusionExecutionOutcome::SharedRuntime(_)));
            assert_eq!(provider.configs.load(Ordering::SeqCst), 0);
            assert_eq!(provider.feasibility.load(Ordering::SeqCst), 0);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn projected_power_routes_share_typed_preparation_and_publication_lifecycle() {
    runmat_gc::gc_test_context(|| {
        for (owner, synchronous, strategy, exponent) in [
            (
                ProviderPrecision::F32,
                true,
                ReductionStrategy::SinglePass,
                0.0_f64,
            ),
            (
                ProviderPrecision::F32,
                false,
                ReductionStrategy::TwoPass,
                2.0,
            ),
            (
                ProviderPrecision::F64,
                true,
                ReductionStrategy::TwoPass,
                1.0,
            ),
            (
                ProviderPrecision::F64,
                false,
                ReductionStrategy::SinglePass,
                2.0,
            ),
        ] {
            let provider = TypedCallerProvider::registered_with_precision(1 << 20, owner);
            *provider.strategies.lock().unwrap() = vec![strategy];
            let _selected = ThreadProviderGuard::set(Some(provider));
            let mut source = power_plan();
            if owner == ProviderPrecision::F32 {
                source
                    .const_values
                    .insert(5, Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()));
                let FusionOp::Builtin {
                    name,
                    origin,
                    inputs,
                    ..
                } = &mut source.operations[1]
                else {
                    unreachable!()
                };
                *name = "min".into();
                *origin = FusionCallOrigin::named("min");
                *inputs = vec![4, 5, 1];
                source.reduction_flavor = None;
            }
            let request = FusionExecutionRequest {
                plan: &source,
                inputs: vec![
                    Value::Tensor(Tensor::from_f32(vec![1.25, -2.0], vec![2, 1]).unwrap()),
                    Value::Num(exponent),
                ],
                placement: None,
                runtime: None,
            };
            let output = if synchronous {
                execute_reduction_with_shape_placed(request, 2, 1, &[1, 1], 8)
            } else {
                futures::executor::block_on(execute_reduction_with_shape_placed_async(
                    request,
                    2,
                    1,
                    &[1, 1],
                    8,
                ))
            }
            .unwrap();
            let FusionExecutionOutcome::Executed(Value::GpuTensor(output)) = output else {
                panic!("projected typed route must execute")
            };
            assert_eq!(output.shape(), &[1, 1]);
            assert_eq!(output.physical().element_type(), NumericElementType::F32);
            assert_eq!(
                provider.uploads.load(Ordering::SeqCst),
                1,
                "original Double control must not upload"
            );
            assert_eq!(
                *provider.uploaded_elements.lock().unwrap(),
                vec![NumericElementType::F32]
            );
            assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 0);
            assert_eq!(*provider.submitted_strategy.lock().unwrap(), Some(strategy));
            let transformed = [1.25_f64, -2.0]
                .into_iter()
                .map(|base| f64::from(base.powf(exponent) as f32))
                .collect::<Vec<_>>();
            let expected = if owner == ProviderPrecision::F32 {
                transformed[0].min(transformed[1])
            } else {
                transformed.iter().sum::<f64>()
            } as f32;
            let downloaded =
                futures::executor::block_on(provider.download_numeric(&output)).unwrap();
            assert_eq!(downloaded.data, HostNumericDataOwned::F32(vec![expected]));
            assert_eq!(provider.probes.lock().unwrap()[0].retirements(), 1);
            drop(output);
            assert_eq!(provider.probes.lock().unwrap()[1].retirements(), 1);
        }
    });
}
