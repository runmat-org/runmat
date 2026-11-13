use super::*;
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use runmat_execution::ResourceByteBound;
use runmat_value::{IntegerStorage, NumericDType, Tensor, Value};
use std::rc::Rc;

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn prepare(
    runtime: &RuntimeContext,
    state: &Value,
    drift: &Value,
    scale: &Value,
    steps: &Value,
    args: &[Value],
) -> Option<StochasticPreparedSource> {
    prepare_stochastic_source(
        runtime,
        StochasticSourceControls {
            state,
            steps: ValueControlRef::Borrowed(steps),
            drift: ValueControlRef::Borrowed(drift),
            scale: ValueControlRef::Borrowed(scale),
            randn_arg_count: args.len(),
            randn_arg: &|index| args.get(index).map(ValueControlRef::Borrowed),
        },
    )
    .unwrap()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_host_preserves_source_order_and_native_precision() {
    for single in [false, true] {
        let runtime = runtime();
        let (state, drift, scale, class) = if single {
            (
                Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap(),
                Value::Tensor(Tensor::from_f32(vec![0.1], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
                "single",
            )
        } else {
            (
                Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
                Value::Num(0.1),
                Value::Num(0.0),
                "double",
            )
        };
        let mut state = Value::Tensor(state);
        let args = [
            Value::Num(2.0),
            Value::Num(1.0),
            Value::String(class.into()),
        ];
        let prepared = prepare(&runtime, &state, &drift, &scale, &Value::Num(3.0), &args)
            .expect("eligible host source");
        let mut phase = 0;
        let mut publish = |checkpoint| {
            match (phase % 3, checkpoint) {
                (0, StochasticCheckpoint::IterationBinding(counter)) => {
                    assert_eq!(counter, f64::from(phase / 3 + 1));
                }
                (1, StochasticCheckpoint::Noise(Value::Tensor(noise))) => {
                    assert_eq!(noise.shape(), &[2, 1]);
                    assert_eq!(
                        noise.numeric_dtype(),
                        if single {
                            NumericDType::F32
                        } else {
                            NumericDType::F64
                        }
                    );
                }
                (2, StochasticCheckpoint::State(value)) => state = value,
                _ => panic!("source checkpoint out of order"),
            }
            phase += 1;
            Ok(())
        };
        let result = futures::executor::block_on(execute_stochastic_source(
            &runtime,
            prepared,
            StochasticSourcePublication {
                host_workspace: ResourceByteBound::Known(0),
                checkpoint: &mut publish,
            },
        ))
        .unwrap();
        assert_eq!(result, StochasticSourceExecution::Completed);
        assert_eq!(phase, 9);
        let Value::Tensor(state) = state else {
            panic!("array result")
        };
        assert_eq!(state.shape(), &[2, 1]);
        if single {
            let factor = 0.1_f32.exp();
            assert_eq!(
                state.as_f32_slice().unwrap(),
                &[factor * factor * factor, 2.0 * factor * factor * factor]
            );
        } else {
            for (actual, expected) in state
                .as_f64_slice()
                .unwrap()
                .iter()
                .zip([0.3_f64.exp(), 2.0 * 0.3_f64.exp()])
            {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_state_and_empty_or_nonpositive_source_loops_decline_before_effects() {
    let runtime = runtime();
    let integer =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I64(vec![3]), vec![1, 1]).unwrap());
    let empty =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I8(Vec::new()), vec![0, 1]).unwrap());
    let args = [Value::Num(1.0), Value::Num(1.0)];
    for (state, steps) in [(&integer, 1.0), (&integer, 0.0), (&empty, 3.0)] {
        assert!(prepare(
            &runtime,
            state,
            &Value::Num(0.1),
            &Value::Num(0.2),
            &Value::Num(steps),
            &args
        )
        .is_none());
    }
    let Value::Tensor(integer) = integer else {
        unreachable!()
    };
    assert_eq!(
        integer.integer_storage(),
        Some(&IntegerStorage::I64(vec![3]))
    );
    let Value::Tensor(empty) = empty else {
        unreachable!()
    };
    assert!(empty.is_empty());
}
