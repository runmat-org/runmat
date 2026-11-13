use super::*;
use crate::builtins::common::test_support::builtin_arguments;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use chrono::NaiveDate;
use std::rc::Rc;

struct FiniteCalendar {
    ledger: ManagedHostEffectLedger,
}

impl RuntimeParallelService for FiniteCalendar {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.ledger.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.ledger.clone())
    }
}

fn fixture() -> (RuntimeContext, ManagedHostEffectLedger) {
    let ledger = ManagedHostEffectLedger::new(1_048_576);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(FiniteCalendar {
            ledger: ledger.clone(),
        })),
    );
    (runtime, ledger)
}

fn serial(year: i32, month: u32, day: u32) -> f64 {
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    super::super::super::datenum_from_naive(date)
}

fn run(runtime: &RuntimeContext, values: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(super::super::super::legacy_builtins::datenum_builtin(
        Some(runtime.clone()),
        builtin_arguments(values),
    ))
}

fn expect_retained(
    runtime: &RuntimeContext,
    ledger: &ManagedHostEffectLedger,
    values: Vec<Value>,
    shape: &[usize],
    expected: &[f64],
) {
    let before = ledger.snapshot().free_bytes();
    let result = run(runtime, values).unwrap();
    let Value::Tensor(tensor) = &result else {
        panic!("non-scalar managed datenum output must be a tensor")
    };
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
    assert_eq!(tensor.as_f64_slice(), Some(expected));
    let quote = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, shape).unwrap();
    assert_eq!(before - ledger.snapshot().free_bytes(), quote);
    let alias = result.clone();
    drop(result);
    assert_eq!(before - ledger.snapshot().free_bytes(), quote);
    drop(alias);
    assert_eq!(before, ledger.snapshot().free_bytes());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn managed_datenum_keeps_native_shape_and_exact_output_claim() {
    let (runtime, ledger) = fixture();
    let a = serial(2024, 1, 2);
    let b = serial(2025, 1, 3);
    expect_retained(
        &runtime,
        &ledger,
        vec![Value::Tensor(Tensor::new(vec![a, b], vec![1, 2]).unwrap())],
        &[1, 2],
        &[a, b],
    );
    // MATLAB-style date vectors are column-major and accept three columns.
    let datevec = Tensor::new(vec![2024.0, 2025.0, 1.0, 1.0, 2.0, 3.0], vec![2, 3]).unwrap();
    expect_retained(
        &runtime,
        &ledger,
        vec![Value::Tensor(datevec)],
        &[2, 1],
        &[a, b],
    );
    expect_retained(
        &runtime,
        &ledger,
        vec![
            Value::Tensor(Tensor::new(vec![2024.0, 2025.0], vec![2, 1]).unwrap()),
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap()),
        ],
        &[2, 1],
        &[a, b],
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn managed_datenum_parses_borrowed_string_and_one_admitted_character_row() {
    let (runtime, ledger) = fixture();
    let a = serial(2024, 1, 2);
    let b = serial(2025, 1, 3);
    let strings =
        runmat_value::StringArray::new(vec!["2024-01-02".into(), "2025-01-03".into()], vec![2, 1])
            .unwrap();
    expect_retained(
        &runtime,
        &ledger,
        vec![Value::StringArray(strings)],
        &[2, 1],
        &[a, b],
    );
    let rows: Vec<char> = "2024-01-02  2025-01-03  ".chars().collect();
    let characters = runmat_value::CharArray::from_unicode_scalars(rows, 2, 12).unwrap();
    expect_retained(
        &runtime,
        &ledger,
        vec![Value::CharArray(characters)],
        &[2, 1],
        &[a, b],
    );
}
