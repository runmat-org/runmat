use super::*;
use crate::builtins::common::test_support::builtin_arguments;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use chrono::NaiveDate;
use runmat_value::{IntegerStorage, Tensor};
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
    crate::builtins::datetime::datenum_from_naive(
        NaiveDate::from_ymd_opt(year, month, day)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_dateshift_preserves_array_shape_format_and_retained_object() {
    let (runtime, ledger) = fixture();
    let source = crate::builtins::datetime::datetime_object_from_serials(
        None,
        vec![serial(2024, 2, 10), serial(2024, 2, 11)],
        vec![2, 1],
        "yyyy-MM-dd",
    )
    .unwrap();
    let before = ledger.snapshot().free_bytes();
    let result = futures::executor::block_on(
        crate::builtins::datetime::dateshift_builtin::dateshift_builtin(
            Some(runtime.clone()),
            source,
            Value::from("start"),
            Value::from("month"),
            builtin_arguments(Vec::new()),
        ),
    )
    .unwrap();
    let serials = crate::builtins::datetime::serials_from_datetime_value(&result).unwrap();
    assert_eq!(serials.shape(), &[2, 1]);
    assert_eq!(
        serials.as_f64_slice(),
        Some(&[serial(2024, 2, 1), serial(2024, 2, 1)][..])
    );
    assert_eq!(
        crate::builtins::datetime::datetime_format_from_value(&result).unwrap(),
        "yyyy-MM-dd"
    );
    assert!(ledger.snapshot().free_bytes() < before);
    let alias = result.clone();
    drop((result, serials));
    assert!(ledger.snapshot().free_bytes() < before);
    drop(alias);
    assert_eq!(ledger.snapshot().free_bytes(), before);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_dateshift_borrows_exact_weekdays_rules_and_empty_geometry() {
    let (runtime, ledger) = fixture();
    let source = crate::builtins::datetime::datetime_object_from_serials(
        None,
        vec![serial(2024, 2, 10)],
        vec![1, 1],
        "yyyy-MM-dd",
    )
    .unwrap();
    let weekdays =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I64(vec![2, 3]), vec![2, 1]).unwrap());
    let before = ledger.snapshot().free_bytes();
    let result = futures::executor::block_on(
        crate::builtins::datetime::dateshift_builtin::dateshift_builtin(
            Some(runtime.clone()),
            source.clone(),
            Value::from("dayofweek"),
            weekdays,
            builtin_arguments(Vec::new()),
        ),
    )
    .unwrap();
    let serials = crate::builtins::datetime::serials_from_datetime_value(&result).unwrap();
    assert_eq!(serials.shape(), &[2, 1]);
    assert_eq!(
        serials.as_f64_slice(),
        Some(&[serial(2024, 2, 12), serial(2024, 2, 13)][..])
    );
    drop((result, serials));
    assert_eq!(ledger.snapshot().free_bytes(), before);

    let empty_rules =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I64(Vec::new()), vec![0, 3]).unwrap());
    let empty = futures::executor::block_on(
        crate::builtins::datetime::dateshift_builtin::dateshift_builtin(
            Some(runtime),
            source,
            Value::from("start"),
            Value::from("month"),
            builtin_arguments(vec![empty_rules]),
        ),
    )
    .unwrap();
    assert_eq!(
        crate::builtins::datetime::serials_from_datetime_value(&empty)
            .unwrap()
            .shape(),
        &[0, 3]
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_dateshift_validates_all_controls_before_empty_output_admission() {
    let (runtime, ledger) = fixture();
    let source = crate::builtins::datetime::datetime_object_from_serials(
        None,
        Vec::new(),
        vec![0, 1],
        "yyyy-MM-dd",
    )
    .unwrap();
    let before = ledger.snapshot().free_bytes();
    let invalid_rule = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
    );
    let rule_error = futures::executor::block_on(
        crate::builtins::datetime::dateshift_builtin::dateshift_builtin(
            Some(runtime.clone()),
            source.clone(),
            Value::from("dayofweek"),
            Value::Num(0.0),
            builtin_arguments(vec![invalid_rule]),
        ),
    )
    .unwrap_err();
    assert!(rule_error.message().contains("rule"));
    assert_eq!(rule_error.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(ledger.snapshot().free_bytes(), before);

    let target_error = futures::executor::block_on(
        crate::builtins::datetime::dateshift_builtin::dateshift_builtin(
            Some(runtime),
            source,
            Value::from("dayofweek"),
            Value::Num(0.0),
            builtin_arguments(Vec::new()),
        ),
    )
    .unwrap_err();
    assert!(target_error.message().contains("weekday"));
    assert_eq!(target_error.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(ledger.snapshot().free_bytes(), before);
}
