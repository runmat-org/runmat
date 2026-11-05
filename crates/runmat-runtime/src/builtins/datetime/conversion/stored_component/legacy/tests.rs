use super::*;
use crate::builtins::common::test_support::builtin_arguments;
use crate::builtins::datetime::component_builtins::{
    day_builtin, hour_builtin, minute_builtin, month_builtin, year_builtin,
};
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use chrono::NaiveDate;
use runmat_value::{NumericDType, Tensor};
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

fn serial(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> f64 {
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .unwrap()
        // A large serial Double quantizes subsecond time. Use an interior
        // minute, not a boundary whose representation may lie just below it.
        .and_hms_opt(hour, minute, 30)
        .unwrap();
    crate::builtins::datetime::datenum_from_naive(date)
}

fn numeric(result: Value, shape: &[usize], expected: &[f64]) -> Value {
    let Value::Tensor(tensor) = &result else {
        panic!("array component must retain tensor");
    };
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
    assert_eq!(tensor.as_f64_slice(), Some(expected));
    result
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_numeric_components_keep_native_array_geometry_and_retained_owner() {
    let (runtime, ledger) = fixture();
    let before = ledger.snapshot().free_bytes();
    let input = Value::Tensor(
        Tensor::new(
            vec![serial(2024, 1, 2, 3, 4), serial(2025, 2, 3, 5, 6)],
            vec![2, 1],
        )
        .unwrap(),
    );
    let cases = [
        futures::executor::block_on(year_builtin(
            Some(runtime.clone()),
            input.clone(),
            builtin_arguments(Vec::new()),
        ))
        .unwrap(),
        futures::executor::block_on(month_builtin(
            Some(runtime.clone()),
            input.clone(),
            builtin_arguments(Vec::new()),
        ))
        .unwrap(),
        futures::executor::block_on(day_builtin(
            Some(runtime.clone()),
            input.clone(),
            builtin_arguments(Vec::new()),
        ))
        .unwrap(),
        futures::executor::block_on(hour_builtin(
            Some(runtime.clone()),
            input.clone(),
            builtin_arguments(Vec::new()),
        ))
        .unwrap(),
        futures::executor::block_on(minute_builtin(
            Some(runtime.clone()),
            input,
            builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    ];
    let expected = [
        [2024.0, 2025.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 5.0],
        [4.0, 6.0],
    ];
    let retained: Vec<_> = cases
        .into_iter()
        .zip(expected)
        .map(|(value, expected)| numeric(value, &[2, 1], &expected))
        .collect();
    let quote = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[2, 1]).unwrap();
    assert_eq!(before - ledger.snapshot().free_bytes(), quote * 5);
    drop(retained);
    assert_eq!(before, ledger.snapshot().free_bytes());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_component_text_uses_borrowed_strings_or_one_character_row_scratch() {
    let (runtime, ledger) = fixture();
    let before = ledger.snapshot().free_bytes();
    let strings =
        runmat_value::StringArray::new(vec!["2024-01-02".into(), "2025-02-03".into()], vec![2, 1])
            .unwrap();
    let years = futures::executor::block_on(year_builtin(
        Some(runtime.clone()),
        Value::StringArray(strings),
        builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let years = numeric(years, &[2, 1], &[2024.0, 2025.0]);
    drop(years);
    let rows: Vec<char> = "2024-01-02  2025-02-03  ".chars().collect();
    let chars = runmat_value::CharArray::from_unicode_scalars(rows, 2, 12).unwrap();
    let months = futures::executor::block_on(month_builtin(
        Some(runtime.clone()),
        Value::CharArray(chars),
        builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let months = numeric(months, &[2, 1], &[1.0, 2.0]);
    drop(months);
    let formatted =
        runmat_value::StringArray::new(vec!["02/01/2024".into(), "03/02/2025".into()], vec![2, 1])
            .unwrap();
    let days = futures::executor::block_on(day_builtin(
        Some(runtime.clone()),
        Value::StringArray(formatted),
        builtin_arguments(vec![Value::String("dd/mm/yyyy".into())]),
    ))
    .unwrap();
    let days = numeric(days, &[2, 1], &[2.0, 3.0]);
    drop(days);
    let minutes = futures::executor::block_on(minute_builtin(
        Some(runtime.clone()),
        Value::StringArray(
            runmat_value::StringArray::new(
                vec!["2024-03-14 09:26".into(), "2024-03-14 09:26:00".into()],
                vec![2, 1],
            )
            .unwrap(),
        ),
        builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let minutes = numeric(minutes, &[2, 1], &[26.0, 26.0]);
    drop(minutes);
    assert_eq!(before, ledger.snapshot().free_bytes());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn exact_format_rewrite_matches_existing_legacy_tokens() {
    for source in ["yyyy-mm-dd", "dd-mmm-yyyy HH:MM:ss.fff", "mmmm yyyy"] {
        let mut scratch = vec![0_u8; source.len()];
        let len = convert_legacy_format(source, &mut scratch);
        assert_eq!(
            std::str::from_utf8(&scratch[..len]).unwrap(),
            crate::builtins::datetime::text_parsing::legacy_date_format_to_strftime(source),
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_named_components_publish_exact_retained_cell_children() {
    let (runtime, ledger) = fixture();
    let before = ledger.snapshot().free_bytes();
    let stored = crate::builtins::datetime::datetime_object_from_serials(
        None,
        vec![serial(2024, 1, 2, 3, 4), serial(2025, 2, 3, 5, 6)],
        vec![2, 1],
        "yyyy-MM-dd",
    )
    .unwrap();
    let months = futures::executor::block_on(month_builtin(
        Some(runtime.clone()),
        stored.clone(),
        builtin_arguments(vec![Value::from("name")]),
    ))
    .unwrap();
    let Value::Cell(months_cell) = &months else {
        panic!("month names must be cells");
    };
    assert_eq!(months_cell.shape(), &[2, 1]);
    let month_names: Vec<_> = months_cell
        .data()
        .iter()
        .map(|value| {
            let Value::CharArray(chars) = value else {
                panic!("name child is a CharArray");
            };
            runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture")
        })
        .collect();
    assert_eq!(month_names, ["January", "February"]);
    let days = futures::executor::block_on(day_builtin(
        Some(runtime.clone()),
        stored,
        builtin_arguments(vec![Value::from("shortname")]),
    ))
    .unwrap();
    let Value::Cell(days_cell) = &days else {
        panic!("weekday names must be cells");
    };
    assert_eq!(days_cell.shape(), &[2, 1]);
    let day_names: Vec<_> = days_cell
        .data()
        .iter()
        .map(|value| {
            let Value::CharArray(chars) = value else {
                panic!("name child is a CharArray");
            };
            runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture")
        })
        .collect();
    assert_eq!(day_names, ["Tue", "Mon"]);
    assert!(ledger.snapshot().free_bytes() < before);
    let alias = months.clone();
    drop(months);
    assert!(ledger.snapshot().free_bytes() < before);
    drop((alias, days));
    assert_eq!(ledger.snapshot().free_bytes(), before);
}
