use super::capabilities::{
    DATETIME_CLASS, DEFAULT_DATETIME_FORMAT, DEFAULT_DATE_FORMAT, FORMAT_FIELD,
    MAX_DATESHIFT_DAY_OCCURRENCE, SECONDS_PER_DAY,
};
use super::*;
use super::{
    business_day_builtins::*, calendar_duration_builtins::*, component_builtins::*,
    dateshift_builtin::*, legacy_builtins::*, operators::*,
};
use runmat_builtins::{BuiltinIntegerInputAvailability, DATESHIFT_INTEGER_INPUTS};

#[path = "tests/second_dateshift.rs"]
mod second_dateshift;

#[path = "tests/cooperation.rs"]
mod cooperation;

async fn datetime_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::operators::datetime_subsref(None, obj, subscript).await
}

async fn datetime_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::operators::datetime_subsasgn(None, obj, subscript, rhs).await
}

async fn datetime_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    super::constructor_builtin::datetime_builtin_source(None, args).await
}

#[test]
fn owned_clock_drives_current_relative_and_timezone_calendar_consumers() {
    use crate::builtins::common::test_support::builtin_arguments;
    use crate::context::{DeterministicTimingService, RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::{cancellation::CancellationSource, host::NativeClockInputs};
    use std::rc::Rc;
    let cancellation = CancellationSource::new();
    let clock = Rc::new(
        DeterministicTimingService::new(
            NativeClockInputs {
                schema_version: runmat_execution::host::NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
                initial_monotonic_nanos: 0,
                initial_wall_unix_seconds: 1_700_000_000,
                initial_wall_nanosecond: 0,
                read_step_nanos: 1_000_000_000,
                local_offset_seconds: 19_800,
                process_cpu: None,
            },
            cancellation.token(),
        )
        .unwrap(),
    );
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_timing(clock));
    let expected = NaiveDate::from_ymd_opt(2023, 11, 15)
        .unwrap()
        .and_hms_opt(3, 43, 20)
        .unwrap();
    assert_eq!(
        now_builtin(Some(runtime.clone())).unwrap(),
        Value::Num(datenum_from_naive(expected))
    );
    let Value::Tensor(vector) =
        futures::executor::block_on(datevec_builtin(Some(runtime.clone()), Value::from("now")))
            .unwrap()
    else {
        panic!("expected calendar vector")
    };
    let components = vector.materialize_f64();
    assert_eq!(&components[..5], &[2023.0, 11.0, 15.0, 3.0, 43.0]);
    assert!((components[5] - 21.0).abs() < 0.0001);
    let object = futures::executor::block_on(super::constructor_builtin::datetime_builtin_source(
        Some(runtime.clone()),
        vec![Value::from("2023-11-14T22:13:20Z")],
    ))
    .unwrap();
    assert_eq!(
        serials_from_datetime_value(&object)
            .unwrap()
            .materialize_f64(),
        vec![datenum_from_naive(expected)]
    );
    assert_eq!(
        current_naive_local(Some(&runtime)).unwrap(),
        expected + Duration::seconds(2),
        "timezone interpretation must not advance current time"
    );
    for (text, seconds, nanos) in [
        ("1969-12-31T23:59:00Z", -60, 0),
        ("2016-12-31T23:59:60.5Z", 1_483_228_799, 1_500_000_000),
    ] {
        let instant =
            crate::context::RuntimeCalendarInstant::from_unix_timestamp(seconds, nanos).unwrap();
        let local = instant.at_fixed_offset(19_800).unwrap();
        let object =
            futures::executor::block_on(super::constructor_builtin::datetime_builtin_source(
                Some(runtime.clone()),
                vec![Value::from(text)],
            ))
            .expect("signed and leap RFC3339 values retain owned calendar conversion");
        assert_eq!(
            serials_from_datetime_value(&object)
                .unwrap()
                .materialize_f64(),
            vec![datenum_from_naive(local)]
        );
    }
    let midnight = expected.date().and_hms_opt(0, 0, 0).unwrap();
    let serial = datenum_from_naive(midnight);
    assert_eq!(
        today_builtin(Some(runtime.clone()), builtin_arguments(vec![])).unwrap(),
        Value::Num(serial)
    );
    assert_eq!(
        today_builtin(
            Some(runtime.clone()),
            builtin_arguments(vec![Value::from("datenum")])
        )
        .unwrap(),
        Value::Num(serial)
    );
    let datetime = today_builtin(
        Some(runtime.clone()),
        builtin_arguments(vec![Value::CharArray(CharArray::new_row("datetime"))]),
    )
    .unwrap();
    assert!(is_datetime_object(&datetime));
    assert_eq!(
        serials_from_datetime_value(&datetime)
            .unwrap()
            .materialize_f64(),
        vec![serial]
    );
    assert_eq!(
        current_naive_local(Some(&runtime)).unwrap(),
        expected + Duration::seconds(6),
        "each today form consumes exactly one calendar read"
    );
    for options in [
        vec![Value::from("unsupported")],
        vec![Value::Num(1.0)],
        vec![Value::from("datenum"), Value::from("datetime")],
    ] {
        assert!(today_builtin(Some(runtime.clone()), builtin_arguments(options)).is_err());
    }
    assert_eq!(
        current_naive_local(Some(&runtime)).unwrap(),
        expected + Duration::seconds(7),
        "invalid today options are rejected before reading the clock"
    );
    assert!(
        current_naive_local(None).is_err(),
        "context-free embedding must not read ambient time"
    );
}

#[test]
fn unadmitted_selected_clock_rejects_current_and_business_relative_text() {
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    let runtime = RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().without_timing());
    let error = now_builtin(Some(runtime.clone())).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Clock:ServiceUnavailable"));
    let error = futures::executor::block_on(isbusday_builtin(
        Some(runtime),
        Value::from("today"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Clock:ServiceUnavailable"));
}

fn run_datetime(args: Vec<Value>) -> Value {
    futures::executor::block_on(datetime_builtin(args)).expect("datetime")
}

fn unregistered_resident(shape: Vec<usize>) -> runmat_accelerate_api::test_support::TestAllocation {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            shape,
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    allocation
}

fn as_datetime(value: Value) -> ObjectInstance {
    match value {
        Value::Object(object) => object,
        other => panic!("expected datetime object, got {other:?}"),
    }
}

fn serial_for_date(year: i32, month: u32, day: u32) -> f64 {
    datenum_from_naive(midnight(NaiveDate::from_ymd_opt(year, month, day).unwrap()))
}

fn integer_tensor(storage: runmat_value::IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

#[test]
fn datetime_descriptor_signatures_cover_constructor_and_methods() {
    let labels: Vec<&str> = runmat_builtins::DATETIME_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"t = datetime()"));
    assert!(labels.contains(&"t = datetime(dateVectors)"));
    assert!(labels.contains(&"t = datetime(year, month, day, hour, minute, second)"));
    assert!(labels.contains(&"t = datetime(year, month, day, hour, minute, second, millisecond)"));
    assert!(labels.contains(&"t = datetime(serialDateNumbers, \"ConvertFrom\", \"datenum\")"));

    assert_eq!(DATETIME_YEAR_DESCRIPTOR.signatures[0].label, "X = year(t)");
    assert!(DATETIME_HOUR_DESCRIPTOR
        .signatures
        .iter()
        .any(|signature| signature.label == "X = hour(t, F)"));
    assert_eq!(
        DATETIME_SUBSREF_DESCRIPTOR.signatures[0].label,
        "out = datetime.subsref(obj, S)"
    );
    for operation in [
        runmat_builtins::DatetimeBinaryOperation::Eq,
        runmat_builtins::DatetimeBinaryOperation::Ne,
        runmat_builtins::DatetimeBinaryOperation::Lt,
        runmat_builtins::DatetimeBinaryOperation::Le,
        runmat_builtins::DatetimeBinaryOperation::Gt,
        runmat_builtins::DatetimeBinaryOperation::Ge,
        runmat_builtins::DatetimeBinaryOperation::Plus,
        runmat_builtins::DatetimeBinaryOperation::Minus,
    ] {
        assert_eq!(
            operation.catalog_entry().descriptor.signatures[0].label,
            format!("out = datetime.{}(lhs, rhs)", operation.method())
        );
    }
}

#[test]
fn hour_supports_datetime_legacy_serial_and_formatted_text_shapes() {
    for text in [
        "2000-01-02 08:00",
        "2000/01/02 08:00",
        "02-Jan-2000 08:00",
        "01/02/2000 08:00",
    ] {
        assert_eq!(
            futures::executor::block_on(hour_builtin(
                None,
                Value::from(text),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
            Value::Num(8.0),
        );
    }
    assert_eq!(
        futures::executor::block_on(minute_builtin(
            None,
            Value::from("2024-03-14 09:26"),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
        Value::Num(26.0),
    );
    let datetime = run_datetime(vec![Value::from("2024-03-14 09:26:53")]);
    assert_eq!(
        futures::executor::block_on(hour_builtin(
            None,
            datetime,
            crate::builtins::common::test_support::builtin_arguments(Vec::new())
        ))
        .unwrap(),
        Value::Num(9.0)
    );

    let serials = Tensor::new(
        vec![
            serial_for_date(2024, 3, 14) + 3.0 / 24.0,
            serial_for_date(2024, 3, 14) + 17.0 / 24.0,
        ],
        vec![1, 2],
    )
    .unwrap();
    let result = futures::executor::block_on(hour_builtin(
        None,
        Value::Tensor(serials),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::Tensor(result) = result else {
        panic!("expected shaped hour result");
    };
    assert_eq!(result.shape(), [1, 2]);
    assert_eq!(result.materialize_f64(), vec![3.0, 17.0]);

    let formatted = futures::executor::block_on(hour_builtin(
        None,
        Value::from("14/03/2024 21:05:00"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
            "dd/MM/yyyy HH:mm:ss",
        )]),
    ))
    .unwrap();
    assert_eq!(formatted, Value::Num(21.0));

    let documented = futures::executor::block_on(hour_builtin(
        None,
        Value::from("2024/14/03 09:26:53.125"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
            "yyyy/dd/mm hh:MM:ss.fff",
        )]),
    ))
    .expect("documented datestr format language");
    assert_eq!(documented, Value::Num(9.0));

    let default_fractional = futures::executor::block_on(hour_builtin(
        None,
        Value::from("2024/03/14 17:26:53.125"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("year-first fractional legacy text");
    assert_eq!(default_fractional, Value::Num(17.0));
}

#[test]
fn hour_gates_uncertain_typed_and_resident_legacy_extensions_before_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let typed = futures::executor::block_on(hour_builtin(
        None,
        integer_tensor(runmat_value::IntegerStorage::U32(vec![739_000]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("typed legacy serial must be gated");
    assert_eq!(
        typed.identifier(),
        Some("RunMat:compatibility:HourTypedLegacySerialExtension")
    );

    let allocation = unregistered_resident(vec![1, 1]);
    let resident = Value::GpuTensor(allocation.handle.clone());
    let resident = futures::executor::block_on(hour_builtin(
        None,
        resident,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("resident legacy input must gate before provider access");
    assert_eq!(
        resident.identifier(),
        Some("RunMat:compatibility:DatetimeGpuInputExtension")
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn hour_minute_and_month_typed_legacy_serials_cover_every_integer_class() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let serial = 42;
    let cases = [
        runmat_value::IntegerStorage::I8(vec![serial as i8]),
        runmat_value::IntegerStorage::I16(vec![serial as i16]),
        runmat_value::IntegerStorage::I32(vec![serial]),
        runmat_value::IntegerStorage::I64(vec![i64::from(serial)]),
        runmat_value::IntegerStorage::U8(vec![serial as u8]),
        runmat_value::IntegerStorage::U16(vec![serial as u16]),
        runmat_value::IntegerStorage::U32(vec![serial as u32]),
        runmat_value::IntegerStorage::U64(vec![serial as u64]),
    ];
    let expected_hour = futures::executor::block_on(hour_builtin(
        None,
        Value::Num(f64::from(serial)),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("double legacy hour");
    let expected_minute = futures::executor::block_on(minute_builtin(
        None,
        Value::Num(f64::from(serial)),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("double legacy minute");
    let expected_month = futures::executor::block_on(month_builtin(
        None,
        Value::Num(f64::from(serial)),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("double legacy month");

    for storage in cases {
        let value = integer_tensor(storage.clone(), vec![1, 1]);
        assert_eq!(
            futures::executor::block_on(hour_builtin(
                None,
                value.clone(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .expect("typed legacy hour"),
            expected_hour
        );
        assert_eq!(
            futures::executor::block_on(minute_builtin(
                None,
                value.clone(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .expect("typed legacy minute"),
            expected_minute
        );
        assert_eq!(
            futures::executor::block_on(month_builtin(
                None,
                value,
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .expect("typed legacy month"),
            expected_month
        );
    }
}

#[test]
fn minute_and_month_gate_typed_and_resident_extensions_before_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for (result, identifier) in [
        (
            futures::executor::block_on(minute_builtin(
                None,
                integer_tensor(runmat_value::IntegerStorage::U16(vec![42]), vec![1, 1]),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
            "RunMat:compatibility:MinuteTypedLegacySerialExtension",
        ),
        (
            futures::executor::block_on(month_builtin(
                None,
                integer_tensor(runmat_value::IntegerStorage::U16(vec![42]), vec![1, 1]),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
            "RunMat:compatibility:MonthTypedLegacySerialExtension",
        ),
    ] {
        assert_eq!(
            result
                .expect_err("typed legacy input must gate")
                .identifier(),
            Some(identifier)
        );
    }

    let allocation = unregistered_resident(vec![1, 1]);
    for result in [
        futures::executor::block_on(minute_builtin(
            None,
            Value::GpuTensor(allocation.handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )),
        futures::executor::block_on(month_builtin(
            None,
            Value::GpuTensor(allocation.handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )),
    ] {
        assert_eq!(
            result
                .expect_err("resident legacy input must gate before provider access")
                .identifier(),
            Some("RunMat:compatibility:DatetimeGpuInputExtension")
        );
    }
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn month_datetime_name_modes_return_shaped_character_cells() {
    let datetime = run_datetime(vec![Value::from("2024-03-14 09:26:53")]);
    for (mode, expected) in [("name", "March"), ("shortname", "Mar")] {
        let options = crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Bool(false),
            Value::from(mode),
        ])
        .into_range(1..)
        .expect("retained logical mode range");
        let result = futures::executor::block_on(month_builtin(None, datetime.clone(), options))
            .expect("month name mode");
        let Value::Cell(cell) = result else {
            panic!("expected month name cell");
        };
        assert_eq!(cell.shape, vec![1, 1]);
        assert_eq!(
            cell.data(),
            vec![Value::CharArray(CharArray::new_row(expected))]
        );
    }
}

#[test]
fn datetime_builds_from_components() {
    let value = run_datetime(vec![Value::Num(2024.0), Value::Num(3.0), Value::Num(14.0)]);
    let object = as_datetime(value);
    assert!(object.class_name().is(DATETIME_CLASS));
    assert_eq!(format_for_object(&object).unwrap(), DEFAULT_DATE_FORMAT);
    let serials = serial_tensor_for_object(&object).expect("serials");
    assert_eq!(serials.materialize_f64().len(), 1);
    let year = futures::executor::block_on(year_builtin(
        None,
        Value::Object(object.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("year");
    assert_eq!(year, Value::Num(2024.0));
}

#[test]
fn year_typed_legacy_serial_is_separately_gated() {
    let serial = serial_for_date(2024, 1, 1) as i64;
    let input = Value::Int(runmat_value::IntValue::I64(serial));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(year_builtin(
        None,
        input.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("strict gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:YearTypedLegacySerialExtension")
    );
    drop(_strict);

    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = futures::executor::block_on(year_builtin(
        None,
        input,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("typed legacy serial");
    assert_eq!(result, Value::Num(2024.0));
}

#[test]
fn datetime_builds_arrays_from_component_vectors() {
    let years = Value::Tensor(Tensor::new(vec![2024.0, 2025.0], vec![1, 2]).unwrap());
    let months = Value::Tensor(Tensor::new(vec![1.0, 6.0], vec![1, 2]).unwrap());
    let days = Value::Tensor(Tensor::new(vec![15.0, 20.0], vec![1, 2]).unwrap());
    let value = run_datetime(vec![years, months, days]);
    let object = as_datetime(value.clone());
    let serials = serial_tensor_for_object(&object).expect("serials");
    assert_eq!(serials.shape(), [1, 2]);
    let rendered = datetime_display_text(&value)
        .expect("display")
        .expect("datetime text");
    assert!(rendered.contains("15-Jan-2024"));
    assert!(rendered.contains("20-Jun-2025"));
}

#[test]
fn datetime_typed_integer_components_and_serials_cross_double_boundary_exactly() {
    let years = integer_tensor(
        runmat_value::IntegerStorage::U16(vec![2024, 2025]),
        vec![1, 2],
    );
    let months = integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 6]), vec![1, 2]);
    let days = integer_tensor(runmat_value::IntegerStorage::I16(vec![15, 20]), vec![1, 2]);
    let value = run_datetime(vec![years, months, days]);
    let rendered = datetime_display_text(&value)
        .expect("display")
        .expect("datetime text");
    assert!(rendered.contains("15-Jan-2024"));
    assert!(rendered.contains("20-Jun-2025"));

    let serial = serial_for_date(2024, 3, 14);
    let object = run_datetime(vec![
        integer_tensor(
            runmat_value::IntegerStorage::U32(vec![serial as u32]),
            vec![1, 1],
        ),
        Value::from("ConvertFrom"),
        Value::from("datenum"),
    ]);
    assert_eq!(
        serials_from_datetime_value(&object)
            .unwrap()
            .materialize_f64(),
        vec![serial.floor()]
    );
}

#[test]
fn datetime_date_vectors_cover_all_integer_classes_and_preaccess_gates() {
    let storages = [
        runmat_value::IntegerStorage::I8(vec![24, 1, 2]),
        runmat_value::IntegerStorage::I16(vec![2024, 1, 2]),
        runmat_value::IntegerStorage::I32(vec![2024, 1, 2]),
        runmat_value::IntegerStorage::I64(vec![2024, 1, 2]),
        runmat_value::IntegerStorage::U8(vec![24, 1, 2]),
        runmat_value::IntegerStorage::U16(vec![2024, 1, 2]),
        runmat_value::IntegerStorage::U32(vec![2024, 1, 2]),
        runmat_value::IntegerStorage::U64(vec![2024, 1, 2]),
    ];
    for storage in storages {
        let small_year = matches!(
            storage,
            runmat_value::IntegerStorage::I8(_) | runmat_value::IntegerStorage::U8(_)
        );
        let value = run_datetime(vec![integer_tensor(storage, vec![1, 3])]);
        assert_eq!(
            datetime_string_array(&value).unwrap().unwrap().data(),
            vec![if small_year {
                "02-Jan-0024"
            } else {
                "02-Jan-2024"
            }
            .to_string()]
        );
    }

    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let logical = futures::executor::block_on(datetime_builtin(vec![Value::Bool(true)]))
        .expect_err("logical input is gated");
    assert_eq!(
        logical.identifier(),
        Some("RunMat:compatibility:DatetimeLogicalInputExtension")
    );
    let implicit_serial =
        futures::executor::block_on(datetime_builtin(vec![Value::Num(739_000.0)]))
            .expect_err("implicit serial input is gated");
    assert_eq!(
        implicit_serial.identifier(),
        Some("RunMat:compatibility:DatetimeImplicitDatenumExtension")
    );
    let legacy_arity = futures::executor::block_on(datetime_builtin(vec![
        Value::Num(2024.0),
        Value::Num(1.0),
        Value::Num(2.0),
        Value::Num(3.0),
    ]))
    .expect_err("four-component constructor is gated");
    assert_eq!(
        legacy_arity.identifier(),
        Some("RunMat:compatibility:DatetimeLegacyComponentArityExtension")
    );
    let allocation = unregistered_resident(vec![1, 3]);
    let resident = Value::GpuTensor(allocation.handle.clone());
    let resident = futures::executor::block_on(datetime_builtin(vec![resident]))
        .expect_err("resident input is gated before provider access");
    assert_eq!(
        resident.identifier(),
        Some("RunMat:compatibility:DatetimeGpuInputExtension")
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn datetime_invalid_options_precede_resident_provider_access() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = unregistered_resident(vec![1, 3]);
    let resident = Value::GpuTensor(allocation.handle.clone());
    for (arguments, expected) in [
        (
            vec![
                resident.clone(),
                Value::from("ConvertFrom"),
                Value::from("posixtime"),
            ],
            "unsupported ConvertFrom",
        ),
        (
            vec![
                resident.clone(),
                Value::from("TimeZone"),
                Value::from("UTC"),
            ],
            "unsupported option",
        ),
        (
            vec![
                resident.clone(),
                Value::from("InputFormat"),
                Value::from("yyyy-MM-dd"),
            ],
            "InputFormat requires exactly one text input",
        ),
        (
            vec![resident.clone(), Value::Num(1.0)],
            "unsupported argument pattern",
        ),
    ] {
        let error = futures::executor::block_on(datetime_builtin(arguments))
            .expect_err("invalid constructor must reject before provider access");
        assert!(error.message().contains(expected), "{error:?}");
        assert_eq!(
            error.identifier(),
            runmat_builtins::DATETIME_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(allocation.probe.retirements(), 0);
    }
    drop(resident);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn datetime_parses_cell_array_of_character_vectors_with_its_shape() {
    let dates = ["2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12"]
        .into_iter()
        .map(|text| Value::CharArray(CharArray::new_row(text)))
        .collect();
    let input = Value::Cell(runmat_value::CellArray::new(dates, 2, 2).expect("cellstr"));
    let result = futures::executor::block_on(datetime_builtin(vec![
        input,
        Value::from("InputFormat"),
        Value::from("yyyy-MM-dd"),
    ]))
    .expect("cellstr datetime");
    let object = as_datetime(result);
    assert_eq!(serial_tensor_for_object(&object).unwrap().shape(), [2, 2]);
    let serials = serial_tensor_for_object(&object).unwrap().materialize_f64();
    assert_eq!(serials[0], serial_for_date(2024, 4, 9));
    assert_eq!(serials[3], serial_for_date(2024, 4, 12));
}

#[test]
fn datetime_preserves_empty_cellstr_geometry() {
    for shape in [vec![0, 2], vec![2, 0], vec![2, 0, 3]] {
        let input = Value::Cell(
            runmat_value::CellArray::new_with_shape(Vec::new(), shape.clone())
                .expect("empty cellstr"),
        );
        let result = futures::executor::block_on(datetime_builtin(vec![input]))
            .expect("empty cellstr datetime");
        let object = as_datetime(result);
        assert_eq!(serial_tensor_for_object(&object).unwrap().shape(), shape);
    }
}

#[test]
fn datetime_cellstr_input_format_preserves_unicode_whitespace_trim_boundaries() {
    let text = format!(
        "{}2024-04-09{}",
        "\u{2003}".repeat(1025),
        "\u{00a0}".repeat(1025)
    );
    let input = Value::Cell(
        runmat_value::CellArray::new(vec![Value::CharArray(CharArray::new_row(&text))], 1, 1)
            .unwrap(),
    );
    let object = as_datetime(
        futures::executor::block_on(datetime_builtin(vec![
            input,
            Value::from("InputFormat"),
            Value::from("yyyy-MM-dd"),
        ]))
        .expect("Unicode whitespace is trimmed before date parsing"),
    );
    assert_eq!(
        serial_tensor_for_object(&object).unwrap().materialize_f64(),
        vec![serial_for_date(2024, 4, 9)]
    );
}

#[test]
fn datetime_rejects_noncharacter_and_paged_cell_elements_before_resident_access() {
    let allocation = unregistered_resident(vec![1, 1]);
    let paged = CharArray::from_unicode_scalars_with_shape(vec!['x'; 20], vec![1, 10, 2])
        .expect("paged char");
    for invalid in [
        Value::Num(1.0),
        Value::Cell(runmat_value::CellArray::new(Vec::new(), 0, 0).unwrap()),
        Value::CharArray(paged),
        Value::GpuTensor(allocation.handle.clone()),
    ] {
        let input = Value::Cell(
            runmat_value::CellArray::new(
                vec![Value::CharArray(CharArray::new_row("2024-04-09")), invalid],
                2,
                1,
            )
            .unwrap(),
        );
        let error = futures::executor::block_on(datetime_builtin(vec![
            input,
            Value::from("InputFormat"),
            Value::from("yyyy-MM-dd"),
        ]))
        .expect_err("cell elements must be character vectors");
        assert_eq!(
            error.identifier(),
            runmat_builtins::DATETIME_ERROR_INVALID_ARGUMENT.identifier
        );
        assert!(error.message().contains("cell element 2"));
        assert_eq!(allocation.probe.retirements(), 0);
    }
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn datetime_constructor_value_errors_use_catalog_invalid_input_identifier() {
    let error = futures::executor::block_on(datetime_builtin(vec![
        Value::Num(2024.5),
        Value::Num(2.0),
        Value::Num(30.0),
    ]))
    .expect_err("fractional year must fail; overflowing calendar days normalize");
    assert_eq!(
        error.identifier(),
        runmat_builtins::DATETIME_ERROR_INVALID_INPUT.identifier
    );
}

#[test]
fn wide_integer_serials_are_rejected_before_lossy_conversion() {
    for storage in [
        runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
        runmat_value::IntegerStorage::I64(vec![i64::MIN]),
    ] {
        let explicit = futures::executor::block_on(datetime_builtin(vec![
            integer_tensor(storage.clone(), vec![1, 1]),
            Value::from("ConvertFrom"),
            Value::from("datenum"),
        ]))
        .expect_err("wide explicit serial must be rejected while exact");
        assert!(explicit.message().contains("supported serial-date range"));

        let legacy_day = futures::executor::block_on(day_builtin(
            None,
            integer_tensor(storage, vec![1, 1]),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("wide legacy day serial must be rejected while exact");
        assert!(legacy_day.message().contains("supported serial-date range"));
    }

    let extreme =
        naive_from_datenum(f64::MAX).expect_err("extreme finite serial must not wrap or panic");
    assert!(extreme.message().contains("outside the supported range"));
}

#[test]
fn datetime_parses_text_and_converts_to_strings() {
    let value = run_datetime(vec![Value::String(
        "2024-03-14 09:26:53".to_string().into(),
    )]);
    let rendered = datetime_string_array(&value)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(rendered.data(), vec!["14-Mar-2024 09:26:53".to_string()]);
}

#[test]
fn datetime_missing_serial_renders_as_nat() {
    let value = datetime_object_from_serial_tensor(
        None,
        Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap(),
        DEFAULT_DATETIME_FORMAT,
    )
    .expect("datetime object");
    let rendered = datetime_string_array(&value)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(rendered.data(), vec!["NaT".to_string()]);
    assert_eq!(
        datetime_display_text(&value).expect("display"),
        Some("NaT".to_string())
    );
}

#[test]
fn datetime_accepts_existing_datetime_input() {
    let value = run_datetime(vec![Value::String("2024-03-14".to_string().into())]);
    let converted = run_datetime(vec![
        value.clone(),
        Value::from("Format"),
        Value::from("yyyy-MM-dd"),
    ]);
    assert_eq!(
        serials_from_datetime_value(&converted)
            .unwrap()
            .materialize_f64(),
        serials_from_datetime_value(&value)
            .unwrap()
            .materialize_f64()
    );
}

#[test]
fn datetime_parses_text_with_input_format() {
    let input = Value::StringArray(
        StringArray::new(
            vec!["2024/03/14".to_string(), "2024/03/15".to_string()],
            vec![2, 1],
        )
        .unwrap(),
    );
    let value = run_datetime(vec![
        input,
        Value::from("InputFormat"),
        Value::from("yyyy/MM/dd"),
        Value::from("Format"),
        Value::from("yyyy-MM-dd"),
    ]);
    let rendered = datetime_string_array(&value)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(
        rendered.data(),
        vec!["2024-03-14".to_string(), "2024-03-15".to_string()]
    );
}

#[test]
fn dateshift_supports_sunday_start_of_week_and_public_month_end() {
    let input = run_datetime(vec![
        Value::StringArray(
            StringArray::new(
                vec!["2024-03-14".to_string(), "2024-03-18".to_string()],
                vec![2, 1],
            )
            .unwrap(),
        ),
        Value::from("Format"),
        Value::from("yyyy-MM-dd"),
    ]);
    let shifted = futures::executor::block_on(dateshift_builtin(
        None,
        input,
        Value::from("start"),
        Value::from("week"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("dateshift start week");
    let rendered = datetime_string_array(&shifted)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(
        rendered.data(),
        vec!["2024-03-10".to_string(), "2024-03-17".to_string()]
    );

    let month_end = futures::executor::block_on(dateshift_builtin(
        None,
        run_datetime(vec![
            Value::from("2024-02-10"),
            Value::from("Format"),
            Value::from("yyyy-MM-dd HH:mm:ss"),
        ]),
        Value::from("end"),
        Value::from("month"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("dateshift end month");
    let rendered = datetime_string_array(&month_end)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(rendered.data(), vec!["2024-02-29 00:00:00".to_string()]);
}

#[test]
fn dateshift_rules_follow_current_week_and_boundary_contracts() {
    let input = run_datetime(vec![Value::from("2024-03-14")]);
    let monday = futures::executor::block_on(dateshift_builtin(
        None,
        input.clone(),
        Value::from("dayofweek"),
        Value::from("monday"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("current")]),
    ))
    .expect("current-week Monday");
    assert_eq!(
        datetime_string_array(&monday).unwrap().unwrap().data(),
        vec!["11-Mar-2024".to_string()]
    );

    let current_zero = futures::executor::block_on(dateshift_builtin(
        None,
        input.clone(),
        Value::from("dayofweek"),
        Value::from("monday"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
            runmat_value::IntValue::I8(0),
        )]),
    ))
    .expect("numeric zero current-week Monday");
    assert_eq!(
        datetime_string_array(&current_zero)
            .unwrap()
            .unwrap()
            .data(),
        vec!["11-Mar-2024".to_string()]
    );

    let next = futures::executor::block_on(dateshift_builtin(
        None,
        run_datetime(vec![Value::from("2024-03-01")]),
        Value::from("start"),
        Value::from("month"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("next")]),
    ))
    .expect("next exact boundary");
    assert_eq!(
        datetime_string_array(&next).unwrap().unwrap().data(),
        vec!["01-Apr-2024".to_string()]
    );

    for weekday in [
        runmat_value::IntValue::I8(2),
        runmat_value::IntValue::I16(2),
        runmat_value::IntValue::I32(2),
        runmat_value::IntValue::I64(2),
        runmat_value::IntValue::U8(2),
        runmat_value::IntValue::U16(2),
        runmat_value::IntValue::U32(2),
        runmat_value::IntValue::U64(2),
    ] {
        let shifted = futures::executor::block_on(dateshift_builtin(
            None,
            input.clone(),
            Value::from("dayofweek"),
            Value::Int(weekday),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("typed weekday");
        assert_eq!(
            datetime_string_array(&shifted).unwrap().unwrap().data(),
            vec!["18-Mar-2024".to_string()]
        );
    }
}

#[test]
fn dateshift_day_occurrences_use_bounded_calendar_arithmetic() {
    let origin = NaiveDate::from_ymd_opt(2024, 3, 14)
        .unwrap()
        .and_hms_opt(12, 30, 0)
        .unwrap();

    let sixth_weekday = shift_day_target(origin, DayTarget::Weekday, DateShiftRule::Occurrence(6))
        .expect("sixth weekday");
    assert_eq!(
        sixth_weekday.date(),
        NaiveDate::from_ymd_opt(2024, 3, 21).unwrap()
    );

    let third_prior_weekend =
        shift_day_target(origin, DayTarget::Weekend, DateShiftRule::Occurrence(-3))
            .expect("third prior weekend day");
    assert_eq!(
        third_prior_weekend.date(),
        NaiveDate::from_ymd_opt(2024, 3, 3).unwrap()
    );

    assert!(shift_day_target(
        origin,
        DayTarget::Exact(Weekday::Mon),
        DateShiftRule::Occurrence(MAX_DATESHIFT_DAY_OCCURRENCE as i64 + 1),
    )
    .is_err());
    assert!(shift_day_target(
        origin,
        DayTarget::Exact(Weekday::Mon),
        DateShiftRule::Occurrence(i64::MIN),
    )
    .is_err());
}

#[test]
fn dateshift_float_integer_conversion_uses_half_open_i64_bounds() {
    const TWO_TO_63: f64 = 9_223_372_036_854_775_808.0;

    assert!(exact_integer_values(&Value::Num(TWO_TO_63), "rule").is_err());
    assert_eq!(
        exact_integer_values(&Value::Num(-TWO_TO_63), "rule")
            .expect("exact i64 minimum")
            .0,
        vec![i64::MIN]
    );

    let values = Value::Tensor(Tensor::new(vec![1.0, TWO_TO_63], vec![1, 2]).unwrap());
    assert!(exact_integer_values(&values, "rule").is_err());
}

#[test]
fn dateshift_boundary_helpers_report_chrono_limits_without_panicking() {
    let maximum = midnight(NaiveDate::MAX);
    assert!(next_unit_start(maximum, DateShiftUnit::Day).is_err());
    assert!(unit_end(maximum, DateShiftUnit::Day).is_err());

    let maximum_year_start = midnight(
        NaiveDate::from_ymd_opt(NaiveDate::MAX.year(), 1, 1)
            .expect("start of Chrono's maximum year"),
    );
    assert!(next_unit_start(maximum_year_start, DateShiftUnit::Year).is_err());
    assert!(unit_end(maximum_year_start, DateShiftUnit::Year).is_err());

    let origin = NaiveDate::from_ymd_opt(2024, 3, 14)
        .unwrap()
        .and_hms_opt(12, 30, 0)
        .unwrap();
    for unit in [
        DateShiftUnit::Week,
        DateShiftUnit::Day,
        DateShiftUnit::Hour,
        DateShiftUnit::Minute,
        DateShiftUnit::Second,
    ] {
        assert!(unit_step(origin, unit, i64::MAX).is_err());
        assert!(unit_step(origin, unit, i64::MIN).is_err());
    }

    let minimum = midnight(NaiveDate::MIN);
    let current = minimum.weekday().num_days_from_monday() as i64;
    let previous_weekday = [
        Weekday::Mon,
        Weekday::Tue,
        Weekday::Wed,
        Weekday::Thu,
        Weekday::Fri,
        Weekday::Sat,
        Weekday::Sun,
    ]
    .into_iter()
    .find(|weekday| (current - weekday.num_days_from_monday() as i64).rem_euclid(7) == 1)
    .expect("a weekday immediately precedes the minimum date");
    assert!(start_of_week(minimum, previous_weekday).is_err());
}

#[test]
fn dateshift_capability_separates_public_form_from_typed_coverage_evidence() {
    let input = &DATESHIFT_INTEGER_INPUTS[0];
    assert_eq!(
        input.availability,
        BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(input.classes.len(), 8);
    assert!(input.notes.contains("without a per-storage-class table"));
    assert!(input.notes.contains("settled compatibility coverage"));
}

#[test]
fn datetime_date_vectors_normalize_and_day_supports_modern_and_legacy_forms() {
    let datetime = run_datetime(vec![integer_tensor(
        runmat_value::IntegerStorage::I64(vec![2024, 13, 1]),
        vec![1, 3],
    )]);
    assert_eq!(
        datetime_string_array(&datetime).unwrap().unwrap().data(),
        vec!["01-Jan-2025".to_string()]
    );
    assert_eq!(
        futures::executor::block_on(day_builtin(
            None,
            datetime.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "dayofyear"
            )]),
        ))
        .unwrap(),
        Value::Num(1.0)
    );
    let names = futures::executor::block_on(day_builtin(
        None,
        datetime,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("shortname")]),
    ))
    .unwrap();
    let Value::Cell(names) = names else {
        panic!("expected cell names")
    };
    let Value::CharArray(name) = &names.data()[0] else {
        panic!("expected char name")
    };
    assert_eq!(
        runmat_value::decode_utf16_code_units(name.data()).unwrap(),
        "Wed"
    );
    assert_eq!(
        futures::executor::block_on(day_builtin(
            None,
            Value::from("2021/28/09"),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "yyyy/dd/mm"
            )]),
        ))
        .unwrap(),
        Value::Num(28.0)
    );
}

#[test]
fn datetime_supports_format_assignment() {
    let value = run_datetime(vec![Value::Num(2024.0), Value::Num(3.0), Value::Num(14.0)]);
    let updated = futures::executor::block_on(datetime_subsasgn(
        value,
        ".".to_string(),
        Value::String(FORMAT_FIELD.to_string().into()),
        Value::String("yyyy-MM-dd".to_string().into()),
    ))
    .expect("subsasgn");
    let rendered = datetime_display_text(&updated)
        .expect("display")
        .expect("datetime text");
    assert_eq!(rendered, "2024-03-14");
}

#[test]
fn datetime_supports_indexing_and_comparison() {
    let years = Value::Tensor(Tensor::new(vec![2024.0, 2025.0], vec![1, 2]).unwrap());
    let months = Value::Tensor(Tensor::new(vec![1.0, 6.0], vec![1, 2]).unwrap());
    let days = Value::Tensor(Tensor::new(vec![15.0, 20.0], vec![1, 2]).unwrap());
    let value = run_datetime(vec![years, months, days]);
    let payload = Value::Cell(runmat_value::CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap());
    let indexed =
        futures::executor::block_on(datetime_subsref(value.clone(), "()".to_string(), payload))
            .expect("subsref");
    let year = indexed
        .with_owned(|value| {
            futures::executor::block_on(year_builtin(
                None,
                value,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
        })
        .expect("year");
    assert_eq!(year, Value::Num(2025.0));

    let lhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(1.0)]);
    let rhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(2.0)]);
    let cmp = futures::executor::block_on(datetime_lt(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![lhs, rhs]),
    ))
    .expect("lt");
    assert_eq!(cmp, Value::Num(1.0));
}

#[test]
fn datetime_typed_integer_index_selectors_are_exact() {
    let years = integer_tensor(
        runmat_value::IntegerStorage::U16(vec![2024, 2025]),
        vec![1, 2],
    );
    let months = integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 6]), vec![1, 2]);
    let days = integer_tensor(runmat_value::IntegerStorage::U8(vec![15, 20]), vec![1, 2]);
    let value = run_datetime(vec![years, months, days]);
    let payload = Value::Cell(
        runmat_value::CellArray::new(
            vec![integer_tensor(
                runmat_value::IntegerStorage::U64(vec![2]),
                vec![1, 1],
            )],
            1,
            1,
        )
        .unwrap(),
    );
    let indexed = futures::executor::block_on(datetime_subsref(value, "()".to_string(), payload))
        .expect("subsref");
    let year = indexed
        .with_owned(|value| {
            futures::executor::block_on(year_builtin(
                None,
                value,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
        })
        .expect("year");
    assert_eq!(year, Value::Num(2025.0));
}

#[test]
fn datetime_and_duration_interoperate() {
    let lhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(1.0)]);
    let rhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(2.0)]);
    let delta = futures::executor::block_on(datetime_minus(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![rhs.clone(), lhs.clone()]),
    ))
    .expect("datetime minus datetime");
    assert_eq!(delta, Value::Num(1.0));

    let duration = crate::builtins::duration::duration_object_from_days_tensor(
        None,
        Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
        crate::builtins::duration::DEFAULT_DURATION_FORMAT,
    )
    .expect("duration");

    let round_trip = futures::executor::block_on(datetime_plus(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            lhs.clone(),
            duration.clone(),
        ]),
    ))
    .expect("plus");
    let round_trip_text = datetime_display_text(&round_trip)
        .expect("datetime display")
        .expect("datetime text");
    assert_eq!(round_trip_text, "02-Jan-2024");

    let restored = futures::executor::block_on(datetime_minus(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![rhs, duration]),
    ))
    .expect("minus duration");
    let restored_text = datetime_display_text(&restored)
        .expect("datetime display")
        .expect("datetime text");
    assert_eq!(restored_text, "01-Jan-2024");
}

#[test]
fn legacy_date_conversion_and_query_helpers_work() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let serial = serial_for_date(2024, 3, 14);
    let date_vector =
        futures::executor::block_on(datevec_builtin(None, Value::Num(serial))).expect("datevec");
    let Value::Tensor(date_vector) = date_vector else {
        panic!("expected datevec tensor");
    };
    assert_eq!(date_vector.shape(), [1, 6]);
    assert_eq!(&date_vector.materialize_f64()[..3], &[2024.0, 3.0, 14.0]);

    let round_trip = futures::executor::block_on(datenum_builtin_source(vec![Value::Tensor(
        date_vector.clone(),
    )]))
    .expect("datenum");
    assert_eq!(round_trip, Value::Num(serial));
    let date_only_round_trip =
        futures::executor::block_on(datenum_builtin_source(vec![Value::Tensor(
            Tensor::new(vec![2024.0, 3.0, 14.0], vec![1, 3]).unwrap(),
        )]))
        .expect("datenum date vector");
    assert_eq!(date_only_round_trip, Value::Num(serial));

    let text = futures::executor::block_on(datestr_builtin_source(
        Value::Num(serial),
        vec![Value::from("yyyy-mm-dd")],
    ))
    .expect("datestr");
    let Value::CharArray(text) = text else {
        panic!("expected datestr char array");
    };
    assert_eq!(
        runmat_value::decode_utf16_code_units(text.data()).unwrap(),
        "2024-03-14"
    );
    let text_from_datevec = futures::executor::block_on(datestr_builtin_source(
        Value::Tensor(Tensor::new(vec![2024.0, 3.0, 14.0], vec![1, 3]).unwrap()),
        vec![Value::from("yyyy-mm-dd")],
    ))
    .expect("datestr date vector");
    let Value::CharArray(text_from_datevec) = text_from_datevec else {
        panic!("expected datestr char array");
    };
    assert_eq!(
        runmat_value::decode_utf16_code_units(text_from_datevec.data()).unwrap(),
        "2024-03-14"
    );

    let january = serial_for_date(2027, 1, 4);
    for (format, expected) in [
        (vec![Value::from("dd-mmm-yyyy")], "04-Jan-2027"),
        (vec![Value::from("yyyy-mm-dd")], "2027-01-04"),
        (vec![Value::Num(1.0)], "04-Jan-2027"),
        (vec![Value::Num(29.0)], "2027-01-04"),
        (vec![Value::from("QQ-YYYY")], "Q1-2027"),
        (vec![Value::Num(27.0)], "Q1-2027"),
        (Vec::new(), "04-Jan-2027"),
    ] {
        let rendered =
            futures::executor::block_on(datestr_builtin_source(Value::Num(january), format))
                .expect("legacy datestr format");
        let Value::CharArray(rendered) = rendered else {
            panic!("expected datestr character row");
        };
        assert_eq!(
            runmat_value::decode_utf16_code_units(rendered.data()).unwrap(),
            expected
        );
    }
    for format in [Value::from("HH:MM PM"), Value::Num(16.0)] {
        let rendered = futures::executor::block_on(datestr_builtin_source(
            Value::Num(january + 0.25),
            vec![format],
        ))
        .expect("space-padded twelve-hour format");
        let Value::CharArray(rendered) = rendered else {
            panic!("expected datestr character row");
        };
        assert_eq!(
            runmat_value::decode_utf16_code_units(rendered.data()).unwrap(),
            " 6:00 AM"
        );
    }
    let midnight_explicit = futures::executor::block_on(datestr_builtin_source(
        Value::Num(january),
        vec![Value::Num(0.0)],
    ))
    .expect("explicit datestr time");
    let Value::CharArray(midnight_explicit) = midnight_explicit else {
        panic!("expected datestr character row");
    };
    assert_eq!(
        runmat_value::decode_utf16_code_units(midnight_explicit.data()).unwrap(),
        "04-Jan-2027 00:00:00"
    );
    let mixed_default = futures::executor::block_on(datestr_builtin_source(
        Value::Tensor(Tensor::new(vec![january, january + 0.5], vec![1, 2]).unwrap()),
        Vec::new(),
    ))
    .expect("one default format for the whole input");
    let Value::CharArray(mixed_default) = mixed_default else {
        panic!("expected datestr character rows");
    };
    assert_eq!(mixed_default.rows, 2);
    let width = mixed_default.cols;
    assert_eq!(
        runmat_value::decode_utf16_code_units(&mixed_default.data()[..width]).unwrap(),
        "04-Jan-2027 00:00:00"
    );
    assert_eq!(
        runmat_value::decode_utf16_code_units(&mixed_default.data()[width..]).unwrap(),
        "04-Jan-2027 12:00:00"
    );
    let subsecond = futures::executor::block_on(datestr_builtin_source(
        Value::Num(january + 0.001 / 86_400.0),
        Vec::new(),
    ))
    .expect("subsecond midnight retains the default time field");
    let Value::CharArray(subsecond) = subsecond else {
        panic!("expected datestr character row");
    };
    assert_eq!(
        runmat_value::decode_utf16_code_units(subsecond.data()).unwrap(),
        "04-Jan-2027 00:00:00"
    );
    for pattern in ["mm-mm-yyyy", "HH PM PM", "QQ-mm-yyyy", "AM"] {
        assert!(
            futures::executor::block_on(datestr_builtin_source(
                Value::Num(january),
                vec![Value::from(pattern)],
            ))
            .is_err(),
            "{pattern}"
        );
    }
    for code in [-2.0, 32.0, 0.5, f64::NAN, f64::INFINITY] {
        assert!(
            futures::executor::block_on(datestr_builtin_source(
                Value::Num(january),
                vec![Value::Num(code)],
            ))
            .is_err(),
            "invalid format code {code}"
        );
    }
    assert!(futures::executor::block_on(datestr_builtin_source(
        Value::Num(january),
        vec![integer_tensor(
            runmat_value::IntegerStorage::I64(vec![1]),
            vec![1, 1]
        )],
    ))
    .is_err());

    let weekday =
        futures::executor::block_on(weekday_builtin_source(Value::Num(serial), Vec::new()))
            .expect("weekday");
    assert_eq!(weekday, Value::Num(5.0));
    assert_eq!(
        futures::executor::block_on(eomday_builtin(Value::Num(2024.0), Value::Num(2.0)))
            .expect("eomday"),
        Value::Num(29.0)
    );
    assert_eq!(
        futures::executor::block_on(etime_builtin(
            Value::Tensor(Tensor::new(vec![2024.0, 1.0, 2.0, 0.0, 0.0, 0.0], vec![1, 6]).unwrap(),),
            Value::Tensor(Tensor::new(vec![2024.0, 1.0, 1.0, 0.0, 0.0, 0.0], vec![1, 6]).unwrap(),),
        ))
        .expect("etime"),
        Value::Num(SECONDS_PER_DAY)
    );
    assert_eq!(
        futures::executor::block_on(isbetween_builtin(
            None,
            Value::Num(serial),
            Value::Num(serial - 1.0),
            Value::Num(serial + 1.0),
        ))
        .expect("isbetween"),
        Value::Bool(true)
    );
}

#[test]
fn legacy_datevec_single_output_is_a_six_column_matrix() {
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let serial = serial_for_date(2024, 3, 14);
    let value = futures::executor::block_on(datevec_builtin(None, Value::Num(serial)))
        .expect("date vector");
    let Value::Tensor(matrix) = value else {
        panic!("single output must be the full date-vector matrix");
    };
    assert_eq!(matrix.shape(), [1, 6]);
    assert_eq!(&matrix.materialize_f64()[..3], &[2024.0, 3.0, 14.0]);
}

#[test]
fn two_by_three_date_vectors_keep_both_rows() {
    let vectors =
        Tensor::new(vec![2024.0, 2024.0, 3.0, 3.0, 14.0, 15.0], vec![2, 3]).expect("date vectors");
    let serials = futures::executor::block_on(datenum_builtin_source(vec![Value::Tensor(vectors)]))
        .expect("two dates");
    let Value::Tensor(serials) = serials else {
        panic!("two input rows must yield two serial dates");
    };
    assert_eq!(serials.shape(), [2, 1]);
    assert_eq!(
        serials.materialize_f64(),
        vec![serial_for_date(2024, 3, 14), serial_for_date(2024, 3, 15)]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn datevec_tensor_preserves_dense_rank_one_row_and_rank_zero_empty_columns() {
    let row = Tensor::new(vec![2024.0, 3.0, 14.0, 0.0, 0.0, 0.0], vec![6]).unwrap();
    let serial = datenum_from_datevec_tensor(&row, "datenum")
        .expect("rank-one six-component date vector is one row");
    assert_eq!(serial.shape(), [1, 1]);
    assert_eq!(serial.materialize_f64(), vec![serial_for_date(2024, 3, 14)]);

    let scalar = Tensor::new(vec![2024.0], Vec::new()).unwrap();
    let error = datenum_from_datevec_tensor(&scalar, "datenum")
        .expect_err("rank-zero date vector has no six-column matrix geometry");
    assert!(error
        .message()
        .contains("date vectors must have six columns"));
}

#[test]
fn legacy_weekday_names_use_short_char_rows_by_default_and_long_on_request() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let serial = serial_for_date(2024, 3, 14);
    for (options, expected) in [
        (Vec::new(), "Thurs"),
        (vec![Value::from("long")], "Thursday"),
        (vec![Value::from("LONG")], "Thursday"),
    ] {
        let result =
            futures::executor::block_on(weekday_builtin_source(Value::Num(serial), options))
                .expect("weekday names");
        let Value::OutputList(outputs) = result else {
            panic!("two outputs expected");
        };
        assert_eq!(outputs[0], Value::Num(5.0));
        let Value::CharArray(names) = &outputs[1] else {
            panic!("day names must be character rows");
        };
        assert_eq!(
            runmat_value::decode_utf16_code_units(names.data()).unwrap(),
            expected
        );
    }
}

#[test]
fn legacy_date_array_shapes_follow_each_operation_contract() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let years = Value::Tensor(Tensor::new(vec![2024.0, 2025.0], vec![2, 1]).unwrap());
    let months = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
    let eomday_error = futures::executor::block_on(eomday_builtin(years, months))
        .expect_err("distinct non-scalar shapes must not broadcast");
    assert!(eomday_error.message().contains("matching sizes"));

    let dates = Value::Tensor(Tensor::new(vec![739_000.0, 739_002.0], vec![2, 1]).unwrap());
    let lower =
        Value::Tensor(Tensor::new(vec![738_999.0, 739_001.0, 739_003.0], vec![1, 3]).unwrap());
    let between =
        futures::executor::block_on(isbetween_builtin(None, dates, lower, Value::Num(739_002.0)))
            .expect("isbetween must implicitly expand compatible shapes");
    let Value::LogicalArray(between) = between else {
        panic!("non-scalar comparison must return a logical array");
    };
    assert_eq!(between.shape, vec![2, 3]);
    assert_eq!(between.data, vec![1, 1, 0, 1, 0, 0]);

    let mismatch = futures::executor::block_on(isbetween_builtin(
        None,
        Value::Tensor(Tensor::new(vec![739_000.0, 739_001.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![738_999.0; 3], vec![3, 1]).unwrap()),
        Value::Num(739_002.0),
    ))
    .expect_err("incompatible dimension sizes must fail");
    assert!(mismatch.message().contains("size mismatch"));

    let empty = futures::executor::block_on(isbetween_builtin(
        None,
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![739_000.0; 3], vec![1, 3]).unwrap()),
        Value::Num(739_002.0),
    ))
    .expect("empty and singleton dimensions are compatible");
    let Value::LogicalArray(empty) = empty else {
        panic!("empty comparison must remain a logical array");
    };
    assert_eq!(empty.shape, vec![0, 3]);
    assert!(empty.data.is_empty());

    let preserved = futures::executor::block_on(isbetween_builtin(
        None,
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 3]).unwrap()),
        Value::Num(739_000.0),
        Value::Num(739_002.0),
    ))
    .expect("empty non-scalar shape must survive scalar bounds");
    let Value::LogicalArray(preserved) = preserved else {
        panic!("empty comparison must remain a logical array");
    };
    assert_eq!(preserved.shape, vec![0, 3]);
}

#[test]
fn legacy_date_extensions_reject_before_provider_access_in_matlab_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let typed = integer_tensor(runmat_value::IntegerStorage::U32(vec![739_000]), vec![1, 1]);
    let typed_error = futures::executor::block_on(datevec_builtin(None, typed))
        .expect_err("typed legacy serial is an extension");
    assert_eq!(
        typed_error.identifier(),
        Some("RunMat:compatibility:LegacyDatetimeTypedNumericExtension")
    );

    let partial = futures::executor::block_on(datenum_builtin_source(vec![
        Value::Num(2024.0),
        Value::Num(3.0),
        Value::Num(14.0),
        Value::Num(1.0),
    ]))
    .expect_err("four-component datenum is an extension");
    assert_eq!(
        partial.identifier(),
        Some("RunMat:compatibility:DatenumPartialComponentsExtension")
    );

    let serial = serial_for_date(2024, 3, 14);
    let between = futures::executor::block_on(isbetween_builtin(
        None,
        Value::Num(serial),
        Value::Num(serial - 1.0),
        Value::Num(serial + 1.0),
    ))
    .expect_err("numeric isbetween is an extension");
    assert_eq!(
        between.identifier(),
        Some("RunMat:compatibility:IsbetweenSerialDateExtension")
    );

    let resident = unregistered_resident(vec![1, 1]);
    let error = futures::executor::block_on(weekday_builtin_source(
        Value::GpuTensor(resident.handle.clone()),
        Vec::new(),
    ))
    .expect_err("resident date is an extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DatetimeGpuInputExtension")
    );
    let bound =
        datetime_object_from_serials(None, vec![serial], vec![1, 1], DEFAULT_DATETIME_FORMAT)
            .expect("datetime bound");
    let between_error = futures::executor::block_on(isbetween_builtin(
        None,
        Value::GpuTensor(resident.handle.clone()),
        bound.clone(),
        bound,
    ))
    .expect_err("resident date comparison must gate before provider access");
    assert_eq!(
        between_error.identifier(),
        Some("RunMat:compatibility:DatetimeGpuInputExtension")
    );
    assert_eq!(resident.probe.retirements(), 0);
    resident.handle.release().expect("release owner allocation");
    assert_eq!(resident.probe.retirements(), 1);
}

#[test]
fn legacy_date_output_counts_reject_unsupported_requests() {
    let serial = serial_for_date(2024, 3, 14);
    {
        let _outputs = crate::output_count::push_output_count(Some(7)).expect("output-count scope");
        let err = futures::executor::block_on(datevec_builtin(None, Value::Num(serial)))
            .expect_err("datevec has six components");
        assert!(err.message().contains("at most six outputs"));
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let err =
            futures::executor::block_on(weekday_builtin_source(Value::Num(serial), Vec::new()))
                .expect_err("weekday has two outputs");
        assert!(err.message().contains("at most two outputs"));
    }
}

#[test]
fn datenum_typed_integer_date_vector_reads_exact_storage() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let serial = serial_for_date(2024, 3, 14);
    let typed_date_vector = Tensor::new_integer(
        runmat_value::IntegerStorage::U16(vec![2024, 3, 14]),
        vec![1, 3],
    )
    .expect("typed date vector");
    let typed_round_trip =
        futures::executor::block_on(datenum_builtin_source(vec![Value::Tensor(
            typed_date_vector,
        )]))
        .expect("datenum typed date vector");
    assert_eq!(typed_round_trip, Value::Num(serial));
}

#[test]
fn datenum_typed_integer_serials_read_exact_storage() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let serial = serial_for_date(2024, 3, 14).floor() as u32;
    let scalar = Tensor::new_integer(runmat_value::IntegerStorage::U32(vec![serial]), vec![1, 1])
        .expect("typed serial");
    let scalar_out =
        futures::executor::block_on(datenum_builtin_source(vec![Value::Tensor(scalar)]))
            .expect("datenum typed scalar serial");
    assert_eq!(scalar_out, Value::Num(f64::from(serial)));

    let vector = Tensor::new_integer(
        runmat_value::IntegerStorage::U32(vec![serial, serial + 1]),
        vec![1, 2],
    )
    .expect("typed serial vector");
    let vector_out =
        futures::executor::block_on(datenum_builtin_source(vec![Value::Tensor(vector)]))
            .expect("datenum typed vector serial");
    let Value::Tensor(vector_out) = vector_out else {
        panic!("expected datenum vector tensor");
    };
    assert_eq!(vector_out.shape(), [1, 2]);
    assert_eq!(
        vector_out.materialize_f64(),
        vec![f64::from(serial), f64::from(serial + 1)]
    );
}

#[test]
fn calendar_duration_helpers_and_datetime_arithmetic_work() {
    let one_month =
        futures::executor::block_on(calmonths_builtin(Value::Num(1.0))).expect("calmonths");
    assert_eq!(
        iscalendarduration_builtin(one_month.clone()).expect("predicate"),
        Value::Bool(true)
    );
    assert_eq!(
        futures::executor::block_on(calmonths_builtin(one_month.clone()))
            .expect("calmonths convert"),
        Value::Num(1.0)
    );

    let jan31 = run_datetime(vec![
        Value::from("2024-01-31"),
        Value::from("Format"),
        Value::from("yyyy-MM-dd"),
    ]);
    let shifted = futures::executor::block_on(datetime_plus(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![jan31, one_month]),
    ))
    .expect("plus");
    let rendered = datetime_string_array(&shifted)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(rendered.data(), vec!["2024-02-29".to_string()]);

    let duration = futures::executor::block_on(calendar_duration_builtin_source(
        None,
        vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
    ))
    .expect("calendarDuration");
    let (months, days) = calendar_duration_tensors_from_value(&duration).expect("components");
    assert_eq!(months.materialize_f64(), vec![14.0]);
    assert_eq!(days.materialize_f64(), vec![3.0]);

    assert!(futures::executor::block_on(calyears_builtin(Value::Num(f64::MAX))).is_err());
    assert!(
        futures::executor::block_on(calendar_duration_builtin_source(
            None,
            vec![Value::Num(f64::MAX), Value::Num(f64::MAX), Value::Num(0.0),]
        ))
        .is_err()
    );
}

#[test]
fn business_day_helpers_use_weekends_and_holidays() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let new_year = serial_for_date(2024, 1, 1);
    let friday = serial_for_date(2024, 1, 5);
    let saturday = serial_for_date(2024, 1, 6);
    let mask = futures::executor::block_on(isbusday_builtin_source(
        Value::Tensor(Tensor::new(vec![new_year, friday, saturday], vec![1, 3]).unwrap()),
        Vec::new(),
    ))
    .expect("isbusday");
    let Value::LogicalArray(mask) = mask else {
        panic!("expected isbusday logical array");
    };
    assert_eq!(mask.data, vec![0, 1, 0]);

    assert_eq!(
        futures::executor::block_on(isbusday_builtin_source(
            Value::Num(friday),
            vec![Value::Num(friday)],
        ))
        .expect("custom holiday"),
        Value::Bool(false)
    );

    assert_eq!(
        futures::executor::block_on(isbusday_builtin_source(
            Value::Num(saturday),
            vec![
                Value::Num(f64::NAN),
                Value::Tensor(
                    Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 7]).unwrap()
                )
            ],
        ))
        .expect("Saturday trading with custom weekend"),
        Value::Bool(true)
    );

    let business_days = futures::executor::block_on(busdays_builtin_source(
        Value::Num(friday),
        Value::Num(friday + 3.0),
        Vec::new(),
    ))
    .expect("busdays");
    let Value::Tensor(business_days) = business_days else {
        panic!("expected busdays tensor");
    };
    assert_eq!(business_days.materialize_f64(), vec![friday, friday + 3.0]);

    let monthly = futures::executor::block_on(busdays_builtin_source(
        Value::Num(serial_for_date(2024, 1, 5)),
        Value::Num(serial_for_date(2024, 3, 8)),
        vec![Value::String("monthly".into())],
    ))
    .expect("monthly business dates");
    let Value::Tensor(monthly) = monthly else {
        panic!("expected monthly serial dates");
    };
    assert_eq!(monthly.shape(), [3, 1]);
    assert_eq!(
        monthly.materialize_f64(),
        vec![
            serial_for_date(2024, 1, 31),
            serial_for_date(2024, 2, 29),
            serial_for_date(2024, 3, 28)
        ]
    );

    assert_eq!(
        futures::executor::block_on(days252bus_builtin_source(
            Value::Num(friday),
            Value::Num(friday + 3.0),
            Vec::new(),
        ))
        .expect("days252bus"),
        Value::Num(2.0)
    );
    assert_eq!(
        futures::executor::block_on(daysdif_builtin_source(
            Value::Num(friday),
            Value::Num(friday + 3.0),
            Vec::new(),
        ))
        .expect("daysdif"),
        Value::Num(3.0)
    );
    let typed_basis =
        Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![1]), vec![1, 1]).expect("basis");
    assert_eq!(
        futures::executor::block_on(daysdif_builtin_source(
            Value::Num(serial_for_date(2024, 1, 30)),
            Value::Num(serial_for_date(2024, 2, 29)),
            vec![Value::Tensor(typed_basis)],
        ))
        .expect("daysdif typed basis"),
        Value::Num(29.0)
    );
    let sia_basis = || vec![Value::Num(1.0)];
    assert_eq!(
        futures::executor::block_on(daysdif_builtin_source(
            Value::Num(serial_for_date(2023, 2, 28)),
            Value::Num(serial_for_date(2023, 3, 1)),
            sia_basis(),
        ))
        .expect("SIA last-February start"),
        Value::Num(1.0)
    );
    assert_eq!(
        futures::executor::block_on(daysdif_builtin_source(
            Value::Num(serial_for_date(2023, 2, 28)),
            Value::Num(serial_for_date(2024, 2, 29)),
            sia_basis(),
        ))
        .expect("SIA both dates last February"),
        Value::Num(360.0)
    );
    assert!(futures::executor::block_on(daysdif_builtin_source(
        Value::Num(friday),
        Value::Num(friday + 3.0),
        vec![Value::Num(13.0)],
    ))
    .is_err());
    assert!(futures::executor::block_on(daysdif_builtin_source(
        Value::Num(friday),
        Value::Num(friday + 3.0),
        vec![Value::Num(0.5)],
    ))
    .is_err());
    assert_eq!(
        futures::executor::block_on(fbusdate_builtin_source(
            Value::Num(2024.0),
            Value::Num(1.0),
            Vec::new(),
        ))
        .expect("fbusdate"),
        Value::Num(serial_for_date(2024, 1, 2))
    );
    assert_eq!(
        futures::executor::block_on(lbusdate_builtin_source(
            Value::Num(2024.0),
            Value::Num(6.0),
            Vec::new(),
        ))
        .expect("lbusdate"),
        Value::Num(serial_for_date(2024, 6, 28))
    );
    assert_eq!(
        futures::executor::block_on(isbusday_builtin_source(
            Value::Num(serial_for_date(2021, 12, 31)),
            Vec::new(),
        ))
        .expect("NYSE did not observe Saturday New Year's Day on Friday"),
        Value::Bool(true)
    );
    assert_eq!(
        futures::executor::block_on(lbusdate_builtin_source(
            Value::Num(2021.0),
            Value::Num(12.0),
            Vec::new(),
        ))
        .expect("2021 December last business day"),
        Value::Num(serial_for_date(2021, 12, 31))
    );
    let weekend =
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 7]).unwrap());
    assert_eq!(
        futures::executor::block_on(fbusdate_builtin_source(
            Value::Num(2003.0),
            Value::Num(3.0),
            vec![
                Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
                weekend
            ],
        ))
        .expect("Saturday trading"),
        Value::Num(serial_for_date(2003, 3, 1))
    );
    let datetime_output = futures::executor::block_on(lbusdate_builtin_source(
        Value::Num(2024.0),
        Value::Num(6.0),
        vec![
            Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
            Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
            Value::String("datetime".into()),
        ],
    ))
    .expect("datetime output type");
    assert!(is_datetime_object(&datetime_output));
    assert_eq!(
        serials_from_datetime_value(&datetime_output)
            .unwrap()
            .materialize_f64(),
        vec![serial_for_date(2024, 6, 28)]
    );

    let holidays = futures::executor::block_on(holidays_builtin_source(vec![Value::Num(2024.0)]))
        .expect("holidays");
    let serials = serials_from_datetime_value(&holidays).expect("holiday serials");
    assert!(serials
        .materialize_f64()
        .contains(&serial_for_date(2024, 1, 1)));

    let serial_holidays = futures::executor::block_on(holidays_builtin_source(vec![
        Value::Num(serial_for_date(2024, 1, 1)),
        Value::Num(serial_for_date(2024, 1, 3)),
    ]))
    .expect("serial holiday range");
    let Value::Tensor(serial_holidays) = serial_holidays else {
        panic!("numeric date range should return serial dates");
    };
    assert_eq!(
        serial_holidays.materialize_f64(),
        vec![serial_for_date(2024, 1, 1)]
    );

    let alternate = futures::executor::block_on(holidays_builtin_source(vec![
        Value::Num(serial_for_date(2024, 1, 1)),
        Value::Num(serial_for_date(2024, 1, 31)),
        Value::Tensor(
            Tensor::new(
                vec![
                    serial_for_date(2023, 12, 31),
                    serial_for_date(2024, 1, 15),
                    serial_for_date(2024, 1, 15),
                    serial_for_date(2024, 2, 1),
                ],
                vec![1, 4],
            )
            .unwrap(),
        ),
    ]))
    .expect("alternate holiday list");
    let Value::Tensor(alternate) = alternate else {
        panic!("numeric alternate holidays should remain serial dates");
    };
    assert_eq!(
        alternate.materialize_f64(),
        vec![serial_for_date(2024, 1, 15)]
    );

    assert!(futures::executor::block_on(fbusdate_builtin_source(
        Value::Num(2024.0),
        Value::Num(2.0),
        vec![Value::Tensor(
            Tensor::new(
                (1..=29).map(|day| serial_for_date(2024, 2, day)).collect(),
                vec![1, 29],
            )
            .unwrap()
        )],
    ))
    .is_err());

    let typed_year =
        Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![2024]), vec![1, 1]).unwrap();
    let holidays =
        futures::executor::block_on(holidays_builtin_source(vec![Value::Tensor(typed_year)]))
            .expect("holidays from typed year");
    let serials = serials_from_datetime_value(&holidays).expect("holiday serials");
    assert!(serials
        .materialize_f64()
        .contains(&serial_for_date(2024, 1, 1)));

    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    assert!(
        futures::executor::block_on(holidays_builtin_source(vec![Value::Num(2024.0)])).is_err()
    );
}

#[test]
fn business_day_extensions_gate_before_resident_access() {
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let date = Value::Num(serial_for_date(2024, 1, 2));
    let typed = integer_tensor(runmat_value::IntegerStorage::U32(vec![739_253]), vec![1, 1]);
    let typed_error =
        futures::executor::block_on(daysdif_builtin_source(typed, date.clone(), Vec::new()))
            .expect_err("typed legacy date is an extension");
    assert_eq!(
        typed_error.identifier(),
        Some("RunMat:compatibility:BusinessDayTypedLegacySerialExtension")
    );

    let logical_error = futures::executor::block_on(daysdif_builtin_source(
        date.clone(),
        date.clone(),
        vec![Value::Bool(true)],
    ))
    .expect_err("logical Basis is an extension");
    assert_eq!(
        logical_error.identifier(),
        Some("RunMat:compatibility:DatetimeLogicalInputExtension")
    );

    let allocation = unregistered_resident(vec![1, 1]);
    let resident_error = futures::executor::block_on(isbusday_builtin_source(
        Value::GpuTensor(allocation.handle.clone()),
        Vec::new(),
    ))
    .expect_err("resident date is gated before provider lookup");
    assert_eq!(
        resident_error.identifier(),
        Some("RunMat:compatibility:DatetimeGpuInputExtension")
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);

    let weekend = runmat_value::LogicalArray::new(vec![1, 0, 0, 0, 0, 0, 1], vec![1, 7]).unwrap();
    assert!(futures::executor::block_on(isbusday_builtin_source(
        date,
        vec![
            Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
            Value::LogicalArray(weekend)
        ],
    ))
    .is_ok());
}

#[test]
fn every_business_day_entrypoint_rejects_resident_input_before_provider_access() {
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = unregistered_resident(vec![1, 1]);
    let resident = Value::GpuTensor(allocation.handle.clone());
    let date = Value::Num(serial_for_date(2024, 1, 2));
    let cases = [
        futures::executor::block_on(isbusday_builtin_source(resident.clone(), Vec::new())),
        futures::executor::block_on(holidays_builtin_source(vec![
            date.clone(),
            resident.clone(),
        ])),
        futures::executor::block_on(busdays_builtin_source(
            date.clone(),
            resident.clone(),
            Vec::new(),
        )),
        futures::executor::block_on(days252bus_builtin_source(
            date.clone(),
            resident.clone(),
            Vec::new(),
        )),
        futures::executor::block_on(daysdif_builtin_source(
            date.clone(),
            resident.clone(),
            Vec::new(),
        )),
        futures::executor::block_on(fbusdate_builtin_source(
            Value::Num(2024.0),
            resident.clone(),
            Vec::new(),
        )),
        futures::executor::block_on(lbusdate_builtin_source(
            Value::Num(2024.0),
            resident,
            Vec::new(),
        )),
    ];
    for result in cases {
        assert_eq!(
            result
                .expect_err("strict-mode resident input must reject")
                .identifier(),
            Some("RunMat:compatibility:DatetimeGpuInputExtension")
        );
    }
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}
