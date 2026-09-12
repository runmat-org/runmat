use super::capabilities::{
    DATETIME_CLASS, DEFAULT_DATETIME_FORMAT, DEFAULT_DATE_FORMAT, FORMAT_FIELD,
    MAX_DATESHIFT_DAY_OCCURRENCE, SECONDS_PER_DAY,
};
use super::descriptors::DATESHIFT_INTEGER_INPUTS;
use super::*;
use super::{
    business_day_builtins::*, calendar_duration_builtins::*, component_builtins::*,
    constructor_builtin::*, dateshift_builtin::*, legacy_builtins::*, operators::*,
};

async fn datetime_subsref(obj: Value, kind: String, payload: Value) -> crate::BuiltinResult<Value> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::operators::datetime_subsref(obj, subscript).await
}

async fn datetime_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::operators::datetime_subsasgn(obj, subscript, rhs).await
}

fn run_datetime(args: Vec<Value>) -> Value {
    futures::executor::block_on(datetime_builtin(args)).expect("datetime")
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
    let labels: Vec<&str> = DATETIME_DESCRIPTOR
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
    assert_eq!(
        DATETIME_BINARY_DESCRIPTOR.signatures[0].label,
        "out = datetime.op(lhs, rhs)"
    );
}

#[test]
fn hour_supports_datetime_legacy_serial_and_formatted_text_shapes() {
    let datetime = run_datetime(vec![Value::from("2024-03-14 09:26:53")]);
    assert_eq!(
        futures::executor::block_on(hour_builtin(datetime, Vec::new())).unwrap(),
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
    let result =
        futures::executor::block_on(hour_builtin(Value::Tensor(serials), Vec::new())).unwrap();
    let Value::Tensor(result) = result else {
        panic!("expected shaped hour result");
    };
    assert_eq!(result.shape, vec![1, 2]);
    assert_eq!(result.materialize_f64(), vec![3.0, 17.0]);

    let formatted = futures::executor::block_on(hour_builtin(
        Value::from("14/03/2024 21:05:00"),
        vec![Value::from("dd/MM/yyyy HH:mm:ss")],
    ))
    .unwrap();
    assert_eq!(formatted, Value::Num(21.0));

    let documented = futures::executor::block_on(hour_builtin(
        Value::from("2024/14/03 09:26:53.125"),
        vec![Value::from("yyyy/dd/mm hh:MM:ss.fff")],
    ))
    .expect("documented datestr format language");
    assert_eq!(documented, Value::Num(9.0));

    let default_fractional = futures::executor::block_on(hour_builtin(
        Value::from("2024/03/14 17:26:53.125"),
        Vec::new(),
    ))
    .expect("year-first fractional legacy text");
    assert_eq!(default_fractional, Value::Num(17.0));
}

#[test]
fn hour_gates_uncertain_typed_and_resident_legacy_extensions_before_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let typed = futures::executor::block_on(hour_builtin(
        integer_tensor(runmat_value::IntegerStorage::U32(vec![739_000]), vec![1, 1]),
        Vec::new(),
    ))
    .expect_err("typed legacy serial must be gated");
    assert_eq!(
        typed.identifier(),
        Some("RunMat:compatibility:HourTypedLegacySerialExtension")
    );

    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_419_002,
        descriptor: Default::default(),
    });
    let resident = futures::executor::block_on(hour_builtin(resident, Vec::new()))
        .expect_err("resident legacy input must gate before provider access");
    assert_eq!(
        resident.identifier(),
        Some("RunMat:compatibility:DatetimeGpuInputExtension")
    );
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
    let expected_hour =
        futures::executor::block_on(hour_builtin(Value::Num(f64::from(serial)), Vec::new()))
            .expect("double legacy hour");
    let expected_minute =
        futures::executor::block_on(minute_builtin(Value::Num(f64::from(serial)), Vec::new()))
            .expect("double legacy minute");
    let expected_month =
        futures::executor::block_on(month_builtin(Value::Num(f64::from(serial)), Vec::new()))
            .expect("double legacy month");

    for storage in cases {
        let value = integer_tensor(storage.clone(), vec![1, 1]);
        assert_eq!(
            futures::executor::block_on(hour_builtin(value.clone(), Vec::new()))
                .expect("typed legacy hour"),
            expected_hour
        );
        assert_eq!(
            futures::executor::block_on(minute_builtin(value.clone(), Vec::new()))
                .expect("typed legacy minute"),
            expected_minute
        );
        assert_eq!(
            futures::executor::block_on(month_builtin(value, Vec::new()))
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
                integer_tensor(runmat_value::IntegerStorage::U16(vec![42]), vec![1, 1]),
                Vec::new(),
            )),
            "RunMat:compatibility:MinuteTypedLegacySerialExtension",
        ),
        (
            futures::executor::block_on(month_builtin(
                integer_tensor(runmat_value::IntegerStorage::U16(vec![42]), vec![1, 1]),
                Vec::new(),
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

    for result in [
        futures::executor::block_on(minute_builtin(
            Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: 0,
                buffer_id: 9_419_003,
                descriptor: Default::default(),
            }),
            Vec::new(),
        )),
        futures::executor::block_on(month_builtin(
            Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: 0,
                buffer_id: 9_419_004,
                descriptor: Default::default(),
            }),
            Vec::new(),
        )),
    ] {
        assert_eq!(
            result
                .expect_err("resident legacy input must gate before provider access")
                .identifier(),
            Some("RunMat:compatibility:DatetimeGpuInputExtension")
        );
    }
}

#[test]
fn month_datetime_name_modes_return_shaped_character_cells() {
    let datetime = run_datetime(vec![Value::from("2024-03-14 09:26:53")]);
    for (mode, expected) in [("name", "March"), ("shortname", "Mar")] {
        let result =
            futures::executor::block_on(month_builtin(datetime.clone(), vec![Value::from(mode)]))
                .expect("month name mode");
        let Value::Cell(cell) = result else {
            panic!("expected month name cell");
        };
        assert_eq!(cell.shape, vec![1, 1]);
        assert_eq!(
            cell.data,
            vec![Value::CharArray(CharArray::new_row(expected))]
        );
    }
}

#[test]
fn datetime_builds_from_components() {
    let value = run_datetime(vec![Value::Num(2024.0), Value::Num(3.0), Value::Num(14.0)]);
    let object = as_datetime(value);
    assert!(object.class_name.is(DATETIME_CLASS));
    assert_eq!(format_for_object(&object), DEFAULT_DATE_FORMAT);
    let serials = serial_tensor_for_object(&object).expect("serials");
    assert_eq!(serials.materialize_f64().len(), 1);
    let year = futures::executor::block_on(year_builtin(Value::Object(object.clone()), Vec::new()))
        .expect("year");
    assert_eq!(year, Value::Num(2024.0));
}

#[test]
fn year_typed_legacy_serial_is_separately_gated() {
    let serial = serial_for_date(2024, 1, 1) as i64;
    let input = Value::Int(runmat_value::IntValue::I64(serial));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(year_builtin(input.clone(), Vec::new()))
        .expect_err("strict gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:YearTypedLegacySerialExtension")
    );
    drop(_strict);

    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let result =
        futures::executor::block_on(year_builtin(input, Vec::new())).expect("typed legacy serial");
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
    assert_eq!(serials.shape, vec![1, 2]);
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
            datetime_string_array(&value).unwrap().unwrap().data,
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
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 3],
        device_id: 0,
        buffer_id: 9_397_001,
        descriptor: Default::default(),
    });
    let resident = futures::executor::block_on(datetime_builtin(vec![resident]))
        .expect_err("resident input is gated before provider access");
    assert_eq!(
        resident.identifier(),
        Some("RunMat:compatibility:DatetimeGpuInputExtension")
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
            integer_tensor(storage, vec![1, 1]),
            Vec::new(),
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
    let value = run_datetime(vec![Value::String("2024-03-14 09:26:53".to_string())]);
    let rendered = datetime_string_array(&value)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(rendered.data, vec!["14-Mar-2024 09:26:53".to_string()]);
}

#[test]
fn datetime_missing_serial_renders_as_nat() {
    let value = datetime_object_from_serial_tensor(
        Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap(),
        DEFAULT_DATETIME_FORMAT,
    )
    .expect("datetime object");
    let rendered = datetime_string_array(&value)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(rendered.data, vec!["NaT".to_string()]);
    assert_eq!(
        datetime_display_text(&value).expect("display"),
        Some("NaT".to_string())
    );
}

#[test]
fn datetime_accepts_existing_datetime_input() {
    let value = run_datetime(vec![Value::String("2024-03-14".to_string())]);
    let converted = run_datetime(vec![
        value.clone(),
        Value::from("InputFormat"),
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
        rendered.data,
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
        input,
        Value::from("start"),
        Value::from("week"),
        Vec::new(),
    ))
    .expect("dateshift start week");
    let rendered = datetime_string_array(&shifted)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(
        rendered.data,
        vec!["2024-03-10".to_string(), "2024-03-17".to_string()]
    );

    let month_end = futures::executor::block_on(dateshift_builtin(
        run_datetime(vec![
            Value::from("2024-02-10"),
            Value::from("Format"),
            Value::from("yyyy-MM-dd HH:mm:ss"),
        ]),
        Value::from("end"),
        Value::from("month"),
        Vec::new(),
    ))
    .expect("dateshift end month");
    let rendered = datetime_string_array(&month_end)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(rendered.data, vec!["2024-02-29 00:00:00".to_string()]);
}

#[test]
fn dateshift_rules_follow_current_week_and_boundary_contracts() {
    let input = run_datetime(vec![Value::from("2024-03-14")]);
    let monday = futures::executor::block_on(dateshift_builtin(
        input.clone(),
        Value::from("dayofweek"),
        Value::from("monday"),
        vec![Value::from("current")],
    ))
    .expect("current-week Monday");
    assert_eq!(
        datetime_string_array(&monday).unwrap().unwrap().data,
        vec!["11-Mar-2024".to_string()]
    );

    let current_zero = futures::executor::block_on(dateshift_builtin(
        input.clone(),
        Value::from("dayofweek"),
        Value::from("monday"),
        vec![Value::Int(runmat_value::IntValue::I8(0))],
    ))
    .expect("numeric zero current-week Monday");
    assert_eq!(
        datetime_string_array(&current_zero).unwrap().unwrap().data,
        vec!["11-Mar-2024".to_string()]
    );

    let next = futures::executor::block_on(dateshift_builtin(
        run_datetime(vec![Value::from("2024-03-01")]),
        Value::from("start"),
        Value::from("month"),
        vec![Value::from("next")],
    ))
    .expect("next exact boundary");
    assert_eq!(
        datetime_string_array(&next).unwrap().unwrap().data,
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
            input.clone(),
            Value::from("dayofweek"),
            Value::Int(weekday),
            Vec::new(),
        ))
        .expect("typed weekday");
        assert_eq!(
            datetime_string_array(&shifted).unwrap().unwrap().data,
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
        datetime_string_array(&datetime).unwrap().unwrap().data,
        vec!["01-Jan-2025".to_string()]
    );
    assert_eq!(
        futures::executor::block_on(day_builtin(
            datetime.clone(),
            vec![Value::from("dayofyear")],
        ))
        .unwrap(),
        Value::Num(1.0)
    );
    let names =
        futures::executor::block_on(day_builtin(datetime, vec![Value::from("shortname")])).unwrap();
    let Value::Cell(names) = names else {
        panic!("expected cell names")
    };
    let Value::CharArray(name) = &names.data[0] else {
        panic!("expected char name")
    };
    assert_eq!(name.data.iter().collect::<String>(), "Wed");
    assert_eq!(
        futures::executor::block_on(day_builtin(
            Value::from("2021/28/09"),
            vec![Value::from("yyyy/dd/mm")],
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
        Value::String(FORMAT_FIELD.to_string()),
        Value::String("yyyy-MM-dd".to_string()),
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
    let year = futures::executor::block_on(year_builtin(indexed, Vec::new())).expect("year");
    assert_eq!(year, Value::Num(2025.0));

    let lhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(1.0)]);
    let rhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(2.0)]);
    let cmp = futures::executor::block_on(datetime_lt(lhs, rhs)).expect("lt");
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
    let year = futures::executor::block_on(year_builtin(indexed, Vec::new())).expect("year");
    assert_eq!(year, Value::Num(2025.0));
}

#[test]
fn datetime_and_duration_interoperate() {
    let lhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(1.0)]);
    let rhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(2.0)]);
    let delta = futures::executor::block_on(datetime_minus(rhs.clone(), lhs.clone()))
        .expect("datetime minus datetime");
    assert_eq!(delta, Value::Num(1.0));

    let duration = crate::builtins::duration::duration_object_from_days_tensor(
        Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
        crate::builtins::duration::DEFAULT_DURATION_FORMAT,
    )
    .expect("duration");

    let round_trip =
        futures::executor::block_on(datetime_plus(lhs.clone(), duration.clone())).expect("plus");
    let round_trip_text = datetime_display_text(&round_trip)
        .expect("datetime display")
        .expect("datetime text");
    assert_eq!(round_trip_text, "02-Jan-2024");

    let restored =
        futures::executor::block_on(datetime_minus(rhs, duration)).expect("minus duration");
    let restored_text = datetime_display_text(&restored)
        .expect("datetime display")
        .expect("datetime text");
    assert_eq!(restored_text, "01-Jan-2024");
}

#[test]
fn legacy_date_conversion_and_query_helpers_work() {
    let serial = serial_for_date(2024, 3, 14);
    let date_vector =
        futures::executor::block_on(datevec_builtin(Value::Num(serial))).expect("datevec");
    let Value::Tensor(date_vector) = date_vector else {
        panic!("expected datevec tensor");
    };
    assert_eq!(date_vector.shape, vec![1, 6]);
    assert_eq!(&date_vector.materialize_f64()[..3], &[2024.0, 3.0, 14.0]);

    let round_trip =
        futures::executor::block_on(datenum_builtin(vec![Value::Tensor(date_vector.clone())]))
            .expect("datenum");
    assert_eq!(round_trip, Value::Num(serial));
    let date_only_round_trip = futures::executor::block_on(datenum_builtin(vec![Value::Tensor(
        Tensor::new(vec![2024.0, 3.0, 14.0], vec![1, 3]).unwrap(),
    )]))
    .expect("datenum date vector");
    assert_eq!(date_only_round_trip, Value::Num(serial));

    let text = futures::executor::block_on(datestr_builtin(
        Value::Num(serial),
        vec![Value::from("yyyy-MM-dd")],
    ))
    .expect("datestr");
    let Value::CharArray(text) = text else {
        panic!("expected datestr char array");
    };
    assert_eq!(text.data.iter().collect::<String>(), "2024-03-14");
    let text_from_datevec = futures::executor::block_on(datestr_builtin(
        Value::Tensor(Tensor::new(vec![2024.0, 3.0, 14.0], vec![1, 3]).unwrap()),
        vec![Value::from("yyyy-MM-dd")],
    ))
    .expect("datestr date vector");
    let Value::CharArray(text_from_datevec) = text_from_datevec else {
        panic!("expected datestr char array");
    };
    assert_eq!(
        text_from_datevec.data.iter().collect::<String>(),
        "2024-03-14"
    );

    let weekday =
        futures::executor::block_on(weekday_builtin(Value::Num(serial))).expect("weekday");
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
            Value::Num(serial),
            Value::Num(serial - 1.0),
            Value::Num(serial + 1.0),
        ))
        .expect("isbetween"),
        Value::Num(1.0)
    );
}

#[test]
fn datenum_typed_integer_date_vector_reads_exact_storage() {
    let serial = serial_for_date(2024, 3, 14);
    let typed_date_vector = Tensor::new_integer(
        runmat_value::IntegerStorage::U16(vec![2024, 3, 14]),
        vec![1, 3],
    )
    .expect("typed date vector");
    let typed_round_trip =
        futures::executor::block_on(datenum_builtin(vec![Value::Tensor(typed_date_vector)]))
            .expect("datenum typed date vector");
    assert_eq!(typed_round_trip, Value::Num(serial));
}

#[test]
fn datenum_typed_integer_serials_read_exact_storage() {
    let serial = serial_for_date(2024, 3, 14).floor() as u32;
    let scalar = Tensor::new_integer(runmat_value::IntegerStorage::U32(vec![serial]), vec![1, 1])
        .expect("typed serial");
    let scalar_out = futures::executor::block_on(datenum_builtin(vec![Value::Tensor(scalar)]))
        .expect("datenum typed scalar serial");
    assert_eq!(scalar_out, Value::Num(f64::from(serial)));

    let vector = Tensor::new_integer(
        runmat_value::IntegerStorage::U32(vec![serial, serial + 1]),
        vec![1, 2],
    )
    .expect("typed serial vector");
    let vector_out = futures::executor::block_on(datenum_builtin(vec![Value::Tensor(vector)]))
        .expect("datenum typed vector serial");
    let Value::Tensor(vector_out) = vector_out else {
        panic!("expected datenum vector tensor");
    };
    assert_eq!(vector_out.shape, vec![1, 2]);
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
    let shifted = futures::executor::block_on(datetime_plus(jan31, one_month)).expect("plus");
    let rendered = datetime_string_array(&shifted)
        .expect("string array")
        .expect("datetime strings");
    assert_eq!(rendered.data, vec!["2024-02-29".to_string()]);

    let duration = futures::executor::block_on(calendar_duration_builtin(vec![
        Value::Num(1.0),
        Value::Num(2.0),
        Value::Num(3.0),
    ]))
    .expect("calendarDuration");
    let (months, days) = calendar_duration_tensors_from_value(&duration).expect("components");
    assert_eq!(months.materialize_f64(), vec![14.0]);
    assert_eq!(days.materialize_f64(), vec![3.0]);

    assert!(futures::executor::block_on(calyears_builtin(Value::Num(f64::MAX))).is_err());
    assert!(futures::executor::block_on(calendar_duration_builtin(vec![
        Value::Num(f64::MAX),
        Value::Num(f64::MAX),
        Value::Num(0.0),
    ]))
    .is_err());
}

#[test]
fn business_day_helpers_use_weekends_and_holidays() {
    let new_year = serial_for_date(2024, 1, 1);
    let friday = serial_for_date(2024, 1, 5);
    let saturday = serial_for_date(2024, 1, 6);
    let mask = futures::executor::block_on(isbusday_builtin(
        Value::Tensor(Tensor::new(vec![new_year, friday, saturday], vec![1, 3]).unwrap()),
        Vec::new(),
    ))
    .expect("isbusday");
    let Value::Tensor(mask) = mask else {
        panic!("expected isbusday tensor");
    };
    assert_eq!(mask.materialize_f64(), vec![0.0, 1.0, 0.0]);

    assert_eq!(
        futures::executor::block_on(isbusday_builtin(
            Value::Num(friday),
            vec![Value::Num(friday)],
        ))
        .expect("custom holiday"),
        Value::Num(0.0)
    );

    let business_days = futures::executor::block_on(busdays_builtin(
        Value::Num(friday),
        Value::Num(friday + 3.0),
        Vec::new(),
    ))
    .expect("busdays");
    let Value::Tensor(business_days) = business_days else {
        panic!("expected busdays tensor");
    };
    assert_eq!(business_days.materialize_f64(), vec![friday, friday + 3.0]);

    assert_eq!(
        futures::executor::block_on(days252bus_builtin(
            Value::Num(friday),
            Value::Num(friday + 3.0),
            Vec::new(),
        ))
        .expect("days252bus"),
        Value::Num(2.0)
    );
    assert_eq!(
        futures::executor::block_on(daysdif_builtin(
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
        futures::executor::block_on(daysdif_builtin(
            Value::Num(serial_for_date(2024, 1, 30)),
            Value::Num(serial_for_date(2024, 2, 29)),
            vec![Value::Tensor(typed_basis)],
        ))
        .expect("daysdif typed basis"),
        Value::Num(29.0)
    );
    assert_eq!(
        futures::executor::block_on(fbusdate_builtin(
            Value::Num(2024.0),
            Value::Num(1.0),
            Vec::new(),
        ))
        .expect("fbusdate"),
        Value::Num(serial_for_date(2024, 1, 2))
    );
    assert_eq!(
        futures::executor::block_on(lbusdate_builtin(
            Value::Num(2024.0),
            Value::Num(6.0),
            Vec::new(),
        ))
        .expect("lbusdate"),
        Value::Num(serial_for_date(2024, 6, 28))
    );

    let holidays =
        futures::executor::block_on(holidays_builtin(vec![Value::Num(2024.0)])).expect("holidays");
    let serials = serials_from_datetime_value(&holidays).expect("holiday serials");
    assert!(serials
        .materialize_f64()
        .contains(&serial_for_date(2024, 1, 1)));

    let typed_year =
        Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![2024]), vec![1, 1]).unwrap();
    let holidays = futures::executor::block_on(holidays_builtin(vec![Value::Tensor(typed_year)]))
        .expect("holidays from typed year");
    let serials = serials_from_datetime_value(&holidays).expect("holiday serials");
    assert!(serials
        .materialize_f64()
        .contains(&serial_for_date(2024, 1, 1)));
}
