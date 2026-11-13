//! RunMat `figure` builtin for selecting and creating plotting windows.

use runmat_builtins::{
    FIGURE_DESCRIPTOR, FIGURE_ERROR_TOO_MANY_OUTPUTS, FIGURE_INTEGER_PROPERTY_EXTENSION,
    FIGURE_INTEGER_TARGET_EXTENSION, FIGURE_NEXT_SELECTOR_EXTENSION,
    FIGURE_SINGLE_TARGET_EXTENSION,
};

use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use super::op_common::handles::parse_optional_figure_handle;
use super::properties::{set_properties_with_runtime, validate_figure_property_value, PlotHandle};
use super::state::{new_figure_handle, select_figure};
use crate::builtins::plotting::plotting_error;

#[runtime_builtin(
    name = "figure",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::figure"
)]
pub fn figure_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    let _scope = runtime.as_ref().map(|runtime| runtime.enter());
    crate::output_count::validate_fixed_outputs(
        &FIGURE_DESCRIPTOR,
        &FIGURE_ERROR_TOO_MANY_OUTPUTS,
        "figure",
    )
    .map_err(super::coordinate_admission::terminal)?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "figure")?;
    work.finish(figure_impl(arguments.view(), runtime.as_ref(), &work))
}

fn figure_impl(
    rest: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&crate::context::RuntimeContext>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    ensure_figure_target_extensions(rest)?;
    let (target_info, property_args) = if rest.is_empty() {
        (None, rest)
    } else {
        match parse_optional_figure_target(&rest[0], rest.len())? {
            Some(handle) => (Some(FigureTarget::Existing(handle)), rest.range(1..)?),
            None if is_next_selector(&rest[0]) => (Some(FigureTarget::New), rest.range(1..)?),
            None => (Some(FigureTarget::New), rest),
        }
    };

    ensure_figure_integer_property_extensions(property_args)?;

    // Validate properties before any state modifications
    if !property_args.is_empty() {
        validate_figure_properties(
            property_args,
            target_info.as_ref().and_then(FigureTarget::target_figure),
        )?;
    }

    work.check()?;
    // Now that validation passed, create/select the figure
    let handle = match target_info {
        Some(FigureTarget::Existing(h)) => {
            select_figure(h);
            h
        }
        Some(FigureTarget::New) | None => new_figure_handle(),
    };

    // Apply properties after figure creation/selection
    if !property_args.is_empty() {
        set_properties_with_runtime(runtime, PlotHandle::Figure(handle), property_args, "figure")?;
    }
    Ok(handle.as_u32() as f64)
}

fn ensure_figure_target_extensions(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<()> {
    let Some(target) = args.first() else {
        return Ok(());
    };
    if is_typed_integer_target(target) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIGURE_INTEGER_TARGET_EXTENSION,
            "figure",
        )?;
    }
    if is_single_target(target) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIGURE_SINGLE_TARGET_EXTENSION,
            "figure",
        )?;
    }
    if is_next_selector(target) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIGURE_NEXT_SELECTOR_EXTENSION,
            "figure",
        )?;
    }
    Ok(())
}

fn is_typed_integer_target(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn is_single_target(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
}

fn ensure_figure_integer_property_extensions(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<()> {
    if !args.len().is_multiple_of(2) {
        return Ok(());
    }
    for pair in args.chunks_exact(2) {
        let Some(property) = figure_property_name(&pair[0]) else {
            continue;
        };
        let value = &pair[1];
        if !is_typed_integer_target(value) || !is_supported_integer_property_route(&property) {
            continue;
        }
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIGURE_INTEGER_PROPERTY_EXTENSION,
            "figure",
        )?;
        if matches!(
            property.as_str(),
            "position" | "color" | "currentaxes" | "sgtitle"
        ) {
            ensure_integer_property_exact_f64(value, &property)?;
        }
    }
    Ok(())
}

fn figure_property_name(value: &Value) -> Option<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .map(|text| text.trim().to_ascii_lowercase())
}

fn is_supported_integer_property_route(property: &str) -> bool {
    matches!(
        property,
        "position" | "color" | "numbertitle" | "visible" | "currentaxes" | "sgtitle"
    )
}

fn ensure_integer_property_exact_f64(value: &Value, property: &str) -> crate::BuiltinResult<()> {
    let exact = crate::builtins::common::validation::integer_is_exact_f64;
    let valid = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(plotting_error(
            "figure",
            format!("figure: integer {property} values must be exactly representable as double"),
        ))
    }
}

enum FigureTarget {
    Existing(super::state::FigureHandle),
    New,
}

impl FigureTarget {
    fn target_figure(&self) -> Option<super::state::FigureHandle> {
        match self {
            FigureTarget::Existing(handle) => Some(*handle),
            FigureTarget::New => None,
        }
    }
}

fn validate_figure_properties(
    args: crate::call::FunctionArgumentView<'_>,
    target_figure: Option<super::state::FigureHandle>,
) -> crate::BuiltinResult<()> {
    if !args.len().is_multiple_of(2) {
        return Err(crate::builtins::plotting::plotting_error(
            "figure",
            "figure: property arguments must be name/value pairs",
        ));
    }
    for pair in args.chunks_exact(2) {
        validate_figure_property_value(&pair[0], &pair[1], target_figure, "figure")?;
    }
    Ok(())
}

fn parse_optional_figure_target(
    value: &Value,
    arg_count: usize,
) -> crate::BuiltinResult<Option<super::state::FigureHandle>> {
    let target =
        if let Some(text) = crate::builtins::common::validation::value_text_scalar_string(value) {
            super::op_common::handles::parse_string_handle_or_next(&text, "figure")
        } else if matches!(
            value,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            Err(plotting_error(
                "figure",
                "figure: target or property name must be scalar text",
            ))
        } else {
            parse_optional_figure_handle(value, "figure")
        };
    match target {
        Ok(target) => Ok(target),
        Err(_) if starts_property_pairs(value, arg_count) => Ok(None),
        Err(_) if is_text(value) && !arg_count.is_multiple_of(2) => Err(plotting_error(
            "figure",
            "figure: property/value arguments must come in pairs",
        )),
        Err(err) => Err(err),
    }
}

fn starts_property_pairs(value: &Value, arg_count: usize) -> bool {
    is_text(value) && arg_count.is_multiple_of(2)
}

fn is_text(value: &Value) -> bool {
    crate::builtins::common::validation::value_is_text_scalar(value)
}

fn is_next_selector(value: &Value) -> bool {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .is_some_and(|text| text.trim().eq_ignore_ascii_case("next"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, figure_handles, reset_hold_state_for_run,
    };
    use runmat_builtins::{
        BuiltinIntegerBackendRule, BuiltinIntegerComputationDomain,
        BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule,
    };
    use runmat_builtins::{
        FIGURE_DESCRIPTOR, FIGURE_INTEGER_CAPABILITIES, FIGURE_INTEGER_PROPERTY_EXTENSION,
        FIGURE_INTEGER_TARGET_EXTENSION, FIGURE_NEXT_SELECTOR_EXTENSION,
        FIGURE_SINGLE_TARGET_EXTENSION,
    };
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn figure_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FIGURE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"fig = figure()"));
        assert!(labels.contains(&"fig = figure(h)"));
        assert!(labels.contains(&"fig = figure(property, value, ...)"));
        assert!(labels.contains(&"fig = figure(h, property, value, ...)"));
    }

    #[test]
    fn figure_creates_and_selects_handles() {
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        assert!(first > 0.0);
        let selected = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(first)]),
        )
        .unwrap();
        assert_eq!(selected, first);
        assert_eq!(current_figure_handle().as_u32() as f64, first);
    }

    #[test]
    fn figure_accepts_property_pairs_without_handle() {
        let _guard = setup();
        let handle = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::StringArray(
                    runmat_value::StringArray::new(vec!["Name".into()], vec![1, 1]).unwrap(),
                ),
                Value::String("demo".into()),
                Value::String("NumberTitle".into()),
                Value::String("off".into()),
                Value::String("Visible".into()),
                Value::String("off".into()),
                Value::String("Color".into()),
                Value::String("black".into()),
            ]),
        )
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .expect("figure should exist");
        assert_eq!(figure.name.as_deref(), Some("demo"));
        assert!(!figure.number_title);
        assert!(!figure.visible);
        assert_eq!(figure.background_color, glam::Vec4::new(0.0, 0.0, 0.0, 1.0));
        let before = figure_handles();
        let paged = runmat_value::CharArray::from_unicode_scalars_with_shape(
            "next".chars().collect(),
            vec![1, 2, 2],
        )
        .unwrap();
        assert!(figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(paged)])
        )
        .is_err());
        assert_eq!(figure_handles(), before);
    }

    #[test]
    fn figure_accepts_position_property_pair() {
        let _guard = setup();
        let position = Tensor::new(vec![100.0, 100.0, 1000.0, 700.0], vec![1, 4]).unwrap();
        let handle = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Position".into()),
                Value::Tensor(position),
            ]),
        )
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .expect("figure should exist");
        assert_eq!(figure.position, [100.0, 100.0, 1000.0, 700.0]);

        let value = crate::builtins::plotting::get::get_builtin(vec![
            Value::Num(handle),
            Value::String("Position".into()),
        ])
        .expect("get position");
        let tensor = Tensor::try_from(&value).expect("position tensor");
        assert_eq!(tensor.materialize_f64(), vec![100.0, 100.0, 1000.0, 700.0]);
    }

    #[test]
    fn figure_accepts_position_column_vector() {
        let _guard = setup();
        let position = Tensor::new(vec![10.0, 20.0, 300.0, 400.0], vec![4, 1]).unwrap();
        let handle = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Position".into()),
                Value::Tensor(position),
            ]),
        )
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .expect("figure should exist");
        assert_eq!(figure.position, [10.0, 20.0, 300.0, 400.0]);
    }

    #[test]
    fn figure_rejects_position_matrix_shape() {
        let _guard = setup();
        let position = Tensor::new(vec![10.0, 20.0, 300.0, 400.0], vec![2, 2]).unwrap();
        let err = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Position".into()),
                Value::Tensor(position),
            ]),
        )
        .expect_err("2-by-2 Position matrix should fail");
        assert!(
            err.to_string()
                .contains("Position must be a 4-element numeric vector"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn figure_selects_explicit_handle_and_applies_property_pairs() {
        let _guard = setup();
        let handle = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(42.0),
                Value::String("Name".into()),
                Value::String("selected".into()),
            ]),
        )
        .unwrap();
        assert_eq!(handle, 42.0);
        assert_eq!(current_figure_handle().as_u32(), 42);
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(42))
            .expect("figure should exist");
        assert_eq!(figure.name.as_deref(), Some("selected"));
    }

    #[test]
    fn figure_next_selector_accepts_property_pairs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        let second = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("next".into()),
                Value::String("Name".into()),
                Value::String("next window".into()),
            ]),
        )
        .unwrap();
        assert_ne!(second, first);
        assert_eq!(current_figure_handle().as_u32() as f64, second);
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            second as u32,
        ))
        .expect("figure should exist");
        assert_eq!(figure.name.as_deref(), Some("next window"));
    }

    #[test]
    fn figure_rejects_dangling_property_name() {
        let _guard = setup();
        let err = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "Name".into(),
            )]),
        )
        .expect_err("dangling property should fail");
        assert!(err
            .message()
            .contains("property/value arguments must come in pairs"));
    }

    #[test]
    fn figure_rejects_invalid_color_name() {
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        let handles_before_error = figure_handles();
        let err = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Color".into()),
                Value::String("banana".into()),
            ]),
        )
        .expect_err("invalid color should fail");
        assert!(err
            .message()
            .contains("unsupported color specification `banana`"));
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before_error);
    }

    #[test]
    fn figure_rejects_oversized_numeric_handle() {
        let _guard = setup();
        let err = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(u32::MAX as f64 + 1.0),
                Value::String("Name".into()),
                Value::String("too large".into()),
            ]),
        )
        .expect_err("oversized handle should fail");
        assert!(err.message().contains("figure handle is too large"));
    }

    #[test]
    fn figure_typed_integer_targets_are_exact_runmat_extensions() {
        let _guard = setup();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let values = [
            IntValue::I8(11),
            IntValue::I16(12),
            IntValue::I32(13),
            IntValue::I64(14),
            IntValue::U8(15),
            IntValue::U16(16),
            IntValue::U32(17),
            IntValue::U64(18),
        ];
        for value in values {
            let expected = value.try_to_u64().expect("positive identifier") as f64;
            assert_eq!(
                figure_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                        value
                    )])
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn figure_integer_target_gate_precedes_graphics_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        let handles_before = figure_handles();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U64(91),
            )]),
        )
        .expect_err("integer target extension");
        assert_eq!(
            err.identifier(),
            FIGURE_INTEGER_TARGET_EXTENSION.error_identifier
        );
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before);
    }

    #[test]
    fn figure_single_target_gate_precedes_graphics_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        let handles_before = figure_handles();
        let single = Tensor::new_with_dtype(vec![92.0], vec![1, 1], NumericDType::F32).unwrap();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(single)]),
        )
        .expect_err("single target extension");
        assert_eq!(
            err.identifier(),
            FIGURE_SINGLE_TARGET_EXTENSION.error_identifier
        );
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before);
    }

    #[test]
    fn figure_next_gate_precedes_graphics_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        let handles_before = figure_handles();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "next".into(),
            )]),
        )
        .expect_err("next selector extension");
        assert_eq!(
            err.identifier(),
            FIGURE_NEXT_SELECTOR_EXTENSION.error_identifier
        );
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before);
    }

    #[test]
    fn figure_rejects_logical_and_resident_targets_without_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        let handles_before = figure_handles();
        assert!(figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)])
        )
        .is_err());
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
                .handle,
        );
        assert!(figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![resident])
        )
        .is_err());
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before);
    }

    #[test]
    fn figure_integer_capability_is_structural_and_host_only() {
        let capability = &FIGURE_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
        assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
        assert_eq!(
            capability.output_class,
            BuiltinIntegerOutputClassRule::NotApplicable
        );
        assert!(capability.notes.contains("representation gap"));
    }

    #[test]
    fn figure_integer_property_routes_gate_before_graphics_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        let handles_before = figure_handles();
        let cases = [
            (
                "Position",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![1, 4]).unwrap(),
                ),
            ),
            (
                "Color",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 0, 0]), vec![1, 3]).unwrap(),
                ),
            ),
            ("NumberTitle", Value::Int(IntValue::U64(u64::MAX))),
            ("Visible", Value::Int(IntValue::I64(i64::MIN))),
            ("CurrentAxes", Value::Int(IntValue::U32(1))),
            ("SgTitle", Value::Int(IntValue::U32(1))),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (property, value) in cases {
            let err = figure_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String(property.into()),
                    value,
                ]),
            )
            .expect_err("integer property extension");
            assert_eq!(
                err.identifier(),
                FIGURE_INTEGER_PROPERTY_EXTENSION.error_identifier,
                "{property}"
            );
            assert_eq!(current_figure_handle().as_u32() as f64, first);
            assert_eq!(figure_handles(), handles_before);
        }
    }

    #[test]
    fn figure_documented_double_and_logical_properties_remain_strict_compatible() {
        let _guard = setup();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let position = Tensor::new(vec![10.0, 20.0, 300.0, 400.0], vec![1, 4]).unwrap();
        let color = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let handle = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Position".into()),
                Value::Tensor(position),
                Value::String("Color".into()),
                Value::Tensor(color),
                Value::String("NumberTitle".into()),
                Value::Bool(false),
                Value::String("Visible".into()),
                Value::Bool(false),
            ]),
        )
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .unwrap();
        assert_eq!(figure.position, [10.0, 20.0, 300.0, 400.0]);
        assert_eq!(figure.background_color, glam::Vec4::new(0.0, 1.0, 0.0, 1.0));
        assert!(!figure.number_title);
        assert!(!figure.visible);
    }

    #[test]
    fn figure_wide_integer_floating_property_boundaries_reject_before_mutation() {
        let _guard = setup();
        let first = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .unwrap();
        let handles_before = figure_handles();
        let wide = 9_007_199_254_740_993_u64;
        let cases = [
            (
                "Position",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![0, 0, wide, 1]), vec![1, 4])
                        .unwrap(),
                ),
            ),
            (
                "Color",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![wide, 0, 0]), vec![1, 3]).unwrap(),
                ),
            ),
            ("CurrentAxes", Value::Int(IntValue::U64(wide))),
            ("SgTitle", Value::Int(IntValue::U64(wide))),
        ];
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for (property, value) in cases {
            let err = figure_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String(property.into()),
                    value,
                ]),
            )
            .expect_err("inexact integer property");
            assert!(
                err.message().contains("exactly representable as double"),
                "{property}: {}",
                err.message()
            );
            assert_eq!(current_figure_handle().as_u32() as f64, first);
            assert_eq!(figure_handles(), handles_before);
        }
    }

    #[test]
    fn figure_integer_position_color_and_switches_use_authoritative_values() {
        let _guard = setup();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let position =
            Tensor::new_integer(IntegerStorage::U64(vec![10, 20, 300, 400]), vec![1, 4]).unwrap();
        let color = Tensor::new_integer(IntegerStorage::U64(vec![1, 0, 0]), vec![1, 3]).unwrap();
        let handle = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Position".into()),
                Value::Tensor(position),
                Value::String("Color".into()),
                Value::Tensor(color),
                Value::String("NumberTitle".into()),
                Value::Int(IntValue::U64(u64::MAX)),
                Value::String("Visible".into()),
                Value::Int(IntValue::I64(i64::MIN)),
            ]),
        )
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .unwrap();
        assert_eq!(figure.position, [10.0, 20.0, 300.0, 400.0]);
        assert_eq!(figure.background_color, glam::Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert!(figure.number_title);
        assert!(figure.visible);
    }

    #[test]
    fn figure_integer_capabilities_classify_every_supported_property_route() {
        assert_eq!(FIGURE_INTEGER_CAPABILITIES.len(), 6);
        for property in [
            "Position",
            "Color",
            "NumberTitle",
            "Visible",
            "CurrentAxes",
            "SgTitle",
        ] {
            let capability = FIGURE_INTEGER_CAPABILITIES
                .iter()
                .find(|capability| capability.form.contains(property))
                .unwrap_or_else(|| panic!("missing {property} capability"));
            assert_eq!(capability.inputs[0].classes.len(), 8);
            assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
            assert_eq!(
                capability.inputs[0].availability,
                BuiltinIntegerInputAvailability::RunMatOnly
            );
        }
    }
}
