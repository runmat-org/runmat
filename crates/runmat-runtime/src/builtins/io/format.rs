//! MATLAB-compatible `format` builtin for controlling numeric display precision.

#[cfg(test)]
use runmat_builtins::FORMAT_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, FormatCommand, FORMAT_ERROR_ARG_CONFIG, FORMAT_ERROR_UNKNOWN_MODE,
};
use runmat_macros::runtime_builtin;
use runmat_value::{set_display_format, FormatMode, Tensor, Value};

use crate::{build_runtime_error, BuiltinResult};

fn format_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("format");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "format",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::format"
)]
async fn format_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().unwrap_or(0) > 0 {
        return Err(format_error_with(
            &runmat_builtins::FORMAT_ERROR_TOO_MANY_OUTPUTS,
            "format: expected no output arguments",
        ));
    }
    let keyword = match (args.len(), args.first()) {
        (0, _) => {
            set_display_format(FormatMode::Short);
            return Ok(empty_value());
        }
        (1, Some(Value::String(runmat_value::StringElement::Text(s)))) => std::borrow::Cow::Borrowed(s.as_str()),
        (1, Some(value @ Value::CharArray(_))) =>
            std::borrow::Cow::Owned(crate::builtins::common::validation::value_text_scalar_string(value)
                .ok_or_else(|| format_error_with(&FORMAT_ERROR_ARG_CONFIG, "format: unrecognized argument; expected a format name such as 'short' or 'long'"))?),
        _ => {
            return Err(format_error_with(
                &FORMAT_ERROR_ARG_CONFIG,
                "format: unrecognized argument; expected a format name such as 'short' or 'long'",
            ));
        }
    };
    let command = FormatCommand::decode(&keyword).ok_or_else(|| {
        format_error_with(
            &FORMAT_ERROR_UNKNOWN_MODE,
            format!(
                "format: unknown format '{}'; numeric modes: short, long, shortE, longE, shortG, longG, rat, hex",
                keyword.trim().to_ascii_lowercase()
            ),
        )
    })?;
    if let FormatCommand::Numeric(mode) = command {
        set_display_format(mode);
    }
    Ok(empty_value())
}

fn empty_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{format_number, get_display_format, FormatMode};
    use std::f64::consts::PI;

    fn with_format<F: FnOnce()>(mode: FormatMode, f: F) {
        let prev = get_display_format();
        set_display_format(mode);
        f();
        set_display_format(prev);
    }

    #[test]
    fn test_parse_numeric_mode_case_insensitive() {
        let labels: Vec<&str> = FORMAT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"format()"));
        assert!(labels.contains(&"format(mode)"));

        assert_eq!(
            FormatCommand::decode("short"),
            Some(FormatCommand::Numeric(FormatMode::Short))
        );
        assert_eq!(
            FormatCommand::decode("long"),
            Some(FormatCommand::Numeric(FormatMode::Long))
        );
        assert_eq!(
            FormatCommand::decode("shorte"),
            Some(FormatCommand::Numeric(FormatMode::ShortE))
        );
        assert_eq!(
            FormatCommand::decode("longe"),
            Some(FormatCommand::Numeric(FormatMode::LongE))
        );
        assert_eq!(
            FormatCommand::decode("shortg"),
            Some(FormatCommand::Numeric(FormatMode::ShortG))
        );
        assert_eq!(
            FormatCommand::decode("longg"),
            Some(FormatCommand::Numeric(FormatMode::LongG))
        );
        assert_eq!(
            FormatCommand::decode("rat"),
            Some(FormatCommand::Numeric(FormatMode::Rational))
        );
        assert_eq!(
            FormatCommand::decode("rational"),
            Some(FormatCommand::Numeric(FormatMode::Rational))
        );
        assert_eq!(
            FormatCommand::decode("hex"),
            Some(FormatCommand::Numeric(FormatMode::Hex))
        );
    }

    #[test]
    fn test_parse_numeric_mode_unknown() {
        assert_eq!(FormatCommand::decode("bank"), None);
        assert_eq!(FormatCommand::decode(""), None);
        assert_eq!(
            FormatCommand::decode("compact"),
            Some(FormatCommand::Compact)
        );
        assert_eq!(FormatCommand::decode("loose"), Some(FormatCommand::Loose));
    }

    #[test]
    fn test_spacing_modes_do_not_change_numeric_format() {
        block_on(async {
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("long".to_string().into())],
            ))
            .await
            .unwrap();
            assert_eq!(get_display_format(), FormatMode::Long);

            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("compact".to_string().into())],
            ))
            .await
            .unwrap();
            assert_eq!(
                get_display_format(),
                FormatMode::Long,
                "compact must not reset numeric format"
            );

            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("loose".to_string().into())],
            ))
            .await
            .unwrap();
            assert_eq!(
                get_display_format(),
                FormatMode::Long,
                "loose must not reset numeric format"
            );

            set_display_format(FormatMode::Short);
        });
    }

    #[test]
    fn test_set_display_format() {
        with_format(FormatMode::Long, || {
            assert_eq!(get_display_format(), FormatMode::Long);
        });
    }

    // --- Behavioral tests: format_builtin + display output ---

    #[test]
    fn format_builtin_long_sets_mode_and_pi_displays_full_precision() {
        block_on(async {
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("long".to_string().into())],
            ))
            .await
            .unwrap();
            assert_eq!(format_number(PI), "3.141592653589793");
            set_display_format(FormatMode::Short);
        });
    }

    #[test]
    fn format_builtin_short_pi_displays_four_decimal_places() {
        block_on(async {
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("short".to_string().into())],
            ))
            .await
            .unwrap();
            assert_eq!(format_number(PI), "3.1416");
            assert_eq!(format_number(0.5), "0.5000");
            assert_eq!(format_number(1.0), "1");
        });
    }

    #[test]
    fn format_builtin_short_e_always_scientific() {
        block_on(async {
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("shortE".to_string().into())],
            ))
            .await
            .unwrap();
            assert_eq!(format_number(PI), "3.1416e+00");
            assert_eq!(format_number(1000.0), "1.0000e+03");
            set_display_format(FormatMode::Short);
        });
    }

    #[test]
    fn format_builtin_no_args_resets_to_short() {
        block_on(async {
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("long".to_string().into())],
            ))
            .await
            .unwrap();
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![],
            ))
            .await
            .unwrap();
            assert_eq!(get_display_format(), FormatMode::Short);
        });
    }

    #[test]
    fn format_builtin_rational_pi_approximation() {
        block_on(async {
            // Use MATLAB's canonical keyword "rat" (not "rational").
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("rat".to_string().into())],
            ))
            .await
            .unwrap();
            // 355/113 is the classic rational approximation of pi
            assert_eq!(format_number(PI), "355/113");
            set_display_format(FormatMode::Short);
        });
    }

    #[test]
    fn format_builtin_hex_pi() {
        block_on(async {
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("hex".to_string().into())],
            ))
            .await
            .unwrap();
            assert_eq!(format_number(PI), "400921fb54442d18");
            set_display_format(FormatMode::Short);
        });
    }

    #[test]
    fn format_builtin_small_value_goes_scientific_in_short() {
        block_on(async {
            format_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("short".to_string().into())],
            ))
            .await
            .unwrap();
            assert_eq!(format_number(1e-5), "1.0000e-05");
        });
    }

    #[test]
    fn captured_output_rejects_before_mode_change() {
        set_display_format(FormatMode::Long);
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let err = block_on(format_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("hex")]),
        ))
        .expect_err("sink output");
        assert_eq!(err.identifier(), Some("RunMat:format:TooManyOutputs"));
        assert_eq!(get_display_format(), FormatMode::Long);
        set_display_format(FormatMode::Short);
    }

    #[test]
    fn paged_character_mode_rejected_but_trailing_singleton_admitted() {
        let paged = runmat_value::CharArray::from_unicode_scalars_with_shape(
            "long".chars().collect(),
            vec![1, 2, 2],
        )
        .expect("paged char");
        assert!(block_on(format_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(paged)])
        ))
        .is_err());
        let trailing = runmat_value::CharArray::from_unicode_scalars_with_shape(
            "long".chars().collect(),
            vec![1, 4, 1],
        )
        .expect("trailing singleton");
        block_on(format_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(
                trailing,
            )]),
        ))
        .expect("valid row");
        assert_eq!(get_display_format(), FormatMode::Long);
        set_display_format(FormatMode::Short);
    }
}
