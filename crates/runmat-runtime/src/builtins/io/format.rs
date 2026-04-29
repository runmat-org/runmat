//! MATLAB-compatible `format` builtin for controlling numeric display precision.

use runmat_builtins::{set_display_format, FormatMode, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult};

#[runtime_builtin(
    name = "format",
    category = "io",
    summary = "Set the numeric display format for console output (format short, format long, etc.).",
    keywords = "format,display,precision,numeric,short,long,scientific",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::format"
)]
async fn format_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let keyword =
        match args.as_slice() {
            [] => {
                set_display_format(FormatMode::Short);
                return Ok(empty_value());
            }
            [Value::String(s)] => s.to_lowercase(),
            [Value::CharArray(ca)] => ca.to_string().to_lowercase(),
            _ => return Err(build_runtime_error(
                "format: unrecognized argument; expected a format name such as 'short' or 'long'",
            )
            .with_builtin("format")
            .build()),
        };
    // Spacing modes: accepted for MATLAB compatibility but not yet implemented.
    // Crucially, they must NOT change the active numeric format.
    if matches!(keyword.trim(), "compact" | "loose") {
        return Ok(empty_value());
    }
    set_display_format(parse_numeric_mode(keyword.trim())?);
    Ok(empty_value())
}

fn parse_numeric_mode(s: &str) -> BuiltinResult<FormatMode> {
    match s {
        "short" => Ok(FormatMode::Short),
        "long" => Ok(FormatMode::Long),
        "shorte" => Ok(FormatMode::ShortE),
        "longe" => Ok(FormatMode::LongE),
        "shortg" => Ok(FormatMode::ShortG),
        "longg" => Ok(FormatMode::LongG),
        "rat" | "rational" => Ok(FormatMode::Rational),
        "hex" => Ok(FormatMode::Hex),
        other => Err(build_runtime_error(format!(
            "format: unknown format '{other}'; numeric modes: short, long, shortE, longE, shortG, longG, rat, hex"
        ))
        .with_builtin("format")
        .build()),
    }
}

fn empty_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{format_number, get_display_format, FormatMode};
    use std::f64::consts::PI;

    fn with_format<F: FnOnce()>(mode: FormatMode, f: F) {
        let prev = get_display_format();
        set_display_format(mode);
        f();
        set_display_format(prev);
    }

    #[test]
    fn test_parse_numeric_mode_case_insensitive() {
        // parse_numeric_mode receives an already-lowercased string from the builtin.
        assert_eq!(parse_numeric_mode("short").unwrap(), FormatMode::Short);
        assert_eq!(parse_numeric_mode("long").unwrap(), FormatMode::Long);
        assert_eq!(parse_numeric_mode("shorte").unwrap(), FormatMode::ShortE);
        assert_eq!(parse_numeric_mode("longe").unwrap(), FormatMode::LongE);
        assert_eq!(parse_numeric_mode("shortg").unwrap(), FormatMode::ShortG);
        assert_eq!(parse_numeric_mode("longg").unwrap(), FormatMode::LongG);
        assert_eq!(parse_numeric_mode("rat").unwrap(), FormatMode::Rational);
        assert_eq!(
            parse_numeric_mode("rational").unwrap(),
            FormatMode::Rational
        );
        assert_eq!(parse_numeric_mode("hex").unwrap(), FormatMode::Hex);
    }

    #[test]
    fn test_parse_numeric_mode_unknown() {
        assert!(parse_numeric_mode("bank").is_err());
        assert!(parse_numeric_mode("").is_err());
        // Spacing modes are handled before parse_numeric_mode is called.
        assert!(parse_numeric_mode("compact").is_err());
        assert!(parse_numeric_mode("loose").is_err());
    }

    #[test]
    fn test_spacing_modes_do_not_change_numeric_format() {
        block_on(async {
            format_builtin(vec![Value::String("long".to_string())])
                .await
                .unwrap();
            assert_eq!(get_display_format(), FormatMode::Long);

            format_builtin(vec![Value::String("compact".to_string())])
                .await
                .unwrap();
            assert_eq!(
                get_display_format(),
                FormatMode::Long,
                "compact must not reset numeric format"
            );

            format_builtin(vec![Value::String("loose".to_string())])
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
            format_builtin(vec![Value::String("long".to_string())])
                .await
                .unwrap();
            assert_eq!(format_number(PI), "3.141592653589793");
            set_display_format(FormatMode::Short);
        });
    }

    #[test]
    fn format_builtin_short_pi_displays_four_decimal_places() {
        block_on(async {
            format_builtin(vec![Value::String("short".to_string())])
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
            format_builtin(vec![Value::String("shortE".to_string())])
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
            format_builtin(vec![Value::String("long".to_string())])
                .await
                .unwrap();
            format_builtin(vec![]).await.unwrap();
            assert_eq!(get_display_format(), FormatMode::Short);
        });
    }

    #[test]
    fn format_builtin_rational_pi_approximation() {
        block_on(async {
            // Use MATLAB's canonical keyword "rat" (not "rational").
            format_builtin(vec![Value::String("rat".to_string())])
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
            format_builtin(vec![Value::String("hex".to_string())])
                .await
                .unwrap();
            assert_eq!(format_number(PI), "400921fb54442d18");
            set_display_format(FormatMode::Short);
        });
    }

    #[test]
    fn format_builtin_small_value_goes_scientific_in_short() {
        block_on(async {
            format_builtin(vec![Value::String("short".to_string())])
                .await
                .unwrap();
            assert_eq!(format_number(1e-5), "1.0000e-05");
        });
    }
}
