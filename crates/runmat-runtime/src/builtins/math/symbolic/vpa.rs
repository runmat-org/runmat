//! MATLAB-compatible scalar `vpa` builtin for variable-precision symbolic decimals.

use num_bigint::BigInt;
use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    SymbolicExpr, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::{
    digits::{current_digits, validate_digits, MAX_DIGITS},
    symbolic_expr_to_value, text_scalar, value_to_symbolic_scalar,
};

const BUILTIN_NAME: &str = "vpa";

const VPA_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic variable-precision value or expression.",
}];

const VPA_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar symbolic or numeric input.",
    },
    BuiltinParamDescriptor {
        name: "d",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("digits"),
        description: "Number of significant decimal digits.",
    },
];

const VPA_INPUTS_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic or numeric input.",
}];

const VPA_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "S = vpa(x)",
        inputs: &VPA_INPUTS_VALUE,
        outputs: &VPA_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = vpa(x, d)",
        inputs: &VPA_INPUTS,
        outputs: &VPA_OUTPUT,
    },
];

const VPA_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.VPA.ARG_COUNT",
        identifier: Some("RunMat:vpa:ArgCount"),
        when: "More than one optional precision argument is supplied.",
        message: "vpa: too many input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.VPA.INVALID_INPUT",
        identifier: Some("RunMat:vpa:InvalidInput"),
        when: "Input cannot be represented as a scalar symbolic expression.",
        message: "vpa: expected a scalar symbolic, numeric, logical, or numeric text input",
    },
    BuiltinErrorDescriptor {
        code: "RM.VPA.INVALID_DIGITS",
        identifier: Some("RunMat:vpa:InvalidDigits"),
        when: "The requested precision is not an integer in the supported range.",
        message: "vpa: expected a positive integer digit count",
    },
];

pub const VPA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VPA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VPA_ERRORS,
};

#[runtime_builtin(
    name = "vpa",
    category = "math/symbolic",
    summary = "Evaluate scalar symbolic or numeric input as a variable-precision decimal.",
    keywords = "vpa,symbolic,variable precision,digits,decimal",
    descriptor(crate::builtins::math::symbolic::vpa::VPA_DESCRIPTOR),
    builtin_path = "crate::builtins::math::symbolic::vpa"
)]
async fn vpa_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(vpa_error(&VPA_ERRORS[0]));
    }
    let digits = match rest.first() {
        Some(value) => parse_precision(value)?,
        None => current_digits(),
    };
    let expr = symbolic_input(&value)?;
    Ok(symbolic_expr_to_value(vpa_expr(&expr, digits)))
}

fn parse_precision(value: &Value) -> BuiltinResult<usize> {
    let parsed = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        _ => return Err(vpa_error(&VPA_ERRORS[2])),
    };
    validate_digits(parsed).map_err(|err| {
        vpa_error_with_message(
            &VPA_ERRORS[2],
            format!("{}: {}", VPA_ERRORS[2].message, err),
        )
    })
}

fn symbolic_input(value: &Value) -> BuiltinResult<SymbolicExpr> {
    if let Some(expr) = value_to_symbolic_scalar(value) {
        return Ok(expr);
    }
    if let Some(text) = text_scalar(value) {
        let trimmed = text.trim();
        if let Some(expr) = parse_rational_literal(trimmed) {
            return Ok(expr);
        }
        if let Ok(value) = trimmed.parse::<f64>() {
            if value.is_finite() {
                return Ok(SymbolicExpr::constant(value));
            }
        }
    }
    Err(vpa_error(&VPA_ERRORS[1]))
}

fn parse_rational_literal(text: &str) -> Option<SymbolicExpr> {
    let (lhs, rhs) = text.split_once('/')?;
    if rhs.contains('/') {
        return None;
    }
    let numerator = lhs.trim().parse::<BigInt>().ok()?;
    let denominator = rhs.trim().parse::<BigInt>().ok()?;
    SymbolicExpr::rational(numerator, denominator)
}

fn vpa_expr(expr: &SymbolicExpr, digits: usize) -> SymbolicExpr {
    match expr {
        SymbolicExpr::Constant(value) => decimal_from_f64(*value, digits),
        SymbolicExpr::Rational {
            numerator,
            denominator,
        } => decimal_from_rational(numerator, denominator, digits),
        SymbolicExpr::DecimalLiteral { value, .. } => decimal_from_f64(*value, digits),
        SymbolicExpr::Variable(_) | SymbolicExpr::FunctionReference(_, _) => expr.clone(),
        SymbolicExpr::FunctionCall(name, args) => SymbolicExpr::FunctionCall(
            name.clone(),
            args.iter().map(|arg| vpa_expr(arg, digits)).collect(),
        ),
        SymbolicExpr::Equation(lhs, rhs) => SymbolicExpr::Equation(
            Box::new(vpa_expr(lhs, digits)),
            Box::new(vpa_expr(rhs, digits)),
        ),
        SymbolicExpr::Derivative {
            expr,
            variable,
            order,
        } => SymbolicExpr::Derivative {
            expr: Box::new(vpa_expr(expr, digits)),
            variable: variable.clone(),
            order: *order,
        },
        SymbolicExpr::Neg(inner) => SymbolicExpr::Neg(Box::new(vpa_expr(inner, digits))),
        SymbolicExpr::Add(lhs, rhs) => SymbolicExpr::Add(
            Box::new(vpa_expr(lhs, digits)),
            Box::new(vpa_expr(rhs, digits)),
        ),
        SymbolicExpr::Sub(lhs, rhs) => SymbolicExpr::Sub(
            Box::new(vpa_expr(lhs, digits)),
            Box::new(vpa_expr(rhs, digits)),
        ),
        SymbolicExpr::Mul(lhs, rhs) => SymbolicExpr::Mul(
            Box::new(vpa_expr(lhs, digits)),
            Box::new(vpa_expr(rhs, digits)),
        ),
        SymbolicExpr::Div(lhs, rhs) => {
            if let Some(SymbolicExpr::Rational {
                numerator,
                denominator,
            }) = expr_as_exact_rational(lhs, rhs)
            {
                decimal_from_rational(&numerator, &denominator, digits)
            } else {
                SymbolicExpr::Div(
                    Box::new(vpa_expr(lhs, digits)),
                    Box::new(vpa_expr(rhs, digits)),
                )
            }
        }
        SymbolicExpr::Pow(lhs, rhs) => {
            if let Some(value) = expr.numeric_constant_value() {
                decimal_from_f64(value, digits)
            } else {
                SymbolicExpr::Pow(
                    Box::new(vpa_expr(lhs, digits)),
                    Box::new(vpa_expr(rhs, digits)),
                )
            }
        }
        SymbolicExpr::Function(function, inner) => {
            if let Some(value) = expr.numeric_constant_value() {
                decimal_from_f64(value, digits)
            } else {
                SymbolicExpr::Function(*function, Box::new(vpa_expr(inner, digits)))
            }
        }
    }
}

fn expr_as_exact_rational(lhs: &SymbolicExpr, rhs: &SymbolicExpr) -> Option<SymbolicExpr> {
    let numerator = exact_integer(lhs)?;
    let denominator = exact_integer(rhs)?;
    SymbolicExpr::rational(numerator, denominator)
}

fn exact_integer(expr: &SymbolicExpr) -> Option<BigInt> {
    match expr {
        SymbolicExpr::Constant(value) if value.is_finite() && value.fract() == 0.0 => {
            BigInt::from_f64(*value)
        }
        SymbolicExpr::Rational {
            numerator,
            denominator,
        } if denominator.is_one() => Some(numerator.clone()),
        _ => None,
    }
}

fn decimal_from_f64(value: f64, digits: usize) -> SymbolicExpr {
    let text = significant_decimal_from_f64(value, digits);
    SymbolicExpr::decimal_literal(text, value, digits)
}

fn decimal_from_rational(numerator: &BigInt, denominator: &BigInt, digits: usize) -> SymbolicExpr {
    let text = significant_decimal_from_rational(numerator, denominator, digits);
    let value = match (numerator.to_f64(), denominator.to_f64()) {
        (Some(numerator), Some(denominator)) => numerator / denominator,
        _ if numerator.sign() == denominator.sign() => f64::INFINITY,
        _ => f64::NEG_INFINITY,
    };
    SymbolicExpr::decimal_literal(text, value, digits)
}

fn significant_decimal_from_f64(value: f64, digits: usize) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf"
        } else {
            "Inf"
        }
        .to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let precision = digits.saturating_sub(1).min(MAX_DIGITS);
    let scientific = format!("{value:.precision$e}");
    scientific_to_decimal(&scientific)
}

fn significant_decimal_from_rational(
    numerator: &BigInt,
    denominator: &BigInt,
    digits: usize,
) -> String {
    let negative = numerator.is_negative() ^ denominator.is_negative();
    let numerator = numerator.abs();
    let denominator = denominator.abs();
    if denominator.is_zero() {
        return "NaN".to_string();
    }
    let integer = &numerator / &denominator;
    let mut remainder = numerator % &denominator;
    let mut out = String::new();
    if negative && (!integer.is_zero() || !remainder.is_zero()) {
        out.push('-');
    }
    let integer_text = integer.to_string();
    out.push_str(&integer_text);
    if remainder.is_zero() {
        return out;
    }

    let mut significant = if integer.is_zero() {
        0
    } else {
        integer_text.len()
    };
    if significant >= digits {
        return out;
    }
    out.push('.');
    let mut fractional_places = 0usize;
    let mut significant_positions = Vec::new();
    while !remainder.is_zero() && significant < digits.saturating_add(1) {
        remainder *= 10;
        let digit = &remainder / &denominator;
        remainder %= &denominator;
        let digit = digit.to_u8().unwrap_or(0);
        out.push(char::from(b'0' + digit));
        fractional_places += 1;
        if digit != 0 || significant > 0 || !integer.is_zero() {
            significant_positions.push(out.len() - 1);
            significant += 1;
        }
        if fractional_places > digits.saturating_add(64) {
            break;
        }
    }
    round_to_significant_digits(&mut out, &significant_positions, digits);
    out
}

fn round_to_significant_digits(text: &mut String, significant_positions: &[usize], digits: usize) {
    if significant_positions.len() <= digits {
        return;
    }
    let guard = significant_positions[digits];
    let round_up = text.as_bytes()[guard] >= b'5';
    text.truncate(guard);
    if !round_up {
        trim_trailing_decimal_zeros(text);
        return;
    }
    for &pos in significant_positions[..digits].iter().rev() {
        let digit = text.as_bytes()[pos];
        if digit < b'9' {
            text.replace_range(pos..=pos, &char::from(digit + 1).to_string());
            trim_trailing_decimal_zeros(text);
            return;
        }
        text.replace_range(pos..=pos, "0");
    }
    let insert_at = if text.starts_with('-') { 1 } else { 0 };
    let unsigned = &text[insert_at..];
    if unsigned.starts_with("0.") {
        text.truncate(insert_at);
        text.push('1');
        return;
    }
    text.insert(insert_at, '1');
    trim_trailing_decimal_zeros(text);
}

fn scientific_to_decimal(scientific: &str) -> String {
    let Some((mantissa, exponent)) = scientific.split_once('e') else {
        return scientific.to_string();
    };
    let exponent = exponent.parse::<i32>().unwrap_or(0);
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches('-').trim_start_matches('+');
    let mut digits = mantissa.replace('.', "");
    while digits.starts_with('0') && digits.len() > 1 {
        digits.remove(0);
    }

    let decimal_pos = 1i32 + exponent;
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if decimal_pos <= 0 {
        out.push_str("0.");
        for _ in 0..decimal_pos.unsigned_abs() {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decimal_pos as usize >= digits.len() {
        out.push_str(&digits);
        for _ in 0..(decimal_pos as usize - digits.len()) {
            out.push('0');
        }
    } else {
        let split = decimal_pos as usize;
        out.push_str(&digits[..split]);
        out.push('.');
        out.push_str(&digits[split..]);
    }
    out
}

fn trim_trailing_decimal_zeros(text: &mut String) {
    if let Some(dot) = text.find('.') {
        while text.len() > dot + 1 && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
}

fn vpa_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    vpa_error_with_message(error, error.message)
}

fn vpa_error_with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(message.to_string()).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_formats_numeric_input_to_requested_digits() {
        let value = block_on(vpa_builtin(
            Value::Num(std::f64::consts::PI),
            vec![Value::Num(50.0)],
        ))
        .expect("vpa");
        let text = value.to_string();
        assert!(text.starts_with("3.141592653589793"));
        assert_eq!(text.chars().filter(|ch| ch.is_ascii_digit()).count(), 50);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_uses_current_digits_when_precision_is_omitted() {
        super::super::digits::set_current_digits_for_test(12);
        let value =
            block_on(vpa_builtin(Value::Num(std::f64::consts::PI), Vec::new())).expect("vpa");
        assert_eq!(value.to_string(), "3.14159265359");
        super::super::digits::set_current_digits_for_test(super::super::digits::DEFAULT_DIGITS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_formats_exact_rational_text() {
        let rational = SymbolicExpr::rational(1, 3).expect("rational");
        let value = block_on(vpa_builtin(
            Value::Symbolic(rational),
            vec![Value::Num(20.0)],
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "0.33333333333333333333");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_rounds_exact_rationals_to_requested_digits() {
        let rational = SymbolicExpr::rational(2, 3).expect("rational");
        let value = block_on(vpa_builtin(
            Value::Symbolic(rational),
            vec![Value::Num(1.0)],
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "0.7");

        let rational = SymbolicExpr::rational(999, 1000).expect("rational");
        let value = block_on(vpa_builtin(
            Value::Symbolic(rational),
            vec![Value::Num(1.0)],
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "1");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_preserves_symbolic_variables() {
        let value = block_on(vpa_builtin(
            Value::Symbolic(SymbolicExpr::variable("x")),
            vec![Value::Num(20.0)],
        ))
        .expect("vpa");
        assert_eq!(value.to_string(), "x");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vpa_rejects_invalid_precision() {
        let err = block_on(vpa_builtin(Value::Num(1.0), vec![Value::Num(0.0)])).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:vpa:InvalidDigits"));
    }
}
