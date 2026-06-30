//! MATLAB-compatible `bi2de` and `de2bi` Communications Toolbox helpers.

use runmat_builtins::{LogicalArray, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BI2DE_NAME: &str = "bi2de";
const DE2BI_NAME: &str = "de2bi";
const INTEGER_TOL: f64 = 1e-9;
const MAX_EXACT_INTEGER: u128 = 1u128 << 52;
const MAX_DE2BI_OUTPUT_ELEMENTS: usize = 10_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DigitOrder {
    RightMsb,
    LeftMsb,
}

#[derive(Debug)]
struct DigitMatrix {
    data: Vec<usize>,
    rows: usize,
    cols: usize,
}

fn bi2de_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BI2DE_NAME)
        .build()
}

fn de2bi_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(DE2BI_NAME)
        .build()
}

fn vector_or_scalar_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::tensor()
}

fn matrix_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::tensor()
}

#[runtime_builtin(
    name = "bi2de",
    category = "comms/conversion",
    summary = "Convert rows of base-p digits to decimal integers.",
    keywords = "bi2de,binary,decimal,base,communications,right-msb,left-msb",
    type_resolver(vector_or_scalar_type),
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn bi2de_builtin(b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let opts = Bi2deOptions::parse(&rest)?;
    let digits = digit_matrix_from_value(b, BI2DE_NAME).await?;
    validate_digit_matrix(&digits, opts.base, BI2DE_NAME)?;

    let mut out = Vec::with_capacity(digits.rows);
    for row in 0..digits.rows {
        let value = row_digits_to_decimal(&digits, row, opts.base, opts.order)?;
        out.push(value as f64);
    }

    if out.len() == 1 {
        Ok(Value::Num(out[0]))
    } else {
        let rows = out.len();
        Ok(Value::Tensor(
            Tensor::new(out, vec![rows, 1]).map_err(|err| bi2de_error(format!("bi2de: {err}")))?,
        ))
    }
}

#[runtime_builtin(
    name = "de2bi",
    category = "comms/conversion",
    summary = "Convert nonnegative decimal integers to rows of base-p digits.",
    keywords = "de2bi,decimal,binary,base,communications,right-msb,left-msb",
    type_resolver(matrix_type),
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn de2bi_builtin(d: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let opts = De2biOptions::parse(&rest)?;
    let decimals = decimal_vector_from_value(d).await?;
    let needed_width = decimals
        .iter()
        .copied()
        .map(|value| digits_required(value, opts.base))
        .max()
        .unwrap_or(0);
    let width = opts.width.unwrap_or(needed_width);
    if opts.width.is_some() && width < needed_width {
        return Err(de2bi_error(format!(
            "de2bi: N is too small to represent an input value in base {}",
            opts.base
        )));
    }

    let rows = decimals.len();
    let output_len = checked_de2bi_output_len(rows, width)?;
    let mut out = vec![0.0; output_len];
    for (row, value) in decimals.into_iter().enumerate() {
        write_digits(value, opts.base, width, opts.order, &mut out, row, rows);
    }

    Ok(Value::Tensor(
        Tensor::new(out, vec![rows, width]).map_err(|err| de2bi_error(format!("de2bi: {err}")))?,
    ))
}

#[derive(Debug)]
struct Bi2deOptions {
    base: usize,
    order: DigitOrder,
}

impl Bi2deOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut base = 2usize;
        let mut order = DigitOrder::RightMsb;
        match args {
            [] => {}
            [one] => {
                if let Some(parsed) = parse_order(one, BI2DE_NAME)? {
                    order = parsed;
                } else {
                    base = parse_base(one, BI2DE_NAME)?;
                }
            }
            [base_value, flag] => {
                base = parse_base(base_value, BI2DE_NAME)?;
                order = parse_required_order(flag, BI2DE_NAME)?;
            }
            _ => return Err(bi2de_error("bi2de: expected at most three inputs")),
        }
        Ok(Self { base, order })
    }
}

#[derive(Debug)]
struct De2biOptions {
    width: Option<usize>,
    base: usize,
    order: DigitOrder,
}

impl De2biOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut width = None;
        let mut base = 2usize;
        let mut order = DigitOrder::RightMsb;
        let mut idx = 0usize;

        if let Some(first) = args.get(idx) {
            if let Some(parsed) = parse_order(first, DE2BI_NAME)? {
                order = parsed;
                idx += 1;
                if idx != args.len() {
                    return Err(de2bi_error(
                        "de2bi: digit order flag must follow width and base arguments",
                    ));
                }
            } else {
                width = parse_width(first)?;
                idx += 1;
            }
        }

        if let Some(second) = args.get(idx) {
            if let Some(parsed) = parse_order(second, DE2BI_NAME)? {
                order = parsed;
                idx += 1;
            } else {
                base = parse_base(second, DE2BI_NAME)?;
                idx += 1;
            }
        }

        if let Some(third) = args.get(idx) {
            order = parse_required_order(third, DE2BI_NAME)?;
            idx += 1;
        }

        if idx != args.len() {
            return Err(de2bi_error("de2bi: expected at most four inputs"));
        }
        Ok(Self { width, base, order })
    }
}

async fn digit_matrix_from_value(
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<DigitMatrix> {
    match value {
        Value::Tensor(tensor) => digit_matrix_from_tensor(tensor, builtin),
        Value::LogicalArray(logical) => digit_matrix_from_logical(logical, builtin),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            digit_matrix_from_tensor(tensor, builtin)
        }
        Value::Num(n) => Ok(DigitMatrix {
            data: vec![number_to_digit(n, builtin, "B")?],
            rows: 1,
            cols: 1,
        }),
        Value::Int(i) => Ok(DigitMatrix {
            data: vec![number_to_digit(i.to_f64(), builtin, "B")?],
            rows: 1,
            cols: 1,
        }),
        Value::Bool(b) => Ok(DigitMatrix {
            data: vec![usize::from(b)],
            rows: 1,
            cols: 1,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(bi2de_error("bi2de: B must contain real digits"))
        }
        other => Err(build_runtime_error(format!(
            "{builtin}: B must be a numeric or logical matrix, got {other:?}"
        ))
        .with_builtin(builtin)
        .build()),
    }
}

fn digit_matrix_from_tensor(tensor: Tensor, builtin: &'static str) -> BuiltinResult<DigitMatrix> {
    ensure_matrix_shape(&tensor.shape, builtin, "B")?;
    let data = tensor
        .data
        .into_iter()
        .map(|value| number_to_digit(value, builtin, "B"))
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(DigitMatrix {
        data,
        rows: tensor.rows,
        cols: tensor.cols,
    })
}

fn digit_matrix_from_logical(
    logical: LogicalArray,
    builtin: &'static str,
) -> BuiltinResult<DigitMatrix> {
    ensure_matrix_shape(&logical.shape, builtin, "B")?;
    let rows = logical.shape.first().copied().unwrap_or(1);
    let cols = logical.shape.get(1).copied().unwrap_or(logical.data.len());
    Ok(DigitMatrix {
        data: logical.data.into_iter().map(usize::from).collect(),
        rows,
        cols,
    })
}

async fn decimal_vector_from_value(value: Value) -> BuiltinResult<Vec<u128>> {
    match value {
        Value::Tensor(tensor) => tensor
            .data
            .into_iter()
            .map(|value| number_to_decimal(value, "D"))
            .collect(),
        Value::LogicalArray(logical) => Ok(logical.data.into_iter().map(u128::from).collect()),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            tensor
                .data
                .into_iter()
                .map(|value| number_to_decimal(value, "D"))
                .collect()
        }
        Value::Num(n) => Ok(vec![number_to_decimal(n, "D")?]),
        Value::Int(i) => Ok(vec![number_to_decimal(i.to_f64(), "D")?]),
        Value::Bool(b) => Ok(vec![if b { 1 } else { 0 }]),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(de2bi_error(
            "de2bi: D must contain real nonnegative integers",
        )),
        other => Err(de2bi_error(format!(
            "de2bi: D must be numeric or logical, got {other:?}"
        ))),
    }
}

fn ensure_matrix_shape(shape: &[usize], builtin: &'static str, name: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(
            build_runtime_error(format!("{builtin}: {name} must be a vector or 2-D matrix"))
                .with_builtin(builtin)
                .build(),
        )
    }
}

fn validate_digit_matrix(
    digits: &DigitMatrix,
    base: usize,
    builtin: &'static str,
) -> BuiltinResult<()> {
    for &digit in &digits.data {
        if digit >= base {
            return Err(build_runtime_error(format!(
                "{builtin}: digit values must be integers in the range [0, {}]",
                base - 1
            ))
            .with_builtin(builtin)
            .build());
        }
    }
    Ok(())
}

fn row_digits_to_decimal(
    digits: &DigitMatrix,
    row: usize,
    base: usize,
    order: DigitOrder,
) -> BuiltinResult<u128> {
    let base_u = base as u128;
    let mut acc = 0u128;
    match order {
        DigitOrder::LeftMsb => {
            for col in 0..digits.cols {
                let digit = digits.data[row + col * digits.rows] as u128;
                acc = acc
                    .checked_mul(base_u)
                    .and_then(|value| value.checked_add(digit))
                    .ok_or_else(|| bi2de_error("bi2de: decimal value overflow"))?;
                ensure_exact_range(acc, BI2DE_NAME)?;
            }
        }
        DigitOrder::RightMsb => {
            let mut place = 1u128;
            for col in 0..digits.cols {
                let digit = digits.data[row + col * digits.rows] as u128;
                let term = digit
                    .checked_mul(place)
                    .ok_or_else(|| bi2de_error("bi2de: decimal value overflow"))?;
                acc = acc
                    .checked_add(term)
                    .ok_or_else(|| bi2de_error("bi2de: decimal value overflow"))?;
                ensure_exact_range(acc, BI2DE_NAME)?;
                if col + 1 < digits.cols {
                    place = place
                        .checked_mul(base_u)
                        .ok_or_else(|| bi2de_error("bi2de: decimal value overflow"))?;
                    ensure_exact_range(place, BI2DE_NAME)?;
                }
            }
        }
    }
    Ok(acc)
}

fn write_digits(
    mut value: u128,
    base: usize,
    width: usize,
    order: DigitOrder,
    out: &mut [f64],
    row: usize,
    rows: usize,
) {
    let base_u = base as u128;
    for offset in 0..width {
        let digit = (value % base_u) as f64;
        value /= base_u;
        let col = match order {
            DigitOrder::RightMsb => offset,
            DigitOrder::LeftMsb => width - 1 - offset,
        };
        out[row + col * rows] = digit;
    }
}

fn digits_required(mut value: u128, base: usize) -> usize {
    if value == 0 {
        return 1;
    }
    let base_u = base as u128;
    let mut count = 0usize;
    while value > 0 {
        value /= base_u;
        count += 1;
    }
    count
}

fn checked_de2bi_output_len(rows: usize, width: usize) -> BuiltinResult<usize> {
    let len = rows
        .checked_mul(width)
        .ok_or_else(|| de2bi_error("de2bi: requested output exceeds the maximum supported size"))?;
    if len > MAX_DE2BI_OUTPUT_ELEMENTS {
        return Err(de2bi_error(format!(
            "de2bi: requested output has {len} elements; limit is {MAX_DE2BI_OUTPUT_ELEMENTS}"
        )));
    }
    Ok(len)
}

fn parse_width(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_numeric(value) {
        return Ok(None);
    }
    let width = parse_nonnegative_integer(value, DE2BI_NAME, "N")?;
    if width > usize::MAX as u128 {
        return Err(de2bi_error("de2bi: N is too large for this platform"));
    }
    Ok(Some(width as usize))
}

fn parse_base(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    let base = parse_nonnegative_integer(value, builtin, "P")?;
    if base < 2 {
        return Err(
            build_runtime_error(format!("{builtin}: P must be an integer greater than 1"))
                .with_builtin(builtin)
                .build(),
        );
    }
    if base > MAX_EXACT_INTEGER || base > usize::MAX as u128 {
        return Err(
            build_runtime_error(format!("{builtin}: P is too large for this platform"))
                .with_builtin(builtin)
                .build(),
        );
    }
    Ok(base as usize)
}

fn parse_nonnegative_integer(
    value: &Value,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<u128> {
    let n = scalar_number(value, builtin, name)?;
    if !n.is_finite() {
        return Err(
            build_runtime_error(format!("{builtin}: {name} must be finite"))
                .with_builtin(builtin)
                .build(),
        );
    }
    let rounded = n.round();
    if (n - rounded).abs() > INTEGER_TOL || rounded < 0.0 {
        return Err(build_runtime_error(format!(
            "{builtin}: {name} must be a nonnegative integer"
        ))
        .with_builtin(builtin)
        .build());
    }
    if rounded > MAX_EXACT_INTEGER as f64 {
        return Err(build_runtime_error(format!(
            "{builtin}: {name} exceeds the maximum exact integer supported by RunMat"
        ))
        .with_builtin(builtin)
        .build());
    }
    Ok(rounded as u128)
}

fn scalar_number(value: &Value, builtin: &'static str, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(
            build_runtime_error(format!("{builtin}: {name} must be a scalar"))
                .with_builtin(builtin)
                .build(),
        ),
    }
}

fn number_to_digit(value: f64, builtin: &'static str, name: &str) -> BuiltinResult<usize> {
    let integer = parse_nonnegative_integer(&Value::Num(value), builtin, name)?;
    if integer > usize::MAX as u128 {
        return Err(build_runtime_error(format!(
            "{builtin}: {name} value is too large for this platform"
        ))
        .with_builtin(builtin)
        .build());
    }
    Ok(integer as usize)
}

fn number_to_decimal(value: f64, name: &str) -> BuiltinResult<u128> {
    parse_nonnegative_integer(&Value::Num(value), DE2BI_NAME, name)
}

fn ensure_exact_range(value: u128, builtin: &'static str) -> BuiltinResult<()> {
    if value > MAX_EXACT_INTEGER {
        Err(build_runtime_error(format!(
            "{builtin}: result exceeds the maximum exact integer supported by RunMat"
        ))
        .with_builtin(builtin)
        .build())
    } else {
        Ok(())
    }
}

fn parse_order(value: &Value, builtin: &'static str) -> BuiltinResult<Option<DigitOrder>> {
    let Some(text) = value_as_string(value) else {
        return Ok(None);
    };
    match normalize_flag(&text).as_str() {
        "rightmsb" => Ok(Some(DigitOrder::RightMsb)),
        "leftmsb" => Ok(Some(DigitOrder::LeftMsb)),
        other => Err(
            build_runtime_error(format!("{builtin}: unsupported digit order '{other}'"))
                .with_builtin(builtin)
                .build(),
        ),
    }
}

fn parse_required_order(value: &Value, builtin: &'static str) -> BuiltinResult<DigitOrder> {
    parse_order(value, builtin)?.ok_or_else(|| {
        build_runtime_error(format!("{builtin}: digit order flag must be a string"))
            .with_builtin(builtin)
            .build()
    })
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn normalize_flag(flag: &str) -> String {
    flag.chars()
        .filter(|ch| *ch != '-' && *ch != '_' && !ch.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.data.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).expect("tensor"))
    }

    fn bi2de(value: Value, rest: Vec<Value>) -> Value {
        block_on(super::bi2de_builtin(value, rest)).expect("bi2de")
    }

    fn de2bi(value: Value, rest: Vec<Value>) -> Tensor {
        match block_on(super::de2bi_builtin(value, rest)).expect("de2bi") {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    fn value_data(value: Value) -> Vec<f64> {
        match value {
            Value::Num(n) => vec![n],
            Value::Tensor(tensor) => tensor.data,
            other => panic!("expected numeric output, got {other:?}"),
        }
    }

    #[test]
    fn bi2de_row_vector_defaults_to_right_msb() {
        assert_eq!(
            value_data(bi2de(tensor(vec![1.0, 0.0, 1.0, 0.0], vec![1, 4]), vec![])),
            vec![5.0]
        );
    }

    #[test]
    fn bi2de_left_msb_flag() {
        assert_eq!(
            value_data(bi2de(
                tensor(vec![1.0, 0.0, 1.0, 0.0], vec![1, 4]),
                vec![Value::from("left-msb")]
            )),
            vec![10.0]
        );
    }

    #[test]
    fn bi2de_converts_each_matrix_row() {
        let out = bi2de(
            tensor(
                vec![
                    1.0, 0.0, 1.0, // first column
                    0.0, 1.0, 1.0, // second column
                    1.0, 1.0, 0.0, // third column
                ],
                vec![3, 3],
            ),
            vec![],
        );
        let Value::Tensor(tensor) = out else {
            panic!("expected column vector")
        };
        assert_eq!(tensor.shape, vec![3, 1]);
        assert_eq!(tensor.data, vec![5.0, 6.0, 3.0]);
    }

    #[test]
    fn bi2de_accepts_arbitrary_base() {
        assert_eq!(
            value_data(bi2de(
                tensor(vec![2.0, 1.0, 0.0], vec![1, 3]),
                vec![Value::Num(3.0)]
            )),
            vec![5.0]
        );
        assert_eq!(
            value_data(bi2de(
                tensor(vec![2.0, 1.0, 0.0], vec![1, 3]),
                vec![Value::Num(3.0), Value::from("left-msb")]
            )),
            vec![21.0]
        );
    }

    #[test]
    fn bi2de_rejects_digits_outside_base() {
        let err = block_on(super::bi2de_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            vec![],
        ))
        .expect_err("invalid binary digit should fail");
        assert!(err.to_string().contains("digit values must be integers"));
    }

    #[test]
    fn de2bi_uses_minimum_width_right_msb() {
        let out = de2bi(tensor(vec![0.0, 1.0, 2.0, 5.0], vec![1, 4]), vec![]);
        assert_eq!(out.shape, vec![4, 3]);
        assert_eq!(
            out.data,
            vec![
                0.0, 1.0, 0.0, 1.0, // 2^0 column
                0.0, 0.0, 1.0, 0.0, // 2^1 column
                0.0, 0.0, 0.0, 1.0, // 2^2 column
            ]
        );
    }

    #[test]
    fn de2bi_respects_width_base_and_left_msb() {
        let out = de2bi(
            tensor(vec![5.0, 21.0], vec![2, 1]),
            vec![Value::Num(4.0), Value::Num(3.0), Value::from("left-msb")],
        );
        assert_eq!(out.shape, vec![2, 4]);
        assert_eq!(
            out.data,
            vec![
                0.0, 0.0, // 3^3 column
                0.0, 2.0, // 3^2 column
                1.0, 1.0, // 3^1 column
                2.0, 0.0, // 3^0 column
            ]
        );
    }

    #[test]
    fn de2bi_round_trips_with_bi2de() {
        let decimals = tensor(vec![0.0, 1.0, 2.0, 7.0, 15.0], vec![5, 1]);
        let bits = de2bi(
            decimals,
            vec![Value::Num(4.0), Value::Num(2.0), Value::from("right-msb")],
        );
        let back = bi2de(Value::Tensor(bits), vec![Value::Num(2.0)]);
        let Value::Tensor(back) = back else {
            panic!("expected tensor")
        };
        assert_eq!(back.data, vec![0.0, 1.0, 2.0, 7.0, 15.0]);
    }

    #[test]
    fn de2bi_accepts_empty_width_placeholder_with_base_and_order() {
        let out = de2bi(
            tensor(vec![5.0, 21.0], vec![2, 1]),
            vec![
                tensor(vec![], vec![0, 0]),
                Value::Num(3.0),
                Value::from("left-msb"),
            ],
        );
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(
            out.data,
            vec![
                0.0, 2.0, // 3^2 column
                1.0, 1.0, // 3^1 column
                2.0, 0.0, // 3^0 column
            ]
        );
    }

    #[test]
    fn de2bi_rejects_too_small_width() {
        let err = block_on(super::de2bi_builtin(Value::Num(8.0), vec![Value::Num(3.0)]))
            .expect_err("width too small should fail");
        assert!(err.to_string().contains("N is too small"));
    }

    #[test]
    fn de2bi_rejects_excessive_output_before_allocation() {
        let err = block_on(super::de2bi_builtin(
            Value::Num(0.0),
            vec![Value::Num((MAX_DE2BI_OUTPUT_ELEMENTS + 1) as f64)],
        ))
        .expect_err("oversized output should fail");
        assert!(err.to_string().contains("requested output"));
    }
}
