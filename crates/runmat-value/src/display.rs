use crate::array::{should_expand_nd_display, write_nd_pages};
use crate::*;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FormatMode {
    /// 4 decimal places, fixed or scientific (MATLAB default).
    #[default]
    Short,
    /// 15 decimal places, fixed or scientific.
    Long,
    /// Always scientific notation, 4 decimal places.
    ShortE,
    /// Always scientific notation, 14 decimal places.
    LongE,
    /// Compact: shorter of fixed/scientific, 5 significant digits.
    ShortG,
    /// Compact: shorter of fixed/scientific, 15 significant digits.
    LongG,
    /// Rational approximation (p/q).
    Rational,
    /// IEEE 754 hexadecimal representation.
    Hex,
}

runmat_thread_local! {
    static DISPLAY_FORMAT: RefCell<FormatMode> = const { RefCell::new(FormatMode::Short) };
}

pub fn set_display_format(mode: FormatMode) {
    DISPLAY_FORMAT.with(|c| *c.borrow_mut() = mode);
}

pub fn get_display_format() -> FormatMode {
    DISPLAY_FORMAT.with(|c| *c.borrow())
}

/// Format a number using the current thread-local display format.
pub fn format_number(value: f64) -> String {
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
    let mode = get_display_format();
    if mode == FormatMode::Hex {
        return fmt_hex(value);
    }
    let v = if value == 0.0 { 0.0 } else { value };
    match mode {
        FormatMode::Short => fmt_short(v),
        FormatMode::Long => fmt_long(v),
        FormatMode::ShortE => fmt_sci(v, 4),
        FormatMode::LongE => fmt_sci(v, 14),
        FormatMode::ShortG => fmt_compact(v, 5),
        FormatMode::LongG => fmt_compact(v, 15),
        FormatMode::Rational => fmt_rational(v),
        FormatMode::Hex => unreachable!("hex mode handled before zero normalization"),
    }
}

/// Reformat Rust's `e`-notation exponent into MATLAB style (`e+02`, `e-03`).
fn matlab_exp(s: &str) -> String {
    if let Some(e_pos) = s.find('e') {
        let mantissa = &s[..e_pos];
        let exp: i32 = s[e_pos + 1..].parse().unwrap_or(0);
        let sign = if exp >= 0 { '+' } else { '-' };
        format!("{mantissa}e{sign}{:02}", exp.unsigned_abs())
    } else {
        s.to_string()
    }
}

fn fmt_sci(v: f64, dec: usize) -> String {
    if v == 0.0 {
        return format!("0.{:0>dec$}e+00", 0, dec = dec);
    }
    let s = format!("{v:.dec$e}");
    matlab_exp(&s)
}

fn fmt_short(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        return "0".to_string();
    }
    if v.fract() == 0.0 && abs < 1e15 {
        return format!("{}", v as i64);
    }
    if (0.001..10000.0).contains(&abs) {
        format!("{:.4}", v)
    } else {
        fmt_sci(v, 4)
    }
}

fn fmt_long(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        return "0".to_string();
    }
    if v.fract() == 0.0 && abs < 1e15 {
        return format!("{}", v as i64);
    }
    if (0.001..10000.0).contains(&abs) {
        format!("{:.15}", v)
    } else {
        fmt_sci(v, 14)
    }
}

fn fmt_compact(v: f64, sig_digits: usize) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        return "0".to_string();
    }
    let use_scientific = !(1e-4..1e6).contains(&abs);
    if use_scientific {
        let dec = sig_digits - 1;
        let s = format!("{v:.dec$e}");
        // trim trailing zeros in mantissa then reformat exponent
        if let Some(e_pos) = s.find('e') {
            let exp_part = &s[e_pos..];
            let mut mantissa = s[..e_pos].to_string();
            if let Some(dot) = mantissa.find('.') {
                let mut end = mantissa.len();
                while end > dot + 1 && mantissa.as_bytes()[end - 1] == b'0' {
                    end -= 1;
                }
                if mantissa.as_bytes()[end - 1] == b'.' {
                    end -= 1;
                }
                mantissa.truncate(end);
            }
            return matlab_exp(&format!("{mantissa}{exp_part}"));
        }
        return matlab_exp(&s);
    }
    let exp10 = abs.log10().floor() as i32;
    let decimals = ((sig_digits as i32 - 1 - exp10).max(0)) as usize;
    let pow = 10f64.powi(decimals as i32);
    let rounded = (v * pow).round() / pow;
    let mut s = format!("{rounded:.decimals$}");
    if let Some(dot) = s.find('.') {
        let mut end = s.len();
        while end > dot + 1 && s.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if s.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        s.truncate(end);
    }
    if s.is_empty() || s == "-0" {
        s = "0".to_string();
    }
    s
}

fn fmt_rational(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let negative = v < 0.0;
    let abs = v.abs();
    if v.fract() == 0.0 && abs < 1e15 {
        return format!("{}", v as i64);
    }
    // Continued fraction convergents; stop at the first one within MATLAB's
    // 5e-7 relative tolerance (matches `format rational` behaviour for pi → 355/113).
    let tol = 5e-7 * abs;
    let max_d = 1_000_000i64;
    let mut n0: i64 = 1;
    let mut n1: i64 = abs.floor() as i64;
    let mut d0: i64 = 0;
    let mut d1: i64 = 1;
    let mut a = abs;
    let mut best_n = n1;
    let mut best_d = d1;
    for _ in 0..50 {
        if (abs - best_n as f64 / best_d as f64).abs() <= tol {
            break;
        }
        let f = a.fract();
        if f < 1e-10 {
            break;
        }
        a = 1.0 / f;
        let q = a.floor() as i64;
        let Some(n2) = q.checked_mul(n1).and_then(|v| v.checked_add(n0)) else {
            break;
        };
        let Some(d2) = q.checked_mul(d1).and_then(|v| v.checked_add(d0)) else {
            break;
        };
        if d2 > max_d {
            break;
        }
        best_n = n2;
        best_d = d2;
        n0 = n1;
        n1 = n2;
        d0 = d1;
        d1 = d2;
    }
    let sign = if negative { "-" } else { "" };
    if best_d == 1 {
        format!("{sign}{best_n}")
    } else {
        format!("{sign}{best_n}/{best_d}")
    }
}

fn fmt_hex(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i.decimal_string()),
            Value::Num(n) => write!(f, "{}", format_number(*n)),
            Value::Complex(re, im) => {
                if *im == 0.0 {
                    write!(f, "{}", format_number(*re))
                } else if *re == 0.0 {
                    write!(f, "{}i", format_number(*im))
                } else if *im < 0.0 {
                    write!(f, "{}-{}i", format_number(*re), format_number(im.abs()))
                } else {
                    write!(f, "{}+{}i", format_number(*re), format_number(*im))
                }
            }
            Value::Bool(b) => write!(f, "{}", if *b { 1 } else { 0 }),
            Value::LogicalArray(la) => write!(f, "{la}"),
            Value::String(s) => write!(f, "'{s}'"),
            Value::StringArray(sa) => write!(f, "{sa}"),
            Value::CharArray(ca) => write!(f, "{ca}"),
            Value::Tensor(m) => write!(f, "{m}"),
            Value::SparseTensor(m) => write!(f, "{m}"),
            Value::ComplexTensor(m) => write!(f, "{m}"),
            Value::Symbolic(expr) => write!(f, "{expr}"),
            Value::SymbolicArray(array) => write!(f, "{array}"),
            Value::Cell(ca) => ca.fmt(f),

            Value::GpuTensor(h) => write!(
                f,
                "GpuTensor(shape={:?}, device={}, buffer={})",
                h.shape, h.device_id, h.buffer_id
            ),
            Value::Object(obj) => write!(f, "{}(props={})", obj.class_name, obj.properties.len()),
            Value::ObjectArray(array) => write!(f, "{array}"),
            Value::HandleObject(h) => {
                write!(
                    f,
                    "<handle {} @0x{:x} valid={}>",
                    h.class_name,
                    h.target.addr(),
                    h.valid
                )
            }
            Value::Listener(l) => {
                write!(
                    f,
                    "<listener id={} {}@0x{:x} '{}' enabled={} valid={}>",
                    l.id,
                    l.class_name(),
                    l.target.addr(),
                    l.event_name,
                    l.enabled,
                    l.valid
                )
            }
            Value::Struct(st) => {
                write!(f, "struct {{")?;
                for (i, (key, val)) in st.fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, val)?;
                }
                write!(f, "}}")
            }
            Value::OutputList(values) => {
                write!(f, "[")?;
                for (i, value) in values.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", value)?;
                }
                write!(f, "]")
            }
            Value::FunctionHandle(name)
            | Value::ExternalFunctionHandle(name)
            | Value::MethodFunctionHandle(name) => {
                write!(f, "@{name}")
            }
            Value::BoundFunctionHandle { name, .. } => write!(f, "@{name}"),
            Value::Closure(c) => write!(
                f,
                "<closure {} captures={}>",
                c.function_name,
                c.captures.len()
            ),
            Value::ClassRef(name) => write!(f, "<class {name}>"),
            Value::MException(e) => write!(
                f,
                "MException(identifier='{}', message='{}')",
                e.identifier, e.message
            ),
            Value::Future(handle) => write!(f, "<future {}>", handle.id),
            Value::Task(handle) => write!(f, "<task {}>", handle.id),
            Value::Pool(handle) => write!(f, "<pool {}>", handle.id),
            Value::Job(handle) => write!(f, "<job {}>", handle.id),
        }
    }
}

impl fmt::Display for ComplexTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape.len() {
            0 | 1 => {
                write!(f, "[")?;
                for i in 0..self.len() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    let s = self.format_element(i);
                    write!(f, "{s}")?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.rows;
                let cols = self.cols;
                write!(f, "[")?;
                for r in 0..rows {
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, " ")?;
                        }
                        let s = self.format_element(r + c * rows);
                        write!(f, "{s}")?;
                    }
                    if r + 1 < rows {
                        write!(f, "; ")?;
                    }
                }
                write!(f, "]")
            }
            _ => {
                if should_expand_nd_display(&self.shape) {
                    write_nd_pages(f, &self.shape, |f, idx| {
                        write!(f, "{}", self.format_element(idx))
                    })
                } else {
                    write!(f, "ComplexTensor(shape={:?})", self.shape)
                }
            }
        }
    }
}

#[cfg(test)]
mod display_tests {
    use super::{
        fmt_rational, format_number, set_display_format, ComplexTensor, FormatMode,
        IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor,
    };

    #[test]
    fn fmt_rational_large_value_with_tiny_fract_does_not_overflow() {
        // abs ~1e15 with a small fractional part: q*n1 would overflow i64 without
        // checked arithmetic.
        let result = std::panic::catch_unwind(|| fmt_rational(1_000_000_000_000_000.000_1));
        assert!(
            result.is_ok(),
            "fmt_rational panicked on large value with tiny fract"
        );

        // Negative counterpart.
        let result = std::panic::catch_unwind(|| fmt_rational(-1_000_000_000_000_000.000_1));
        assert!(
            result.is_ok(),
            "fmt_rational panicked on negative large value with tiny fract"
        );
    }

    #[test]
    fn tensor_nd_display_uses_page_headers() {
        let tensor = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![2, 3, 2],
        )
        .expect("tensor");
        let rendered = tensor.to_string();
        assert!(rendered.contains("(:, :, 1) ="));
        assert!(rendered.contains("(:, :, 2) ="));
        assert!(rendered.contains("  1  0  0"));
    }

    #[test]
    fn dense_integer_tensor_display_uses_exact_storage_values() {
        let vector = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![2],
        )
        .expect("uint64 vector");
        assert_eq!(
            vector.to_string(),
            "[18446744073709551615 9007199254740993]"
        );

        let matrix = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MIN, -1, 1, i64::MAX]),
            vec![2, 2],
        )
        .expect("int64 matrix");
        let rendered = matrix.to_string();
        assert!(rendered.contains("-9223372036854775808"));
        assert!(rendered.contains("9223372036854775807"));
    }

    #[test]
    fn dense_integer_nd_display_uses_exact_storage_values() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 7, 8]),
            vec![1, 2, 2],
        )
        .expect("uint64 nd tensor");
        let rendered = tensor.to_string();
        assert!(rendered.contains("(:, :, 1) ="));
        assert!(rendered.contains("(:, :, 2) ="));
        assert!(rendered.contains("18446744073709551615"));
        assert!(rendered.contains("9007199254740993"));
    }

    #[test]
    fn tensor_nd_display_falls_back_for_large_arrays() {
        let tensor = Tensor::new(vec![0.0; 4097], vec![1, 1, 4097]).expect("tensor");
        assert_eq!(tensor.to_string(), "Tensor(shape=[1, 1, 4097])");
    }

    #[test]
    fn logical_nd_display_uses_headers_and_fallback_summary() {
        let logical =
            LogicalArray::new(vec![1, 0, 0, 1, 1, 0, 0, 1], vec![2, 2, 2]).expect("logical");
        let rendered = logical.to_string();
        assert!(rendered.contains("(:, :, 1) ="));
        assert!(rendered.contains("(:, :, 2) ="));

        let large = LogicalArray::new(vec![1; 4097], vec![1, 1, 4097]).expect("large logical");
        assert_eq!(large.to_string(), "1x1x4097 logical array");
    }

    #[test]
    fn complex_nd_display_uses_page_headers() {
        let complex = ComplexTensor::new(
            vec![(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
            vec![2, 1, 2],
        )
        .expect("complex");
        let rendered = complex.to_string();
        assert!(rendered.contains("(:, :, 1) ="));
        assert!(rendered.contains("(:, :, 2) ="));
    }

    #[test]
    fn typed_complex_integer_display_uses_exact_components() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
            IntegerStorage::U64(vec![7, 0]),
        )
        .expect("matching components");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).expect("typed complex");
        assert_eq!(
            tensor.to_string(),
            format!("[{}+7i {}]", u64::MAX, 1_u64 << 63)
        );

        let negative_imaginary = IntegerComplexStorage::new(
            IntegerStorage::I64(vec![1]),
            IntegerStorage::I64(vec![i64::MIN]),
        )
        .expect("matching components");
        let tensor =
            ComplexTensor::new_integer(negative_imaginary, vec![1, 1]).expect("typed complex");
        assert_eq!(
            tensor.to_string(),
            format!("[1-{}i]", i64::MIN.unsigned_abs())
        );
    }

    #[test]
    fn format_hex_preserves_negative_zero_sign_bit() {
        set_display_format(FormatMode::Hex);
        assert_eq!(format_number(-0.0), "8000000000000000");
        assert_eq!(format_number(0.0), "0000000000000000");
        set_display_format(FormatMode::Short);
    }
}
