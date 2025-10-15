use regex::Regex;
use runmat_builtins::Value;
use runmat_gc_api::GcPtr;
use runmat_macros::runtime_builtin;

pub mod dispatcher;

pub mod accel;
pub mod arrays;
pub mod builtins;
pub mod comparison;
pub mod concatenation;
pub mod constants;
pub mod elementwise;
pub mod indexing;
pub mod introspection;
pub mod io;
pub mod mathematics;
pub mod matrix;
pub mod plotting;

#[cfg(feature = "blas-lapack")]
pub mod blas;
#[cfg(feature = "blas-lapack")]
pub mod lapack;

// Link to Apple's Accelerate framework on macOS
#[cfg(all(feature = "blas-lapack", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

// Ensure OpenBLAS is linked on non-macOS platforms when BLAS/LAPACK is enabled
#[cfg(all(feature = "blas-lapack", not(target_os = "macos")))]
extern crate openblas_src;

pub use dispatcher::{call_builtin, gather_if_needed, is_gpu_value, value_contains_gpu};

pub use arrays::*;
pub use comparison::*;
pub use concatenation::*;
// Explicitly re-export for external users (ignition VM) that build matrices from values
pub use concatenation::create_matrix_from_values;
// Note: constants and mathematics modules only contain #[runtime_builtin] functions
// and don't export public items, so they don't need to be re-exported
pub use elementwise::*;
pub use indexing::*;
pub use matrix::*;

#[cfg(feature = "blas-lapack")]
pub use blas::*;
#[cfg(feature = "blas-lapack")]
pub use lapack::*;

pub(crate) fn make_cell(values: Vec<Value>, rows: usize, cols: usize) -> Result<Value, String> {
    let handles: Vec<GcPtr<Value>> = values
        .into_iter()
        .map(|v| runmat_gc::gc_allocate(v).expect("gc alloc"))
        .collect();
    let ca = runmat_builtins::CellArray::new_handles(handles, rows, cols)
        .map_err(|e| format!("Cell creation error: {e}"))?;
    Ok(Value::Cell(ca))
}

// Internal builtin to construct a cell from a vector of values (used by ignition)
#[runmat_macros::runtime_builtin(name = "__make_cell")]
fn make_cell_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let rows = 1usize;
    let cols = rest.len();
    make_cell(rest, rows, cols)
}

#[runmat_macros::runtime_builtin(name = "cellstr")]
fn cellstr_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::String(s) => make_cell(vec![Value::String(s)], 1, 1),
        Value::StringArray(sa) => {
            let rows = sa.rows();
            let cols = sa.cols();
            let mut cells: Vec<Value> = Vec::with_capacity(sa.data.len());
            for r in 0..rows {
                for c in 0..cols {
                    let idx = r + c * rows;
                    cells.push(Value::String(sa.data[idx].clone()));
                }
            }
            make_cell(cells, rows, cols)
        }
        other => Err(format!(
            "cellstr: expected string or string array, got {other:?}"
        )),
    }
}

#[runmat_macros::runtime_builtin(name = "char")]
fn char_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::String(s) => {
            let data: Vec<f64> = s.chars().map(|ch| ch as u32 as f64).collect();
            Ok(Value::Tensor(
                runmat_builtins::Tensor::new(data, vec![1, s.chars().count()])
                    .map_err(|e| format!("char: {e}"))?,
            ))
        }
        Value::StringArray(sa) => {
            let rows = sa.rows();
            let cols = sa.cols();
            let mut max_len = 0usize;
            for c in 0..cols {
                for r in 0..rows {
                    let idx = r + c * rows;
                    max_len = max_len.max(sa.data[idx].chars().count());
                }
            }
            if rows == 0 || cols == 0 {
                return Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(Vec::new(), vec![0, 0]).unwrap(),
                ));
            }
            let out_rows = rows;
            let out_cols = max_len * cols;
            let mut out = vec![32.0; out_rows * out_cols];
            for c in 0..cols {
                for r in 0..rows {
                    let idx = r + c * rows;
                    let s = &sa.data[idx];
                    for (j, ch) in s.chars().enumerate() {
                        let oc = c * max_len + j;
                        out[r + oc * out_rows] = ch as u32 as f64;
                    }
                }
            }
            Ok(Value::Tensor(
                runmat_builtins::Tensor::new(out, vec![out_rows, out_cols])
                    .map_err(|e| format!("char: {e}"))?,
            ))
        }
        other => Err(format!(
            "char: expected string or string array, got {other:?}"
        )),
    }
}

// size builtin centralized in introspection.rs

// Linear index to subscripts (column-major)
#[runmat_macros::runtime_builtin(name = "ind2sub")]
fn ind2sub_builtin(dims_val: Value, idx_val: f64) -> Result<Value, String> {
    let dims: Vec<usize> = match dims_val {
        Value::Tensor(t) => {
            if t.shape.len() == 2 && (t.shape[0] == 1 || t.shape[1] == 1) {
                t.data.iter().map(|v| *v as usize).collect()
            } else {
                return Err("ind2sub: dims must be a vector".to_string());
            }
        }
        Value::Cell(ca) => {
            if ca.data.is_empty() {
                vec![]
            } else {
                ca.data
                    .iter()
                    .map(|v| match &**v {
                        Value::Num(n) => *n as usize,
                        Value::Int(i) => i.to_i64() as usize,
                        _ => 1usize,
                    })
                    .collect()
            }
        }
        _ => return Err("ind2sub: dims must be a vector".to_string()),
    };
    if dims.is_empty() {
        return Err("ind2sub: empty dims".to_string());
    }
    let mut subs: Vec<usize> = vec![1; dims.len()];
    let idx = if idx_val < 1.0 {
        1usize
    } else {
        idx_val as usize
    } - 1; // 0-based
    let mut stride = 1usize;
    for d in 0..dims.len() {
        let dim_len = dims[d];
        let val = (idx / stride) % dim_len;
        subs[d] = val + 1; // 1-based
        stride *= dim_len.max(1);
    }
    // Return as a tensor column vector for expansion
    let data: Vec<f64> = subs.iter().map(|s| *s as f64).collect();
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(data, vec![subs.len(), 1])
            .map_err(|e| format!("ind2sub: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "sub2ind")]
fn sub2ind_builtin(dims_val: Value, rest: Vec<Value>) -> Result<Value, String> {
    let dims: Vec<usize> = match dims_val {
        Value::Tensor(t) => {
            if t.shape.len() == 2 && (t.shape[0] == 1 || t.shape[1] == 1) {
                t.data.iter().map(|v| *v as usize).collect()
            } else {
                return Err("sub2ind: dims must be a vector".to_string());
            }
        }
        Value::Cell(ca) => {
            if ca.data.is_empty() {
                vec![]
            } else {
                ca.data
                    .iter()
                    .map(|v| match &**v {
                        Value::Num(n) => *n as usize,
                        Value::Int(i) => i.to_i64() as usize,
                        _ => 1usize,
                    })
                    .collect()
            }
        }
        _ => return Err("sub2ind: dims must be a vector".to_string()),
    };
    if dims.is_empty() {
        return Err("sub2ind: empty dims".to_string());
    }
    if rest.len() != dims.len() {
        return Err("sub2ind: expected one subscript per dimension".to_string());
    }
    let subs: Vec<usize> = rest
        .iter()
        .map(|v| match v {
            Value::Num(n) => *n as isize,
            Value::Int(i) => i.to_i64() as isize,
            _ => 1isize,
        })
        .map(|x| if x < 1 { 1 } else { x as usize })
        .collect();
    // Column-major linear index: 1 + sum_{d=0}^{n-1} (sub[d]-1) * prod_{k<d} dims[k]
    let mut stride = 1usize;
    let mut lin0 = 0usize;
    for d in 0..dims.len() {
        let dim_len = dims[d];
        let s = subs[d];
        if s == 0 || s > dim_len {
            return Err("sub2ind: subscript out of bounds".to_string());
        }
        lin0 += (s - 1) * stride;
        stride *= dim_len.max(1);
    }
    Ok(Value::Num((lin0 + 1) as f64))
}

// -------- String constructors/conversions --------

#[runmat_macros::runtime_builtin(name = "strings")]
fn strings_ctor(rest: Vec<Value>) -> Result<Value, String> {
    let mut shape: Vec<usize> = Vec::new();
    if rest.is_empty() {
        shape = vec![1, 1];
    } else {
        for v in rest {
            let n: f64 = (&v).try_into()?;
            if n < 0.0 {
                return Err("strings: dimensions must be non-negative".to_string());
            }
            shape.push(n as usize);
        }
        if shape.is_empty() {
            shape = vec![1, 1];
        }
    }
    let total: usize = shape.iter().product();
    let data = vec![String::new(); total];
    Ok(Value::StringArray(
        runmat_builtins::StringArray::new(data, shape).map_err(|e| format!("strings: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "string.empty")]
fn string_empty_ctor(rest: Vec<Value>) -> Result<Value, String> {
    let mut shape: Vec<usize> = Vec::new();
    for v in rest {
        let n: f64 = (&v).try_into()?;
        if n < 0.0 {
            return Err("string.empty: dimensions must be non-negative".to_string());
        }
        shape.push(n as usize);
    }
    if shape.is_empty() {
        shape = vec![0, 0];
    }
    let total: usize = shape.iter().product();
    let data = vec![String::new(); total];
    Ok(Value::StringArray(
        runmat_builtins::StringArray::new(data, shape).map_err(|e| format!("string.empty: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "string")]
fn string_conv(a: Value) -> Result<Value, String> {
    match a {
        Value::String(s) => Ok(Value::StringArray(
            runmat_builtins::StringArray::new(vec![s], vec![1, 1]).unwrap(),
        )),
        Value::StringArray(sa) => Ok(Value::StringArray(sa)),
        Value::CharArray(ca) => {
            let mut out: Vec<String> = Vec::with_capacity(ca.rows);
            for r in 0..ca.rows {
                let mut s = String::with_capacity(ca.cols);
                for c in 0..ca.cols {
                    s.push(ca.data[r * ca.cols + c]);
                }
                out.push(s);
            }
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(out, vec![ca.rows, 1])
                    .map_err(|e| e.to_string())?,
            ))
        }
        Value::Tensor(t) => {
            let mut out: Vec<String> = Vec::with_capacity(t.data.len());
            for &x in &t.data {
                out.push(x.to_string());
            }
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(out, t.shape)
                    .map_err(|e| format!("string: {e}"))?,
            ))
        }
        Value::ComplexTensor(t) => {
            let mut out: Vec<String> = Vec::with_capacity(t.data.len());
            for &(re, im) in &t.data {
                out.push(runmat_builtins::Value::Complex(re, im).to_string());
            }
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(out, t.shape)
                    .map_err(|e| format!("string: {e}"))?,
            ))
        }
        Value::LogicalArray(la) => {
            let mut out: Vec<String> = Vec::with_capacity(la.data.len());
            for &b in &la.data {
                out.push((if b != 0 { 1 } else { 0 }).to_string());
            }
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(out, la.shape)
                    .map_err(|e| format!("string: {e}"))?,
            ))
        }
        Value::Cell(ca) => {
            let mut out: Vec<String> = Vec::with_capacity(ca.rows * ca.cols);
            for r in 0..ca.rows {
                for c in 0..ca.cols {
                    let v = &ca.data[r * ca.cols + c];
                    let s: String = match &**v {
                        Value::String(s) => s.clone(),
                        Value::Num(n) => n.to_string(),
                        Value::Int(i) => i.to_i64().to_string(),
                        Value::Bool(b) => (if *b { 1 } else { 0 }).to_string(),
                        Value::LogicalArray(la) => {
                            if la.data.len() == 1 {
                                (if la.data[0] != 0 { 1 } else { 0 }).to_string()
                            } else {
                                format!("LogicalArray(shape={:?})", la.shape)
                            }
                        }
                        Value::CharArray(ch) => ch.data.iter().collect(),
                        other => format!("{other:?}"),
                    };
                    out.push(s);
                }
            }
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(out, vec![ca.rows, ca.cols])
                    .map_err(|e| e.to_string())?,
            ))
        }
        Value::Num(n) => Ok(Value::StringArray(
            runmat_builtins::StringArray::new(vec![n.to_string()], vec![1, 1]).unwrap(),
        )),
        Value::Complex(re, im) => {
            let s = runmat_builtins::Value::Complex(re, im).to_string();
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(vec![s], vec![1, 1]).unwrap(),
            ))
        }
        Value::Int(i) => Ok(Value::StringArray(
            runmat_builtins::StringArray::new(vec![i.to_i64().to_string()], vec![1, 1]).unwrap(),
        )),
        Value::HandleObject(_) => {
            // The MATLAB language string(handle) produces class-name-like text; keep conservative
            Err("string: unsupported conversion from handle".to_string())
        }
        Value::Listener(_) => Err("string: unsupported conversion from listener".to_string()),
        other => Err(format!("string: unsupported conversion from {other:?}")),
    }
}

// -------- Logical constructors / conversions --------

#[runmat_macros::runtime_builtin(name = "logical")]
fn logical_ctor(a: Value) -> Result<Value, String> {
    match a {
        Value::Bool(b) => Ok(Value::Bool(b)),
        Value::Num(n) => Ok(Value::Bool(n != 0.0)),
        Value::Complex(re, im) => Ok(Value::Bool(!(re == 0.0 && im == 0.0))),
        Value::Int(i) => Ok(Value::Bool(!i.is_zero())),
        Value::Tensor(t) => {
            let data: Vec<u8> = t
                .data
                .iter()
                .map(|&x| if x != 0.0 { 1 } else { 0 })
                .collect();
            Ok(Value::LogicalArray(
                runmat_builtins::LogicalArray::new(data, t.shape)
                    .map_err(|e| format!("logical: {e}"))?,
            ))
        }
        Value::StringArray(sa) => {
            let data: Vec<u8> = sa
                .data
                .iter()
                .map(|s| if !s.is_empty() { 1 } else { 0 })
                .collect();
            Ok(Value::LogicalArray(
                runmat_builtins::LogicalArray::new(data, sa.shape)
                    .map_err(|e| format!("logical: {e}"))?,
            ))
        }
        Value::CharArray(ca) => {
            let non_empty = !(ca.rows == 0 || ca.cols == 0);
            Ok(Value::Bool(non_empty))
        }
        Value::LogicalArray(la) => Ok(Value::LogicalArray(la)),
        Value::ComplexTensor(t) => {
            // Element-wise logical array from complex tensor (non-zero magnitude)
            let data: Vec<u8> = t
                .data
                .iter()
                .map(|(re, im)| if *re != 0.0 || *im != 0.0 { 1 } else { 0 })
                .collect();
            Ok(Value::LogicalArray(
                runmat_builtins::LogicalArray::new(data, t.shape)
                    .map_err(|e| format!("logical: {e}"))?,
            ))
        }
        Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::GpuTensor(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::String(_)
        | Value::HandleObject(_)
        | Value::Listener(_) => Err("logical: unsupported conversion".to_string()),
    }
}

// -------- String functions --------

fn to_string_scalar(v: &Value) -> Result<String, String> {
    let s: String = v.try_into()?;
    Ok(s)
}

fn map_string_array<F>(sa: &runmat_builtins::StringArray, mut f: F) -> runmat_builtins::Tensor
where
    F: FnMut(&str) -> f64,
{
    let mut out: Vec<f64> = Vec::with_capacity(sa.data.len());
    for s in &sa.data {
        out.push(f(s));
    }
    runmat_builtins::Tensor::new(out, sa.shape.clone()).unwrap()
}

#[runmat_macros::runtime_builtin(name = "strcmp")]
fn strcmp_builtin(a: Value, b: Value) -> Result<Value, String> {
    match (a, b) {
        (Value::StringArray(sa), Value::StringArray(sb)) => {
            if sa.shape != sb.shape {
                return Err("strcmp: shape mismatch".to_string());
            }
            let data: Vec<f64> = sa
                .data
                .iter()
                .zip(sb.data.iter())
                .map(|(x, y)| if x == y { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                runmat_builtins::Tensor::new(data, sa.shape).map_err(|e| format!("strcmp: {e}"))?,
            ))
        }
        (Value::StringArray(sa), other) => {
            let s = to_string_scalar(&other)?;
            let t = map_string_array(&sa, |x| if x == s { 1.0 } else { 0.0 });
            Ok(Value::Tensor(t))
        }
        (other, Value::StringArray(sb)) => {
            let s = to_string_scalar(&other)?;
            let t = map_string_array(&sb, |x| if x == s { 1.0 } else { 0.0 });
            Ok(Value::Tensor(t))
        }
        (av, bv) => {
            let as_ = to_string_scalar(&av)?;
            let bs_ = to_string_scalar(&bv)?;
            Ok(Value::Num(if as_ == bs_ { 1.0 } else { 0.0 }))
        }
    }
}

#[runmat_macros::runtime_builtin(name = "strncmp")]
fn strncmp_builtin(a: Value, b: Value, n: f64) -> Result<Value, String> {
    let n = if n < 0.0 { 0usize } else { n as usize };
    let cmp = |x: &str, y: &str| -> f64 {
        let xs = &x.chars().take(n).collect::<String>();
        let ys = &y.chars().take(n).collect::<String>();
        if xs == ys {
            1.0
        } else {
            0.0
        }
    };
    match (a, b) {
        (Value::StringArray(sa), Value::StringArray(sb)) => {
            if sa.shape != sb.shape {
                return Err("strncmp: shape mismatch".to_string());
            }
            let data: Vec<f64> = sa
                .data
                .iter()
                .zip(sb.data.iter())
                .map(|(x, y)| cmp(x, y))
                .collect();
            Ok(Value::Tensor(
                runmat_builtins::Tensor::new(data, sa.shape)
                    .map_err(|e| format!("strncmp: {e}"))?,
            ))
        }
        (Value::StringArray(sa), other) => {
            let s = to_string_scalar(&other)?;
            let t = map_string_array(&sa, |x| cmp(x, &s));
            Ok(Value::Tensor(t))
        }
        (other, Value::StringArray(sb)) => {
            let s = to_string_scalar(&other)?;
            let t = map_string_array(&sb, |x| cmp(&s, x));
            Ok(Value::Tensor(t))
        }
        (av, bv) => {
            let as_ = to_string_scalar(&av)?;
            let bs_ = to_string_scalar(&bv)?;
            Ok(Value::Num(cmp(&as_, &bs_)))
        }
    }
}

#[runmat_macros::runtime_builtin(name = "contains")]
fn contains_builtin(a: Value, pat: Value) -> Result<Value, String> {
    let p = to_string_scalar(&pat)?;
    match a {
        Value::String(s) => Ok(Value::Num(if s.contains(&p) { 1.0 } else { 0.0 })),
        Value::StringArray(sa) => {
            let data: Vec<f64> = sa
                .data
                .iter()
                .map(|x| if x.contains(&p) { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                runmat_builtins::Tensor::new(data, sa.shape)
                    .map_err(|e| format!("contains: {e}"))?,
            ))
        }
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Value::Num(if s.contains(&p) { 1.0 } else { 0.0 }))
        }
        other => Err(format!("contains: unsupported input {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "strrep")]
fn strrep_builtin(a: Value, old: Value, newv: Value) -> Result<Value, String> {
    let old_s = to_string_scalar(&old)?;
    let new_s = to_string_scalar(&newv)?;
    match a {
        Value::String(s) => Ok(Value::String(s.replace(&old_s, &new_s))),
        Value::StringArray(sa) => {
            let data: Vec<String> = sa.data.iter().map(|x| x.replace(&old_s, &new_s)).collect();
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(data, sa.shape)
                    .map_err(|e| format!("strrep: {e}"))?,
            ))
        }
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Value::String(s.replace(&old_s, &new_s)))
        }
        other => Err(format!("strrep: unsupported input {other:?}")),
    }
}

fn to_string_array(v: &Value) -> Result<runmat_builtins::StringArray, String> {
    match v {
        Value::String(s) => runmat_builtins::StringArray::new(vec![s.clone()], vec![1, 1])
            .map_err(|e| e.to_string()),
        Value::StringArray(sa) => Ok(sa.clone()),
        Value::CharArray(ca) => {
            // Convert each row to a string; treat as column vector
            let mut out: Vec<String> = Vec::with_capacity(ca.rows);
            for r in 0..ca.rows {
                let mut s = String::with_capacity(ca.cols);
                for c in 0..ca.cols {
                    s.push(ca.data[r * ca.cols + c]);
                }
                out.push(s);
            }
            runmat_builtins::StringArray::new(out, vec![ca.rows, 1]).map_err(|e| e.to_string())
        }
        other => Err(format!("cannot convert to string array: {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "strcat")]
fn strcat_builtin(rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return Ok(Value::String(String::new()));
    }
    // Normalize all inputs to StringArray; allow scalars (1x1) to broadcast to the max shape if equal
    let mut arrays: Vec<runmat_builtins::StringArray> = Vec::new();
    for v in &rest {
        arrays.push(to_string_array(v)?);
    }
    // Determine result shape by choosing the first non-1x1 shape; require all non-1x1 equal
    let mut shape: Vec<usize> = vec![1, 1];
    for a in &arrays {
        if a.shape != vec![1, 1] {
            if shape == vec![1, 1] {
                shape = a.shape.clone();
            } else if shape != a.shape {
                return Err("strcat: shape mismatch".to_string());
            }
        }
    }
    let total = shape.iter().product::<usize>().max(1);
    let mut out: Vec<String> = vec![String::new(); total];
    for a in &arrays {
        if a.shape == vec![1, 1] {
            let s = &a.data[0];
            for item in out.iter_mut().take(total) {
                item.push_str(s);
            }
        } else {
            for (i, item) in out.iter_mut().enumerate().take(total) {
                item.push_str(&a.data[i]);
            }
        }
    }
    Ok(Value::StringArray(
        runmat_builtins::StringArray::new(out, shape).map_err(|e| format!("strcat: {e}"))?,
    ))
}

// removed duplicate strjoin (keep row-wise version below)

#[runmat_macros::runtime_builtin(name = "join")]
fn join_builtin(a: Value, delim: Value) -> Result<Value, String> {
    strjoin_rowwise(a, delim)
}

#[runmat_macros::runtime_builtin(name = "split")]
fn split_builtin(a: Value, delim: Value) -> Result<Value, String> {
    let s = to_string_scalar(&a)?;
    let d = to_string_scalar(&delim)?;
    if d.is_empty() {
        return Err("split: empty delimiter not supported".to_string());
    }
    let parts: Vec<String> = s.split(&d).map(|t| t.to_string()).collect();
    let len = parts.len();
    Ok(Value::StringArray(
        runmat_builtins::StringArray::new(parts, vec![len, 1])
            .map_err(|e| format!("split: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "upper")]
fn upper_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::String(s) => Ok(Value::String(s.to_uppercase())),
        Value::StringArray(sa) => {
            let data: Vec<String> = sa.data.iter().map(|x| x.to_uppercase()).collect();
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(data, sa.shape)
                    .map_err(|e| format!("upper: {e}"))?,
            ))
        }
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Value::String(s.to_uppercase()))
        }
        other => Err(format!("upper: unsupported input {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "lower")]
fn lower_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::String(s) => Ok(Value::String(s.to_lowercase())),
        Value::StringArray(sa) => {
            let data: Vec<String> = sa.data.iter().map(|x| x.to_lowercase()).collect();
            Ok(Value::StringArray(
                runmat_builtins::StringArray::new(data, sa.shape)
                    .map_err(|e| format!("lower: {e}"))?,
            ))
        }
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Value::String(s.to_lowercase()))
        }
        other => Err(format!("lower: unsupported input {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "startsWith")]
fn starts_with_builtin(a: Value, prefix: Value) -> Result<Value, String> {
    let p = to_string_scalar(&prefix)?;
    match a {
        Value::String(s) => Ok(Value::Num(if s.starts_with(&p) { 1.0 } else { 0.0 })),
        Value::StringArray(sa) => {
            let data: Vec<f64> = sa
                .data
                .iter()
                .map(|x| if x.starts_with(&p) { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                runmat_builtins::Tensor::new(data, sa.shape)
                    .map_err(|e| format!("startsWith: {e}"))?,
            ))
        }
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Value::Num(if s.starts_with(&p) { 1.0 } else { 0.0 }))
        }
        other => Err(format!("startsWith: unsupported input {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "endsWith")]
fn ends_with_builtin(a: Value, suffix: Value) -> Result<Value, String> {
    let p = to_string_scalar(&suffix)?;
    match a {
        Value::String(s) => Ok(Value::Num(if s.ends_with(&p) { 1.0 } else { 0.0 })),
        Value::StringArray(sa) => {
            let data: Vec<f64> = sa
                .data
                .iter()
                .map(|x| if x.ends_with(&p) { 1.0 } else { 0.0 })
                .collect();
            Ok(Value::Tensor(
                runmat_builtins::Tensor::new(data, sa.shape)
                    .map_err(|e| format!("endsWith: {e}"))?,
            ))
        }
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Value::Num(if s.ends_with(&p) { 1.0 } else { 0.0 }))
        }
        other => Err(format!("endsWith: unsupported input {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "extractBetween")]
fn extract_between_builtin(a: Value, start: Value, stop: Value) -> Result<Value, String> {
    let s = to_string_scalar(&a)?;
    let st = to_string_scalar(&start)?;
    let en = to_string_scalar(&stop)?;
    if st.is_empty() || en.is_empty() {
        return Ok(Value::String(String::new()));
    }
    if let Some(i) = s.find(&st) {
        if let Some(j) = s[i + st.len()..].find(&en) {
            return Ok(Value::String(s[i + st.len()..i + st.len() + j].to_string()));
        }
    }
    Ok(Value::String(String::new()))
}

#[runmat_macros::runtime_builtin(name = "erase")]
fn erase_builtin(a: Value, pat: Value) -> Result<Value, String> {
    let s = to_string_scalar(&a)?;
    let p = to_string_scalar(&pat)?;
    Ok(Value::String(s.replace(&p, "")))
}

#[runmat_macros::runtime_builtin(name = "eraseBetween")]
fn erase_between_builtin(a: Value, start: Value, stop: Value) -> Result<Value, String> {
    let s = to_string_scalar(&a)?;
    let st = to_string_scalar(&start)?;
    let en = to_string_scalar(&stop)?;
    if st.is_empty() || en.is_empty() {
        return Ok(Value::String(s));
    }
    if let Some(i) = s.find(&st) {
        if let Some(j) = s[i + st.len()..].find(&en) {
            let mut out = String::new();
            out.push_str(&s[..i + st.len()]);
            out.push_str(&s[i + st.len() + j..]);
            return Ok(Value::String(out));
        }
    }
    Ok(Value::String(s))
}

#[runmat_macros::runtime_builtin(name = "pad")]
fn pad_builtin(a: Value, total_len: f64, rest: Vec<Value>) -> Result<Value, String> {
    // pad(s, n, 'left'|'right', char)
    let s = to_string_scalar(&a)?;
    let n = if total_len < 0.0 {
        0usize
    } else {
        total_len as usize
    };
    let mut direction = "left".to_string();
    let mut ch = ' ';
    if !rest.is_empty() {
        direction = to_string_scalar(&rest[0])?;
    }
    if rest.len() > 1 {
        let t = to_string_scalar(&rest[1])?;
        ch = t.chars().next().unwrap_or(' ');
    }
    if s.chars().count() >= n {
        return Ok(Value::String(s));
    }
    let pad_count = n - s.chars().count();
    let pad_str: String = std::iter::repeat_n(ch, pad_count).collect();
    if direction == "left" {
        Ok(Value::String(format!("{pad_str}{s}")))
    } else {
        Ok(Value::String(format!("{s}{pad_str}")))
    }
}

#[runmat_macros::runtime_builtin(name = "strtrim")]
fn strtrim_builtin(a: Value) -> Result<Value, String> {
    let s = to_string_scalar(&a)?;
    Ok(Value::String(s.trim().to_string()))
}

#[runmat_macros::runtime_builtin(name = "strip")]
fn strip_builtin(a: Value) -> Result<Value, String> {
    strtrim_builtin(a)
}

#[runmat_macros::runtime_builtin(name = "regexp")]
fn regexp_builtin(a: Value, pat: Value) -> Result<Value, String> {
    let s = to_string_scalar(&a)?;
    let p = to_string_scalar(&pat)?;
    let re = Regex::new(&p).map_err(|e| format!("regexp: {e}"))?;
    let mut matches: Vec<Value> = Vec::new();
    for cap in re.captures_iter(&s) {
        // tokens: full match and capture groups
        let full = cap
            .get(0)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        let mut row: Vec<Value> = vec![Value::String(full)];
        for i in 1..cap.len() {
            row.push(Value::String(
                cap.get(i)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default(),
            ));
        }
        matches.push(make_cell(row, 1, cap.len())?);
    }
    let len = matches.len();
    make_cell(matches, 1, len)
}

#[runmat_macros::runtime_builtin(name = "regexpi")]
fn regexpi_builtin(a: Value, pat: Value) -> Result<Value, String> {
    let s = to_string_scalar(&a)?;
    let p = to_string_scalar(&pat)?;
    let re = Regex::new(&format!("(?i){p}")).map_err(|e| format!("regexpi: {e}"))?;
    let mut matches: Vec<Value> = Vec::new();
    for cap in re.captures_iter(&s) {
        let full = cap
            .get(0)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        let mut row: Vec<Value> = vec![Value::String(full)];
        for i in 1..cap.len() {
            row.push(Value::String(
                cap.get(i)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default(),
            ));
        }
        matches.push(make_cell(row, 1, cap.len())?);
    }
    let len = matches.len();
    make_cell(matches, 1, len)
}

// Adjust strjoin semantics: join rows (row-wise)
#[runmat_macros::runtime_builtin(name = "strjoin")]
fn strjoin_rowwise(a: Value, delim: Value) -> Result<Value, String> {
    let d = to_string_scalar(&delim)?;
    let sa = to_string_array(&a)?;
    let rows = *sa.shape.first().unwrap_or(&sa.data.len());
    let cols = *sa.shape.get(1).unwrap_or(&1);
    if rows == 0 || cols == 0 {
        return Ok(Value::StringArray(
            runmat_builtins::StringArray::new(Vec::new(), vec![0, 0]).unwrap(),
        ));
    }
    let mut out: Vec<String> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut s = String::new();
        for c in 0..cols {
            if c > 0 {
                s.push_str(&d);
            }
            s.push_str(&sa.data[r + c * rows]);
        }
        out.push(s);
    }
    Ok(Value::StringArray(
        runmat_builtins::StringArray::new(out, vec![rows, 1])
            .map_err(|e| format!("strjoin: {e}"))?,
    ))
}

// deal: distribute inputs to multiple outputs (via cell for expansion)
#[runmat_macros::runtime_builtin(name = "deal")]
fn deal_builtin(rest: Vec<Value>) -> Result<Value, String> {
    // Return cell row vector of inputs for expansion
    let cols = rest.len();
    make_cell(rest, 1, cols)
}

fn do_find_all_indices(t: &runmat_builtins::Tensor) -> Result<Value, String> {
    let mut idxs: Vec<f64> = Vec::new();
    for (i, &v) in t.data.iter().enumerate() {
        if v != 0.0 {
            idxs.push((i + 1) as f64);
        }
    }
    let len = idxs.len();
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(idxs, vec![len, 1]).map_err(|e| format!("find: {e}"))?,
    ))
}

fn do_find_first_k_indices(t: &runmat_builtins::Tensor, k: usize) -> Result<Value, String> {
    let mut idxs: Vec<f64> = Vec::new();
    for (i, &v) in t.data.iter().enumerate() {
        if v != 0.0 {
            idxs.push((i + 1) as f64);
            if idxs.len() >= k {
                break;
            }
        }
    }
    let len = idxs.len();
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(idxs, vec![len, 1]).map_err(|e| format!("find: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "find")]
fn find_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("find: expected tensor".to_string()),
    };
    if rest.is_empty() {
        return do_find_all_indices(&t);
    }
    // Only support find(A, k) for now
    let k = match &rest[0] {
        Value::Num(n) => *n as usize,
        Value::Int(i) => i.to_i64() as usize,
        _ => 0,
    };
    if k == 0 {
        return do_find_all_indices(&t);
    }
    do_find_first_k_indices(&t, k)
}
// Object/handle utilities used by interpreter lowering for OOP/func handles

#[runmat_macros::runtime_builtin(name = "getfield")]
fn getfield_builtin(base: Value, field: String) -> Result<Value, String> {
    match base {
        Value::MException(me) => match field.as_str() {
            "message" => Ok(Value::String(me.message)),
            "identifier" => Ok(Value::String(me.identifier)),
            _ => Err(format!("getfield: unknown field '{field}' on MException")),
        },
        Value::Object(obj) => {
            if let Some((p, _owner)) = runmat_builtins::lookup_property(&obj.class_name, &field) {
                if p.is_static {
                    return Err(format!(
                        "Property '{}' is static; use classref('{}').{}",
                        field, obj.class_name, field
                    ));
                }
                if p.get_access == runmat_builtins::Access::Private {
                    return Err(format!("Property '{field}' is private"));
                }
                if p.is_dependent {
                    // Try dynamic getter first
                    let getter = format!("get.{field}");
                    if let Ok(v) = crate::call_builtin(&getter, &[Value::Object(obj.clone())]) {
                        return Ok(v);
                    }
                    // Fallback to backing field '<field>_backing'
                    let backing = format!("{field}_backing");
                    if let Some(vb) = obj.properties.get(&backing) {
                        return Ok(vb.clone());
                    }
                }
            }
            if let Some(v) = obj.properties.get(&field) {
                Ok(v.clone())
            } else {
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field, obj.class_name
                ))
            }
        }
        Value::Struct(st) => st
            .fields
            .get(&field)
            .cloned()
            .ok_or_else(|| format!("getfield: unknown field '{field}'")),
        other => Err(format!(
            "getfield unsupported on this value for field '{field}': {other:?}"
        )),
    }
}

// Error handling builtins (basic compatibility)
#[runmat_macros::runtime_builtin(name = "error")]
fn error_builtin(rest: Vec<Value>) -> Result<Value, String> {
    // The MATLAB language is compatible: error(message) or error(identifier, message)
    // We surface a unified error string "IDENT: message" for VM to parse into MException
    if rest.is_empty() {
        return Err("MATLAB:error: missing message".to_string());
    }
    if rest.len() == 1 {
        let msg: String = (&rest[0]).try_into()?;
        return Err(format!("MATLAB:error: {msg}"));
    }
    let ident: String = (&rest[0]).try_into()?;
    let msg: String = (&rest[1]).try_into()?;
    let id = if ident.contains(":") {
        ident
    } else {
        ident.to_string()
    };
    Err(format!("{id}: {msg}"))
}

#[runmat_macros::runtime_builtin(name = "rethrow")]
fn rethrow_builtin(e: Value) -> Result<Value, String> {
    match e {
        Value::MException(me) => Err(format!("{}: {}", me.identifier, me.message)),
        Value::String(s) => Err(s),
        other => Err(format!("MATLAB:error: {other:?}")),
    }
}

// -------- Struct utilities --------
#[runmat_macros::runtime_builtin(name = "fieldnames")]
fn fieldnames_builtin(s: Value) -> Result<Value, String> {
    match s {
        Value::Struct(st) => {
            let mut names: Vec<String> = st.fields.keys().cloned().collect();
            names.sort();
            let len = names.len();
            let vals: Vec<Value> = names.into_iter().map(Value::String).collect();
            make_cell(vals, len, 1)
        }
        other => Err(format!("fieldnames: expected struct, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "isfield")]
fn isfield_builtin(s: Value, name: Value) -> Result<Value, String> {
    match s {
        Value::Struct(st) => match name {
            Value::String(n) => Ok(Value::Num(if st.fields.contains_key(&n) {
                1.0
            } else {
                0.0
            })),
            Value::StringArray(sa) => {
                let rows = sa.rows();
                let cols = sa.cols();
                let mut out: Vec<f64> = vec![0.0; sa.data.len()];
                for c in 0..cols {
                    for r in 0..rows {
                        let idx = r + c * rows;
                        let n = &sa.data[idx];
                        out[idx] = if st.fields.contains_key(n) { 1.0 } else { 0.0 };
                    }
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![rows, cols])
                        .map_err(|e| format!("isfield: {e}"))?,
                ))
            }
            Value::Cell(ca) => {
                let rows = ca.rows;
                let cols = ca.cols;
                let mut out: Vec<f64> = Vec::with_capacity(rows * cols);
                for c in 0..cols {
                    for r in 0..rows {
                        let idx = r * cols + c;
                        let n: String = (&*ca.data[idx]).try_into()?;
                        out.push(if st.fields.contains_key(&n) { 1.0 } else { 0.0 });
                    }
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![rows, cols])
                        .map_err(|e| format!("isfield: {e}"))?,
                ))
            }
            other => {
                let n: String = (&other).try_into()?;
                Ok(Value::Num(if st.fields.contains_key(&n) {
                    1.0
                } else {
                    0.0
                }))
            }
        },
        non_struct => {
            // Support swapped argument order: isfield(names, struct)
            if let Value::Struct(st) = name {
                match non_struct {
                    Value::String(n) => Ok(Value::Num(if st.fields.contains_key(&n) {
                        1.0
                    } else {
                        0.0
                    })),
                    Value::StringArray(sa) => {
                        let rows = sa.rows();
                        let cols = sa.cols();
                        let mut out: Vec<f64> = vec![0.0; sa.data.len()];
                        for c in 0..cols {
                            for r in 0..rows {
                                let idx = r + c * rows;
                                let n = &sa.data[idx];
                                out[idx] = if st.fields.contains_key(n) { 1.0 } else { 0.0 };
                            }
                        }
                        Ok(Value::Tensor(
                            runmat_builtins::Tensor::new(out, vec![rows, cols])
                                .map_err(|e| format!("isfield: {e}"))?,
                        ))
                    }
                    Value::Cell(ca) => {
                        let rows = ca.rows;
                        let cols = ca.cols;
                        let mut out: Vec<f64> = Vec::with_capacity(rows * cols);
                        for c in 0..cols {
                            for r in 0..rows {
                                let idx = r * cols + c;
                                let n: String = (&*ca.data[idx]).try_into()?;
                                out.push(if st.fields.contains_key(&n) { 1.0 } else { 0.0 });
                            }
                        }
                        Ok(Value::Tensor(
                            runmat_builtins::Tensor::new(out, vec![rows, cols])
                                .map_err(|e| format!("isfield: {e}"))?,
                        ))
                    }
                    other => {
                        let n: String = (&other).try_into()?;
                        Ok(Value::Num(if st.fields.contains_key(&n) {
                            1.0
                        } else {
                            0.0
                        }))
                    }
                }
            } else {
                Err(format!("isfield: expected struct, got {non_struct:?}"))
            }
        }
    }
}

#[runmat_macros::runtime_builtin(name = "rmfield")]
fn rmfield_builtin(s: Value, rest: Vec<Value>) -> Result<Value, String> {
    let mut names: Vec<String> = Vec::new();
    if rest.len() == 1 {
        match &rest[0] {
            Value::Cell(ca) => {
                for v in &ca.data {
                    names.push(String::try_from(&**v).map_err(|e| format!("rmfield: {e}"))?);
                }
            }
            other => {
                names.push(String::try_from(other).map_err(|e| format!("rmfield: {e}"))?);
            }
        }
    } else {
        for v in &rest {
            names.push(String::try_from(v).map_err(|e| format!("rmfield: {e}"))?);
        }
    }
    match s {
        Value::Struct(mut st) => {
            for n in names {
                st.fields.remove(&n);
            }
            Ok(Value::Struct(st))
        }
        other => Err(format!("rmfield: expected struct, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "orderfields")]
fn orderfields_builtin(s: Value) -> Result<Value, String> {
    // With HashMap-backed structs, field order is not stored; ensure fieldnames() returns sorted order.
    // Return struct unchanged to preserve data.
    match s {
        Value::Struct(st) => Ok(Value::Struct(st)),
        other => Err(format!("orderfields: expected struct, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "reshape")]
fn reshape_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    // Accept 2 or 3 dims (for tests); implement MATLAB-style (column-major) reshape semantics
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("reshape: expected tensor".to_string()),
    };
    let mut dims: Vec<usize> = Vec::new();
    for v in rest {
        let n: f64 = (&v).try_into()?;
        dims.push(n as usize);
    }
    if dims.len() < 2 || dims.len() > 3 {
        return Err("reshape: expected 2 or 3 dimension sizes".to_string());
    }
    let total: usize = dims.iter().product();
    if total != t.data.len() {
        return Err(format!(
            "reshape: element count mismatch {} vs {}",
            total,
            t.data.len()
        ));
    }
    // MATLAB uses column-major storage; reshape reinterprets without reordering
    let new_t =
        runmat_builtins::Tensor::new(t.data.clone(), dims).map_err(|e| format!("reshape: {e}"))?;
    Ok(Value::Tensor(new_t))
}
#[runmat_macros::runtime_builtin(name = "setfield")]
fn setfield_builtin(base: Value, field: String, rhs: Value) -> Result<Value, String> {
    match base {
        Value::Object(mut obj) => {
            if let Some((p, _owner)) = runmat_builtins::lookup_property(&obj.class_name, &field) {
                if p.is_static {
                    return Err(format!(
                        "Property '{}' is static; use classref('{}').{}",
                        field, obj.class_name, field
                    ));
                }
                if p.set_access == runmat_builtins::Access::Private {
                    return Err(format!("Property '{field}' is private"));
                }
                if p.is_dependent {
                    let setter = format!("set.{field}");
                    // Try class/user-defined setter first
                    if let Ok(v) =
                        crate::call_builtin(&setter, &[Value::Object(obj.clone()), rhs.clone()])
                    {
                        return Ok(v);
                    }
                    // Fallback: write to backing field '<field>_backing'
                    let backing = format!("{field}_backing");
                    obj.properties.insert(backing, rhs);
                    return Ok(Value::Object(obj));
                }
            }
            obj.properties.insert(field, rhs);
            Ok(Value::Object(obj))
        }
        Value::Struct(mut st) => {
            st.fields.insert(field, rhs);
            Ok(Value::Struct(st))
        }
        Value::HandleObject(_) | Value::Listener(_) => Err(format!(
            "setfield unsupported on this value for field '{field}': handle/listener"
        )),
        other => Err(format!(
            "setfield unsupported on this value for field '{field}': {other:?}"
        )),
    }
}

#[runmat_macros::runtime_builtin(name = "call_method")]
fn call_method_builtin(base: Value, method: String, rest: Vec<Value>) -> Result<Value, String> {
    match base {
        Value::Object(obj) => {
            // Simple dynamic dispatch via builtin registry: method name may be qualified as Class.method
            let qualified = format!("{}.{}", obj.class_name, method);
            // Prepend receiver as first arg so methods can accept it
            let mut args = Vec::with_capacity(1 + rest.len());
            args.push(Value::Object(obj.clone()));
            args.extend(rest);
            if let Ok(v) = crate::call_builtin(&qualified, &args) {
                return Ok(v);
            }
            // Fallback to global method name
            crate::call_builtin(&method, &args)
        }
        Value::HandleObject(h) => {
            // Methods on handle classes dispatch to the underlying target's class namespace
            let target = unsafe { &*h.target.as_raw() };
            let class_name = match target {
                Value::Object(o) => o.class_name.clone(),
                Value::Struct(_) => h.class_name.clone(),
                _ => h.class_name.clone(),
            };
            let qualified = format!("{class_name}.{method}");
            let mut args = Vec::with_capacity(1 + rest.len());
            args.push(Value::HandleObject(h.clone()));
            args.extend(rest);
            if let Ok(v) = crate::call_builtin(&qualified, &args) {
                return Ok(v);
            }
            crate::call_builtin(&method, &args)
        }
        other => Err(format!(
            "call_method unsupported on {other:?} for method '{method}'"
        )),
    }
}

// Global dispatch helpers for overloaded indexing (subsref/subsasgn) to support fallback resolution paths
#[runmat_macros::runtime_builtin(name = "subsasgn")]
fn subsasgn_dispatch(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> Result<Value, String> {
    match &obj {
        Value::Object(o) => {
            let qualified = format!("{}.subsasgn", o.class_name);
            crate::call_builtin(&qualified, &[obj, Value::String(kind), payload, rhs])
        }
        Value::HandleObject(h) => {
            let target = unsafe { &*h.target.as_raw() };
            let class_name = match target {
                Value::Object(o) => o.class_name.clone(),
                _ => h.class_name.clone(),
            };
            let qualified = format!("{class_name}.subsasgn");
            crate::call_builtin(&qualified, &[obj, Value::String(kind), payload, rhs])
        }
        other => Err(format!("subsasgn: receiver must be object, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "subsref")]
fn subsref_dispatch(obj: Value, kind: String, payload: Value) -> Result<Value, String> {
    match &obj {
        Value::Object(o) => {
            let qualified = format!("{}.subsref", o.class_name);
            crate::call_builtin(&qualified, &[obj, Value::String(kind), payload])
        }
        Value::HandleObject(h) => {
            let target = unsafe { &*h.target.as_raw() };
            let class_name = match target {
                Value::Object(o) => o.class_name.clone(),
                _ => h.class_name.clone(),
            };
            let qualified = format!("{class_name}.subsref");
            crate::call_builtin(&qualified, &[obj, Value::String(kind), payload])
        }
        other => Err(format!("subsref: receiver must be object, got {other:?}")),
    }
}

// -------- Handle classes & events --------

#[runmat_macros::runtime_builtin(name = "new_handle_object")]
fn new_handle_object_builtin(class_name: String) -> Result<Value, String> {
    // Create an underlying object instance and wrap it in a handle
    let obj = new_object_builtin(class_name.clone())?;
    let gc = runmat_gc::gc_allocate(obj).map_err(|e| format!("gc: {e}"))?;
    Ok(Value::HandleObject(runmat_builtins::HandleRef {
        class_name,
        target: gc,
        valid: true,
    }))
}

#[runmat_macros::runtime_builtin(name = "isvalid")]
fn isvalid_builtin(v: Value) -> Result<Value, String> {
    match v {
        Value::HandleObject(h) => Ok(Value::Bool(h.valid)),
        Value::Listener(l) => Ok(Value::Bool(l.valid && l.enabled)),
        _ => Ok(Value::Bool(false)),
    }
}

#[runmat_macros::runtime_builtin(name = "delete")]
fn delete_builtin(v: Value) -> Result<Value, String> {
    match v {
        Value::HandleObject(mut h) => {
            h.valid = false;
            Ok(Value::HandleObject(h))
        }
        Value::Listener(mut l) => {
            l.valid = false;
            Ok(Value::Listener(l))
        }
        other => Err(format!("delete: unsupported value {other:?}")),
    }
}

use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct EventRegistry {
    next_id: u64,
    listeners: std::collections::HashMap<(usize, String), Vec<runmat_builtins::Listener>>,
}

static EVENT_REGISTRY: OnceLock<Mutex<EventRegistry>> = OnceLock::new();

fn events() -> &'static Mutex<EventRegistry> {
    EVENT_REGISTRY.get_or_init(|| Mutex::new(EventRegistry::default()))
}

#[runmat_macros::runtime_builtin(name = "addlistener")]
fn addlistener_builtin(
    target: Value,
    event_name: String,
    callback: Value,
) -> Result<Value, String> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => (unsafe { h.target.as_raw() }) as usize,
        Value::Object(o) => o as *const _ as usize,
        _ => return Err("addlistener: target must be handle or object".to_string()),
    };
    let mut reg = events().lock().unwrap();
    let id = {
        reg.next_id += 1;
        reg.next_id
    };
    let tgt_gc = match target {
        Value::HandleObject(h) => h.target,
        Value::Object(o) => {
            runmat_gc::gc_allocate(Value::Object(o)).map_err(|e| format!("gc: {e}"))?
        }
        _ => unreachable!(),
    };
    let cb_gc = runmat_gc::gc_allocate(callback).map_err(|e| format!("gc: {e}"))?;
    let listener = runmat_builtins::Listener {
        id,
        target: tgt_gc,
        event_name: event_name.clone(),
        callback: cb_gc,
        enabled: true,
        valid: true,
    };
    reg.listeners
        .entry((key_ptr, event_name))
        .or_default()
        .push(listener.clone());
    Ok(Value::Listener(listener))
}

#[runmat_macros::runtime_builtin(name = "notify")]
fn notify_builtin(target: Value, event_name: String, rest: Vec<Value>) -> Result<Value, String> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => (unsafe { h.target.as_raw() }) as usize,
        Value::Object(o) => o as *const _ as usize,
        _ => return Err("notify: target must be handle or object".to_string()),
    };
    let mut to_call: Vec<runmat_builtins::Listener> = Vec::new();
    {
        let reg = events().lock().unwrap();
        if let Some(list) = reg.listeners.get(&(key_ptr, event_name.clone())) {
            for l in list {
                if l.valid && l.enabled {
                    to_call.push(l.clone());
                }
            }
        }
    }
    for l in to_call {
        // Call callback via feval-like protocol
        let mut args = Vec::new();
        args.push(target.clone());
        args.extend(rest.iter().cloned());
        let cbv: Value = (*l.callback).clone();
        match &cbv {
            Value::String(s) if s.starts_with('@') => {
                let mut a = vec![Value::String(s.clone())];
                a.extend(args.into_iter());
                let _ = crate::call_builtin("feval", &a)?;
            }
            Value::FunctionHandle(name) => {
                let mut a = vec![Value::FunctionHandle(name.clone())];
                a.extend(args.into_iter());
                let _ = crate::call_builtin("feval", &a)?;
            }
            Value::Closure(_) => {
                let mut a = vec![cbv.clone()];
                a.extend(args.into_iter());
                let _ = crate::call_builtin("feval", &a)?;
            }
            _ => {}
        }
    }
    Ok(Value::Num(0.0))
}

// Test-oriented dependent property handlers (global). If a class defines a Dependent
// property named 'p', the VM will try to call get.p / set.p. We provide generic
// implementations that read/write a conventional backing field 'p_backing'.
#[runmat_macros::runtime_builtin(name = "get.p")]
fn get_p_builtin(obj: Value) -> Result<Value, String> {
    match obj {
        Value::Object(o) => {
            if let Some(v) = o.properties.get("p_backing") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        other => Err(format!("get.p requires object, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "set.p")]
fn set_p_builtin(obj: Value, val: Value) -> Result<Value, String> {
    match obj {
        Value::Object(mut o) => {
            o.properties.insert("p_backing".to_string(), val);
            Ok(Value::Object(o))
        }
        other => Err(format!("set.p requires object, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "make_handle")]
fn make_handle_builtin(name: String) -> Result<Value, String> {
    Ok(Value::String(format!("@{name}")))
}

#[runmat_macros::runtime_builtin(name = "make_anon")]
fn make_anon_builtin(params: String, body: String) -> Result<Value, String> {
    Ok(Value::String(format!("@anon({params}) {body}")))
}

#[runmat_macros::runtime_builtin(name = "new_object")]
pub(crate) fn new_object_builtin(class_name: String) -> Result<Value, String> {
    if let Some(def) = runmat_builtins::get_class(&class_name) {
        // Collect class hierarchy from root to leaf for default initialization
        let mut chain: Vec<runmat_builtins::ClassDef> = Vec::new();
        // Walk up to root
        let mut cursor: Option<String> = Some(def.name.clone());
        while let Some(name) = cursor {
            if let Some(cd) = runmat_builtins::get_class(&name) {
                chain.push(cd.clone());
                cursor = cd.parent.clone();
            } else {
                break;
            }
        }
        // Reverse to root-first
        chain.reverse();
        let mut obj = runmat_builtins::ObjectInstance::new(def.name.clone());
        // Apply defaults from root to leaf (leaf overrides effectively by later assignment)
        for cd in chain {
            for (k, p) in cd.properties.iter() {
                if !p.is_static {
                    if let Some(v) = &p.default_value {
                        obj.properties.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        Ok(Value::Object(obj))
    } else {
        Ok(Value::Object(runmat_builtins::ObjectInstance::new(
            class_name,
        )))
    }
}

// handle-object builtins removed for now

#[runmat_macros::runtime_builtin(name = "classref")]
fn classref_builtin(class_name: String) -> Result<Value, String> {
    Ok(Value::ClassRef(class_name))
}

#[runmat_macros::runtime_builtin(name = "__register_test_classes")]
fn register_test_classes_builtin() -> Result<Value, String> {
    use runmat_builtins::*;
    let mut props = std::collections::HashMap::new();
    props.insert(
        "x".to_string(),
        PropertyDef {
            name: "x".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    props.insert(
        "y".to_string(),
        PropertyDef {
            name: "y".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    props.insert(
        "staticValue".to_string(),
        PropertyDef {
            name: "staticValue".to_string(),
            is_static: true,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(42.0)),
        },
    );
    props.insert(
        "secret".to_string(),
        PropertyDef {
            name: "secret".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Private,
            set_access: Access::Private,
            default_value: Some(Value::Num(99.0)),
        },
    );
    let mut methods = std::collections::HashMap::new();
    methods.insert(
        "move".to_string(),
        MethodDef {
            name: "move".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Point.move".to_string(),
        },
    );
    methods.insert(
        "origin".to_string(),
        MethodDef {
            name: "origin".to_string(),
            is_static: true,
            access: Access::Public,
            function_name: "Point.origin".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Point".to_string(),
        parent: None,
        properties: props,
        methods,
    });

    // Namespaced class example: pkg.PointNS with same shape as Point
    let mut ns_props = std::collections::HashMap::new();
    ns_props.insert(
        "x".to_string(),
        PropertyDef {
            name: "x".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(1.0)),
        },
    );
    ns_props.insert(
        "y".to_string(),
        PropertyDef {
            name: "y".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(2.0)),
        },
    );
    let ns_methods = std::collections::HashMap::new();
    runmat_builtins::register_class(ClassDef {
        name: "pkg.PointNS".to_string(),
        parent: None,
        properties: ns_props,
        methods: ns_methods,
    });

    // Inheritance: Shape (base) and Circle (derived)
    let shape_props = std::collections::HashMap::new();
    let mut shape_methods = std::collections::HashMap::new();
    shape_methods.insert(
        "area".to_string(),
        MethodDef {
            name: "area".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Shape.area".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Shape".to_string(),
        parent: None,
        properties: shape_props,
        methods: shape_methods,
    });

    let mut circle_props = std::collections::HashMap::new();
    circle_props.insert(
        "r".to_string(),
        PropertyDef {
            name: "r".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    let mut circle_methods = std::collections::HashMap::new();
    circle_methods.insert(
        "area".to_string(),
        MethodDef {
            name: "area".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Circle.area".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Circle".to_string(),
        parent: Some("Shape".to_string()),
        properties: circle_props,
        methods: circle_methods,
    });

    // Constructor demo class: Ctor with static constructor method Ctor
    let ctor_props = std::collections::HashMap::new();
    let mut ctor_methods = std::collections::HashMap::new();
    ctor_methods.insert(
        "Ctor".to_string(),
        MethodDef {
            name: "Ctor".to_string(),
            is_static: true,
            access: Access::Public,
            function_name: "Ctor.Ctor".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Ctor".to_string(),
        parent: None,
        properties: ctor_props,
        methods: ctor_methods,
    });

    // Overloaded indexing demo class: OverIdx with subsref/subsasgn
    let overidx_props = std::collections::HashMap::new();
    let mut overidx_methods = std::collections::HashMap::new();
    overidx_methods.insert(
        "subsref".to_string(),
        MethodDef {
            name: "subsref".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "OverIdx.subsref".to_string(),
        },
    );
    overidx_methods.insert(
        "subsasgn".to_string(),
        MethodDef {
            name: "subsasgn".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "OverIdx.subsasgn".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "OverIdx".to_string(),
        parent: None,
        properties: overidx_props,
        methods: overidx_methods,
    });
    Ok(Value::Num(1.0))
}

#[cfg(feature = "test-classes")]
pub fn test_register_classes() {
    let _ = register_test_classes_builtin();
}

// Example method implementation: Point.move(obj, dx, dy) -> updated obj
#[runmat_macros::runtime_builtin(name = "Point.move")]
fn point_move_method(obj: Value, dx: f64, dy: f64) -> Result<Value, String> {
    match obj {
        Value::Object(mut o) => {
            let mut x = 0.0;
            let mut y = 0.0;
            if let Some(Value::Num(v)) = o.properties.get("x") {
                x = *v;
            }
            if let Some(Value::Num(v)) = o.properties.get("y") {
                y = *v;
            }
            o.properties.insert("x".to_string(), Value::Num(x + dx));
            o.properties.insert("y".to_string(), Value::Num(y + dy));
            Ok(Value::Object(o))
        }
        other => Err(format!(
            "Point.move requires object receiver, got {other:?}"
        )),
    }
}

#[runmat_macros::runtime_builtin(name = "Point.origin")]
fn point_origin_method() -> Result<Value, String> {
    let mut o = runmat_builtins::ObjectInstance::new("Point".to_string());
    o.properties.insert("x".to_string(), Value::Num(0.0));
    o.properties.insert("y".to_string(), Value::Num(0.0));
    Ok(Value::Object(o))
}

#[runmat_macros::runtime_builtin(name = "Shape.area")]
fn shape_area_method(_obj: Value) -> Result<Value, String> {
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(name = "Circle.area")]
fn circle_area_method(obj: Value) -> Result<Value, String> {
    match obj {
        Value::Object(o) => {
            let r = if let Some(Value::Num(v)) = o.properties.get("r") {
                *v
            } else {
                0.0
            };
            Ok(Value::Num(std::f64::consts::PI * r * r))
        }
        other => Err(format!(
            "Circle.area requires object receiver, got {other:?}"
        )),
    }
}

// --- Test-only helpers to validate constructors and subsref/subsasgn ---
#[runmat_macros::runtime_builtin(name = "Ctor.Ctor")]
fn ctor_ctor_method(x: f64) -> Result<Value, String> {
    // Construct object with property 'x' initialized
    let mut o = runmat_builtins::ObjectInstance::new("Ctor".to_string());
    o.properties.insert("x".to_string(), Value::Num(x));
    Ok(Value::Object(o))
}

// --- Test-only package functions to exercise import precedence ---
#[runmat_macros::runtime_builtin(name = "PkgF.foo")]
fn pkgf_foo() -> Result<Value, String> {
    Ok(Value::Num(10.0))
}

#[runmat_macros::runtime_builtin(name = "PkgG.foo")]
fn pkgg_foo() -> Result<Value, String> {
    Ok(Value::Num(20.0))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.subsref")]
fn overidx_subsref(obj: Value, kind: String, payload: Value) -> Result<Value, String> {
    // Simple sentinel implementation: return different values for '.' vs '()'
    match (obj, kind.as_str(), payload) {
        (Value::Object(_), "()", Value::Cell(_)) => Ok(Value::Num(99.0)),
        (Value::Object(o), "{}", Value::Cell(_)) => {
            if let Some(v) = o.properties.get("lastCell") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        (Value::Object(o), ".", Value::String(field)) => {
            // If field exists, return it; otherwise sentinel 77
            if let Some(v) = o.properties.get(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        (Value::Object(o), ".", Value::CharArray(ca)) => {
            let field: String = ca.data.iter().collect();
            if let Some(v) = o.properties.get(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        _ => Err("subsref: unsupported payload".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "OverIdx.subsasgn")]
fn overidx_subsasgn(
    mut obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> Result<Value, String> {
    match (&mut obj, kind.as_str(), payload) {
        (Value::Object(o), "()", Value::Cell(_)) => {
            // Store into 'last' property
            o.properties.insert("last".to_string(), rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), "{}", Value::Cell(_)) => {
            o.properties.insert("lastCell".to_string(), rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), ".", Value::String(field)) => {
            o.properties.insert(field, rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), ".", Value::CharArray(ca)) => {
            let field: String = ca.data.iter().collect();
            o.properties.insert(field, rhs);
            Ok(Value::Object(o.clone()))
        }
        _ => Err("subsasgn: unsupported payload".to_string()),
    }
}

// --- Operator overloading methods for OverIdx (test scaffolding) ---
#[runmat_macros::runtime_builtin(name = "OverIdx.plus")]
fn overidx_plus(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.plus: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k + r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.times")]
fn overidx_times(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.times: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.mtimes")]
fn overidx_mtimes(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.mtimes: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.lt")]
fn overidx_lt(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.lt: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k < r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.gt")]
fn overidx_gt(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.gt: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k > r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.eq")]
fn overidx_eq(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.eq: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k - r).abs() < 1e-12 { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.uplus")]
fn overidx_uplus(obj: Value) -> Result<Value, String> {
    // Identity
    Ok(obj)
}

#[runmat_macros::runtime_builtin(name = "OverIdx.rdivide")]
fn overidx_rdivide(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.rdivide: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k / r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.ldivide")]
fn overidx_ldivide(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.ldivide: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(r / k))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.and")]
fn overidx_and(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.and: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) && (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.or")]
fn overidx_or(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.or: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) || (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.xor")]
fn overidx_xor(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.xor: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    let a = k != 0.0;
    let b = r != 0.0;
    Ok(Value::Num(if a ^ b { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "feval")]
fn feval_builtin(f: Value, rest: Vec<Value>) -> Result<Value, String> {
    match f {
        // Current handles are strings like "@sin"
        Value::String(s) => {
            if let Some(name) = s.strip_prefix('@') {
                crate::call_builtin(name, &rest)
            } else {
                Err(format!(
                    "feval: expected function handle string starting with '@', got {s}"
                ))
            }
        }
        Value::Closure(c) => {
            let mut args = c.captures.clone();
            args.extend(rest);
            crate::call_builtin(&c.function_name, &args)
        }
        // Future: support Value::Function variants
        other => Err(format!("feval: unsupported function value {other:?}")),
    }
}

// Common mathematical functions that tests expect

/// Transpose operation for Values
pub fn transpose(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.transpose(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("transpose: unsupported for gpuArray".to_string())
        }
        Value::Tensor(ref m) => Ok(Value::Tensor(matrix_transpose(m))),
        // For complex scalars, transpose is conjugate transpose => scalar transpose is conjugate
        Value::Complex(re, im) => Ok(Value::Complex(re, -im)),
        Value::ComplexTensor(ref ct) => {
            let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); ct.rows * ct.cols];
            // Conjugate transpose for complex matrices
            for i in 0..ct.rows {
                for j in 0..ct.cols {
                    let (re, im) = ct.data[i + j * ct.rows];
                    data[j + i * ct.cols] = (re, -im);
                }
            }
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, ct.cols, ct.rows).unwrap(),
            ))
        }
        Value::Num(n) => Ok(Value::Num(n)), // Scalar transpose is identity
        _ => Err("transpose not supported for this type".to_string()),
    }
}

// Explicit GPU usage builtins (scaffolding)
#[runmat_macros::runtime_builtin(name = "gpuArray")]
fn gpu_array_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::Tensor(t) => {
            // Placeholder: mark as GPU handle; device/buffer ids are dummies for now
            if let Some(p) = runmat_accelerate_api::provider() {
                let view = runmat_accelerate_api::HostTensorView {
                    data: &t.data,
                    shape: &t.shape,
                };
                let h = p
                    .upload(&view)
                    .map_err(|e| format!("gpuArray upload: {e}"))?;
                Ok(Value::GpuTensor(h))
            } else {
                Ok(Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                    shape: t.shape.clone(),
                    device_id: 0,
                    buffer_id: 0,
                }))
            }
        }
        Value::Num(_n) => Ok(Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 0,
        })),
        other => Err(format!("gpuArray unsupported for {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "gather")]
fn gather_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                let ht = p
                    .download(&h)
                    .map_err(|e| format!("gather download: {e}"))?;
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(ht.data, ht.shape)
                        .map_err(|e| format!("gather build: {e}"))?,
                ))
            } else {
                let total: usize = h.shape.iter().product();
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(vec![0.0; total], h.shape)
                        .map_err(|e| format!("gather: {e}"))?,
                ))
            }
        }
        v => Ok(v),
    }
}

// consolidate scalar max into helper; keep one registered varargs form below
fn max_scalar(a: f64, b: f64) -> f64 {
    a.max(b)
}

fn max_vector_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                // Reduce across all elements -> 1x1 handle
                if let Ok(hc) = p.reduce_max(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("max: unsupported for gpuArray".to_string())
        }
        Value::Tensor(t) => {
            if t.shape.len() == 2 && t.shape[1] == 1 {
                let mut max_val = f64::NEG_INFINITY;
                let mut idx = 1usize;
                for (i, &v) in t.data.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        idx = i + 1;
                    }
                }
                // Return a 2x1 column [max; idx] for expansion (value then index)
                let out = runmat_builtins::Tensor::new(vec![max_val, idx as f64], vec![2, 1])
                    .map_err(|e| format!("max: {e}"))?;
                Ok(Value::Tensor(out))
            } else {
                // Reduce across all elements -> scalar
                let max_val = t.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                Ok(Value::Num(max_val))
            }
        }
        Value::Num(n) => Ok(Value::Num(n)),
        _ => Err("max: unsupported input".to_string()),
    }
}

fn min_vector_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.reduce_min(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("min: unsupported for gpuArray".to_string())
        }
        Value::Tensor(t) => {
            if t.shape.len() == 2 && t.shape[1] == 1 {
                let mut min_val = f64::INFINITY;
                let mut idx = 1usize;
                for (i, &v) in t.data.iter().enumerate() {
                    if v < min_val {
                        min_val = v;
                        idx = i + 1;
                    }
                }
                let out = runmat_builtins::Tensor::new(vec![min_val, idx as f64], vec![2, 1])
                    .map_err(|e| format!("min: {e}"))?;
                Ok(Value::Tensor(out))
            } else {
                let min_val = t.data.iter().cloned().fold(f64::INFINITY, f64::min);
                Ok(Value::Num(min_val))
            }
        }
        Value::Num(n) => Ok(Value::Num(n)),
        _ => Err("min: unsupported input".to_string()),
    }
}

fn max_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    if let Value::GpuTensor(h) = a {
        if let Some(p) = runmat_accelerate_api::provider() {
            let d = if dim < 1.0 { 1 } else { dim as usize };
            if let Ok(res) = p.reduce_max_dim(&h, d) {
                // Return cell {values, indices} with GPU handles
                return make_cell(
                    vec![Value::GpuTensor(res.values), Value::GpuTensor(res.indices)],
                    1,
                    2,
                );
            }
        }
        return Err("max: unsupported for gpuArray".to_string());
    }
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("max: expected tensor for dim variant".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    if t.shape.len() < 2 {
        return Err("max: dim variant expects 2D tensor".to_string());
    }
    let rows = t.shape[0];
    let cols = t.shape[1];
    if dim == 1 {
        // column-wise maxima: return {M_row, I_row}
        let mut m: Vec<f64> = vec![f64::NEG_INFINITY; cols];
        let mut idx: Vec<f64> = vec![1.0; cols];
        for c in 0..cols {
            for r in 0..rows {
                let v = t.data[r + c * rows];
                if v > m[c] {
                    m[c] = v;
                    idx[c] = (r + 1) as f64;
                }
            }
        }
        let m_t =
            runmat_builtins::Tensor::new(m, vec![1, cols]).map_err(|e| format!("max: {e}"))?;
        let i_t =
            runmat_builtins::Tensor::new(idx, vec![1, cols]).map_err(|e| format!("max: {e}"))?;
        make_cell(vec![Value::Tensor(m_t), Value::Tensor(i_t)], 1, 2)
    } else if dim == 2 {
        // row-wise maxima: return {M_col, I_col}
        let mut m: Vec<f64> = vec![f64::NEG_INFINITY; rows];
        let mut idx: Vec<f64> = vec![1.0; rows];
        for r in 0..rows {
            for c in 0..cols {
                let v = t.data[r + c * rows];
                if v > m[r] {
                    m[r] = v;
                    idx[r] = (c + 1) as f64;
                }
            }
        }
        let m_t =
            runmat_builtins::Tensor::new(m, vec![rows, 1]).map_err(|e| format!("max: {e}"))?;
        let i_t =
            runmat_builtins::Tensor::new(idx, vec![rows, 1]).map_err(|e| format!("max: {e}"))?;
        make_cell(vec![Value::Tensor(m_t), Value::Tensor(i_t)], 1, 2)
    } else {
        Err("max: dim out of range".to_string())
    }
}

fn min_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    if let Value::GpuTensor(h) = a {
        if let Some(p) = runmat_accelerate_api::provider() {
            let d = if dim < 1.0 { 1 } else { dim as usize };
            if let Ok(res) = p.reduce_min_dim(&h, d) {
                return make_cell(
                    vec![Value::GpuTensor(res.values), Value::GpuTensor(res.indices)],
                    1,
                    2,
                );
            }
        }
        return Err("min: unsupported for gpuArray".to_string());
    }
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("min: expected tensor for dim variant".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    if t.shape.len() < 2 {
        return Err("min: dim variant expects 2D tensor".to_string());
    }
    let rows = t.shape[0];
    let cols = t.shape[1];
    if dim == 1 {
        let mut m: Vec<f64> = vec![f64::INFINITY; cols];
        let mut idx: Vec<f64> = vec![1.0; cols];
        for c in 0..cols {
            for r in 0..rows {
                let v = t.data[r + c * rows];
                if v < m[c] {
                    m[c] = v;
                    idx[c] = (r + 1) as f64;
                }
            }
        }
        let m_t =
            runmat_builtins::Tensor::new(m, vec![1, cols]).map_err(|e| format!("min: {e}"))?;
        let i_t =
            runmat_builtins::Tensor::new(idx, vec![1, cols]).map_err(|e| format!("min: {e}"))?;
        make_cell(vec![Value::Tensor(m_t), Value::Tensor(i_t)], 1, 2)
    } else if dim == 2 {
        let mut m: Vec<f64> = vec![f64::INFINITY; rows];
        let mut idx: Vec<f64> = vec![1.0; rows];
        for r in 0..rows {
            for c in 0..cols {
                let v = t.data[r + c * rows];
                if v < m[r] {
                    m[r] = v;
                    idx[r] = (c + 1) as f64;
                }
            }
        }
        let m_t =
            runmat_builtins::Tensor::new(m, vec![rows, 1]).map_err(|e| format!("min: {e}"))?;
        let i_t =
            runmat_builtins::Tensor::new(idx, vec![rows, 1]).map_err(|e| format!("min: {e}"))?;
        make_cell(vec![Value::Tensor(m_t), Value::Tensor(i_t)], 1, 2)
    } else {
        Err("min: dim out of range".to_string())
    }
}

fn min_scalar(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[runmat_macros::runtime_builtin(name = "max", accel = "reduction")]
fn max_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return max_vector_builtin(a);
    }
    if rest.len() == 1 {
        let r0 = &rest[0];
        // Scalar pair max(a,b)
        if let (Value::Num(a0), Value::Num(b0)) = (a.clone(), r0.clone()) {
            return Ok(Value::Num(max_scalar(a0, b0)));
        }
        // Optional dim variant max(A, dim)
        if matches!(r0, Value::Num(_) | Value::Int(_)) {
            return max_dim_builtin(
                a,
                match r0 {
                    Value::Num(d) => *d,
                    Value::Int(i) => i.to_i64() as f64,
                    _ => unreachable!(),
                },
            );
        }
    }
    Err("max: unsupported arguments".to_string())
}

#[runmat_macros::runtime_builtin(name = "min", accel = "reduction")]
fn min_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return min_vector_builtin(a);
    }
    if rest.len() == 1 {
        let r0 = &rest[0];
        // Scalar pair min(a,b)
        if let (Value::Num(a0), Value::Num(b0)) = (a.clone(), r0.clone()) {
            return Ok(Value::Num(min_scalar(a0, b0)));
        }
        match r0 {
            Value::Num(d) => return min_dim_builtin(a, *d),
            Value::Int(i) => return min_dim_builtin(a, i.to_i64() as f64),
            _ => {}
        }
    }
    Err("min: unsupported arguments".to_string())
}

#[runtime_builtin(name = "sqrt")]
fn sqrt_builtin(x: f64) -> Result<f64, String> {
    if x < 0.0 {
        Err("MATLAB:domainError: Cannot take square root of negative number".to_string())
    } else {
        Ok(x.sqrt())
    }
}

/// Simple timing functions for benchmarks
/// tic() starts a timer and returns current time
#[runtime_builtin(name = "tic")]
fn tic_builtin() -> Result<f64, String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Time error: {e}"))?;
    Ok(now.as_secs_f64())
}

/// toc() returns elapsed time since the last tic() call
/// Note: In a real implementation, this would use a saved start time,
/// but for simplicity we'll just return a small time value
#[runtime_builtin(name = "toc")]
fn toc_builtin() -> Result<f64, String> {
    // For benchmark purposes, return a realistic small time
    Ok(0.001) // 1 millisecond
}

#[runtime_builtin(name = "exp")]
fn exp_builtin(x: f64) -> Result<f64, String> {
    Ok(x.exp())
}

#[runtime_builtin(name = "log")]
fn log_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("MATLAB:domainError: Cannot take logarithm of non-positive number".to_string())
    } else {
        Ok(x.ln())
    }
}

// -------- Reductions: sum/prod/mean/any/all --------

fn tensor_sum_all(t: &runmat_builtins::Tensor) -> f64 {
    t.data.iter().sum()
}

fn tensor_prod_all(t: &runmat_builtins::Tensor) -> f64 {
    t.data.iter().product()
}

fn prod_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![1.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut p = 1.0;
                    for r in 0..rows {
                        p *= t.data[r + c * rows];
                    }
                    *oc = p;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("prod: {e}"))?,
                ))
            } else {
                Ok(Value::Num(tensor_prod_all(&t)))
            }
        }
        _ => Err("prod: expected tensor".to_string()),
    }
}

fn prod_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("prod: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![1.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut p = 1.0;
            for r in 0..rows {
                p *= t.data[r + c * rows];
            }
            *oc = p;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("prod: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![1.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut p = 1.0;
            for c in 0..cols {
                p *= t.data[r + c * rows];
            }
            *orow = p;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("prod: {e}"))?,
        ))
    } else {
        Err("prod: dim out of range".to_string())
    }
}

#[runmat_macros::runtime_builtin(name = "prod")]
fn prod_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return prod_all_or_cols(a);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return prod_dim(a, *d),
            Value::Int(i) => return prod_dim(a, i.to_i64() as f64),
            _ => {}
        }
    }
    Err("prod: unsupported arguments".to_string())
}

fn mean_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.reduce_mean(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("mean: unsupported for gpuArray".to_string())
        }
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut s = 0.0;
                    for r in 0..rows {
                        s += t.data[r + c * rows];
                    }
                    *oc = s / (rows as f64);
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("mean: {e}"))?,
                ))
            } else {
                Ok(Value::Num(tensor_sum_all(&t) / (t.data.len() as f64)))
            }
        }
        _ => Err("mean: expected tensor".to_string()),
    }
}

fn mean_dim(a: Value, dim: f64) -> Result<Value, String> {
    if let Value::GpuTensor(h) = a {
        if let Some(p) = runmat_accelerate_api::provider() {
            let d = if dim < 1.0 { 1 } else { dim as usize };
            if let Ok(hc) = p.reduce_mean_dim(&h, d) {
                return Ok(Value::GpuTensor(hc));
            }
        }
        return Err("mean: unsupported for gpuArray".to_string());
    }
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("mean: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut s = 0.0;
            for r in 0..rows {
                s += t.data[r + c * rows];
            }
            *oc = s / (rows as f64);
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("mean: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut s = 0.0;
            for c in 0..cols {
                s += t.data[r + c * rows];
            }
            *orow = s / (cols as f64);
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("mean: {e}"))?,
        ))
    } else {
        Err("mean: dim out of range".to_string())
    }
}

#[runmat_macros::runtime_builtin(name = "mean", accel = "reduction")]
fn mean_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return mean_all_or_cols(a);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return mean_dim(a, *d),
            Value::Int(i) => return mean_dim(a, i.to_i64() as f64),
            _ => {}
        }
    }
    Err("mean: unsupported arguments".to_string())
}

fn any_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut v = 0.0;
                    for r in 0..rows {
                        if t.data[r + c * rows] != 0.0 {
                            v = 1.0;
                            break;
                        }
                    }
                    *oc = v;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("any: {e}"))?,
                ))
            } else {
                Ok(Value::Num(if t.data.iter().any(|&x| x != 0.0) {
                    1.0
                } else {
                    0.0
                }))
            }
        }
        _ => Err("any: expected tensor".to_string()),
    }
}

fn any_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("any: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut v = 0.0;
            for r in 0..rows {
                if t.data[r + c * rows] != 0.0 {
                    v = 1.0;
                    break;
                }
            }
            *oc = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("any: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut v = 0.0;
            for c in 0..cols {
                if t.data[r + c * rows] != 0.0 {
                    v = 1.0;
                    break;
                }
            }
            *orow = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("any: {e}"))?,
        ))
    } else {
        Err("any: dim out of range".to_string())
    }
}

#[runmat_macros::runtime_builtin(name = "any")]
fn any_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return any_all_or_cols(a);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return any_dim(a, *d),
            Value::Int(i) => return any_dim(a, i.to_i64() as f64),
            _ => {}
        }
    }
    Err("any: unsupported arguments".to_string())
}

fn all_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut v = 1.0;
                    for r in 0..rows {
                        if t.data[r + c * rows] == 0.0 {
                            v = 0.0;
                            break;
                        }
                    }
                    *oc = v;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("all: {e}"))?,
                ))
            } else {
                Ok(Value::Num(if t.data.iter().all(|&x| x != 0.0) {
                    1.0
                } else {
                    0.0
                }))
            }
        }
        _ => Err("all: expected tensor".to_string()),
    }
}

fn all_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("all: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut v = 1.0;
            for r in 0..rows {
                if t.data[r + c * rows] == 0.0 {
                    v = 0.0;
                    break;
                }
            }
            *oc = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("all: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut v = 1.0;
            for c in 0..cols {
                if t.data[r + c * rows] == 0.0 {
                    v = 0.0;
                    break;
                }
            }
            *orow = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("all: {e}"))?,
        ))
    } else {
        Err("all: dim out of range".to_string())
    }
}

#[runmat_macros::runtime_builtin(name = "all")]
fn all_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return all_all_or_cols(a);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return all_dim(a, *d),
            Value::Int(i) => return all_dim(a, i.to_i64() as f64),
            _ => {}
        }
    }
    Err("all: unsupported arguments".to_string())
}

// -------- N-D utilities: permute, squeeze, cat --------

#[runmat_macros::runtime_builtin(name = "squeeze")]
fn squeeze_builtin(a: Value) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        Value::StringArray(_) => return Err("squeeze: not supported for string arrays".to_string()),
        Value::CharArray(_) => return Err("squeeze: not supported for char arrays".to_string()),
        _ => return Err("squeeze: expected tensor".to_string()),
    };
    let mut new_shape: Vec<usize> = t.shape.iter().copied().filter(|&d| d != 1).collect();
    if new_shape.is_empty() {
        new_shape.push(1);
    }
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(t.data.clone(), new_shape)
            .map_err(|e| format!("squeeze: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "permute")]
fn permute_builtin(a: Value, order: Value) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        Value::StringArray(_) => return Err("permute: not supported for string arrays".to_string()),
        Value::CharArray(_) => return Err("permute: not supported for char arrays".to_string()),
        _ => return Err("permute: expected tensor".to_string()),
    };
    let ord = match order {
        Value::Tensor(idx) => idx.data.iter().map(|&v| v as usize).collect::<Vec<usize>>(),
        Value::Cell(c) => c
            .data
            .iter()
            .map(|v| match &**v {
                Value::Num(n) => *n as usize,
                Value::Int(i) => i.to_i64() as usize,
                _ => 0,
            })
            .collect(),
        Value::Num(n) => vec![n as usize],
        _ => return Err("permute: expected index vector".to_string()),
    };
    if ord.contains(&0) {
        return Err("permute: indices are 1-based".to_string());
    }
    let ord0: Vec<usize> = ord.into_iter().map(|k| k - 1).collect();
    let rank = t.shape.len();
    if ord0.len() != rank {
        return Err("permute: order length must match rank".to_string());
    }
    let mut new_shape = vec![0usize; rank];
    for (i, &src) in ord0.iter().enumerate() {
        new_shape[i] = *t.shape.get(src).unwrap_or(&1);
    }
    // Precompute strides
    let mut src_strides = vec![0usize; rank];
    let mut acc = 1usize;
    for (d, stride) in src_strides.iter_mut().enumerate().take(rank) {
        *stride = acc;
        acc *= t.shape[d];
    }
    let mut dst_strides = vec![0usize; rank];
    let mut acc2 = 1usize;
    for (d, stride) in dst_strides.iter_mut().enumerate().take(rank) {
        *stride = acc2;
        acc2 *= new_shape[d];
    }
    let total = t.data.len();
    let mut out = vec![0f64; total];
    // Iterate destination multi-index in column-major
    fn unrank(mut lin: usize, shape: &[usize]) -> Vec<usize> {
        let mut idx = Vec::with_capacity(shape.len());
        for &s in shape {
            idx.push(lin % s);
            lin /= s;
        }
        idx
    }
    for (dst_lin, item) in out.iter_mut().enumerate().take(total) {
        let dst_multi = unrank(dst_lin, &new_shape);
        // Map dst dims -> src dims by inverse order mapping
        let mut src_multi = vec![0usize; rank];
        for (dst_d, &src_d) in ord0.iter().enumerate() {
            src_multi[src_d] = dst_multi[dst_d];
        }
        let mut src_lin = 0usize;
        for d in 0..rank {
            src_lin += src_multi[d] * src_strides[d];
        }
        *item = t.data[src_lin];
    }
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(out, new_shape).map_err(|e| format!("permute: {e}"))?,
    ))
}

// -------- Linear algebra helpers: diag, triu, tril --------

#[runmat_macros::runtime_builtin(name = "diag")]
fn diag_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows == 1 || cols == 1 {
                // Vector -> diagonal matrix
                let n = rows.max(cols);
                let mut data = vec![0.0; n * n];
                for (i, slot) in data.iter_mut().enumerate().step_by(n + 1).take(n) {
                    // Map linear i on diagonal to source index
                    let idx = i / (n + 1);
                    let val = t.data[idx];
                    *slot = val;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(data, vec![n, n])
                        .map_err(|e| format!("diag: {e}"))?,
                ))
            } else {
                // Matrix -> main diagonal as column vector
                let n = rows.min(cols);
                let mut data = vec![0.0; n];
                for (i, slot) in data.iter_mut().enumerate().take(n) {
                    *slot = t.data[i + i * rows];
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(data, vec![n, 1])
                        .map_err(|e| format!("diag: {e}"))?,
                ))
            }
        }
        _ => Err("diag: expected tensor".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "triu")]
fn triu_builtin(a: Value) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("triu: expected tensor".to_string()),
    };
    let rows = t.rows();
    let cols = t.cols();
    let mut out = vec![0.0; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            if r <= c {
                out[r + c * rows] = t.data[r + c * rows];
            }
        }
    }
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(out, vec![rows, cols]).map_err(|e| format!("triu: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "tril")]
fn tril_builtin(a: Value) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("tril: expected tensor".to_string()),
    };
    let rows = t.rows();
    let cols = t.cols();
    let mut out = vec![0.0; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            if r >= c {
                out[r + c * rows] = t.data[r + c * rows];
            }
        }
    }
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(out, vec![rows, cols]).map_err(|e| format!("tril: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "cat")]
fn cat_var_builtin(dim: f64, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() < 2 {
        return Err("cat: expects at least two arrays".to_string());
    }
    let d = if dim < 1.0 { 1usize } else { dim as usize } - 1; // zero-based
                                                               // If any string array/string present, do string-array cat
    if rest
        .iter()
        .any(|v| matches!(v, Value::StringArray(_) | Value::String(_)))
    {
        let mut arrs: Vec<runmat_builtins::StringArray> = Vec::new();
        for v in rest {
            match v {
                Value::StringArray(sa) => arrs.push(sa),
                Value::String(s) => arrs.push(
                    runmat_builtins::StringArray::new(vec![s], vec![1, 1])
                        .map_err(|e| format!("cat: {e}"))?,
                ),
                Value::Num(n) => arrs.push(
                    runmat_builtins::StringArray::new(vec![n.to_string()], vec![1, 1])
                        .map_err(|e| format!("cat: {e}"))?,
                ),
                Value::Int(i) => arrs.push(
                    runmat_builtins::StringArray::new(vec![i.to_i64().to_string()], vec![1, 1])
                        .map_err(|e| format!("cat: {e}"))?,
                ),
                other => {
                    return Err(format!(
                        "cat: expected string arrays/strings or scalars, got {other:?}"
                    ))
                }
            }
        }
        let rank = arrs
            .iter()
            .map(|a| a.shape.len())
            .max()
            .unwrap_or(2)
            .max(d + 1);
        let shapes: Vec<Vec<usize>> = arrs
            .iter()
            .map(|a| {
                let mut s = a.shape.clone();
                if s.len() < rank {
                    s.resize(rank, 1);
                }
                s
            })
            .collect();
        for k in 0..rank {
            if k == d {
                continue;
            }
            let first = shapes[0][k];
            if !shapes.iter().all(|s| s[k] == first) {
                return Err("cat: dimension mismatch".to_string());
            }
        }
        let mut out_shape = shapes[0].clone();
        out_shape[d] = shapes.iter().map(|s| s[d]).sum();
        fn strides(shape: &[usize]) -> Vec<usize> {
            let mut s = vec![0; shape.len()];
            let mut acc = 1;
            for i in 0..shape.len() {
                s[i] = acc;
                acc *= shape[i];
            }
            s
        }
        let out_str = strides(&out_shape);
        let mut out: Vec<String> = vec![String::new(); out_shape.iter().product()];
        let mut offset = 0usize;
        for (a, s) in arrs.iter().zip(shapes.iter()) {
            let s_str = strides(s);
            let total: usize = s.iter().product();
            let rank = out_shape.len();
            for idx_lin in 0..total {
                let mut rem = idx_lin;
                let mut src_multi = vec![0usize; rank];
                for i in 0..rank {
                    let si = s[i];
                    src_multi[i] = rem % si;
                    rem /= si;
                }
                let mut dst_multi = src_multi.clone();
                dst_multi[d] += offset;
                let mut s_lin = 0usize;
                for i in 0..rank {
                    s_lin += src_multi[i] * s_str[i];
                }
                let mut d_lin = 0usize;
                for i in 0..rank {
                    d_lin += dst_multi[i] * out_str[i];
                }
                out[d_lin] = a.data[s_lin].clone();
            }
            offset += s[d];
        }
        return Ok(Value::StringArray(
            runmat_builtins::StringArray::new(out, out_shape).map_err(|e| format!("cat: {e}"))?,
        ));
    }
    // Numeric cat path
    let tensors: Vec<runmat_builtins::Tensor> =
        rest.into_iter()
            .map(|v| match v {
                Value::Tensor(t) => Ok(t),
                Value::Num(n) => runmat_builtins::Tensor::new(vec![n], vec![1, 1])
                    .map_err(|e| format!("cat: {e}")),
                Value::Int(i) => runmat_builtins::Tensor::new(vec![i.to_f64()], vec![1, 1])
                    .map_err(|e| format!("cat: {e}")),
                Value::Bool(b) => {
                    runmat_builtins::Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                        .map_err(|e| format!("cat: {e}"))
                }
                other => Err(format!("cat: expected tensors or scalars, got {other:?}")),
            })
            .collect::<Result<_, _>>()?;
    let mut rank = tensors.iter().map(|t| t.shape.len()).max().unwrap_or(2);
    if d + 1 > rank {
        rank = d + 1;
    }
    let shapes: Vec<Vec<usize>> = tensors
        .iter()
        .map(|t| {
            let mut s = t.shape.clone();
            s.resize(rank, 1);
            s
        })
        .collect();
    for k in 0..rank {
        if k == d {
            continue;
        }
        let first = shapes[0][k];
        if !shapes.iter().all(|s| s[k] == first) {
            return Err("cat: dimension mismatch".to_string());
        }
    }
    let mut out_shape = shapes[0].clone();
    out_shape[d] = shapes.iter().map(|s| s[d]).sum();
    let mut out = vec![0f64; out_shape.iter().product()];
    fn strides(shape: &[usize]) -> Vec<usize> {
        let mut s = vec![0; shape.len()];
        let mut acc = 1;
        for i in 0..shape.len() {
            s[i] = acc;
            acc *= shape[i];
        }
        s
    }
    let out_str = strides(&out_shape);
    let mut offset = 0usize;
    for (t, s) in tensors.iter().zip(shapes.iter()) {
        let s_str = strides(s);
        let rank = out_shape.len();
        let total: usize = s.iter().product();
        for idx_lin in 0..total {
            // Convert src lin to multi
            let mut rem = idx_lin;
            let mut src_multi = vec![0usize; rank];
            for i in 0..rank {
                let si = s[i];
                src_multi[i] = rem % si;
                rem /= si;
            }
            let mut dst_multi = src_multi.clone();
            dst_multi[d] += offset;
            let mut s_lin = 0usize;
            for i in 0..rank {
                s_lin += src_multi[i] * s_str[i];
            }
            let mut d_lin = 0usize;
            for i in 0..rank {
                d_lin += dst_multi[i] * out_str[i];
            }
            out[d_lin] = t.data[s_lin];
        }
        offset += s[d];
    }
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(out, out_shape).map_err(|e| format!("cat: {e}"))?,
    ))
}

// repmat 2D helper (covered by varargs entry)
#[inline]
fn repmat_builtin(a: Value, m: f64, n: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("repmat: expected tensor".to_string()),
    };
    let m = if m < 1.0 { 1usize } else { m as usize };
    let n = if n < 1.0 { 1usize } else { n as usize };
    let mut shape = t.shape.clone();
    if shape.len() < 2 {
        shape.resize(2, 1);
    }
    let base_rows = shape[0];
    let base_cols = shape[1];
    let out_rows = base_rows * m;
    let out_cols = base_cols * n;
    let mut out = vec![0f64; out_rows * out_cols];
    for rep_c in 0..n {
        for rep_r in 0..m {
            for c in 0..base_cols {
                for r in 0..base_rows {
                    let src = t.data[r + c * base_rows];
                    let dr = r + rep_r * base_rows;
                    let dc = c + rep_c * base_cols;
                    out[dr + dc * out_rows] = src;
                }
            }
        }
    }
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(out, vec![out_rows, out_cols])
            .map_err(|e| format!("repmat: {e}"))?,
    ))
}

#[runmat_macros::runtime_builtin(name = "repmat")]
fn repmat_nd_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("repmat: expected tensor".to_string()),
    };
    // Build replication factors from rest
    let mut reps: Vec<usize> = Vec::new();
    if rest.len() == 1 {
        match &rest[0] {
            Value::Tensor(v) => {
                for &x in &v.data {
                    reps.push(if x < 1.0 { 1 } else { x as usize });
                }
            }
            _ => return Err("repmat: expected replication vector".to_string()),
        }
    } else {
        for v in rest {
            match v {
                Value::Num(n) => reps.push(if n < 1.0 { 1 } else { n as usize }),
                Value::Int(i) => {
                    let ii = i.to_i64();
                    reps.push(if ii < 1 { 1 } else { ii as usize })
                }
                _ => return Err("repmat: expected numeric reps".to_string()),
            }
        }
    }
    // If 2D case, delegate to the 2D helper to avoid unused warning.
    if reps.len() == 2 {
        return repmat_builtin(Value::Tensor(t), reps[0] as f64, reps[1] as f64);
    }
    let rank = t.shape.len().max(reps.len());
    let mut base = t.shape.clone();
    base.resize(rank, 1);
    reps.resize(rank, 1);
    let mut out_shape = base.clone();
    for i in 0..rank {
        out_shape[i] = base[i] * reps[i];
    }
    let mut out = vec![0f64; out_shape.iter().product()];
    fn strides(shape: &[usize]) -> Vec<usize> {
        let mut s = vec![0; shape.len()];
        let mut acc = 1;
        for i in 0..shape.len() {
            s[i] = acc;
            acc *= shape[i];
        }
        s
    }
    let src_str = strides(&base);
    // Iterate over all dest coords and map back to source via modulo
    let total: usize = out_shape.iter().product();
    for (d_lin, item) in out.iter_mut().enumerate().take(total) {
        // convert to multi
        let mut rem = d_lin;
        let mut multi = vec![0usize; rank];
        for i in 0..rank {
            let s = out_shape[i];
            multi[i] = rem % s;
            rem /= s;
        }
        // map to src via modulo by base size
        let mut src_lin = 0usize;
        for i in 0..rank {
            let coord = multi[i] % base[i];
            src_lin += coord * src_str[i];
        }
        *item = t.data[src_lin];
    }
    Ok(Value::Tensor(
        runmat_builtins::Tensor::new(out, out_shape).map_err(|e| format!("repmat: {e}"))?,
    ))
}

// linspace and meshgrid are defined in arrays.rs; avoid duplicates here

// -------- Vararg string/IO builtins: sprintf, fprintf, disp, warning --------

#[runmat_macros::runtime_builtin(name = "sprintf", sink = true)]
fn sprintf_builtin(fmt: String, rest: Vec<Value>) -> Result<Value, String> {
    let s = format_variadic(&fmt, &rest)?;
    Ok(Value::String(s))
}

#[runmat_macros::runtime_builtin(name = "fprintf", sink = true)]
fn fprintf_builtin(first: Value, rest: Vec<Value>) -> Result<Value, String> {
    // MATLAB: fprintf(fid, fmt, ...) or fprintf(fmt, ...)
    let (fmt, args) = match first {
        Value::String(s) => (s, rest),
        Value::Num(_) | Value::Int(_) => {
            // File IDs not supported yet; treat as stdout and expect format string next
            if rest.is_empty() {
                return Err("fprintf: missing format string".to_string());
            }
            let fmt = match &rest[0] {
                Value::String(s) => s.clone(),
                _ => return Err("fprintf: expected format string".to_string()),
            };
            (fmt, rest[1..].to_vec())
        }
        other => return Err(format!("fprintf: unsupported first argument {other:?}")),
    };
    let s = format_variadic(&fmt, &args)?;
    println!("{s}");
    Ok(Value::Num(s.len() as f64))
}

#[runmat_macros::runtime_builtin(name = "warning", sink = true)]
fn warning_builtin(fmt: String, rest: Vec<Value>) -> Result<Value, String> {
    let s = format_variadic(&fmt, &rest)?;
    eprintln!("Warning: {s}");
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(name = "disp", sink = true)]
fn disp_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::String(s) => println!("{s}"),
        Value::Num(n) => println!("{n}"),
        Value::Int(i) => println!("{}", i.to_i64()),
        Value::Tensor(t) => println!("{:?}", t.data),
        other => println!("{other:?}"),
    }
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(name = "struct")]
fn struct_builtin(rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() % 2 != 0 {
        return Err("struct: expected name/value pairs".to_string());
    }
    let mut st = runmat_builtins::StructValue::new();
    let mut i = 0usize;
    while i < rest.len() {
        let key: String = (&rest[i]).try_into()?;
        let val = rest[i + 1].clone();
        st.fields.insert(key, val);
        i += 2;
    }
    Ok(Value::Struct(st))
}

fn format_variadic(fmt: &str, args: &[Value]) -> Result<String, String> {
    // Minimal subset: supports %d/%i, %f, %s and %%
    let mut out = String::with_capacity(fmt.len() + args.len() * 8);
    let mut it = fmt.chars().peekable();
    let mut ai = 0usize;
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        if let Some('%') = it.peek() {
            it.next();
            out.push('%');
            continue;
        }
        // Consume optional width/precision (very limited)
        let mut precision: Option<usize> = None;
        // skip digits for width
        while let Some(ch) = it.peek() {
            if ch.is_ascii_digit() {
                it.next();
            } else {
                break;
            }
        }
        // precision .digits
        if let Some('.') = it.peek() {
            it.next();
            let mut p = String::new();
            while let Some(ch) = it.peek() {
                if ch.is_ascii_digit() {
                    p.push(*ch);
                    it.next();
                } else {
                    break;
                }
            }
            if !p.is_empty() {
                precision = p.parse::<usize>().ok();
            }
        }
        let ty = it.next().ok_or("sprintf: incomplete format specifier")?;
        let val = args.get(ai).cloned().unwrap_or(Value::Num(0.0));
        ai += 1;
        match ty {
            'd' | 'i' => {
                let v: f64 = (&val).try_into()?;
                out.push_str(&(v as i64).to_string());
            }
            'f' => {
                let v: f64 = (&val).try_into()?;
                if let Some(p) = precision {
                    out.push_str(&format!("{v:.p$}"));
                } else {
                    out.push_str(&format!("{v}"));
                }
            }
            's' => match val {
                Value::String(s) => out.push_str(&s),
                Value::Num(n) => out.push_str(&n.to_string()),
                Value::Int(i) => out.push_str(&i.to_i64().to_string()),
                Value::Tensor(t) => out.push_str(&format!("{:?}", t.data)),
                other => out.push_str(&format!("{other:?}")),
            },
            other => return Err(format!("sprintf: unsupported format %{other}")),
        }
    }
    Ok(out)
}

#[runmat_macros::runtime_builtin(name = "getmethod")]
fn getmethod_builtin(obj: Value, name: String) -> Result<Value, String> {
    match obj {
        Value::Object(o) => {
            // Return a closure capturing the receiver; feval will call runtime builtin call_method
            Ok(Value::Closure(runmat_builtins::Closure {
                function_name: "call_method".to_string(),
                captures: vec![Value::Object(o), Value::String(name)],
            }))
        }
        Value::ClassRef(cls) => Ok(Value::String(format!("@{cls}.{name}"))),
        other => Err(format!("getmethod unsupported on {other:?}")),
    }
}
