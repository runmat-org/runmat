//! MATLAB-compatible `ismember` builtin with GPU-aware semantics for RunMat.

use std::collections::HashMap;

use runmat_accelerate_api::{
    GpuTensorHandle, HostLogicalOwned, HostTensorOwned, IsMemberOptions as ProviderIsMemberOptions,
    IsMemberResult,
};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use super::type_resolvers::logical_output_type;
use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::ismember")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ismember",
    op_kind: GpuOpKind::Custom("ismember"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("ismember")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may supply dedicated membership kernels; until then RunMat gathers GPU tensors to host memory.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::ismember"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ismember",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`ismember` materialises logical outputs and terminates fusion chains; upstream tensors are gathered when necessary.",
};

fn ismember_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("ismember")
        .build()
}

#[runtime_builtin(
    name = "ismember",
    category = "array/sorting_sets",
    summary = "Identify array elements or rows that appear in another array while returning first-match indices.",
    keywords = "ismember,membership,set,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    type_resolver(logical_output_type),
    builtin_path = "crate::builtins::array::sorting_sets::ismember"
)]
async fn ismember_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    Ok(evaluate(a, b, &rest).await?.into_mask_value())
}

/// Evaluate the `ismember` builtin once and expose all outputs.
pub async fn evaluate(
    a: Value,
    b: Value,
    rest: &[Value],
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let opts = parse_options(rest)?;
    match (a, b) {
        (Value::GpuTensor(handle_a), Value::GpuTensor(handle_b)) => {
            ismember_gpu_pair(handle_a, handle_b, &opts).await
        }
        (Value::GpuTensor(handle_a), other) => {
            ismember_gpu_mixed(handle_a, other, &opts, true).await
        }
        (other, Value::GpuTensor(handle_b)) => {
            ismember_gpu_mixed(handle_b, other, &opts, false).await
        }
        (left, right) => ismember_host(left, right, &opts),
    }
}

#[derive(Debug, Clone, Copy)]
struct IsMemberOptions {
    rows: bool,
}

impl IsMemberOptions {
    fn into_provider_options(self) -> ProviderIsMemberOptions {
        ProviderIsMemberOptions { rows: self.rows }
    }
}

fn parse_options(rest: &[Value]) -> crate::BuiltinResult<IsMemberOptions> {
    let mut opts = IsMemberOptions { rows: false };
    for arg in rest {
        let text = tensor::value_to_string(arg)
            .ok_or_else(|| ismember_error("ismember: expected string option arguments"))?;
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "rows" => opts.rows = true,
            "legacy" | "r2012a" => {
                return Err(ismember_error(
                    "ismember: the 'legacy' behaviour is not supported",
                ))
            }
            other => {
                return Err(ismember_error(format!(
                    "ismember: unrecognised option '{other}'"
                )))
            }
        }
    }
    Ok(opts)
}

async fn ismember_gpu_pair(
    handle_a: GpuTensorHandle,
    handle_b: GpuTensorHandle,
    opts: &IsMemberOptions,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let provider_opts = opts.into_provider_options();
        match provider
            .ismember(&handle_a, &handle_b, &provider_opts)
            .await
        {
            Ok(result) => return IsMemberEvaluation::from_provider_result(result),
            Err(_) => {
                // Fall back to host gather when the provider lacks an ismember implementation.
            }
        }
    }
    let tensor_a = gpu_helpers::gather_tensor_async(&handle_a).await?;
    let tensor_b = gpu_helpers::gather_tensor_async(&handle_b).await?;
    ismember_numeric_tensors(tensor_a, tensor_b, opts)
}

async fn ismember_gpu_mixed(
    handle_gpu: GpuTensorHandle,
    other: Value,
    opts: &IsMemberOptions,
    gpu_is_a: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let tensor_gpu = gpu_helpers::gather_tensor_async(&handle_gpu).await?;
    if gpu_is_a {
        ismember_host(Value::Tensor(tensor_gpu), other, opts)
    } else {
        ismember_host(other, Value::Tensor(tensor_gpu), opts)
    }
}

fn ismember_host(
    a: Value,
    b: Value,
    opts: &IsMemberOptions,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    match (a, b) {
        (Value::ComplexTensor(at), Value::ComplexTensor(bt)) => ismember_complex(at, bt, opts.rows),
        (Value::ComplexTensor(at), Value::Complex(re, im)) => {
            let bt = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| ismember_error(format!("ismember: {e}")))?;
            ismember_complex(at, bt, opts.rows)
        }
        (Value::Complex(a_re, a_im), Value::ComplexTensor(bt)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| ismember_error(format!("ismember: {e}")))?;
            ismember_complex(at, bt, opts.rows)
        }
        (Value::Complex(a_re, a_im), Value::Complex(b_re, b_im)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| ismember_error(format!("ismember: {e}")))?;
            let bt = ComplexTensor::new(vec![(b_re, b_im)], vec![1, 1])
                .map_err(|e| ismember_error(format!("ismember: {e}")))?;
            ismember_complex(at, bt, opts.rows)
        }

        (Value::CharArray(ac), Value::CharArray(bc)) => ismember_char(ac, bc, opts.rows),

        (Value::StringArray(astring), Value::StringArray(bstring)) => {
            ismember_string(astring, bstring, opts.rows)
        }
        (Value::StringArray(astring), Value::String(b)) => {
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| ismember_error(format!("ismember: {e}")))?;
            ismember_string(astring, bstring, opts.rows)
        }
        (Value::String(a), Value::StringArray(bstring)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| ismember_error(format!("ismember: {e}")))?;
            ismember_string(astring, bstring, opts.rows)
        }
        (Value::String(a), Value::String(b)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| ismember_error(format!("ismember: {e}")))?;
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| ismember_error(format!("ismember: {e}")))?;
            ismember_string(astring, bstring, opts.rows)
        }

        (left, right) => {
            let tensor_a =
                tensor::value_into_tensor_for("ismember", left).map_err(|e| ismember_error(e))?;
            let tensor_b =
                tensor::value_into_tensor_for("ismember", right).map_err(|e| ismember_error(e))?;
            ismember_numeric_tensors(tensor_a, tensor_b, opts)
        }
    }
}

fn ismember_numeric_tensors(
    a: Tensor,
    b: Tensor,
    opts: &IsMemberOptions,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if opts.rows {
        ismember_numeric_rows(a, b)
    } else {
        ismember_numeric_elements(a, b)
    }
}

/// Helper exposed for acceleration providers handling numeric tensors on the host.
pub fn ismember_numeric_from_tensors(
    a: Tensor,
    b: Tensor,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let opts = IsMemberOptions { rows };
    ismember_numeric_tensors(a, b, &opts)
}

fn ismember_numeric_elements(a: Tensor, b: Tensor) -> crate::BuiltinResult<IsMemberEvaluation> {
    let mut map: HashMap<u64, usize> = HashMap::new();
    for (idx, &value) in b.data.iter().enumerate() {
        map.entry(canonicalize_f64(value)).or_insert(idx + 1);
    }

    let mut mask_data = Vec::<u8>::with_capacity(a.data.len());
    let mut loc_data = Vec::<f64>::with_capacity(a.data.len());

    for &value in &a.data {
        let key = canonicalize_f64(value);
        if let Some(&pos) = map.get(&key) {
            mask_data.push(1);
            loc_data.push(pos as f64);
        } else {
            mask_data.push(0);
            loc_data.push(0.0);
        }
    }

    let logical = LogicalArray::new(mask_data, a.shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, a.shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_numeric_rows(a: Tensor, b: Tensor) -> crate::BuiltinResult<IsMemberEvaluation> {
    let (rows_a, cols_a) = tensor_rows_cols(&a, "ismember")?;
    let (rows_b, cols_b) = tensor_rows_cols(&b, "ismember")?;
    if cols_a != cols_b {
        return Err(ismember_error(
            "ismember: inputs must have the same number of columns when using 'rows'",
        ));
    }

    let mut map: HashMap<NumericRowKey, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols_b);
        for c in 0..cols_b {
            let idx = r + c * rows_b;
            row_values.push(b.data[idx]);
        }
        let key = NumericRowKey::from_slice(&row_values);
        map.entry(key).or_insert(r + 1);
    }

    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = vec![0.0f64; rows_a];

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols_a);
        for c in 0..cols_a {
            let idx = r + c * rows_a;
            row_values.push(a.data[idx]);
        }
        let key = NumericRowKey::from_slice(&row_values);
        if let Some(&pos) = map.get(&key) {
            mask_data[r] = 1;
            loc_data[r] = pos as f64;
        }
    }

    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    let loc_tensor =
        Tensor::new(loc_data, shape).map_err(|e| ismember_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if rows {
        ismember_complex_rows(a, b)
    } else {
        ismember_complex_elements(a, b)
    }
}

fn ismember_complex_elements(
    a: ComplexTensor,
    b: ComplexTensor,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let mut map: HashMap<ComplexKey, usize> = HashMap::new();
    for (idx, &value) in b.data.iter().enumerate() {
        map.entry(ComplexKey::new(value)).or_insert(idx + 1);
    }

    let mut mask_data = Vec::<u8>::with_capacity(a.data.len());
    let mut loc_data = Vec::<f64>::with_capacity(a.data.len());

    for &value in &a.data {
        let key = ComplexKey::new(value);
        if let Some(&pos) = map.get(&key) {
            mask_data.push(1);
            loc_data.push(pos as f64);
        } else {
            mask_data.push(0);
            loc_data.push(0.0);
        }
    }

    let logical = LogicalArray::new(mask_data, a.shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, a.shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_complex_rows(
    a: ComplexTensor,
    b: ComplexTensor,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let (rows_a, cols_a) = complex_rows_cols(&a)?;
    let (rows_b, cols_b) = complex_rows_cols(&b)?;
    if cols_a != cols_b {
        return Err(ismember_error(
            "ismember: complex inputs must have the same number of columns when using 'rows'",
        )
        .into());
    }

    let mut map: HashMap<Vec<ComplexKey>, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_keys = Vec::with_capacity(cols_b);
        for c in 0..cols_b {
            let idx = r + c * rows_b;
            row_keys.push(ComplexKey::new(b.data[idx]));
        }
        map.entry(row_keys).or_insert(r + 1);
    }

    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = vec![0.0f64; rows_a];

    for r in 0..rows_a {
        let mut row_keys = Vec::with_capacity(cols_a);
        for c in 0..cols_a {
            let idx = r + c * rows_a;
            row_keys.push(ComplexKey::new(a.data[idx]));
        }
        if let Some(&pos) = map.get(&row_keys) {
            mask_data[r] = 1;
            loc_data[r] = pos as f64;
        }
    }

    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    let loc_tensor =
        Tensor::new(loc_data, shape).map_err(|e| ismember_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_char(
    a: CharArray,
    b: CharArray,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if rows {
        ismember_char_rows(a, b)
    } else {
        ismember_char_elements(a, b)
    }
}

fn ismember_char_elements(a: CharArray, b: CharArray) -> crate::BuiltinResult<IsMemberEvaluation> {
    let rows_b = b.rows;
    let cols_b = b.cols;
    let mut map: HashMap<char, usize> = HashMap::new();

    for col in 0..cols_b {
        for row in 0..rows_b {
            let data_idx = row * cols_b + col;
            let ch = b.data[data_idx];
            let linear_idx = row + col * rows_b;
            map.entry(ch).or_insert(linear_idx + 1);
        }
    }

    let rows_a = a.rows;
    let cols_a = a.cols;
    let mut mask_data = vec![0u8; rows_a * cols_a];
    let mut loc_data = vec![0.0f64; rows_a * cols_a];

    for col in 0..cols_a {
        for row in 0..rows_a {
            let data_idx = row * cols_a + col;
            let ch = a.data[data_idx];
            let linear_idx = row + col * rows_a;
            if let Some(&pos) = map.get(&ch) {
                mask_data[linear_idx] = 1;
                loc_data[linear_idx] = pos as f64;
            }
        }
    }

    let shape = vec![rows_a, cols_a];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    let loc_tensor =
        Tensor::new(loc_data, shape).map_err(|e| ismember_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_char_rows(a: CharArray, b: CharArray) -> crate::BuiltinResult<IsMemberEvaluation> {
    if a.cols != b.cols {
        return Err(ismember_error(
            "ismember: character inputs must have the same number of columns when using 'rows'",
        )
        .into());
    }

    let rows_b = b.rows;
    let cols = b.cols;
    let mut map: HashMap<RowCharKey, usize> = HashMap::new();

    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(b.data[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        map.entry(key).or_insert(r + 1);
    }

    let rows_a = a.rows;
    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = vec![0.0f64; rows_a];

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(a.data[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        if let Some(&pos) = map.get(&key) {
            mask_data[r] = 1;
            loc_data[r] = pos as f64;
        }
    }

    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    let loc_tensor =
        Tensor::new(loc_data, shape).map_err(|e| ismember_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_string(
    a: StringArray,
    b: StringArray,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if rows {
        ismember_string_rows(a, b)
    } else {
        ismember_string_elements(a, b)
    }
}

fn ismember_string_elements(
    a: StringArray,
    b: StringArray,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let mut map: HashMap<String, usize> = HashMap::new();
    for (idx, value) in b.data.iter().enumerate() {
        map.entry(value.clone()).or_insert(idx + 1);
    }

    let mut mask_data = Vec::<u8>::with_capacity(a.data.len());
    let mut loc_data = Vec::<f64>::with_capacity(a.data.len());

    for value in &a.data {
        if let Some(&pos) = map.get(value) {
            mask_data.push(1);
            loc_data.push(pos as f64);
        } else {
            mask_data.push(0);
            loc_data.push(0.0);
        }
    }

    let logical = LogicalArray::new(mask_data, a.shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, a.shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_string_rows(
    a: StringArray,
    b: StringArray,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(ismember_error(
            "ismember: 'rows' option requires 2-D string arrays",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(ismember_error(
            "ismember: string inputs must have the same number of columns when using 'rows'",
        )
        .into());
    }

    let rows_a = a.shape[0];
    let cols = a.shape[1];
    let rows_b = b.shape[0];

    let mut map: HashMap<RowStringKey, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b.data[idx].clone());
        }
        let key = RowStringKey(row_values);
        map.entry(key).or_insert(r + 1);
    }

    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = vec![0.0f64; rows_a];

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a.data[idx].clone());
        }
        let key = RowStringKey(row_values);
        if let Some(&pos) = map.get(&key) {
            mask_data[r] = 1;
            loc_data[r] = pos as f64;
        }
    }

    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_error(format!("ismember: {e}")))?;
    let loc_tensor =
        Tensor::new(loc_data, shape).map_err(|e| ismember_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn tensor_rows_cols(t: &Tensor, name: &str) -> crate::BuiltinResult<(usize, usize)> {
    match t.shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((t.shape[0], 1)),
        2 => Ok((t.shape[0], t.shape[1])),
        _ => Err(ismember_error(format!(
            "{name}: 'rows' option requires 2-D numeric matrices"
        ))
        .into()),
    }
}

fn complex_rows_cols(t: &ComplexTensor) -> crate::BuiltinResult<(usize, usize)> {
    match t.shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((t.shape[0], 1)),
        2 => Ok((t.shape[0], t.shape[1])),
        _ => Err(ismember_error(
            "ismember: 'rows' option requires 2-D complex matrices",
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NumericRowKey(Vec<u64>);

impl NumericRowKey {
    fn from_slice(values: &[f64]) -> Self {
        NumericRowKey(values.iter().map(|&v| canonicalize_f64(v)).collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ComplexKey {
    re: u64,
    im: u64,
}

impl ComplexKey {
    fn new(value: (f64, f64)) -> Self {
        Self {
            re: canonicalize_f64(value.0),
            im: canonicalize_f64(value.1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowCharKey(Vec<u32>);

impl RowCharKey {
    fn from_slice(values: &[char]) -> Self {
        RowCharKey(values.iter().map(|&ch| ch as u32).collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowStringKey(Vec<String>);

fn canonicalize_f64(value: f64) -> u64 {
    if value.is_nan() {
        0x7ff8_0000_0000_0000u64
    } else if value == 0.0 {
        0u64
    } else {
        value.to_bits()
    }
}

#[derive(Debug, Clone)]
pub struct IsMemberEvaluation {
    mask: LogicalArray,
    loc: Tensor,
}

impl IsMemberEvaluation {
    fn new(mask: LogicalArray, loc: Tensor) -> Self {
        Self { mask, loc }
    }

    pub fn from_provider_result(result: IsMemberResult) -> crate::BuiltinResult<Self> {
        let mask = LogicalArray::new(result.mask.data, result.mask.shape)
            .map_err(|e| ismember_error(format!("ismember: {e}")))?;
        let loc = Tensor::new(result.loc.data, result.loc.shape)
            .map_err(|e| ismember_error(format!("ismember: {e}")))?;
        Ok(IsMemberEvaluation::new(mask, loc))
    }

    pub fn into_numeric_ismember_result(self) -> crate::BuiltinResult<IsMemberResult> {
        let IsMemberEvaluation { mask, loc } = self;
        Ok(IsMemberResult {
            mask: HostLogicalOwned {
                data: mask.data,
                shape: mask.shape,
            },
            loc: HostTensorOwned {
                data: loc.data,
                shape: loc.shape,
            },
        })
    }

    pub fn into_mask_value(self) -> Value {
        logical_array_into_value(self.mask)
    }

    pub fn mask_value(&self) -> Value {
        logical_array_into_value(self.mask.clone())
    }

    pub fn into_pair(self) -> (Value, Value) {
        let mask = logical_array_into_value(self.mask);
        let loc = tensor::tensor_into_value(self.loc);
        (mask, loc)
    }

    pub fn loc_value(&self) -> Value {
        tensor::tensor_into_value(self.loc.clone())
    }
}

fn logical_array_into_value(logical: LogicalArray) -> Value {
    if logical.data.len() == 1 {
        Value::Bool(logical.data[0] != 0)
    } else {
        Value::LogicalArray(logical)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ResolveContext, Tensor, Type};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn evaluate_sync(
        a: Value,
        b: Value,
        rest: &[Value],
    ) -> crate::BuiltinResult<IsMemberEvaluation> {
        futures::executor::block_on(evaluate(a, b, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_membership_basic() {
        let a = Tensor::new(vec![5.0, 7.0, 2.0, 7.0], vec![1, 4]).unwrap();
        let b = Tensor::new(vec![7.0, 9.0, 5.0], vec![1, 3]).unwrap();
        let eval = ismember_numeric_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1, 0, 1]);
        assert_eq!(eval.loc.data, vec![3.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn ismember_type_resolver_logical() {
        assert_eq!(
            logical_output_type(
                &[Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new()),
            ),
            Type::logical()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_nan_membership() {
        let a = Tensor::new(vec![f64::NAN, 1.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![f64::NAN, 2.0], vec![1, 2]).unwrap();
        let eval = ismember_numeric_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 0]);
        assert_eq!(eval.loc.data, vec![1.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_rows_membership() {
        let a = Tensor::new(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 5.0, 1.0, 4.0, 6.0, 2.0], vec![3, 2]).unwrap();
        let eval = ismember_numeric_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1, 1]);
        assert_eq!(eval.loc.data, vec![3.0, 1.0, 3.0]);
        assert_eq!(eval.loc.shape, vec![3, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_membership() {
        let a = ComplexTensor::new(vec![(1.0, 2.0), (0.0, 0.0)], vec![1, 2]).unwrap();
        let b = ComplexTensor::new(vec![(0.0, 0.0), (1.0, 2.0)], vec![1, 2]).unwrap();
        let eval = ismember_complex_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.data, vec![2.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rows_membership() {
        let a = ComplexTensor::new(
            vec![(1.0, 1.0), (3.0, 0.0), (2.0, 0.0), (4.0, 4.0)],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::new(
            vec![
                (1.0, 1.0),
                (5.0, 0.0),
                (3.0, 0.0),
                (2.0, 0.0),
                (6.0, 0.0),
                (4.0, 4.0),
            ],
            vec![3, 2],
        )
        .unwrap();
        let eval = ismember_complex_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.data, vec![1.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_membership() {
        let a = CharArray::new(vec!['r', 'u', 'n', 'm'], 2, 2).unwrap();
        let b = CharArray::new(vec!['m', 'a', 'r', 'u'], 2, 2).unwrap();
        let eval = ismember_char_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 0, 1, 1]);
        assert_eq!(eval.loc.data, vec![2.0, 0.0, 4.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rows_membership() {
        let a = CharArray::new(vec!['m', 'a', 't', 'l'], 2, 2).unwrap();
        let b = CharArray::new(vec!['m', 'a', 'g', 'e', 't', 'l'], 3, 2).unwrap();
        let eval = ismember_char_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.data, vec![1.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_membership() {
        let a = StringArray::new(
            vec![
                "apple".to_string(),
                "pear".to_string(),
                "banana".to_string(),
            ],
            vec![1, 3],
        )
        .unwrap();
        let b = StringArray::new(
            vec![
                "pear".to_string(),
                "orange".to_string(),
                "apple".to_string(),
            ],
            vec![1, 3],
        )
        .unwrap();
        let eval = ismember_string_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1, 0]);
        assert_eq!(eval.loc.data, vec![3.0, 1.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_rows_membership() {
        let a = StringArray::new(
            vec![
                "alpha".to_string(),
                "gamma".to_string(),
                "beta".to_string(),
                "delta".to_string(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let b = StringArray::new(
            vec![
                "alpha".to_string(),
                "theta".to_string(),
                "gamma".to_string(),
                "beta".to_string(),
                "eta".to_string(),
                "delta".to_string(),
            ],
            vec![3, 2],
        )
        .unwrap();
        let eval = ismember_string_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.data, vec![1.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn options_reject_legacy() {
        let err = error_message(parse_options(&[Value::from("legacy")]).unwrap_err());
        assert!(err.contains("legacy"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unknown_option() {
        let err = error_message(
            evaluate_sync(Value::Num(1.0), Value::Num(1.0), &[Value::from("stable")]).unwrap_err(),
        );
        assert!(err.contains("unrecognised option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismember_runtime_numeric() {
        let a = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap());
        let (mask, loc) = evaluate_sync(a, b, &[]).unwrap().into_pair();
        match mask {
            Value::LogicalArray(arr) => assert_eq!(arr.data, vec![1, 0, 1]),
            other => panic!("expected logical array, got {other:?}"),
        }
        match loc {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 0.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_inputs_promoted() {
        let a = Value::Bool(true);
        let logical_b =
            LogicalArray::new(vec![1, 0], vec![2, 1]).expect("logical array construction");
        let eval = evaluate_sync(a, Value::LogicalArray(logical_b), &[]).expect("ismember");
        assert_eq!(eval.mask_value(), Value::Bool(true));
        assert_eq!(eval.loc_value(), Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismember_rows_shape_checks() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        assert!(ismember_numeric_rows(a.clone(), b.clone()).is_ok());
        let bad = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = error_message(ismember_numeric_rows(a, bad).unwrap_err());
        assert!(err.contains("same number of columns"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismember_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 4.0], vec![4, 1]).unwrap();
            let set = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
            let view_a = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let view_b = runmat_accelerate_api::HostTensorView {
                data: &set.data,
                shape: &set.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload a");
            let handle_b = provider.upload(&view_b).expect("upload b");
            let eval = evaluate_sync(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[])
                .expect("ismember");
            assert_eq!(eval.mask.data, vec![0, 1, 0, 1]);
            assert_eq!(eval.loc.data, vec![0.0, 1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismember_gpu_rows_roundtrip() {
        test_support::with_test_provider(|provider| {
            let rows = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let bank = Tensor::new(vec![1.0, 5.0, 3.0, 2.0, 6.0, 4.0], vec![3, 2]).unwrap();
            let view_a = runmat_accelerate_api::HostTensorView {
                data: &rows.data,
                shape: &rows.shape,
            };
            let view_b = runmat_accelerate_api::HostTensorView {
                data: &bank.data,
                shape: &bank.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload a");
            let handle_b = provider.upload(&view_b).expect("upload b");
            let eval = evaluate_sync(
                Value::GpuTensor(handle_a.clone()),
                Value::GpuTensor(handle_b.clone()),
                &[Value::from("rows")],
            )
            .expect("ismember");
            assert_eq!(eval.mask.data, vec![1, 1]);
            assert_eq!(eval.loc.data, vec![1.0, 3.0]);
            let _ = provider.free(&handle_a);
            let _ = provider.free(&handle_b);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ismember_wgpu_numeric_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 4.0], vec![4, 1]).unwrap();
        let set = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let cpu_eval =
            ismember_numeric_from_tensors(tensor.clone(), set.clone(), false).expect("cpu");

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view_a = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let view_b = HostTensorView {
            data: &set.data,
            shape: &set.shape,
        };
        let handle_a = provider.upload(&view_a).expect("upload a");
        let handle_b = provider.upload(&view_b).expect("upload b");

        let eval = evaluate_sync(
            Value::GpuTensor(handle_a.clone()),
            Value::GpuTensor(handle_b.clone()),
            &[],
        )
        .expect("gpu evaluate");
        assert_eq!(eval.mask.data, cpu_eval.mask.data);
        assert_eq!(eval.loc.data, cpu_eval.loc.data);

        let _ = provider.free(&handle_a);
        let _ = provider.free(&handle_b);

        let matrix = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let bank = Tensor::new(vec![1.0, 7.0, 3.0, 2.0, 9.0, 4.0], vec![3, 2]).unwrap();
        let cpu_rows =
            ismember_numeric_from_tensors(matrix.clone(), bank.clone(), true).expect("cpu rows");
        let view_matrix = HostTensorView {
            data: &matrix.data,
            shape: &matrix.shape,
        };
        let view_bank = HostTensorView {
            data: &bank.data,
            shape: &bank.shape,
        };
        let handle_matrix = provider.upload(&view_matrix).expect("upload matrix");
        let handle_bank = provider.upload(&view_bank).expect("upload bank");
        let eval_rows = evaluate_sync(
            Value::GpuTensor(handle_matrix.clone()),
            Value::GpuTensor(handle_bank.clone()),
            &[Value::from("rows")],
        )
        .expect("gpu rows evaluate");
        assert_eq!(eval_rows.mask.data, cpu_rows.mask.data);
        assert_eq!(eval_rows.loc.data, cpu_rows.loc.data);
        let _ = provider.free(&handle_matrix);
        let _ = provider.free(&handle_bank);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_return_is_bool() {
        let a = Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap());
        let mask = evaluate_sync(a, b, &[]).unwrap().into_mask_value();
        assert_eq!(mask, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_rows_option() {
        let opts = parse_options(&[Value::from("rows")]).unwrap();
        assert!(opts.rows);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_rows_with_nan() {
        let a = Tensor::new(vec![f64::NAN, 1.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![f64::NAN, 2.0], vec![2, 1]).unwrap();
        let eval = ismember_numeric_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 0]);
        assert_eq!(eval.loc.data, vec![1.0, 0.0]);
    }
}
