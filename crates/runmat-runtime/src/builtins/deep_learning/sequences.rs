use runmat_builtins::{CellArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{gather_if_needed_async, BuiltinResult};

use super::{
    any_type, deep_learning_error, gather_args, numeric_scalar, parse_name_values, positive_usize,
    scalar_text, tensor_value, MAX_COMBVEC_COLUMNS, MAX_PAD_ELEMENTS,
};

#[runtime_builtin(
    name = "combvec",
    category = "deep_learning",
    summary = "Generate all column combinations from input row vectors.",
    keywords = "combvec,neural network,deep learning,cartesian product",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::sequences"
)]
pub(super) async fn combvec_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args(args).await?;
    if args.is_empty() {
        return Err(deep_learning_error(
            "combvec",
            "combvec: expected at least one numeric vector",
        ));
    }
    let mut matrices = Vec::with_capacity(args.len());
    let mut total_cols = 1usize;
    let mut total_rows = 0usize;
    for (idx, arg) in args.iter().enumerate() {
        let matrix = CombvecInput::from_value(arg, idx + 1)?;
        total_cols = total_cols.checked_mul(matrix.cols).ok_or_else(|| {
            deep_learning_error("combvec", "combvec: output column count overflows usize")
        })?;
        if total_cols > MAX_COMBVEC_COLUMNS {
            return Err(deep_learning_error(
                "combvec",
                format!("combvec: output exceeds {MAX_COMBVEC_COLUMNS} columns"),
            ));
        }
        total_rows = total_rows.checked_add(matrix.rows).ok_or_else(|| {
            deep_learning_error("combvec", "combvec: output row count overflows usize")
        })?;
        matrices.push(matrix);
    }
    let total_elements = total_rows.checked_mul(total_cols).ok_or_else(|| {
        deep_learning_error("combvec", "combvec: output element count overflows usize")
    })?;
    if total_elements > MAX_PAD_ELEMENTS {
        return Err(deep_learning_error(
            "combvec",
            format!("combvec: output exceeds {MAX_PAD_ELEMENTS} elements"),
        ));
    }
    let mut data = vec![0.0; total_elements];
    for col in 0..total_cols {
        let mut divisor = 1usize;
        let mut row_offset = 0usize;
        for matrix in &matrices {
            let source_col = (col / divisor) % matrix.cols;
            let source_start = source_col * matrix.rows;
            let target_start = row_offset + col * total_rows;
            data[target_start..target_start + matrix.rows]
                .copy_from_slice(&matrix.data[source_start..source_start + matrix.rows]);
            row_offset += matrix.rows;
            divisor *= matrix.cols;
        }
    }
    tensor_value(data, vec![total_rows, total_cols], "combvec")
}

#[runtime_builtin(
    name = "padsequences",
    category = "deep_learning",
    summary = "Pad or truncate numeric sequence cell arrays.",
    keywords = "padsequences,deep learning,sequences,padding",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::sequences"
)]
pub(super) async fn padsequences_builtin(
    sequences: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let sequences = gather_if_needed_async(&sequences).await?;
    let args = gather_args(rest).await?;
    if args.is_empty() {
        return Err(deep_learning_error(
            "padsequences",
            "padsequences: expected padding dimension as the second argument",
        ));
    }
    let padding_dim = positive_usize(&args[0], "padsequences", "paddingDim")? - 1;
    let mut options = PadOptions::default();
    let mut parsed = parse_name_values(args[1..].to_vec(), "padsequences")?;
    if let Some(value) = parsed.remove("paddingvalue") {
        options.padding_value = numeric_scalar(&value, "padsequences", "PaddingValue")?;
    }
    if let Some(value) = parsed
        .remove("paddingdirection")
        .or_else(|| parsed.remove("direction"))
    {
        options.direction = PaddingDirection::parse(&scalar_text(&value, "padsequences")?)?;
    }
    if let Some(value) = parsed.remove("length") {
        options.length = SequenceLength::parse(&value)?;
    }
    if let Some(value) = parsed.remove("uniformoutput") {
        options.uniform_output = parse_bool(&value, "padsequences", "UniformOutput")?;
    }
    if !parsed.is_empty() {
        let first = parsed.keys().next().cloned().unwrap_or_default();
        return Err(deep_learning_error(
            "padsequences",
            format!("padsequences: unsupported option '{first}'"),
        ));
    }
    let Value::Cell(cell) = sequences else {
        return Err(deep_learning_error(
            "padsequences",
            "padsequences: expected a cell array of numeric sequences",
        ));
    };
    let mut seqs = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        seqs.push(SequenceInput::from_value(item, padding_dim)?);
    }
    let max_rank = seqs
        .iter()
        .map(|seq| seq.shape.len())
        .max()
        .unwrap_or(padding_dim + 1)
        .max(padding_dim + 1);
    for seq in &mut seqs {
        while seq.shape.len() < max_rank {
            seq.shape.push(1);
        }
    }
    let target_len = match options.length {
        SequenceLength::Longest => seqs
            .iter()
            .map(|seq| seq.shape[padding_dim])
            .max()
            .unwrap_or(0),
        SequenceLength::Shortest => seqs
            .iter()
            .map(|seq| seq.shape[padding_dim])
            .min()
            .unwrap_or(0),
        SequenceLength::Fixed(len) => len,
    };
    let mut padded = Vec::with_capacity(seqs.len());
    for seq in &seqs {
        padded.push(pad_sequence(seq, padding_dim, target_len, options)?);
    }
    if !options.uniform_output {
        let cells = padded
            .into_iter()
            .map(Value::Tensor)
            .collect::<Vec<Value>>();
        return CellArray::new_with_shape(cells, cell.shape)
            .map(Value::Cell)
            .map_err(|err| deep_learning_error("padsequences", err));
    }
    let Some(first) = padded.first() else {
        return tensor_value(Vec::new(), vec![0, 0], "padsequences");
    };
    for seq in &padded[1..] {
        if !non_padding_dims_match(&first.shape, &seq.shape, padding_dim) {
            return Err(deep_learning_error(
                "padsequences",
                "padsequences: non-padding dimensions must match when UniformOutput is true",
            ));
        }
    }
    let mut output_shape = first.shape.clone();
    output_shape.extend(cell.shape.iter().copied());
    let block_len = checked_numel(&first.shape, "padsequences")?;
    let total = block_len.checked_mul(padded.len()).ok_or_else(|| {
        deep_learning_error("padsequences", "padsequences: output size overflows usize")
    })?;
    check_pad_elements(total)?;
    let mut data = Vec::with_capacity(total);
    for seq in padded {
        data.extend(seq.data);
    }
    tensor_value(data, output_shape, "padsequences")
}

#[derive(Clone, Copy)]
struct PadOptions {
    padding_value: f64,
    direction: PaddingDirection,
    length: SequenceLength,
    uniform_output: bool,
}

impl Default for PadOptions {
    fn default() -> Self {
        Self {
            padding_value: 0.0,
            direction: PaddingDirection::Right,
            length: SequenceLength::Longest,
            uniform_output: true,
        }
    }
}

#[derive(Clone, Copy)]
enum PaddingDirection {
    Left,
    Right,
    Both,
}

impl PaddingDirection {
    fn parse(value: &str) -> BuiltinResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "left" => Ok(Self::Left),
            "right" => Ok(Self::Right),
            "both" => Ok(Self::Both),
            other => Err(deep_learning_error(
                "padsequences",
                format!(
                    "padsequences: PaddingDirection must be 'left', 'right', or 'both', got '{other}'"
                ),
            )),
        }
    }
}

#[derive(Clone, Copy)]
enum SequenceLength {
    Longest,
    Shortest,
    Fixed(usize),
}

impl SequenceLength {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
                match scalar_text(value, "padsequences")?.to_ascii_lowercase().as_str() {
                    "longest" => Ok(Self::Longest),
                    "shortest" => Ok(Self::Shortest),
                    other => Err(deep_learning_error(
                        "padsequences",
                        format!("padsequences: Length must be 'longest', 'shortest', or a positive integer, got '{other}'"),
                    )),
                }
            }
            _ => positive_usize(value, "padsequences", "Length").map(Self::Fixed),
        }
    }
}

struct CombvecInput {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl CombvecInput {
    fn from_value(value: &Value, index: usize) -> BuiltinResult<Self> {
        match value {
            Value::Num(n) if n.is_finite() => Ok(Self {
                data: vec![*n],
                rows: 1,
                cols: 1,
            }),
            Value::Int(i) => Ok(Self {
                data: vec![i.to_f64()],
                rows: 1,
                cols: 1,
            }),
            Value::Tensor(t) if t.shape.len() <= 2 => {
                if t.rows == 0 || t.cols == 0 {
                    return Err(deep_learning_error(
                        "combvec",
                        format!("combvec: input {index} must have at least one column"),
                    ));
                }
                for value in &t.data {
                    if !value.is_finite() {
                        return Err(deep_learning_error(
                            "combvec",
                            format!("combvec: input {index} must contain finite values"),
                        ));
                    }
                }
                Ok(Self {
                    data: t.data.clone(),
                    rows: t.rows,
                    cols: t.cols,
                })
            }
            other => Err(deep_learning_error(
                "combvec",
                format!("combvec: input {index} must be a finite numeric scalar or 2-D matrix, got {other:?}"),
            )),
        }
    }
}

struct SequenceInput {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl SequenceInput {
    fn from_value(value: &Value, padding_dim: usize) -> BuiltinResult<Self> {
        let (data, mut shape) = match value {
            Value::Num(n) if n.is_finite() => (vec![*n], vec![1, 1]),
            Value::Int(i) => (vec![i.to_f64()], vec![1, 1]),
            Value::Tensor(t) => {
                for item in &t.data {
                    if !item.is_finite() {
                        return Err(deep_learning_error(
                            "padsequences",
                            "padsequences: sequence values must be finite",
                        ));
                    }
                }
                (t.data.clone(), t.shape.clone())
            }
            other => {
                return Err(deep_learning_error(
                    "padsequences",
                    format!("padsequences: sequences must contain numeric arrays, got {other:?}"),
                ));
            }
        };
        while shape.len() <= padding_dim {
            shape.push(1);
        }
        Ok(Self { data, shape })
    }
}

fn pad_sequence(
    seq: &SequenceInput,
    padding_dim: usize,
    target_len: usize,
    options: PadOptions,
) -> BuiltinResult<Tensor> {
    let mut out_shape = seq.shape.clone();
    out_shape[padding_dim] = target_len;
    let total = checked_numel(&out_shape, "padsequences")?;
    check_pad_elements(total)?;
    let mut out = vec![options.padding_value; total];
    let out_strides = strides_for(&out_shape)?;
    let in_strides = strides_for(&seq.shape)?;
    for (out_linear, slot) in out.iter_mut().enumerate() {
        if let Some(in_linear) = map_padded_index(
            out_linear,
            seq,
            &out_shape,
            &out_strides,
            &in_strides,
            padding_dim,
            target_len,
            options.direction,
        )? {
            *slot = seq.data[in_linear];
        }
    }
    Tensor::new(out, out_shape).map_err(|err| deep_learning_error("padsequences", err))
}

fn map_padded_index(
    mut out_linear: usize,
    seq: &SequenceInput,
    out_shape: &[usize],
    out_strides: &[usize],
    in_strides: &[usize],
    padding_dim: usize,
    target_len: usize,
    direction: PaddingDirection,
) -> BuiltinResult<Option<usize>> {
    let source_len = seq.shape[padding_dim];
    let axis_coord = (out_linear / out_strides[padding_dim]) % out_shape[padding_dim];
    let source_axis = match direction {
        PaddingDirection::Right => {
            if axis_coord >= source_len.min(target_len) {
                return Ok(None);
            }
            axis_coord
        }
        PaddingDirection::Left => {
            if source_len <= target_len {
                let pad_before = target_len - source_len;
                if axis_coord < pad_before {
                    return Ok(None);
                }
                axis_coord - pad_before
            } else {
                axis_coord + (source_len - target_len)
            }
        }
        PaddingDirection::Both => {
            if source_len <= target_len {
                let pad_before = (target_len - source_len) / 2;
                if axis_coord < pad_before || axis_coord >= pad_before + source_len {
                    return Ok(None);
                }
                axis_coord - pad_before
            } else {
                axis_coord + ((source_len - target_len) / 2)
            }
        }
    };
    let mut source_linear = 0usize;
    for dim in 0..out_shape.len() {
        let coord = out_linear % out_shape[dim].max(1);
        out_linear /= out_shape[dim].max(1);
        let source_coord = if dim == padding_dim {
            source_axis
        } else if coord < seq.shape[dim] {
            coord
        } else {
            return Ok(None);
        };
        source_linear = source_linear
            .checked_add(source_coord.checked_mul(in_strides[dim]).ok_or_else(|| {
                deep_learning_error("padsequences", "padsequences: index calculation overflows")
            })?)
            .ok_or_else(|| {
                deep_learning_error("padsequences", "padsequences: index calculation overflows")
            })?;
    }
    Ok(Some(source_linear))
}

fn non_padding_dims_match(left: &[usize], right: &[usize], padding_dim: usize) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .enumerate()
            .all(|(idx, (a, b))| idx == padding_dim || a == b)
}

fn checked_numel(shape: &[usize], function: &'static str) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            deep_learning_error(function, format!("{function}: shape overflows usize"))
        })
    })
}

fn check_pad_elements(total: usize) -> BuiltinResult<()> {
    if total > MAX_PAD_ELEMENTS {
        return Err(deep_learning_error(
            "padsequences",
            format!("padsequences: output exceeds {MAX_PAD_ELEMENTS} elements"),
        ));
    }
    Ok(())
}

fn strides_for(shape: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for dim in shape {
        strides.push(stride);
        stride = stride.checked_mul((*dim).max(1)).ok_or_else(|| {
            deep_learning_error("padsequences", "padsequences: stride calculation overflows")
        })?;
    }
    Ok(strides)
}

fn parse_bool(value: &Value, function: &'static str, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(n) if *n == 0.0 => Ok(false),
        Value::Num(n) if *n == 1.0 => Ok(true),
        Value::Int(i) if i.to_f64() == 0.0 => Ok(false),
        Value::Int(i) if i.to_f64() == 1.0 => Ok(true),
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            match scalar_text(value, function)?.to_ascii_lowercase().as_str() {
                "true" | "on" | "yes" => Ok(true),
                "false" | "off" | "no" => Ok(false),
                other => Err(deep_learning_error(
                    function,
                    format!("{function}: {label} must be logical, got '{other}'"),
                )),
            }
        }
        other => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be logical, got {other:?}"),
        )),
    }
}
