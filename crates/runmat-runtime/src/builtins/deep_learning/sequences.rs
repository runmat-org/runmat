use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, LogicalArray, NumericDType, NumericScalar, NumericStorage, Tensor, Value,
};

use crate::{builtins::common::tensor, gather_if_needed_async, BuiltinResult};

use super::{
    deep_learning_error, logical_scalar, numeric_scalar, parse_name_values, positive_usize,
    scalar_text, tensor_value, MAX_COMBVEC_COLUMNS, MAX_PAD_ELEMENTS,
};

#[runtime_builtin(
    name = "combvec",
    category = "deep_learning",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::sequences"
)]
pub(super) async fn combvec_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(deep_learning_error(
            "combvec",
            "combvec: expected at least one numeric vector",
        ));
    }
    if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::COMBVEC_INTEGER_EXTENSION,
            "combvec",
        )?;
    }
    // Parse host arguments and check resident shapes before the first provider
    // operation. Host-invalid later inputs must not download an earlier input.
    let mut parsed = Vec::with_capacity(args.len());
    let mut known_rows = 0usize;
    let mut known_columns = 1usize;
    for (index, arg) in args.iter().enumerate() {
        let input = match arg {
            Value::GpuTensor(handle) => {
                if handle.shape().len() > 2
                    || handle.shape().first().copied() == Some(0)
                    || handle.shape().get(1).copied() == Some(0)
                    || runmat_accelerate_api::handle_is_logical(handle)
                    || runmat_accelerate_api::handle_storage(handle)
                        != runmat_accelerate_api::GpuTensorStorage::Real
                {
                    return Err(deep_learning_error(
                        "combvec",
                        format!(
                            "combvec: input {} must be a nonempty real numeric 2-D matrix",
                            index + 1
                        ),
                    ));
                }
                None
            }
            _ => Some(CombvecInput::from_value(arg, index + 1)?),
        };
        let (rows, columns) = match &input {
            Some(matrix) => (matrix.rows, matrix.cols),
            None => {
                let Value::GpuTensor(handle) = arg else {
                    unreachable!()
                };
                (
                    handle.shape().first().copied().unwrap_or(1),
                    handle.shape().get(1).copied().unwrap_or(1),
                )
            }
        };
        known_rows = known_rows.checked_add(rows).ok_or_else(|| {
            deep_learning_error("combvec", "combvec: output row count overflows usize")
        })?;
        known_columns = known_columns.checked_mul(columns).ok_or_else(|| {
            deep_learning_error("combvec", "combvec: output column count overflows usize")
        })?;
        if known_columns > MAX_COMBVEC_COLUMNS
            || known_rows
                .checked_mul(known_columns)
                .is_none_or(|total| total > MAX_PAD_ELEMENTS)
        {
            return Err(deep_learning_error(
                "combvec",
                "combvec: output exceeds supported capacity",
            ));
        }
        parsed.push(input);
    }
    let mut matrices = Vec::with_capacity(args.len());
    let mut total_cols = 1usize;
    let mut total_rows = 0usize;
    for (idx, (arg, host)) in args.iter().zip(parsed).enumerate() {
        let matrix = match host {
            Some(host) => host,
            None => CombvecInput::from_value(&gather_if_needed_async(arg).await?, idx + 1)?,
        };
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
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::sequences"
)]
pub(super) async fn padsequences_builtin(
    sequences: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    if requested_outputs > 2 {
        return Err(deep_learning_error(
            "padsequences",
            "padsequences: at most two outputs are supported",
        ));
    }
    if rest.is_empty() {
        return Err(deep_learning_error(
            "padsequences",
            "padsequences: expected padding dimension as the second argument",
        ));
    }
    let Value::Cell(cell) = sequences else {
        return Err(deep_learning_error(
            "padsequences",
            "padsequences: expected a cell array of numeric sequences",
        ));
    };
    let mut parsed = parse_name_values(rest[1..].to_vec(), "padsequences")?;
    for (name, value) in &parsed {
        if !matches!(
            name.as_str(),
            "paddingvalue" | "paddingdirection" | "direction" | "length" | "uniformoutput"
        ) {
            return Err(deep_learning_error(
                "padsequences",
                format!("padsequences: unsupported option '{name}'"),
            ));
        }
        if matches!(value, Value::GpuTensor(_)) {
            validate_resident_pad_option(name, value)?;
            continue;
        }
        validate_pad_option(name, value)?;
    }
    let mut seqs = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        seqs.push(SequenceInput::from_value(item)?);
    }
    let dtype = seqs
        .first()
        .map_or(NumericDType::F64, |seq| seq.data.numeric_dtype());
    let dtype = if seqs.iter().all(|seq| seq.data.numeric_dtype() == dtype) {
        dtype
    } else {
        // Retain the existing mixed-class RunMat form as a double result.
        NumericDType::F64
    };
    let host_dim = if matches!(rest[0], Value::GpuTensor(_)) {
        validate_resident_padding_dim(&rest[0])?;
        None
    } else {
        Some(positive_usize(&rest[0], "padsequences", "paddingDim")? - 1)
    };
    let padding_dim = match host_dim {
        Some(dim) => dim,
        None => {
            positive_usize(
                &gather_if_needed_async(&rest[0]).await?,
                "padsequences",
                "paddingDim",
            )? - 1
        }
    };
    for seq in &mut seqs {
        while seq.shape.len() <= padding_dim {
            seq.shape.push(1);
        }
    }
    for value in parsed.values_mut() {
        if matches!(value, Value::GpuTensor(_)) {
            *value = gather_if_needed_async(value).await?;
        }
    }
    let mut options = PadOptions::default();
    if let Some(value) = parsed.remove("paddingvalue") {
        options.padding_value = numeric_pad_value(&value)?;
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
        padded.push(pad_sequence(
            seq,
            padding_dim,
            target_len,
            options,
            requested_outputs == 2,
        )?);
    }
    if !options.uniform_output {
        let mut cells = Vec::with_capacity(padded.len());
        let mut masks = Vec::with_capacity(padded.len());
        for (tensor, mask) in padded {
            cells.push(Value::Tensor(tensor));
            if let Some(mask) = mask {
                masks.push(Value::LogicalArray(mask));
            }
        }
        let output = CellArray::new_with_shape(cells, cell.shape.clone())
            .map(Value::Cell)
            .map_err(|err| deep_learning_error("padsequences", err))?;
        if requested_outputs == 2 {
            let mask = CellArray::new_with_shape(masks, cell.shape)
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("padsequences", err))?;
            return Ok(Value::OutputList(vec![output, mask]));
        }
        return Ok(if requested_outputs == 0 {
            Value::OutputList(Vec::new())
        } else {
            output
        });
    }
    let Some(first) = padded.first() else {
        let output = Value::Tensor(
            Tensor::from_numeric_storage(NumericStorage::zeros(dtype, 0), vec![0, 0])
                .map_err(|err| deep_learning_error("padsequences", err))?,
        );
        return Ok(match requested_outputs {
            0 => Value::OutputList(Vec::new()),
            2 => Value::OutputList(vec![
                output,
                Value::LogicalArray(
                    LogicalArray::new(Vec::new(), vec![0, 0])
                        .map_err(|err| deep_learning_error("padsequences", err))?,
                ),
            ]),
            _ => output,
        });
    };
    for seq in &padded[1..] {
        if !non_padding_dims_match(&first.0.shape, &seq.0.shape, padding_dim) {
            return Err(deep_learning_error(
                "padsequences",
                "padsequences: non-padding dimensions must match when UniformOutput is true",
            ));
        }
    }
    let mut output_shape = first.0.shape.clone();
    output_shape.push(padded.len());
    let block_len = checked_numel(&first.0.shape, "padsequences")?;
    let total = block_len.checked_mul(padded.len()).ok_or_else(|| {
        deep_learning_error("padsequences", "padsequences: output size overflows usize")
    })?;
    check_pad_elements(total)?;
    let mut output =
        Tensor::from_numeric_storage(NumericStorage::zeros(dtype, total), output_shape.clone())
            .map_err(|err| deep_learning_error("padsequences", err))?;
    let mut mask_data = (requested_outputs == 2).then(|| Vec::with_capacity(total));
    for (block, (seq, mask)) in padded.into_iter().enumerate() {
        for index in 0..block_len {
            let value = seq.numeric_value_at(index).ok_or_else(|| {
                deep_learning_error("padsequences", "padsequences: padded index out of bounds")
            })?;
            output
                .set_numeric_assignment_at(block * block_len + index, value)
                .map_err(|err| deep_learning_error("padsequences", err))?;
        }
        if let (Some(data), Some(mask)) = (&mut mask_data, mask) {
            data.extend_from_slice(mask.data.as_slice());
        }
    }
    let output = Value::Tensor(output);
    Ok(match requested_outputs {
        0 => Value::OutputList(Vec::new()),
        2 => Value::OutputList(vec![
            output,
            Value::LogicalArray(
                LogicalArray::new(mask_data.unwrap_or_default(), output_shape)
                    .map_err(|err| deep_learning_error("padsequences", err))?,
            ),
        ]),
        _ => output,
    })
}

fn numeric_pad_value(value: &Value) -> BuiltinResult<NumericScalar> {
    numeric_scalar(value, "padsequences", "PaddingValue")?;
    match value {
        Value::Num(value) => Ok(NumericScalar::F64(*value)),
        Value::Int(value) => Ok(NumericScalar::from(value.clone())),
        Value::Tensor(value) => value.numeric_value_at(0).ok_or_else(|| {
            deep_learning_error("padsequences", "padsequences: PaddingValue is empty")
        }),
        _ => unreachable!("numeric scalar admitted by validator"),
    }
}

fn validate_pad_option(name: &str, value: &Value) -> BuiltinResult<()> {
    match name {
        "paddingvalue" => {
            numeric_scalar(value, "padsequences", "PaddingValue")?;
        }
        "paddingdirection" | "direction" => {
            PaddingDirection::parse(&scalar_text(value, "padsequences")?)?;
        }
        "length" => {
            SequenceLength::parse(value)?;
        }
        "uniformoutput" => {
            parse_bool(value, "padsequences", "UniformOutput")?;
        }
        _ => unreachable!("option name admitted by caller"),
    }
    Ok(())
}

fn validate_resident_padding_dim(value: &Value) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        unreachable!()
    };
    if handle.shape().iter().all(|dimension| *dimension == 1)
        && runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && !runmat_accelerate_api::handle_is_logical(handle)
    {
        Ok(())
    } else {
        Err(deep_learning_error(
            "padsequences",
            "padsequences: paddingDim must be a positive numeric scalar",
        ))
    }
}

fn validate_resident_pad_option(name: &str, value: &Value) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        unreachable!()
    };
    let numeric_scalar = handle.shape().iter().all(|dimension| *dimension == 1)
        && runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::Real;
    let accepted = match name {
        "paddingvalue" | "length" => {
            numeric_scalar && !runmat_accelerate_api::handle_is_logical(handle)
        }
        "uniformoutput" => numeric_scalar,
        // A numeric device buffer cannot be a text direction token.
        "paddingdirection" | "direction" => false,
        _ => unreachable!("option name admitted by caller"),
    };
    if accepted {
        Ok(())
    } else {
        Err(deep_learning_error(
            "padsequences",
            format!("padsequences: {name} has an invalid resident type or shape"),
        ))
    }
}

#[derive(Clone, Copy)]
struct PadOptions {
    padding_value: NumericScalar,
    direction: PaddingDirection,
    length: SequenceLength,
    uniform_output: bool,
}

impl Default for PadOptions {
    fn default() -> Self {
        Self {
            padding_value: NumericScalar::F64(0.0),
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
pub(super) enum SequenceLength {
    Longest,
    Shortest,
    Fixed(usize),
}

impl SequenceLength {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
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
                let data = tensor::tensor_values_f64(t);
                for value in &data {
                    if !value.is_finite() {
                        return Err(deep_learning_error(
                            "combvec",
                            format!("combvec: input {index} must contain finite values"),
                        ));
                    }
                }
                Ok(Self {
                    data,
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
    data: Tensor,
    shape: Vec<usize>,
}

impl SequenceInput {
    fn from_value(value: &Value) -> BuiltinResult<Self> {
        let (data, shape) = match value {
            Value::Num(n) if n.is_finite() => (
                Tensor::new(vec![*n], vec![1, 1])
                    .map_err(|err| deep_learning_error("padsequences", err))?,
                vec![1, 1],
            ),
            Value::Int(i) => {
                let mut storage = NumericStorage::zeros(i.numeric_dtype(), 1);
                storage
                    .set_value(0, NumericScalar::from(i.clone()))
                    .map_err(|err| deep_learning_error("padsequences", err))?;
                (
                    Tensor::from_numeric_storage(storage, vec![1, 1])
                        .map_err(|err| deep_learning_error("padsequences", err))?,
                    vec![1, 1],
                )
            }
            Value::Tensor(t) => {
                for index in 0..t.host_buffer().len() {
                    if !t
                        .numeric_value_at(index)
                        .is_some_and(NumericScalar::is_finite)
                    {
                        return Err(deep_learning_error(
                            "padsequences",
                            "padsequences: sequence values must be finite",
                        ));
                    }
                }
                (t.clone(), t.shape.clone())
            }
            other => {
                return Err(deep_learning_error(
                    "padsequences",
                    format!("padsequences: sequences must contain numeric arrays, got {other:?}"),
                ));
            }
        };
        Ok(Self { data, shape })
    }
}

fn pad_sequence(
    seq: &SequenceInput,
    padding_dim: usize,
    target_len: usize,
    options: PadOptions,
    with_mask: bool,
) -> BuiltinResult<(Tensor, Option<LogicalArray>)> {
    let mut out_shape = seq.shape.clone();
    out_shape[padding_dim] = target_len;
    let total = checked_numel(&out_shape, "padsequences")?;
    check_pad_elements(total)?;
    let mut out = Tensor::from_numeric_storage(
        NumericStorage::zeros(seq.data.numeric_dtype(), total),
        out_shape.clone(),
    )
    .map_err(|err| deep_learning_error("padsequences", err))?;
    let mut mask = with_mask.then(|| vec![0_u8; total]);
    let out_strides = strides_for(&out_shape)?;
    let in_strides = strides_for(&seq.shape)?;
    for out_linear in 0..total {
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
            let value = seq.data.numeric_value_at(in_linear).ok_or_else(|| {
                deep_learning_error("padsequences", "padsequences: source index out of bounds")
            })?;
            out.set_numeric_assignment_at(out_linear, value)
                .map_err(|err| deep_learning_error("padsequences", err))?;
            if let Some(mask) = &mut mask {
                mask[out_linear] = 1;
            }
        } else if !options.padding_value.is_zero() {
            out.set_numeric_assignment_at(out_linear, options.padding_value)
                .map_err(|err| deep_learning_error("padsequences", err))?;
        }
    }
    let mask = mask
        .map(|bits| LogicalArray::new(bits, out_shape))
        .transpose()
        .map_err(|err| deep_learning_error("padsequences", err))?;
    Ok((out, mask))
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
        _ => logical_scalar(value, function, label).map_err(|_| {
            deep_learning_error(function, format!("{function}: {label} must be logical"))
        }),
    }
}
