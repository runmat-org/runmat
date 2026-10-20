use runmat_builtins::PadOptionKey;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, LogicalArray, NumericDType, NumericStorage, Tensor, Value};

use crate::{gather_if_needed_async, BuiltinResult};

use super::super::{deep_learning_error, parse_name_values, positive_usize};

mod admission;
mod execution;
mod options;

use admission::{
    admit_pad_option, admit_padding_rank, preflight_uniform_shapes, validate_resident_padding_dim,
    validate_resident_sequence, SequenceInput,
};
use execution::{check_pad_elements, checked_numel, non_padding_dims_match, pad_sequence};
pub(in crate::builtins::deep_learning) use options::SequenceLength;
use options::{AdmittedPadOption, AdmittedPadOptions, PadOptions};

#[runtime_builtin(
    name = "padsequences",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::sequences"
)]
pub(in crate::builtins::deep_learning) async fn padsequences_builtin(
    sequences: Value,
    mut rest: crate::call::BuiltinArguments,
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
    let mut parsed = AdmittedPadOptions::new();
    // Preserve the existing lowered-name sort and last-wins duplicate
    // behavior while decoding each public spelling exactly once.
    let padding_dimension = rest
        .pop_front()
        .expect("nonempty padsequences argument carrier");
    for (spelling, value) in parse_name_values(rest, "padsequences")? {
        let name = PadOptionKey::parse(&spelling).ok_or_else(|| {
            deep_learning_error(
                "padsequences",
                format!("padsequences: unsupported option '{spelling}'"),
            )
        })?;
        parsed.insert(name, admit_pad_option(name, value)?);
    }
    if parsed.contains_key(&PadOptionKey::PaddingDirection)
        && parsed.contains_key(&PadOptionKey::DirectionAlias)
    {
        return Err(deep_learning_error(
            "padsequences",
            "padsequences: unsupported option 'direction'",
        ));
    }
    // Admit every host element and resident descriptor before the first
    // download. A malformed later host sequence must not download an earlier
    // provider-owned element.
    let mut prepared = Vec::with_capacity(cell.data().len());
    for item in cell.data() {
        prepared.push(match item {
            Value::GpuTensor(handle) => {
                validate_resident_sequence(handle)?;
                None
            }
            _ => Some(SequenceInput::from_value(item)?),
        });
    }
    let host_dim = if matches!(padding_dimension, Value::GpuTensor(_)) {
        validate_resident_padding_dim(&padding_dimension)?;
        None
    } else {
        Some(positive_usize(&padding_dimension, "padsequences", "paddingDim")? - 1)
    };
    if let Some(dim) = host_dim {
        admit_padding_rank(dim, cell.data(), &prepared)?;
        preflight_uniform_shapes(cell.data(), &prepared, &parsed, dim)?;
    }
    let padding_dim = match host_dim {
        Some(dim) => dim,
        None => {
            positive_usize(
                &gather_if_needed_async(&padding_dimension).await?,
                "padsequences",
                "paddingDim",
            )? - 1
        }
    };
    if host_dim.is_none() {
        admit_padding_rank(padding_dim, cell.data(), &prepared)?;
        preflight_uniform_shapes(cell.data(), &prepared, &parsed, padding_dim)?;
    }
    let mut seqs = Vec::with_capacity(prepared.len());
    for (parsed, item) in prepared.into_iter().zip(cell.data()) {
        seqs.push(match parsed {
            Some(sequence) => sequence,
            None => SequenceInput::from_value(&gather_if_needed_async(item).await?)?,
        });
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
    for seq in &mut seqs {
        while seq.shape.len() <= padding_dim {
            seq.shape.push(1);
        }
    }
    for (&name, value) in &mut parsed {
        if let AdmittedPadOption::Resident(resident) = value {
            let gathered = gather_if_needed_async(resident).await?;
            if matches!(gathered, Value::GpuTensor(_)) {
                return Err(deep_learning_error(
                    "padsequences",
                    "padsequences: option gather left the value resident",
                ));
            }
            *value = admit_pad_option(name, gathered)?;
        }
    }
    let mut options = PadOptions::default();
    if let Some(AdmittedPadOption::PaddingValue(value)) = parsed.remove(&PadOptionKey::PaddingValue)
    {
        options.padding_value = value;
    }
    if let Some(AdmittedPadOption::Direction(value)) = parsed
        .remove(&PadOptionKey::PaddingDirection)
        .or_else(|| parsed.remove(&PadOptionKey::DirectionAlias))
    {
        options.direction = value;
    }
    if let Some(AdmittedPadOption::Length(value)) = parsed.remove(&PadOptionKey::Length) {
        options.length = value;
    }
    if let Some(AdmittedPadOption::UniformOutput(value)) =
        parsed.remove(&PadOptionKey::UniformOutput)
    {
        options.uniform_output = value;
    }
    if !parsed.is_empty() {
        let first = parsed
            .keys()
            .next()
            .map(|key| key.spelling())
            .unwrap_or_default();
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
            if options.uniform_output {
                dtype
            } else {
                seq.data.numeric_dtype()
            },
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
        let output = CellArray::new_with_shape(cells, cell.shape().to_vec())
            .map(Value::Cell)
            .map_err(|err| deep_learning_error("padsequences", err))?;
        if requested_outputs == 2 {
            let mask = CellArray::new_with_shape(masks, cell.shape().to_vec())
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("padsequences", err))?;
            return Ok(Value::output_list(vec![output, mask]));
        }
        return Ok(if requested_outputs == 0 {
            Value::output_list(Vec::new())
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
            0 => Value::output_list(Vec::new()),
            2 => Value::output_list(vec![
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
        if !non_padding_dims_match(first.0.shape(), seq.0.shape(), padding_dim) {
            return Err(deep_learning_error(
                "padsequences",
                "padsequences: non-padding dimensions must match when UniformOutput is true",
            ));
        }
    }
    let mut output_shape = first.0.shape().to_vec();
    output_shape.push(padded.len());
    let block_len = checked_numel(first.0.shape(), "padsequences")?;
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
        0 => Value::output_list(Vec::new()),
        2 => Value::output_list(vec![
            output,
            Value::LogicalArray(
                LogicalArray::new(mask_data.unwrap_or_default(), output_shape)
                    .map_err(|err| deep_learning_error("padsequences", err))?,
            ),
        ]),
        _ => output,
    })
}
