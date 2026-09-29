use runmat_value::{LogicalArray, NumericDType, NumericStorage, Tensor};

use crate::BuiltinResult;

use super::super::super::{deep_learning_error, MAX_PAD_ELEMENTS};
use super::{
    admission::SequenceInput,
    options::{PadOptions, PaddingDirection},
};

pub(super) fn pad_sequence(
    seq: &SequenceInput,
    padding_dim: usize,
    target_len: usize,
    output_dtype: NumericDType,
    options: PadOptions,
    with_mask: bool,
) -> BuiltinResult<(Tensor, Option<LogicalArray>)> {
    let mut out_shape = seq.shape.clone();
    out_shape[padding_dim] = target_len;
    let total = checked_numel(&out_shape, "padsequences")?;
    check_pad_elements(total)?;
    let mut out = Tensor::from_numeric_storage(
        NumericStorage::zeros(output_dtype, total),
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

pub(super) fn non_padding_dims_match(left: &[usize], right: &[usize], padding_dim: usize) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .enumerate()
            .all(|(idx, (a, b))| idx == padding_dim || a == b)
}

pub(super) fn checked_numel(shape: &[usize], function: &'static str) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            deep_learning_error(function, format!("{function}: shape overflows usize"))
        })
    })
}

pub(super) fn check_pad_elements(total: usize) -> BuiltinResult<()> {
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
