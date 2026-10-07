use runmat_builtins::PadOptionKey;
use runmat_value::{NumericScalar, NumericStorage, Tensor, Value};

use crate::BuiltinResult;

use super::super::super::deep_learning_error;

// Metadata growth is charged against the same bounded work budget as the
// padded result. Shape copies and strides must not become an unbounded
// allocation when paddingDim names a distant singleton axis.
const MAX_PAD_METADATA_BYTES: usize = super::super::super::MAX_PAD_ELEMENTS;

pub(super) fn admit_padding_rank(
    padding_dim: usize,
    values: &[Value],
    prepared: &[Option<SequenceInput>],
) -> BuiltinResult<()> {
    let existing_rank = values
        .iter()
        .zip(prepared)
        .map(|(value, parsed)| match parsed {
            Some(sequence) => sequence.shape.len(),
            None => match value {
                Value::GpuTensor(handle) => handle.shape().len(),
                _ => unreachable!("resident sequence admitted above"),
            },
        })
        .max()
        .unwrap_or(0);
    let rank = padding_dim.saturating_add(1).max(existing_rank);
    let shape_vectors = values.len().saturating_mul(2).saturating_add(4);
    let bytes = rank
        .checked_mul(shape_vectors)
        .and_then(|count| count.checked_mul(std::mem::size_of::<usize>()))
        .ok_or_else(|| {
            deep_learning_error(
                "padsequences",
                "padsequences: padding rank exceeds metadata capacity",
            )
        })?;
    if bytes > MAX_PAD_METADATA_BYTES {
        return Err(deep_learning_error(
            "padsequences",
            "padsequences: padding rank exceeds metadata capacity",
        ));
    }
    Ok(())
}
use super::{
    execution::non_padding_dims_match,
    options::{parse_bool, parse_direction, AdmittedPadOption, AdmittedPadOptions, SequenceLength},
};

pub(super) fn admit_pad_option(
    name: PadOptionKey,
    value: Value,
) -> BuiltinResult<AdmittedPadOption> {
    if matches!(value, Value::GpuTensor(_)) {
        validate_resident_pad_option(name, &value)?;
        return Ok(AdmittedPadOption::Resident(value));
    }
    match name {
        PadOptionKey::PaddingValue => {
            super::options::numeric_pad_value(&value).map(AdmittedPadOption::PaddingValue)
        }
        PadOptionKey::PaddingDirection | PadOptionKey::DirectionAlias => {
            parse_direction(&value).map(AdmittedPadOption::Direction)
        }
        PadOptionKey::Length => SequenceLength::parse(&value).map(AdmittedPadOption::Length),
        PadOptionKey::UniformOutput => parse_bool(&value, "padsequences", "UniformOutput")
            .map(AdmittedPadOption::UniformOutput),
    }
}

pub(super) fn validate_resident_padding_dim(value: &Value) -> BuiltinResult<()> {
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

pub(super) fn validate_resident_sequence(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<()> {
    if runmat_accelerate_api::handle_storage(handle)
        != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Err(deep_learning_error(
            "padsequences",
            "padsequences: sequence cells must contain real numeric data",
        ));
    }
    Ok(())
}

pub(super) fn preflight_uniform_shapes(
    values: &[Value],
    prepared: &[Option<SequenceInput>],
    options: &AdmittedPadOptions,
    padding_dim: usize,
) -> BuiltinResult<()> {
    let uniform = match options.get(&PadOptionKey::UniformOutput) {
        None => true,
        Some(AdmittedPadOption::Resident(_)) => return Ok(()),
        Some(AdmittedPadOption::UniformOutput(value)) => *value,
        Some(_) => unreachable!("typed uniform output option"),
    };
    if !uniform {
        return Ok(());
    }
    let mut shapes = values
        .iter()
        .zip(prepared)
        .map(|(value, parsed)| match parsed {
            Some(sequence) => sequence.shape.clone(),
            None => match value {
                Value::GpuTensor(handle) => handle.shape().to_vec(),
                _ => unreachable!("resident sequence admitted above"),
            },
        })
        .collect::<Vec<_>>();
    let rank = shapes
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0)
        .max(padding_dim + 1);
    for shape in &mut shapes {
        shape.resize(rank, 1);
    }
    if let Some(first) = shapes.first() {
        for shape in &shapes[1..] {
            if !non_padding_dims_match(first, shape, padding_dim) {
                return Err(deep_learning_error(
                    "padsequences",
                    "padsequences: non-padding dimensions must match when UniformOutput is true",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_resident_pad_option(name: PadOptionKey, value: &Value) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        unreachable!()
    };
    let numeric_scalar = handle.shape().iter().all(|dimension| *dimension == 1)
        && runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::Real;
    let accepted = match name {
        PadOptionKey::PaddingValue | PadOptionKey::Length => {
            numeric_scalar && !runmat_accelerate_api::handle_is_logical(handle)
        }
        PadOptionKey::UniformOutput => numeric_scalar,
        // A numeric device buffer cannot be a text direction token.
        PadOptionKey::PaddingDirection | PadOptionKey::DirectionAlias => false,
    };
    if accepted {
        Ok(())
    } else {
        Err(deep_learning_error(
            "padsequences",
            format!(
                "padsequences: {} has an invalid resident type or shape",
                name.spelling()
            ),
        ))
    }
}

pub(super) struct SequenceInput {
    pub(super) data: Tensor,
    pub(super) shape: Vec<usize>,
}

impl SequenceInput {
    pub(super) fn from_value(value: &Value) -> BuiltinResult<Self> {
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
                (t.clone(), t.shape().to_vec())
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
