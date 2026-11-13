use super::{char_value, scalar_text};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_async::GpuGatherRetry;
use runmat_builtins::{
    BuiltinErrorDescriptor, MEMMAPFILE_ERROR_INVALID_CONTROL, MEMMAPFILE_ERROR_INVALID_FORMAT,
    MEMMAPFILE_ERROR_INVALID_OPTION,
};
use runmat_value::{
    CellArray, IntegerStorage, MemmapElementType, MemmapFieldState, MemmapFormatPresentationState,
    MemmapReceiverState, NumericDType, NumericScalar, Tensor, Value,
};

mod authority;
mod encode;
pub(crate) mod live;
pub(super) use authority::issue as issue_authority;
pub(super) use live::ensure_class_registered;

pub(super) fn typed_error(
    descriptor: &BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("memmapfile")
        .with_identifier(descriptor.identifier.expect("typed memmapfile error"))
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

pub(super) fn format_error(message: impl Into<String>) -> RuntimeError {
    typed_error(&MEMMAPFILE_ERROR_INVALID_FORMAT, message)
}

fn control_error(message: impl Into<String>) -> RuntimeError {
    typed_error(&MEMMAPFILE_ERROR_INVALID_CONTROL, message)
}

fn option_error(message: impl Into<String>) -> RuntimeError {
    typed_error(&MEMMAPFILE_ERROR_INVALID_OPTION, message)
}

fn text_with_category(
    value: &Value,
    argument: &str,
    descriptor: &BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    scalar_text(value, "memmapfile", argument).map_err(|source| {
        build_runtime_error(source.message().to_owned())
            .with_builtin("memmapfile")
            .with_identifier(descriptor.identifier.expect("typed memmapfile error"))
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .with_source(source)
            .build()
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MemmapDataType {
    Numeric(NumericDType),
    Character,
}

impl MemmapDataType {
    pub(super) fn parse(name: &str) -> BuiltinResult<Self> {
        let dtype = match name.to_ascii_lowercase().as_str() {
            "double" => Self::Numeric(NumericDType::F64),
            "single" => Self::Numeric(NumericDType::F32),
            "int8" => Self::Numeric(NumericDType::I8),
            "int16" => Self::Numeric(NumericDType::I16),
            "int32" => Self::Numeric(NumericDType::I32),
            "int64" => Self::Numeric(NumericDType::I64),
            "uint8" => Self::Numeric(NumericDType::U8),
            "uint16" => Self::Numeric(NumericDType::U16),
            "uint32" => Self::Numeric(NumericDType::U32),
            "uint64" => Self::Numeric(NumericDType::U64),
            "char" => Self::Character,
            other => {
                return Err(format_error(format!(
                    "memmapfile: unsupported Format type '{other}'"
                )));
            }
        };
        Ok(dtype)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Numeric(dtype) => dtype.class_name(),
            Self::Character => "char",
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::Numeric(dtype) => dtype.byte_size(),
            Self::Character => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MemmapOption {
    Writable,
    Offset,
    Format,
    Repeat,
}

impl MemmapOption {
    pub(super) fn from_value(value: &Value) -> BuiltinResult<Self> {
        let name = text_with_category(value, "option", &MEMMAPFILE_ERROR_INVALID_OPTION)?;
        match name.to_ascii_lowercase().as_str() {
            "writable" => Ok(Self::Writable),
            "offset" => Ok(Self::Offset),
            "format" => Ok(Self::Format),
            "repeat" => Ok(Self::Repeat),
            _ => Err(option_error(format!(
                "memmapfile: unsupported option '{name}'"
            ))),
        }
    }

    pub(super) fn validate_host_value(self, value: &Value) -> BuiltinResult<()> {
        match self {
            Self::Writable => Ok(()),
            Self::Offset => numeric_usize(value, "Offset").map(|_| ()),
            Self::Repeat => repeat_count(value).map(|_| ()),
            Self::Format => {
                let format = MemmapFormat::from_value(value)?;
                format.validate_layout(None).map(|_| ())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct MemmapFormat {
    pub(super) dtype: MemmapDataType,
    pub(super) shape: Vec<usize>,
    pub(super) field: Option<String>,
    pub(super) additional_fields: Vec<MemmapField>,
    pub(super) presentation: MemmapFormatPresentation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MemmapFormatPresentation {
    TypeName,
    Cell { columns: usize },
}

#[derive(Clone, Debug)]
pub(super) struct MemmapField {
    pub(super) dtype: MemmapDataType,
    pub(super) shape: Vec<usize>,
    name: String,
}

impl Default for MemmapFormat {
    fn default() -> Self {
        Self {
            dtype: MemmapDataType::Numeric(NumericDType::U8),
            shape: vec![1, 1],
            field: None,
            additional_fields: Vec::new(),
            presentation: MemmapFormatPresentation::TypeName,
        }
    }
}

impl MemmapFormat {
    pub(super) fn receiver_fields(&self) -> Vec<MemmapFieldState> {
        std::iter::once(MemmapFieldState {
            name: self.field.clone(),
            element: self.dtype.into(),
            shape: self.shape.clone(),
        })
        .chain(self.additional_fields.iter().map(|field| MemmapFieldState {
            name: Some(field.name.clone()),
            element: field.dtype.into(),
            shape: field.shape.clone(),
        }))
        .collect()
    }

    pub(super) fn receiver_presentation(&self) -> MemmapFormatPresentationState {
        match self.presentation {
            MemmapFormatPresentation::TypeName => MemmapFormatPresentationState::TypeName,
            MemmapFormatPresentation::Cell { columns } => {
                MemmapFormatPresentationState::Cell { columns }
            }
        }
    }

    pub(super) fn from_receiver(receiver: &MemmapReceiverState) -> BuiltinResult<Self> {
        let fields = &receiver.fields;
        let first = fields
            .first()
            .ok_or_else(|| format_error("memmapfile: sealed Format is empty"))?;
        if first.name.is_none() && fields.len() != 1 {
            return Err(format_error("memmapfile: unnamed Format has extra fields"));
        }
        let to_dtype = |element| match element {
            MemmapElementType::Numeric(dtype) => MemmapDataType::Numeric(dtype),
            MemmapElementType::Character => MemmapDataType::Character,
        };
        let format = Self {
            dtype: to_dtype(first.element),
            shape: first.shape.clone(),
            field: first.name.clone(),
            additional_fields: fields[1..]
                .iter()
                .map(|field| {
                    field.name.as_ref().map(|name| MemmapField {
                        dtype: to_dtype(field.element),
                        shape: field.shape.clone(),
                        name: name.clone(),
                    })
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| format_error("memmapfile: sealed Format field is unnamed"))?,
            presentation: match receiver.format_presentation {
                MemmapFormatPresentationState::TypeName => MemmapFormatPresentation::TypeName,
                MemmapFormatPresentationState::Cell { columns } => {
                    MemmapFormatPresentation::Cell { columns }
                }
            },
        };
        format.validate_layout(None)?;
        Ok(format)
    }

    pub(super) fn validate_layout(&self, repeat: Option<usize>) -> BuiltinResult<usize> {
        let mut record_bytes = memmap_field_bytes(self.dtype, &self.shape)?;
        if let Some(name) = &self.field {
            if name.is_empty() {
                return Err(format_error("memmapfile: Format field is empty"));
            }
        }
        let mut names = std::collections::HashSet::new();
        if let Some(name) = &self.field {
            names.insert(name.as_str());
        }
        for segment in &self.additional_fields {
            if segment.name.is_empty() || !names.insert(segment.name.as_str()) {
                return Err(format_error(
                    "memmapfile: Format field names must be nonempty and unique",
                ));
            }
            record_bytes = record_bytes
                .checked_add(memmap_field_bytes(segment.dtype, &segment.shape)?)
                .ok_or_else(|| format_error("memmapfile: Format record is too large"))?;
        }
        if let Some(count) = repeat {
            count
                .checked_mul(record_bytes)
                .ok_or_else(|| control_error("memmapfile: Repeat exceeds addressable size"))?;
        }
        Ok(record_bytes)
    }

    pub(super) fn from_value(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => Ok(Self {
                dtype: MemmapDataType::parse(&text_with_category(
                    value,
                    "Format",
                    &MEMMAPFILE_ERROR_INVALID_FORMAT,
                )?)?,
                ..Self::default()
            }),
            Value::Cell(cell) => {
                if cell.data().is_empty() {
                    return Err(format_error("memmapfile: Format cell is empty"));
                }
                if cell.shape.as_slice() != [cell.rows, cell.cols]
                    || cell.rows.checked_mul(cell.cols) != Some(cell.data().len())
                    || cell.cols > 3
                    || (cell.rows > 1 && cell.cols != 3)
                {
                    return Err(format_error(
                        "memmapfile: Format must be a three-column format table",
                    ));
                }
                let dtype = MemmapDataType::parse(&text_with_category(
                    &cell.data()[0],
                    "Format type",
                    &MEMMAPFILE_ERROR_INVALID_FORMAT,
                )?)?;
                let shape = if cell.cols >= 2 {
                    shape_from_value(&cell.data()[1])?
                } else {
                    vec![1, 1]
                };
                let field = if cell.cols == 3 {
                    Some(text_with_category(
                        &cell.data()[2],
                        "Format field",
                        &MEMMAPFILE_ERROR_INVALID_FORMAT,
                    )?)
                } else {
                    None
                };
                let mut additional_fields = Vec::with_capacity(cell.rows.saturating_sub(1));
                for row in 1..cell.rows {
                    let start = row * 3;
                    additional_fields.push(MemmapField {
                        dtype: MemmapDataType::parse(&text_with_category(
                            &cell.data()[start],
                            "Format type",
                            &MEMMAPFILE_ERROR_INVALID_FORMAT,
                        )?)?,
                        shape: shape_from_value(&cell.data()[start + 1])?,
                        name: text_with_category(
                            &cell.data()[start + 2],
                            "Format field",
                            &MEMMAPFILE_ERROR_INVALID_FORMAT,
                        )?,
                    });
                }
                Ok(Self {
                    dtype,
                    shape,
                    field,
                    additional_fields,
                    presentation: MemmapFormatPresentation::Cell { columns: cell.cols },
                })
            }
            _ => Err(format_error(
                "memmapfile: Format must be a type name or format cell array",
            )),
        }
    }

    pub(super) fn to_value(&self) -> BuiltinResult<Value> {
        let MemmapFormatPresentation::Cell { columns } = self.presentation else {
            return Ok(char_value(self.dtype.name()));
        };
        let rows = self.additional_fields.len() + 1;
        let mut cells = Vec::with_capacity(rows * columns);
        cells.push(char_value(self.dtype.name()));
        if columns >= 2 {
            cells.push(memmap_shape_value(&self.shape)?);
        }
        if columns == 3 {
            cells.push(char_value(self.field.as_deref().ok_or_else(|| {
                format_error("memmapfile: Format field is missing")
            })?));
        }
        for segment in &self.additional_fields {
            cells.extend([
                char_value(segment.dtype.name()),
                memmap_shape_value(&segment.shape)?,
                char_value(&segment.name),
            ]);
        }
        Ok(Value::Cell(
            CellArray::new(cells, rows, columns).map_err(format_error)?,
        ))
    }
}

impl From<MemmapDataType> for MemmapElementType {
    fn from(value: MemmapDataType) -> Self {
        match value {
            MemmapDataType::Numeric(dtype) => Self::Numeric(dtype),
            MemmapDataType::Character => Self::Character,
        }
    }
}

fn memmap_field_bytes(dtype: MemmapDataType, shape: &[usize]) -> BuiltinResult<usize> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(format_error(
            "memmapfile: Format shape must contain positive integers",
        ));
    }
    let elements = shape
        .iter()
        .try_fold(1usize, |product, dim| product.checked_mul(*dim))
        .ok_or_else(|| format_error("memmapfile: Format shape is too large"))?;
    elements
        .checked_mul(dtype.byte_size())
        .ok_or_else(|| format_error("memmapfile: Format field is too large"))
}

fn memmap_shape_value(shape: &[usize]) -> BuiltinResult<Value> {
    if shape.iter().any(|value| (*value as u128) > (1u128 << 53)) {
        let exact = shape
            .iter()
            .map(|value| u64::try_from(*value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format_error(error.to_string()))?;
        return Ok(Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(exact), vec![1, shape.len()])
                .map_err(format_error)?,
        ));
    }
    let values = shape.iter().map(|value| *value as f64).collect();
    Ok(Value::Tensor(
        Tensor::new(values, vec![1, shape.len()]).map_err(format_error)?,
    ))
}

pub(super) fn memmap_count_value(count: usize) -> BuiltinResult<Value> {
    if (count as u128) > (1u128 << 53) {
        return Ok(Value::Int(runmat_value::IntValue::U64(
            u64::try_from(count).map_err(|error| control_error(error.to_string()))?,
        )));
    }
    Ok(Value::Num(count as f64))
}

pub(super) fn shape_from_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(v) => Ok(vec![positive_platform_usize(*v, "Format shape")?, 1]),
        Value::Int(v) => Ok(vec![positive_integer_shape_dim(v)?, 1]),
        Value::Tensor(tensor) => {
            let len = tensor
                .shape()
                .iter()
                .try_fold(1usize, |product, dim| product.checked_mul(*dim))
                .ok_or_else(|| format_error("memmapfile: Format shape is too large"))?;
            let mut shape = Vec::with_capacity(len);
            for index in 0..len {
                let value = tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| format_error("memmapfile: invalid Format shape storage"))?;
                shape.push(shape_dim_from_numeric_scalar(value)?);
            }
            Ok(shape)
        }
        _ => Err(format_error(
            "memmapfile: Format shape must be a positive numeric vector",
        )),
    }
}

fn shape_dim_from_numeric_scalar(value: NumericScalar) -> BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => positive_platform_usize(value, "Format shape"),
        NumericScalar::F32(value) => positive_platform_usize(f64::from(value), "Format shape"),
        integer => positive_integer_shape_dim(
            &integer
                .into_int_value()
                .expect("non-floating numeric scalar must be integer"),
        ),
    }
}

fn positive_integer_shape_dim(value: &runmat_value::IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|size| *size > 0)
        .ok_or_else(|| format_error("memmapfile: Format shape must contain positive integers"))
}

fn positive_platform_usize(value: f64, arg: &str) -> BuiltinResult<usize> {
    let Some(size) = nonnegative_platform_usize(value) else {
        return Err(format_error(format!(
            "memmapfile: {arg} must contain positive integers"
        )));
    };
    if size == 0 {
        return Err(format_error(format!(
            "memmapfile: {arg} must contain positive integers"
        )));
    }
    Ok(size)
}

fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

mod decode;
#[cfg(test)]
#[path = "memmapfile/tests.rs"]
mod tests;

pub(super) fn numeric_usize(value: &Value, arg: &str) -> BuiltinResult<usize> {
    match value {
        Value::Num(v) => nonnegative_platform_usize(*v).ok_or_else(|| {
            control_error(format!("memmapfile: {arg} must be a nonnegative integer"))
        }),
        Value::Int(v) => v.try_to_usize().ok_or_else(|| {
            control_error(format!("memmapfile: {arg} must be a nonnegative integer"))
        }),
        _ => Err(control_error(format!(
            "memmapfile: {arg} must be a nonnegative integer"
        ))),
    }
}

/// None denotes complete records through the opened file extent.
pub(super) fn repeat_count(value: &Value) -> BuiltinResult<Option<usize>> {
    if matches!(value, Value::Num(value) if *value == f64::INFINITY) {
        return Ok(None);
    }
    numeric_usize(value, "Repeat").map(Some)
}
