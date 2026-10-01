use super::{
    invalid, typed_integer, Distance, EmptyAction, KmeansDisplay, KmeansOption, KmeansStart,
    KmeansStatsetField, KmeansToggle, StartSpec, MAX_WORK_CELLS,
};
use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::BuiltinResult;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

pub(super) struct OptionHeader {
    pub(super) kind: KmeansOption,
    pub(super) value: Value,
    pub(super) statset_fields: Option<Vec<StatsetHeader>>,
}

pub(super) struct StatsetHeader {
    pub(super) kind: KmeansStatsetField,
    pub(super) value: Value,
}

/// Public spellings are decoded at the boundary. Ordered entries retain the
/// validation and last-wins behavior of repeated name-value arguments.
pub(super) enum OptionInput {
    Distance(Distance),
    EmptyAction(EmptyAction),
    MaxIter(Value),
    Replicates(Value),
    Start(StartInput),
    Display,
    OnlinePhase(KmeansToggle),
    Statset(Vec<StatsetInput>),
    EmptyStatset,
}

pub(super) enum StartInput {
    Named(KmeansStart),
    Prepared(StartSpec),
    Resident(Value),
}

pub(super) enum StatsetInput {
    MaxIter(Value),
    Display,
    UseParallel(Value),
    UseSubstreams(Value),
    Streams,
    TolX(Value),
    TolFun(Value),
}

pub(super) fn parse_option_headers(args: &[Value]) -> BuiltinResult<Vec<OptionHeader>> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid("kmeans: options must be name-value pairs"));
    }
    args.chunks_exact(2)
        .map(|pair| {
            let name =
                keyword_of(&pair[0]).ok_or_else(|| invalid("kmeans: option name must be text"))?;
            let kind = KmeansOption::from_keyword(&name)
                .ok_or_else(|| invalid(format!("kmeans: unsupported option '{name}'")))?;
            let value = pair[1].clone();
            let statset_fields = match (&kind, &value) {
                (KmeansOption::Options, Value::Struct(value)) => Some(
                    value
                        .fields
                        .iter()
                        .filter(|(_, value)| !is_empty_option_value(value))
                        .map(|(name, value)| StatsetHeader {
                            kind: KmeansStatsetField::from_keyword(name),
                            value: value.clone(),
                        })
                        .collect(),
                ),
                _ => None,
            };
            Ok(OptionHeader {
                kind,
                value,
                statset_fields,
            })
        })
        .collect()
}

/// Decode each value after compatibility admission but before provider access.
/// The already typed option and statset identities are never parsed again.
pub(super) fn admit_option_values(headers: Vec<OptionHeader>) -> BuiltinResult<Vec<OptionInput>> {
    headers
        .into_iter()
        .map(|header| {
            let OptionHeader {
                kind,
                value,
                statset_fields,
            } = header;
            let input = match kind {
                KmeansOption::Distance => OptionInput::Distance(
                    Distance::from_keyword(&required_text(&value, "Distance")?)
                        .ok_or_else(|| invalid("kmeans: unsupported Distance"))?,
                ),
                KmeansOption::EmptyAction => OptionInput::EmptyAction(
                    EmptyAction::from_keyword(&required_text(&value, "EmptyAction")?)
                        .ok_or_else(|| invalid("kmeans: unsupported EmptyAction"))?,
                ),
                KmeansOption::MaxIter => {
                    preflight_scalar(&value, "MaxIter")?;
                    OptionInput::MaxIter(value)
                }
                KmeansOption::Replicates => {
                    preflight_scalar(&value, "Replicates")?;
                    OptionInput::Replicates(value)
                }
                KmeansOption::Start => {
                    if let Some(text) = keyword_of(&value) {
                        OptionInput::Start(StartInput::Named(
                            KmeansStart::from_keyword(&text).ok_or_else(|| {
                                invalid(format!("kmeans: unsupported Start '{text}'"))
                            })?,
                        ))
                    } else {
                        preflight_numeric_start(&value)?;
                        if matches!(&value, Value::GpuTensor(_)) {
                            OptionInput::Start(StartInput::Resident(value))
                        } else {
                            if !numeric_value_is_exact(&value) {
                                return Err(invalid("kmeans: typed-integer Start values must be exactly representable as double"));
                            }
                            OptionInput::Start(StartInput::Prepared(super::options::parse_numeric_start(&value)?))
                        }
                    }
                }
                KmeansOption::Display => {
                    KmeansDisplay::from_keyword(&required_text(&value, "Display")?)
                        .ok_or_else(|| invalid("kmeans: unsupported Display"))?;
                    OptionInput::Display
                }
                KmeansOption::OnlinePhase => OptionInput::OnlinePhase(
                    KmeansToggle::from_keyword(&required_text(&value, "OnlinePhase")?)
                        .ok_or_else(|| invalid("kmeans: unsupported OnlinePhase"))?,
                ),
                KmeansOption::Options => match value {
                    Value::Struct(_) => OptionInput::Statset(parse_statset(
                        statset_fields.ok_or_else(|| invalid("kmeans: missing statset fields"))?,
                    )?),
                    _ if is_empty_numeric(&value) => OptionInput::EmptyStatset,
                    _ => {
                        return Err(invalid(
                            "kmeans: Options must be [] or a statset options struct",
                        ))
                    }
                },
            };
            match &input {
                OptionInput::MaxIter(value) => validate_host_positive(value, "MaxIter")?,
                OptionInput::Replicates(value) => validate_host_positive(value, "Replicates")?,
                _ => {}
            }
            Ok(input)
        })
        .collect()
}

fn validate_host_positive(value: &Value, label: &str) -> BuiltinResult<()> {
    if !matches!(value, Value::GpuTensor(_)) {
        super::options::parse_positive_usize(value, label)?;
    }
    Ok(())
}

fn validate_host_parallel(value: &Value, label: &str) -> BuiltinResult<()> {
    if !matches!(value, Value::GpuTensor(_)) && super::options::bool_option(value)? {
        return Err(invalid(format!(
            "kmeans: {label} is not supported by the CPU runtime"
        )));
    }
    Ok(())
}

fn validate_host_tolerance(value: &Value, label: &str) -> BuiltinResult<()> {
    if !matches!(value, Value::GpuTensor(_)) {
        super::options::validate_tolerance(value, label)?;
    }
    Ok(())
}

fn required_text(value: &Value, role: &str) -> BuiltinResult<String> {
    keyword_of(value).ok_or_else(|| invalid(format!("kmeans: {role} must be text")))
}

fn parse_statset(value: Vec<StatsetHeader>) -> BuiltinResult<Vec<StatsetInput>> {
    let mut fields = Vec::new();
    for StatsetHeader { kind, value } in value {
        let field = match kind {
            KmeansStatsetField::MaxIter => {
                preflight_scalar(&value, "Options.MaxIter")?;
                validate_host_positive(&value, "Options.MaxIter")?;
                StatsetInput::MaxIter(value)
            }
            KmeansStatsetField::Display => {
                KmeansDisplay::from_keyword(&required_text(&value, "Options.Display")?)
                    .ok_or_else(|| invalid("kmeans: unsupported Options.Display"))?;
                StatsetInput::Display
            }
            KmeansStatsetField::UseParallel => {
                preflight_scalar(&value, "Options.UseParallel")?;
                validate_host_parallel(&value, "Options.UseParallel")?;
                StatsetInput::UseParallel(value)
            }
            KmeansStatsetField::UseSubstreams => {
                preflight_scalar(&value, "Options.UseSubstreams")?;
                validate_host_parallel(&value, "Options.UseSubstreams")?;
                StatsetInput::UseSubstreams(value)
            }
            KmeansStatsetField::Streams => {
                if !is_empty_numeric(&value) {
                    return Err(invalid(
                        "kmeans: Options.Streams is not supported by the CPU runtime",
                    ));
                }
                StatsetInput::Streams
            }
            KmeansStatsetField::TolX => {
                preflight_scalar(&value, "Options.TolX")?;
                validate_host_tolerance(&value, "Options.TolX")?;
                StatsetInput::TolX(value)
            }
            KmeansStatsetField::TolFun => {
                preflight_scalar(&value, "Options.TolFun")?;
                validate_host_tolerance(&value, "Options.TolFun")?;
                StatsetInput::TolFun(value)
            }
            KmeansStatsetField::Other => continue,
        };
        fields.push(field);
    }
    Ok(fields)
}

fn preflight_scalar(value: &Value, role: &str) -> BuiltinResult<()> {
    let scalar = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::GpuTensor(handle) => {
            handle.physical().storage() == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
                && handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |product, dim| product.checked_mul(*dim))
                    == Some(1)
        }
        _ => false,
    };
    if scalar {
        Ok(())
    } else {
        Err(invalid(format!("kmeans: {role} must be a numeric scalar")))
    }
}

fn preflight_numeric_start(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape.as_slice(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) if handle.physical().storage() == GpuTensorStorage::Real => {
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok(()),
        _ => {
            return Err(invalid(
                "kmeans: numeric Start must be a real matrix or 3-D array",
            ))
        }
    };
    if shape.len() > 3 {
        return Err(invalid(
            "kmeans: numeric Start must be a matrix or 3-D array",
        ));
    }
    let cells = shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| invalid("kmeans: numeric Start size overflow"))?;
    if cells > MAX_WORK_CELLS {
        return Err(invalid("kmeans: requested clustering work is too large"));
    }
    Ok(())
}

pub(super) fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0)
        || matches!(value, Value::GpuTensor(handle) if handle.shape().contains(&0)
            && handle.physical().storage() == GpuTensorStorage::Real
            && !runmat_accelerate_api::handle_is_logical(handle))
}

pub(super) fn is_empty_option_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::StringArray(array) => array.data.is_empty(),
        Value::CharArray(array) => array.data.is_empty(),
        Value::GpuTensor(handle) => {
            handle.shape().contains(&0) && handle.physical().storage() == GpuTensorStorage::Real
        }
        _ => false,
    }
}

pub(super) struct BorrowedOperands(Vec<GpuTensorHandle>);

impl BorrowedOperands {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }

    pub(super) fn record(&mut self, value: &Value) {
        if let Value::GpuTensor(handle) = value {
            self.0.push(handle.clone());
        }
    }

    pub(super) fn handles(&self) -> &[GpuTensorHandle] {
        &self.0
    }
}

pub(super) fn numeric_value_is_exact(value: &Value) -> bool {
    !typed_integer(value)
        || crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
}
