use super::{
    invalid, Distance, EmptyAction, KmeansDisplay, KmeansOption, KmeansStart, KmeansStatsetField,
    KmeansToggle, StartSpec, MAX_WORK_CELLS,
};
use crate::builtins::common::{arg_tokens::scalar_keyword_of, tensor};
use crate::BuiltinResult;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

pub(super) enum OptionHeader {
    Argument { kind: KmeansOption, value: Value },
    Statset(Vec<StatsetHeader>),
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

fn keyword(value: &Value) -> BuiltinResult<Option<String>> {
    scalar_keyword_of(value, "kmeans").map_err(|source| {
        let mut error = invalid("kmeans: text control is not valid UTF-16");
        error.source = Some(Box::new(source));
        error
    })
}

pub(super) fn parse_option_headers(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Vec<OptionHeader>> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid("kmeans: options must be name-value pairs"));
    }
    let mut values = args.into_iter();
    let mut headers = Vec::new();
    while let Some(name_value) = values.next() {
        let name =
            keyword(&name_value)?.ok_or_else(|| invalid("kmeans: option name must be text"))?;
        let kind = KmeansOption::from_keyword(&name)
            .ok_or_else(|| invalid(format!("kmeans: unsupported option '{name}'")))?;
        let value = values
            .next()
            .ok_or_else(|| invalid("kmeans: options must be name-value pairs"))?;
        let header = match (kind, value) {
            (KmeansOption::Options, Value::Struct(value)) => OptionHeader::Statset(
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
            (kind, value) => OptionHeader::Argument { kind, value },
        };
        headers.push(header);
    }
    Ok(headers)
}

/// Decode each value after compatibility admission but before provider access.
/// The already typed option and statset identities are never parsed again.
#[cfg(test)]
pub(super) fn admit_option_values(headers: Vec<OptionHeader>) -> BuiltinResult<Vec<OptionInput>> {
    futures::executor::block_on(admit_option_values_cooperative(
        headers,
        &mut super::cooperation::KmeansCooperation::new(None),
    ))
}

pub(super) async fn admit_option_values_cooperative(
    headers: Vec<OptionHeader>,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<OptionInput>> {
    let mut admitted = Vec::new();
    admitted
        .try_reserve_exact(headers.len())
        .map_err(|source| {
            super::internal_with_source("kmeans: options exceed available memory", source)
        })?;
    for header in headers {
        let (kind, value) = match header {
            OptionHeader::Argument { kind, value } => (kind, value),
            OptionHeader::Statset(fields) => {
                admitted.push(OptionInput::Statset(parse_statset(fields)?));
                cooperation.completed(1).await?;
                continue;
            }
        };
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
                if let Some(text) = keyword(&value)? {
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
                        if !numeric_value_is_exact_cooperative(&value, cooperation).await? {
                            return Err(invalid("kmeans: typed-integer Start values must be exactly representable as double"));
                        }
                        OptionInput::Start(StartInput::Prepared(
                            super::options::parse_numeric_start_cooperative(value, cooperation)
                                .await?,
                        ))
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
            KmeansOption::Options if is_empty_numeric(&value) => OptionInput::EmptyStatset,
            KmeansOption::Options => {
                return Err(invalid(
                    "kmeans: Options must be [] or a statset options struct",
                ))
            }
        };
        match &input {
            OptionInput::MaxIter(value) => validate_host_positive(value, "MaxIter")?,
            OptionInput::Replicates(value) => validate_host_positive(value, "Replicates")?,
            _ => {}
        }
        admitted.push(input);
        cooperation.completed(1).await?;
    }
    Ok(admitted)
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
    keyword(value)?.ok_or_else(|| invalid(format!("kmeans: {role} must be text")))
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
        Value::Tensor(tensor) => tensor.shape(),
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
        Value::Cell(cell) => cell.data().is_empty(),
        Value::StringArray(array) => array.data().is_empty(),
        Value::CharArray(array) => array.data().is_empty(),
        Value::GpuTensor(handle) => {
            handle.shape().contains(&0) && handle.physical().storage() == GpuTensorStorage::Real
        }
        _ => false,
    }
}

pub(super) struct BorrowedOperands {
    protected: Vec<GpuTensorHandle>,
    intent_sources: Vec<GpuTensorHandle>,
}

impl BorrowedOperands {
    pub(super) fn new() -> Self {
        Self {
            protected: Vec::new(),
            intent_sources: Vec::new(),
        }
    }

    pub(super) fn protect_value(&mut self, value: &Value) -> BuiltinResult<()> {
        collect_resident_handles(value, &mut self.protected)
    }

    pub(super) fn record_consumed(&mut self, value: &Value) -> BuiltinResult<()> {
        // A numerical operand actually materialized for clustering carries its
        // explicit device intent into the result. Merely present metadata,
        // including ignored statset fields, is protected from aliasing but
        // does not determine result placement.
        if let Value::GpuTensor(handle) = value {
            super::reserve_borrowed(
                &mut self.intent_sources,
                1,
                "kmeans: consumed operands exceed available memory",
            )?;
            self.intent_sources.push(handle.clone());
        }
        Ok(())
    }

    pub(super) fn protected(&self) -> &[GpuTensorHandle] {
        &self.protected
    }

    pub(super) fn intent_sources(&self) -> &[GpuTensorHandle] {
        &self.intent_sources
    }
}

fn collect_resident_handles(value: &Value, into: &mut Vec<GpuTensorHandle>) -> BuiltinResult<()> {
    match value {
        Value::GpuTensor(handle) => {
            super::reserve_borrowed(into, 1, "kmeans: borrowed operands exceed available memory")?;
            into.push(handle.clone());
        }
        Value::Cell(cell) => {
            for value in cell.data() {
                collect_resident_handles(value, into)?;
            }
        }
        Value::Struct(value) => {
            for value in value.fields.values() {
                collect_resident_handles(value, into)?;
            }
        }
        Value::StructArray(value) => {
            let mut result = Ok(());
            value.for_each_value(|value| {
                if result.is_ok() {
                    result = collect_resident_handles(value, into);
                }
            });
            result?;
        }
        Value::Object(value) => {
            for value in value.properties().values() {
                collect_resident_handles(value, into)?;
            }
        }
        Value::ObjectArray(value) => {
            for value in value.data() {
                collect_resident_handles(value, into)?;
            }
        }
        Value::Closure(value) => {
            for value in &value.captures {
                collect_resident_handles(value, into)?;
            }
        }
        Value::OutputList(values) => {
            for value in values {
                collect_resident_handles(value, into)?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(super) async fn numeric_value_is_exact_cooperative(
    value: &Value,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<bool> {
    match value {
        Value::Int(integer) => Ok(crate::builtins::common::validation::integer_is_exact_f64(
            integer,
        )),
        Value::Tensor(tensor) => {
            let Some(storage) = tensor.integer_storage() else {
                return Ok(true);
            };
            for index in 0..storage.len() {
                let integer = storage
                    .value_at(index)
                    .expect("integer storage index is within its length");
                let exact = crate::builtins::common::validation::integer_is_exact_f64(&integer);
                cooperation.completed(1).await?;
                if !exact {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(true),
    }
}
