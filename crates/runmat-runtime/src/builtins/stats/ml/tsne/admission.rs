//! Typed ordered option admission before any provider materialization.

use super::{
    invalid, Algorithm, DistanceMetric, TsneCacheSizeKeyword, TsneDisplay, TsneOption,
    TsneStatsetField,
};
use crate::builtins::common::random_args::keyword_of;
use crate::BuiltinResult;
use runmat_value::Value;

mod materialize;
mod exactness;
pub(super) use exactness::{ensure_exact_tsne_floating_options, ensure_exact_tsne_integer_value};
pub(super) use materialize::materialize_options;

pub(super) type OptionHeader = (TsneOption, Value);

fn reserve_admission<T>(
    values: &mut Vec<T>,
    count: usize,
    message: &'static str,
) -> BuiltinResult<()> {
    values
        .try_reserve_exact(count)
        .map_err(|source| super::internal_with_source(message, source))
}

#[derive(Clone)]
pub(super) enum AdmittedOption {
    Algorithm(Algorithm),
    CacheSizeMaximal,
    CacheSizeNumeric(Value),
    Distance(DistanceMetric),
    Exaggeration(Value),
    NumDimensions(Value),
    NumPcaComponents(Value),
    Perplexity(Value),
    Standardize(Value),
    InitialY(Value),
    LearnRate(Value),
    NumPrint(Value),
    Options(Vec<AdmittedStatsetField>),
    OptionsEmptyResident(Value),
    Theta(Value),
    Verbose(Value),
}

#[derive(Clone)]
pub(super) enum AdmittedStatsetField {
    MaxIter(Value),
    TolFun(Value),
    Display,
    EmptyOutputFcn,
    EmptyOutputFcnResident(Value),
}

pub(super) fn option_headers(
    values: crate::call::BuiltinArguments,
) -> BuiltinResult<Vec<OptionHeader>> {
    if !values.len().is_multiple_of(2) {
        return Err(invalid("tsne: name-value options must be paired"));
    }
    let mut headers = Vec::new();
    reserve_admission(
        &mut headers,
        values.len() / 2,
        "tsne: option headers exceed available memory",
    )?;
    let mut values = values.into_iter();
    while let (Some(name), Some(value)) = (values.next(), values.next()) {
        let name = keyword_of(&name)?.ok_or_else(|| invalid("tsne: option name must be text"))?;
        let option = TsneOption::from_keyword(&name)
            .ok_or_else(|| invalid(format!("tsne: unknown option '{name}'")))?;
        headers.push((option, value));
    }
    Ok(headers)
}

pub(super) fn admit_options(headers: Vec<OptionHeader>) -> BuiltinResult<Vec<AdmittedOption>> {
    let mut admitted = Vec::new();
    reserve_admission(
        &mut admitted,
        headers.len(),
        "tsne: admitted options exceed available memory",
    )?;
    for (option, value) in headers {
        let admitted_option = match option {
            TsneOption::Algorithm => {
                let text = text(&value, "Algorithm")?;
                AdmittedOption::Algorithm(
                    Algorithm::from_keyword(&text)
                        .ok_or_else(|| invalid(format!("tsne: unsupported Algorithm '{text}'")))?,
                )
            }
            TsneOption::Distance => {
                if matches!(
                    value,
                    Value::FunctionHandle(_)
                        | Value::ExternalFunctionHandle(_)
                        | Value::MethodFunctionHandle(_)
                        | Value::BoundFunctionHandle(_)
                        | Value::ScopedFunctionHandle(_)
                ) {
                    return Err(invalid(
                        "tsne: custom distance function handles are not supported yet",
                    ));
                }
                let text = text(&value, "Distance")?;
                AdmittedOption::Distance(
                    DistanceMetric::from_keyword(&text)
                        .ok_or_else(|| invalid(format!("tsne: unsupported Distance '{text}'")))?,
                )
            }
            TsneOption::CacheSize => {
                if let Some(text) = keyword_of(&value)? {
                    if TsneCacheSizeKeyword::from_keyword(&text)
                        != Some(TsneCacheSizeKeyword::Maximal)
                    {
                        return Err(invalid("tsne: CacheSize must be positive or 'maximal'"));
                    }
                    AdmittedOption::CacheSizeMaximal
                } else {
                    scalar_shape(&value, "CacheSize")?;
                    AdmittedOption::CacheSizeNumeric(value)
                }
            }
            TsneOption::Options => {
                if resident_empty_numeric(&value) {
                    AdmittedOption::OptionsEmptyResident(value)
                } else {
                    AdmittedOption::Options(statset_fields(value)?)
                }
            }
            TsneOption::InitialY => {
                numeric_shape(&value, "InitialY", false)?;
                AdmittedOption::InitialY(value)
            }
            TsneOption::Exaggeration => {
                scalar(value, "Exaggeration", AdmittedOption::Exaggeration)?
            }
            TsneOption::NumDimensions => {
                scalar(value, "NumDimensions", AdmittedOption::NumDimensions)?
            }
            TsneOption::NumPcaComponents => {
                scalar(value, "NumPCAComponents", AdmittedOption::NumPcaComponents)?
            }
            TsneOption::Perplexity => scalar(value, "Perplexity", AdmittedOption::Perplexity)?,
            TsneOption::Standardize => scalar(value, "Standardize", AdmittedOption::Standardize)?,
            TsneOption::LearnRate => scalar(value, "LearnRate", AdmittedOption::LearnRate)?,
            TsneOption::NumPrint => scalar(value, "NumPrint", AdmittedOption::NumPrint)?,
            TsneOption::Theta => scalar(value, "Theta", AdmittedOption::Theta)?,
            TsneOption::Verbose => scalar(value, "Verbose", AdmittedOption::Verbose)?,
        };
        admitted.push(admitted_option);
    }
    validate_host_controls(&admitted)?;
    Ok(admitted)
}

fn validate_host_controls(options: &[AdmittedOption]) -> BuiltinResult<()> {
    // The execution parser is also the host-visible preflight authority.
    // A resident scalar cannot be range-checked until its owning provider
    // materializes it, but an unrelated malformed host control must not
    // allow that provider call to happen first.
    let mut visible = Vec::new();
    reserve_admission(
        &mut visible,
        options.len(),
        "tsne: host-visible options exceed available memory",
    )?;
    for option in options {
        match option {
            AdmittedOption::CacheSizeNumeric(value) if !matches!(value, Value::GpuTensor(_)) => {
                visible.push(option.clone())
            }
            AdmittedOption::Exaggeration(value)
            | AdmittedOption::NumDimensions(value)
            | AdmittedOption::NumPcaComponents(value)
            | AdmittedOption::Perplexity(value)
            | AdmittedOption::Standardize(value)
            | AdmittedOption::LearnRate(value)
            | AdmittedOption::NumPrint(value)
            | AdmittedOption::Theta(value)
            | AdmittedOption::Verbose(value)
                if !matches!(value, Value::GpuTensor(_)) =>
            {
                visible.push(option.clone())
            }
            AdmittedOption::Options(fields) => {
                let mut visible_fields = Vec::new();
                reserve_admission(
                    &mut visible_fields,
                    fields.len(),
                    "tsne: host-visible statset fields exceed available memory",
                )?;
                for field in fields.iter().filter(|field| match field {
                    AdmittedStatsetField::MaxIter(value) | AdmittedStatsetField::TolFun(value) => {
                        !matches!(value, Value::GpuTensor(_))
                    }
                    AdmittedStatsetField::Display | AdmittedStatsetField::EmptyOutputFcn => true,
                    AdmittedStatsetField::EmptyOutputFcnResident(_) => false,
                }) {
                    visible_fields.push(field.clone());
                }
                visible.push(AdmittedOption::Options(visible_fields));
            }
            AdmittedOption::Algorithm(_)
            | AdmittedOption::CacheSizeMaximal
            | AdmittedOption::CacheSizeNumeric(_)
            | AdmittedOption::Distance(_)
            | AdmittedOption::Exaggeration(_)
            | AdmittedOption::NumDimensions(_)
            | AdmittedOption::NumPcaComponents(_)
            | AdmittedOption::Perplexity(_)
            | AdmittedOption::Standardize(_)
            | AdmittedOption::InitialY(_)
            | AdmittedOption::LearnRate(_)
            | AdmittedOption::NumPrint(_)
            | AdmittedOption::OptionsEmptyResident(_)
            | AdmittedOption::Theta(_)
            | AdmittedOption::Verbose(_) => {}
        }
    }
    super::options::parse_options(visible)?;
    Ok(())
}

fn scalar(
    value: Value,
    name: &str,
    constructor: fn(Value) -> AdmittedOption,
) -> BuiltinResult<AdmittedOption> {
    scalar_shape(&value, name)?;
    Ok(constructor(value))
}

fn text(value: &Value, name: &str) -> BuiltinResult<String> {
    keyword_of(value)?.ok_or_else(|| invalid(format!("tsne: {name} must be text")))
}

fn scalar_shape(value: &Value, name: &str) -> BuiltinResult<()> {
    numeric_shape(value, name, true)
}

fn numeric_shape(value: &Value, name: &str, scalar: bool) -> BuiltinResult<()> {
    let shape = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok(()),
        Value::Tensor(tensor) => tensor.shape(),
        Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            handle.shape()
        }
        _ => {
            return Err(invalid(format!(
                "tsne: {name} must be a real numeric {}",
                if scalar { "scalar" } else { "array" }
            )))
        }
    };
    let count = shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| invalid(format!("tsne: {name} shape overflow")))?;
    if scalar && count != 1 {
        return Err(invalid(format!("tsne: {name} must be a numeric scalar")));
    }
    if !scalar && shape.len() > 2 {
        return Err(invalid("tsne: InitialY must be a 2-D numeric matrix"));
    }
    Ok(())
}

fn statset_fields(value: Value) -> BuiltinResult<Vec<AdmittedStatsetField>> {
    let Value::Struct(fields) = value else {
        if super::options::is_empty_numeric(&value) {
            return Ok(Vec::new());
        }
        return Err(invalid("tsne: Options must be a struct"));
    };
    let mut admitted = Vec::new();
    reserve_admission(
        &mut admitted,
        fields.fields.len(),
        "tsne: statset fields exceed available memory",
    )?;
    for (name, value) in fields.into_fields_unbounded() {
        if super::options::is_empty_option_value(&value) {
            continue;
        }
        let field = TsneStatsetField::from_keyword(&name)
            .ok_or_else(|| invalid(format!("tsne: unsupported Options field '{name}'")))?;
        admitted.push(match field {
            TsneStatsetField::MaxIter => {
                scalar_shape(&value, "Options.MaxIter")?;
                AdmittedStatsetField::MaxIter(value)
            }
            TsneStatsetField::TolFun => {
                scalar_shape(&value, "Options.TolFun")?;
                AdmittedStatsetField::TolFun(value)
            }
            TsneStatsetField::Display => {
                let display = text(&value, "Options.Display")?;
                TsneDisplay::from_keyword(&display).ok_or_else(|| {
                    invalid(format!("tsne: unsupported Options.Display '{display}'"))
                })?;
                AdmittedStatsetField::Display
            }
            TsneStatsetField::OutputFcn => {
                if resident_empty_numeric(&value) {
                    AdmittedStatsetField::EmptyOutputFcnResident(value)
                } else if super::options::is_empty_numeric(&value) {
                    AdmittedStatsetField::EmptyOutputFcn
                } else {
                    return Err(invalid(
                        "tsne: Options.OutputFcn is not supported by the CPU runtime",
                    ));
                }
            }
        });
    }
    Ok(admitted)
}

fn resident_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle)
        if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
            && handle.shape().contains(&0))
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn option_capacity_failure_keeps_original_operational_cause() {
        let mut headers = Vec::<u8>::new();
        let error = reserve_admission(
            &mut headers,
            usize::MAX,
            "tsne: option headers exceed available memory",
        )
        .expect_err("capacity overflow must reject before allocation");
        assert_eq!(error.identifier(), Some("RunMat:tsne:Internal"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error
            .source()
            .expect("original reservation failure")
            .is::<std::collections::TryReserveError>());
    }
}
