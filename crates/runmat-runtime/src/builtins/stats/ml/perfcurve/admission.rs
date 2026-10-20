//! Typed option admission before provider transfer and one-time decoding.

use super::*;
use runmat_accelerate_api::GpuTensorStorage;

pub(super) enum DecodedOption {
    XCrit(Criterion),
    YCrit(Criterion),
    NegClass(LabelVector),
    NegClassPrepared(Vec<Label>),
    Weights(Vec<f64>),
    Prior(PriorOption),
    Cost([[f64; 2]; 2]),
    TVals(Vec<f64>),
    XVals(Vec<f64>),
    UseNearest(bool),
    ProcessNaN(ProcessNaN),
}

pub(super) enum AdmittedOption {
    Host(DecodedOption),
    Resident(OptionName, Value),
}

pub(super) enum AdmittedInput<T> {
    Host(T),
    Resident(Value),
}

pub(super) struct AdmittedCall {
    pub(super) labels: AdmittedInput<LabelVector>,
    pub(super) scores: AdmittedInput<Vec<f64>>,
    pub(super) posclass: AdmittedInput<Label>,
    pub(super) controls: Vec<AdmittedOption>,
    pub(super) precomputed_negatives: Option<Vec<Label>>,
}

impl AdmittedCall {
    pub(super) async fn finish(
        self,
    ) -> BuiltinResult<(
        LabelVector,
        Vec<f64>,
        Label,
        Vec<DecodedOption>,
        Option<Vec<Label>>,
    )> {
        let labels = match self.labels {
            AdmittedInput::Host(labels) => labels,
            AdmittedInput::Resident(value) => labels_from_value(gather(value).await?, "labels")?,
        };
        let scores = match self.scores {
            AdmittedInput::Host(scores) => scores,
            AdmittedInput::Resident(value) => {
                let gathered = gather(value).await?;
                crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                    &gathered,
                    &PERFCURVE_INTEGER_SCORES_EXTENSION,
                    NAME,
                    "score",
                )
                .await?;
                score_vector(gathered)?
            }
        };
        let posclass = match self.posclass {
            AdmittedInput::Host(label) => label,
            AdmittedInput::Resident(value) => {
                scalar_label_from_value(gather(value).await?, "posclass")?
            }
        };
        let mut controls = Vec::with_capacity(self.controls.len());
        for control in self.controls {
            controls.push(control.finish().await?);
        }
        Ok((
            labels,
            scores,
            posclass,
            controls,
            self.precomputed_negatives,
        ))
    }
}

impl AdmittedOption {
    pub(super) async fn finish(self) -> BuiltinResult<DecodedOption> {
        match self {
            Self::Host(decoded) => Ok(decoded),
            Self::Resident(name, source) => {
                let gathered = gather(source).await?;
                if name.is_floating_numeric_control() {
                    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                        &gathered,
                        &PERFCURVE_INTEGER_NUMERIC_OPTIONS_EXTENSION,
                        NAME,
                        "numeric option",
                    )
                    .await?;
                }
                decode_host(name, gathered)
            }
        }
    }
}

fn count(shape: &[usize]) -> BuiltinResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| invalid("perfcurve: input dimensions overflow"))
}

fn vector(shape: &[usize], role: &str) -> BuiltinResult<usize> {
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid(format!("perfcurve: {role} must be a vector")));
    }
    count(shape)
}

fn numeric_shape(value: &Value, role: &str) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => vector(tensor.shape(), role),
        Value::Num(_) | Value::Int(_) => Ok(1),
        Value::GpuTensor(handle)
            if handle.physical().storage() == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            vector(handle.shape(), role)
        }
        _ => Err(invalid(format!(
            "perfcurve: {role} must be a real numeric vector"
        ))),
    }
}

fn labels_shape(value: &Value, role: &str) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Tensor(tensor) => vector(tensor.shape(), role).map(Some),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => Ok(Some(1)),
        Value::LogicalArray(array) => vector(&array.shape, role).map(Some),
        Value::StringArray(array) => vector(&array.shape, role).map(Some),
        Value::CharArray(array) => Ok(Some(array.rows)),
        Value::Cell(cell) => vector(&[cell.rows, cell.cols], role).map(Some),
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => Ok(None),
        Value::GpuTensor(handle) if handle.physical().storage() == GpuTensorStorage::Real => {
            vector(handle.shape(), role).map(Some)
        }
        _ => Err(invalid(format!(
            "perfcurve: {role} must contain admitted labels"
        ))),
    }
}

fn preflight_option(name: OptionName, value: Value) -> BuiltinResult<AdmittedOption> {
    if let Value::GpuTensor(handle) = &value {
        match name {
            OptionName::XCrit | OptionName::YCrit | OptionName::ProcessNaN => {
                return Err(invalid(
                    "perfcurve: criterion controls must be text scalars",
                ))
            }
            OptionName::NegClass => {
                labels_shape(&value, "NegClass")?;
            }
            OptionName::Weights | OptionName::TVals | OptionName::XVals => {
                numeric_shape(&value, "numeric option")?;
            }
            OptionName::Prior => {
                if numeric_shape(&value, "Prior")? != 2 {
                    return Err(invalid(
                        "perfcurve: numeric Prior must contain positive and negative priors",
                    ));
                }
            }
            OptionName::Cost => {
                if handle.physical().storage() != GpuTensorStorage::Real
                    || runmat_accelerate_api::handle_is_logical(handle)
                    || handle.shape() != [2, 2]
                {
                    return Err(invalid("perfcurve: Cost must be a 2-by-2 numeric matrix"));
                }
            }
            OptionName::UseNearest => {
                if handle.physical().storage() != GpuTensorStorage::Real
                    || count(handle.shape())? != 1
                {
                    return Err(invalid("perfcurve: UseNearest must be logical"));
                }
            }
            OptionName::UnsupportedBootstrap => {
                return Err(unsupported(
                    "perfcurve: confidence-bound and bootstrap modes are not implemented",
                ))
            }
        }
        return Ok(AdmittedOption::Resident(name, value));
    }
    decode_host(name, value).map(AdmittedOption::Host)
}

fn decode_host(name: OptionName, value: Value) -> BuiltinResult<DecodedOption> {
    Ok(match name {
        OptionName::XCrit => DecodedOption::XCrit(parse_criterion(&value, "XCrit")?),
        OptionName::YCrit => DecodedOption::YCrit(parse_criterion(&value, "YCrit")?),
        OptionName::NegClass => DecodedOption::NegClass(labels_from_value(value, "NegClass")?),
        OptionName::Weights => {
            let weights = numeric_vector(&value, "Weights")?;
            if weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            {
                return Err(invalid(
                    "perfcurve: Weights must be finite nonnegative values",
                ));
            }
            DecodedOption::Weights(weights)
        }
        OptionName::Prior => DecodedOption::Prior(parse_prior(&value)?),
        OptionName::Cost => DecodedOption::Cost(parse_cost(&value)?),
        OptionName::TVals => {
            let values = numeric_vector(&value, "TVals")?;
            if values.is_empty() || values.iter().any(|value| value.is_nan()) {
                return Err(invalid("perfcurve: TVals must contain non-NaN thresholds"));
            }
            DecodedOption::TVals(values)
        }
        OptionName::XVals => {
            let values = numeric_vector(&value, "XVals")?;
            if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
                return Err(invalid("perfcurve: XVals must contain finite values"));
            }
            DecodedOption::XVals(values)
        }
        OptionName::UseNearest => DecodedOption::UseNearest(scalar_bool(&value, "UseNearest")?),
        OptionName::ProcessNaN => DecodedOption::ProcessNaN(parse_process_nan(&value)?),
        OptionName::UnsupportedBootstrap => {
            return Err(unsupported(
                "perfcurve: confidence-bound and bootstrap modes are not implemented",
            ))
        }
    })
}

pub(super) fn preflight(
    labels: Value,
    scores: Value,
    posclass: Value,
    rest: crate::call::BuiltinArguments,
    names: Vec<OptionName>,
) -> BuiltinResult<AdmittedCall> {
    let label_count = labels_shape(&labels, "labels")?;
    let score_count = numeric_shape(&scores, "scores")?;
    if label_count.is_some_and(|count| count != score_count) {
        return Err(invalid(
            "perfcurve: labels and scores must contain the same number of observations",
        ));
    }
    if labels_shape(&posclass, "posclass")?.is_some_and(|count| count != 1) {
        return Err(invalid("perfcurve: posclass must be a scalar label"));
    }
    let mut tvals = false;
    let mut xvals = false;
    let mut admitted = Vec::with_capacity(names.len());
    for (name, value) in names.into_iter().zip(rest.into_iter().skip(1).step_by(2)) {
        if name == OptionName::Weights
            && label_count.is_some_and(|count| {
                numeric_shape(&value, "Weights")
                    .ok()
                    .is_some_and(|weights| weights != count)
            })
        {
            return Err(invalid(
                "perfcurve: Weights must have one value per observation",
            ));
        }
        tvals |= name == OptionName::TVals;
        xvals |= name == OptionName::XVals;
        admitted.push(preflight_option(name, value)?);
    }
    if tvals && xvals {
        return Err(invalid(
            "perfcurve: TVals and XVals cannot be specified together",
        ));
    }
    let labels = match labels {
        Value::GpuTensor(_) => AdmittedInput::Resident(labels),
        value => AdmittedInput::Host(labels_from_value(value, "labels")?),
    };
    let scores = match scores {
        Value::GpuTensor(_) => AdmittedInput::Resident(scores),
        value => AdmittedInput::Host(score_vector(value)?),
    };
    let posclass = match posclass {
        Value::GpuTensor(_) => AdmittedInput::Resident(posclass),
        value => AdmittedInput::Host(scalar_label_from_value(value, "posclass")?),
    };
    let mut explicit = None;
    let mut deferred_explicit = false;
    if let AdmittedInput::Host(labels) = &labels {
        for control in &mut admitted {
            match control {
                AdmittedOption::Host(DecodedOption::NegClass(values)) => {
                    let selected = dedupe_labels(coerce_labels_to_kind(
                        std::mem::take(&mut values.labels),
                        labels.kind,
                        "NegClass",
                    )?);
                    if selected.is_empty() {
                        return Err(invalid("perfcurve: NegClass cannot be empty"));
                    }
                    explicit = Some(selected.clone());
                    *control = AdmittedOption::Host(DecodedOption::NegClassPrepared(selected));
                }
                AdmittedOption::Resident(OptionName::NegClass, _) => deferred_explicit = true,
                _ => {}
            }
        }
    }
    let mut precomputed_negatives = None;
    if let (AdmittedInput::Host(labels), AdmittedInput::Host(posclass)) = (&labels, &posclass) {
        let positive = coerce_label_to_kind(posclass.clone(), labels.kind, "posclass")?;
        if !labels
            .labels
            .iter()
            .any(|label| same_label(label, &positive) && !is_missing_label(label))
        {
            return Err(invalid("perfcurve: posclass must appear in labels"));
        }
        if !deferred_explicit {
            let classes = negative_classes(labels, &positive, explicit.as_ref())?;
            if classes.is_empty() {
                return Err(invalid(
                    "perfcurve: at least one negative class must be present",
                ));
            }
            precomputed_negatives = Some(classes);
        }
    }
    Ok(AdmittedCall {
        labels,
        scores,
        posclass,
        controls: admitted,
        precomputed_negatives,
    })
}
