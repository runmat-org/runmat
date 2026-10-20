//! Classification performance curve compatibility surface.

mod options;
mod admission;
mod labels;
#[cfg(test)]
mod tests;

use labels::*;

use options::OptionName;

use runmat_value::IntValue;
use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinErrorDescriptor, PerfcurveBoolean, PerfcurveCriterion as Criterion,
    PerfcurvePriorKeyword, PerfcurveProcessNaN as ProcessNaN,
    PERFCURVE_ERROR_INTERNAL as ERROR_INTERNAL,
    PERFCURVE_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    PERFCURVE_ERROR_UNSUPPORTED as ERROR_UNSUPPORTED, PERFCURVE_INTEGER_LABELS_EXTENSION,
    PERFCURVE_INTEGER_LOGICAL_CONTROLS_EXTENSION, PERFCURVE_INTEGER_NUMERIC_OPTIONS_EXTENSION,
    PERFCURVE_INTEGER_SCORES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, ObjectInstance, Tensor, Value};

use crate::builtins::common::tensor;
use crate::{
    build_runtime_error, gather_if_needed_async, output_count, BuiltinResult, RuntimeError,
};

const NAME: &str = "perfcurve";

fn error(message: impl Into<String>, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INVALID_ARGUMENT)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_UNSUPPORTED)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INTERNAL)
}

mod curve;
use curve::*;

#[runtime_builtin(
    name = "perfcurve",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::perfcurve"
)]
async fn perfcurve_builtin(
    labels: Value,
    scores: Value,
    posclass: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let inspected = options::inspect(rest.view());
    gate_integer_extensions(&labels, &scores, &posclass, rest.view(), &inspected)?;
    let requested_outputs = output_count::current_output_count();
    if matches!(requested_outputs, Some(count) if count > 7) {
        return Err(invalid("perfcurve: too many output arguments"));
    }
    let names = options::validate(rest.view(), inspected)?;
    gate_host_integer_exactness(&scores, rest.view(), &names).await?;
    let admitted = admission::preflight(labels, scores, posclass, rest, names)?;
    if let Some(0) = requested_outputs {
        return Ok(Value::output_list(Vec::new()));
    }

    let (labels, scores, posclass, controls, precomputed_negatives) = admitted.finish().await?;
    let output_limit = requested_outputs.unwrap_or(1).max(1);
    let outputs = perfcurve_compute(
        labels,
        scores,
        posclass,
        controls,
        precomputed_negatives,
        output_limit,
    )?;
    match requested_outputs {
        None => Ok(outputs.into_iter().next().expect("perfcurve publishes X")),
        Some(count) => Ok(output_count::output_list_with_padding(count, outputs)),
    }
}

fn gate_integer_extensions(
    labels: &Value,
    scores: &Value,
    posclass: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    inspected: &[options::OptionSlot],
) -> BuiltinResult<()> {
    for value in [labels, scores, posclass] {
        crate::builtins::common::validation::reject_typed_complex_integer(value, NAME)?;
    }
    for value in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, NAME)?;
    }
    for value in [labels, posclass] {
        if crate::builtins::common::validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &PERFCURVE_INTEGER_LABELS_EXTENSION,
                NAME,
            )?;
        }
    }
    if crate::builtins::common::validation::value_has_native_integer_class(scores) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PERFCURVE_INTEGER_SCORES_EXTENSION,
            NAME,
        )?;
    }
    for (identity, pair) in options::identities(inspected).zip(rest.chunks_exact(2)) {
        let Some(identity) = identity else { continue };
        let value = &pair[1];
        match identity {
            OptionName::NegClass => {
                if crate::builtins::common::validation::value_has_native_integer_class(value) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &PERFCURVE_INTEGER_LABELS_EXTENSION,
                        NAME,
                    )?;
                }
            }
            OptionName::Weights
            | OptionName::Prior
            | OptionName::Cost
            | OptionName::TVals
            | OptionName::XVals => {
                if crate::builtins::common::validation::value_has_native_integer_class(value) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &PERFCURVE_INTEGER_NUMERIC_OPTIONS_EXTENSION,
                        NAME,
                    )?;
                }
            }
            OptionName::UseNearest
                if crate::builtins::common::validation::value_has_native_integer_class(value) =>
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &PERFCURVE_INTEGER_LOGICAL_CONTROLS_EXTENSION,
                    NAME,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

async fn gate_host_integer_exactness(
    scores: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    names: &[OptionName],
) -> BuiltinResult<()> {
    if !matches!(scores, Value::GpuTensor(_)) {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            scores,
            &PERFCURVE_INTEGER_SCORES_EXTENSION,
            NAME,
            "score",
        )
        .await?;
    }
    for (name, pair) in names.iter().zip(rest.chunks_exact(2)) {
        let value = &pair[1];
        if name.is_floating_numeric_control() && !matches!(value, Value::GpuTensor(_)) {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &PERFCURVE_INTEGER_NUMERIC_OPTIONS_EXTENSION,
                NAME,
                "numeric option",
            )
            .await?;
        }
    }
    Ok(())
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await.map_err(|cause| {
        let span = cause.span;
        let retry = cause.gpu_gather_retry();
        let mut mapped = invalid(format!("perfcurve: {cause}"));
        mapped.span = span;
        mapped.gpu_gather_retry = retry;
        mapped.source = Some(Box::new(cause));
        mapped
    })
}

fn perfcurve_compute(
    labels: LabelVector,
    scores: Vec<f64>,
    posclass: Label,
    rest: Vec<admission::DecodedOption>,
    precomputed_negatives: Option<Vec<Label>>,
    output_limit: usize,
) -> BuiltinResult<Vec<Value>> {
    if labels.labels.len() != scores.len() {
        return Err(invalid(
            "perfcurve: labels and scores must contain the same number of observations",
        ));
    }
    let posclass = coerce_label_to_kind(posclass, labels.kind, "posclass")?;
    if !labels
        .labels
        .iter()
        .any(|label| same_label(label, &posclass) && !is_missing_label(label))
    {
        return Err(invalid("perfcurve: posclass must appear in labels"));
    }
    let options = parse_options(rest, labels.kind)?;
    let neg_classes = match precomputed_negatives {
        Some(classes) => classes,
        None => negative_classes(&labels, &posclass, options.neg_class.as_ref())?,
    };
    if neg_classes.is_empty() {
        return Err(invalid(
            "perfcurve: at least one negative class must be present",
        ));
    }

    let observations = prepare_observations(&labels, &scores, &posclass, &neg_classes, &options)?;
    if observations.iter().filter(|obs| obs.is_positive).count() == 0 {
        return Err(invalid(
            "perfcurve: at least one positive observation is required",
        ));
    }
    if observations.iter().filter(|obs| !obs.is_positive).count() == 0 {
        return Err(invalid(
            "perfcurve: at least one negative observation is required",
        ));
    }

    let thresholds = thresholds_for_scores(&observations, options.tvals.as_ref())?;
    let full_curve = compute_curve(
        &observations,
        &thresholds,
        options.xcrit,
        options.ycrit,
        &options.cost,
    )?;
    let auc = if output_limit >= 4 {
        if let Some(xvals) = &options.xvals {
            auc_interval(&full_curve.x, &full_curve.y, xvals)
        } else {
            auc(&full_curve.x, &full_curve.y)
        }
    } else {
        0.0
    };
    let mut curve = if let Some(xvals) = &options.xvals {
        apply_xvals(&full_curve, xvals, options.use_nearest)?
    } else {
        full_curve
    };
    curve.auc = auc;
    let opt = (output_limit >= 5).then(|| optimal_point(&curve, &options));

    let mut outputs = vec![tensor_column(curve.x)?];
    if output_limit >= 2 {
        outputs.push(tensor_column(curve.y)?);
    }
    if output_limit >= 3 {
        outputs.push(tensor_column(curve.thresholds)?);
    }
    if output_limit >= 4 {
        outputs.push(Value::Num(curve.auc));
    }
    if output_limit >= 5 {
        let opt = opt.expect("requested optimal point was computed");
        outputs.push(tensor_row(vec![opt.0, opt.1])?);
    }
    if output_limit >= 6 {
        outputs.push(suby_output(
            &observations,
            &thresholds,
            &neg_classes,
            options.xcrit,
            options.ycrit,
            &options.cost,
            options.xvals.as_ref(),
            options.use_nearest,
        )?);
    }
    if output_limit >= 7 {
        outputs.push(labels_to_cell_value(&neg_classes)?);
    }
    Ok(outputs)
}

fn parse_options(
    rest: Vec<admission::DecodedOption>,
    label_kind: LabelKind,
) -> BuiltinResult<Options> {
    let mut options = Options {
        xcrit: Criterion::Fpr,
        ycrit: Criterion::Tpr,
        neg_class: None,
        weights: None,
        prior: PriorOption::Empirical,
        cost: [[0.0, 1.0], [1.0, 0.0]],
        tvals: None,
        xvals: None,
        use_nearest: true,
        process_nan: ProcessNaN::Ignore,
    };
    for control in rest {
        match control {
            admission::DecodedOption::XCrit(criterion) => options.xcrit = criterion,
            admission::DecodedOption::YCrit(criterion) => options.ycrit = criterion,
            admission::DecodedOption::NegClass(labels) => {
                let deduped = dedupe_labels(coerce_labels_to_kind(
                    labels.labels,
                    label_kind,
                    "NegClass",
                )?);
                if deduped.is_empty() {
                    return Err(invalid("perfcurve: NegClass cannot be empty"));
                }
                options.neg_class = Some(deduped);
            }
            admission::DecodedOption::NegClassPrepared(labels) => options.neg_class = Some(labels),
            admission::DecodedOption::Weights(weights) => options.weights = Some(weights),
            admission::DecodedOption::Prior(prior) => options.prior = prior,
            admission::DecodedOption::Cost(cost) => options.cost = cost,
            admission::DecodedOption::TVals(values) => options.tvals = Some(values),
            admission::DecodedOption::XVals(values) => options.xvals = Some(values),
            admission::DecodedOption::UseNearest(value) => options.use_nearest = value,
            admission::DecodedOption::ProcessNaN(value) => options.process_nan = value,
        }
    }
    if let Some(tvals) = &options.tvals {
        if tvals.is_empty() || tvals.iter().any(|value| value.is_nan()) {
            return Err(invalid("perfcurve: TVals must contain non-NaN thresholds"));
        }
    }
    if let Some(xvals) = &options.xvals {
        if xvals.is_empty() || xvals.iter().any(|value| !value.is_finite()) {
            return Err(invalid("perfcurve: XVals must contain finite values"));
        }
    }
    if options.tvals.is_some() && options.xvals.is_some() {
        return Err(invalid(
            "perfcurve: TVals and XVals cannot be specified together",
        ));
    }
    if !options.use_nearest {
        if let Some(xvals) = &mut options.xvals {
            xvals.sort_by(f64::total_cmp);
        }
    }
    Ok(options)
}

fn parse_criterion(value: &Value, name: &str) -> BuiltinResult<Criterion> {
    let text = scalar_text(value, name)?;
    Criterion::from_public(&text).ok_or_else(|| {
        invalid(format!(
            "perfcurve: unsupported {name} criterion '{other}'",
            other = canonical(&text)
        ))
    })
}

fn parse_process_nan(value: &Value) -> BuiltinResult<ProcessNaN> {
    let text = scalar_text(value, "ProcessNaN")?;
    ProcessNaN::from_public(&text).ok_or_else(|| {
        invalid(format!(
            "perfcurve: unsupported ProcessNaN value '{other}'",
            other = canonical(&text)
        ))
    })
}

fn parse_prior(value: &Value) -> BuiltinResult<PriorOption> {
    if let Ok(text) = scalar_text(value, "Prior") {
        return match PerfcurvePriorKeyword::from_public(&text) {
            Some(PerfcurvePriorKeyword::Empirical) => Ok(PriorOption::Empirical),
            Some(PerfcurvePriorKeyword::Uniform) => Ok(PriorOption::Uniform),
            None => Err(invalid(format!(
                "perfcurve: unsupported Prior value '{other}'",
                other = canonical(&text)
            ))),
        };
    }
    let values = numeric_vector(value, "Prior")?;
    if values.len() != 2 {
        return Err(invalid(
            "perfcurve: numeric Prior must contain positive and negative priors",
        ));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(
            "perfcurve: Prior values must be finite and nonnegative",
        ));
    }
    let scale = values[0].max(values[1]);
    if scale <= 0.0 {
        return Err(invalid("perfcurve: Prior values must have positive sum"));
    }
    let positive = values[0] / scale;
    let negative = values[1] / scale;
    let total = positive + negative;
    Ok(PriorOption::Values(positive / total, negative / total))
}

fn parse_cost(value: &Value) -> BuiltinResult<[[f64; 2]; 2]> {
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        _ => return Err(invalid("perfcurve: Cost must be a 2-by-2 numeric matrix")),
    };
    if tensor.shape() != [2, 2] {
        return Err(invalid("perfcurve: Cost must be a 2-by-2 numeric matrix"));
    }
    let values = tensor::tensor_values_f64(tensor);
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(
            "perfcurve: Cost entries must be finite and nonnegative",
        ));
    }
    Ok([[values[0], values[2]], [values[1], values[3]]])
}

fn score_vector(value: Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(tensor.shape(), "scores")?;
            Ok(tensor::tensor_into_values_f64(tensor))
        }
        Value::Num(value) => Ok(vec![value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        other => Err(invalid(format!(
            "perfcurve: scores must be a numeric vector; got {other:?}"
        ))),
    }
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(tensor.shape(), name)?;
            Ok(tensor::tensor_values_f64(tensor))
        }
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        other => Err(invalid(format!(
            "perfcurve: {name} must be a numeric vector; got {other:?}"
        ))),
    }
}

fn scalar_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Int(value) if value.is_zero() => Ok(false),
        Value::Int(value) if value.try_to_u64() == Some(1) => Ok(true),
        _ => {
            let text = scalar_text(value, name)?;
            PerfcurveBoolean::from_public(&text)
                .map(PerfcurveBoolean::value)
                .ok_or_else(|| {
                    invalid(format!(
                        "perfcurve: {name} must be logical; got '{}'",
                        canonical(&text)
                    ))
                })
        }
    }
}

fn tensor_column(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    Ok(Value::Tensor(
        Tensor::new(data, vec![rows, 1]).map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}

fn tensor_row(data: Vec<f64>) -> BuiltinResult<Value> {
    let cols = data.len();
    Ok(Value::Tensor(
        Tensor::new(data, vec![1, cols]).map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}
