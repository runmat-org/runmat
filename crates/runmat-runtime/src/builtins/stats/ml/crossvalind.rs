//! MATLAB-compatible `crossvalind` builtin.

use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinErrorDescriptor, CrossvalIndMethod as Method, CrossvalIndOption, PartitionCountGrammar,
    CROSSVALIND_ERROR_INTERNAL, CROSSVALIND_ERROR_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    decode_utf16_code_units, CharArray, LogicalArray, StringArray, Tensor, Utf16DecodeError, Value,
};

use super::partition_shared::PartitionLabel;
use crate::builtins::common::random;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "crossvalind";
const MAX_OBSERVATIONS: usize = 10_000_000;
const EPS: f64 = 1.0e-12;

fn builtin_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    builtin_error(message, &CROSSVALIND_ERROR_INVALID_ARGUMENT)
}

fn invalid_utf16(context: &str, error: Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "crossvalind: {context} is not valid UTF-16 text: {error}"
    ))
    .with_builtin(NAME)
    .with_source(error);
    if let Some(identifier) = CROSSVALIND_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    builtin_error(message, &CROSSVALIND_ERROR_INTERNAL)
}

#[derive(Clone, Debug, Default)]
struct Options {
    classes: Option<Vec<Option<PartitionLabel>>>,
    min: Option<usize>,
}

#[derive(Clone, Debug)]
enum PartitionOutput {
    Folds(Value),
    TrainTest { train: Value, test: Value },
}

#[runtime_builtin(
    name = "crossvalind",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::crossvalind"
)]
async fn crossvalind_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let admitted = admit_host_controls(args.view())?;
    let args = gather_values(args, runtime.as_ref()).await?;
    let mut turn = if matches!(admitted.method, Method::Resubstitution) {
        None
    } else {
        Some(random::acquire_turn(runtime.as_ref(), NAME).await?)
    };
    let output = crossvalind_compute(args.view(), admitted, turn.as_mut())?;
    output_value(output)
}

struct AdmittedControls {
    method: Method,
    option_start: usize,
    option_names: Vec<CrossvalIndOption>,
}

fn admit_host_controls(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<AdmittedControls> {
    if args.len() < 2 {
        return Err(invalid_argument(
            "crossvalind: method and observation count are required",
        ));
    }
    let method_text = scalar_text(&args[0], "method")?;
    let method = Method::parse(&method_text).ok_or_else(|| {
        invalid_argument(format!(
            "crossvalind: unsupported cross-validation method '{}'",
            canonical(&method_text)
        ))
    })?;
    let (option_start, max_outputs) = match method {
        Method::KFold => (3, 1),
        Method::HoldOut | Method::LeaveMOut => (3, 2),
        Method::Resubstitution => (2, 2),
    };
    if args.len() < option_start {
        return Err(invalid_argument(format!(
            "crossvalind: {} requires a partition value",
            method.name()
        )));
    }
    if crate::output_count::current_output_count().is_some_and(|count| count > max_outputs) {
        return Err(invalid_argument(if max_outputs == 1 {
            "crossvalind: KFold returns one output argument"
        } else {
            "crossvalind: too many output arguments; maximum is 2"
        }));
    }
    if !(args.len() - option_start).is_multiple_of(2) {
        return Err(invalid_argument(
            "crossvalind: name-value options must be supplied in pairs",
        ));
    }
    let n = if matches!(&args[1], Value::GpuTensor(_)) {
        None
    } else {
        let n = observation_count(&args[1])?;
        validate_n(n)?;
        Some(n)
    };
    if let Some(value) = args.get(2).filter(|_| method != Method::Resubstitution) {
        if !matches!(value, Value::GpuTensor(_)) {
            match method {
                Method::KFold => {
                    let k = positive_integer(value, "KFold")?;
                    if k < 2 || n.is_some_and(|n| k > n) {
                        return Err(invalid_argument(
                            "crossvalind: KFold fold count must be an integer in [2, n]",
                        ));
                    }
                }
                Method::HoldOut => {
                    if let Some(n) = n {
                        holdout_count(value, n)?;
                    }
                }
                Method::LeaveMOut => {
                    let m = positive_integer(value, "LeaveMOut")?;
                    if n.is_some_and(|n| m >= n) {
                        return Err(invalid_argument(
                            "crossvalind: LeaveMOut must leave between 1 and n-1 observations out",
                        ));
                    }
                }
                Method::Resubstitution => {}
            }
        }
    }
    let mut option_names = Vec::with_capacity((args.len() - option_start) / 2);
    for index in (option_start..args.len()).step_by(2) {
        let name = scalar_text(&args[index], "option name")?;
        let option = CrossvalIndOption::parse(&name).ok_or_else(|| {
            invalid_argument(format!(
                "crossvalind: unsupported option '{}'",
                canonical(&name)
            ))
        })?;
        if method == Method::Resubstitution {
            return Err(invalid_argument(
                "crossvalind: Resubstitution does not accept Classes or Min options",
            ));
        }
        let value = &args[index + 1];
        if !matches!(value, Value::GpuTensor(_)) {
            match option {
                CrossvalIndOption::Min => {
                    positive_integer(value, "Min")?;
                }
                CrossvalIndOption::Classes => {
                    if let Some(n) = n {
                        classes_from_value(value, n)?;
                    }
                }
            }
        }
        option_names.push(option);
    }
    Ok(AdmittedControls {
        method,
        option_start,
        option_names,
    })
}

async fn gather_values(
    values: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let gathered = values
        .try_map_preserving_roots_async(values.len(), runtime, |index| {
            gather_if_needed_async(&values[index])
        })
        .await?;
    drop(values);
    Ok(gathered)
}

fn crossvalind_compute(
    args: crate::call::FunctionArgumentView<'_>,
    admitted: AdmittedControls,
    turn: Option<&mut random::RandomTurn>,
) -> BuiltinResult<PartitionOutput> {
    let method = admitted.method;
    let n = observation_count(&args[1])?;
    let (value, option_start) = match method {
        Method::Resubstitution => (None, 2usize),
        Method::KFold | Method::HoldOut | Method::LeaveMOut => {
            let Some(value) = args.get(2) else {
                return Err(invalid_argument(format!(
                    "crossvalind: {} requires a partition value",
                    method.name()
                )));
            };
            (Some(value), 3usize)
        }
    };
    debug_assert_eq!(option_start, admitted.option_start);
    let options = parse_options(args.range(option_start..)?, n, &admitted.option_names)?;
    validate_n(n)?;

    match method {
        Method::KFold => {
            let k = positive_integer(value.expect("KFold value"), "KFold")?;
            let folds = kfold_indices(
                n,
                k,
                options.classes.as_deref(),
                options.min,
                turn.expect("random method has an ordered turn"),
            )?;
            Ok(PartitionOutput::Folds(tensor_value(folds, vec![n, 1])?))
        }
        Method::HoldOut => {
            let test_count = holdout_count(value.expect("HoldOut value"), n)?;
            let test = random_test_mask(
                n,
                test_count,
                options.classes.as_deref(),
                options.min,
                turn.expect("random method has an ordered turn"),
            )?;
            train_test_from_test_mask(test)
        }
        Method::LeaveMOut => {
            let m = positive_integer(value.expect("LeaveMOut value"), "LeaveMOut")?;
            if m >= n {
                return Err(invalid_argument(
                    "crossvalind: LeaveMOut must leave between 1 and n-1 observations out",
                ));
            }
            let test = random_test_mask(
                n,
                m,
                options.classes.as_deref(),
                options.min,
                turn.expect("random method has an ordered turn"),
            )?;
            train_test_from_test_mask(test)
        }
        Method::Resubstitution => {
            if options.classes.is_some() || options.min.is_some() {
                return Err(invalid_argument(
                    "crossvalind: Resubstitution does not accept Classes or Min options",
                ));
            }
            Ok(PartitionOutput::TrainTest {
                train: logical_value(vec![1; n], vec![n, 1])?,
                test: logical_value(vec![1; n], vec![n, 1])?,
            })
        }
    }
}

fn output_value(output: PartitionOutput) -> BuiltinResult<Value> {
    match output {
        PartitionOutput::Folds(folds) => match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::output_list(Vec::new())),
            Some(1) => Ok(Value::output_list(vec![folds])),
            Some(_) => Err(invalid_argument(
                "crossvalind: KFold returns one output argument",
            )),
            None => Ok(folds),
        },
        PartitionOutput::TrainTest { train, test } => {
            match crate::output_count::current_output_count() {
                Some(0) => Ok(Value::output_list(Vec::new())),
                Some(1) => Ok(Value::output_list(vec![train])),
                Some(2) => Ok(Value::output_list(vec![train, test])),
                Some(_) => Err(invalid_argument(
                    "crossvalind: too many output arguments; maximum is 2",
                )),
                None => Ok(train),
            }
        }
    }
}

fn parse_options(
    rest: crate::call::FunctionArgumentView<'_>,
    n: usize,
    names: &[CrossvalIndOption],
) -> BuiltinResult<Options> {
    debug_assert_eq!(rest.len() / 2, names.len());
    let mut options = Options::default();
    let mut idx = 0usize;
    while idx < rest.len() {
        let value = &rest[idx + 1];
        match names[idx / 2] {
            CrossvalIndOption::Classes => options.classes = Some(classes_from_value(value, n)?),
            CrossvalIndOption::Min => options.min = Some(positive_integer(value, "Min")?),
        }
        idx += 2;
    }
    Ok(options)
}

fn observation_count(value: &Value) -> BuiltinResult<usize> {
    positive_integer(value, "n").map_err(|_| {
        invalid_argument(format!(
            "crossvalind: observation count must be a positive scalar, got {value:?}"
        ))
    })
}

fn validate_n(n: usize) -> BuiltinResult<()> {
    if n == 0 {
        return Err(invalid_argument("crossvalind: n must be positive"));
    }
    if n > MAX_OBSERVATIONS {
        return Err(internal_error(format!(
            "crossvalind: requested partition has {n} observations; limit is {MAX_OBSERVATIONS}"
        )));
    }
    Ok(())
}

fn kfold_indices(
    n: usize,
    k: usize,
    classes: Option<&[Option<PartitionLabel>]>,
    min: Option<usize>,
    turn: &mut random::RandomTurn,
) -> BuiltinResult<Vec<f64>> {
    if k < 2 || k > n {
        return Err(invalid_argument(
            "crossvalind: KFold fold count must be an integer in [2, n]",
        ));
    }
    let mut folds = vec![0.0; n];
    if let Some(classes) = classes {
        validate_class_min(classes, min.unwrap_or(1), k)?;
        for indices in grouped_indices(classes).values() {
            let shuffled = shuffled_indices(indices, turn)?;
            for (offset, row) in shuffled.into_iter().enumerate() {
                folds[row] = (offset % k + 1) as f64;
            }
        }
        for (row, value) in folds.iter_mut().enumerate() {
            if *value == 0.0 && classes[row].is_none() {
                *value = ((row % k) + 1) as f64;
            }
        }
    } else {
        let indices: Vec<usize> = (0..n).collect();
        for (offset, row) in shuffled_indices(&indices, turn)?.into_iter().enumerate() {
            folds[row] = (offset % k + 1) as f64;
        }
    }
    Ok(folds)
}

fn random_test_mask(
    n: usize,
    test_count: usize,
    classes: Option<&[Option<PartitionLabel>]>,
    min: Option<usize>,
    turn: &mut random::RandomTurn,
) -> BuiltinResult<Vec<u8>> {
    if test_count == 0 || test_count >= n {
        return Err(invalid_argument(
            "crossvalind: test partition must select between 1 and n-1 observations",
        ));
    }
    let mut test = vec![0u8; n];
    if let Some(classes) = classes {
        let min = min.unwrap_or(1);
        validate_class_min(classes, min, 2)?;
        let groups = grouped_indices(classes);
        let mut selected = 0usize;
        for indices in groups.values() {
            let target = ((indices.len() * test_count) + (n / 2)) / n;
            let target = target.clamp(min.min(indices.len()), indices.len().saturating_sub(min));
            for row in shuffled_indices(indices, turn)?.into_iter().take(target) {
                if selected < test_count {
                    test[row] = 1;
                    selected += 1;
                }
            }
        }
        if selected < test_count {
            let candidates: Vec<usize> = (0..n).filter(|row| test[*row] == 0).collect();
            for row in shuffled_indices(&candidates, turn)? {
                if selected >= test_count {
                    break;
                }
                test[row] = 1;
                selected += 1;
            }
        }
    } else {
        let indices: Vec<usize> = (0..n).collect();
        for row in shuffled_indices(&indices, turn)?
            .into_iter()
            .take(test_count)
        {
            test[row] = 1;
        }
    }
    Ok(test)
}

fn train_test_from_test_mask(test: Vec<u8>) -> BuiltinResult<PartitionOutput> {
    let train = test
        .iter()
        .map(|flag| u8::from(*flag == 0))
        .collect::<Vec<_>>();
    let shape = vec![test.len(), 1];
    Ok(PartitionOutput::TrainTest {
        train: logical_value(train, shape.clone())?,
        test: logical_value(test, shape)?,
    })
}

fn shuffled_indices(indices: &[usize], turn: &mut random::RandomTurn) -> BuiltinResult<Vec<usize>> {
    let mut out = indices.to_vec();
    if out.len() <= 1 {
        return Ok(out);
    }
    let uniforms = random::generate_uniform(turn, out.len() - 1, NAME)?;
    for (i, u) in uniforms.into_iter().enumerate() {
        let span = out.len() - i;
        let mut offset = (u * span as f64).floor() as usize;
        if offset >= span {
            offset = span - 1;
        }
        out.swap(i, i + offset);
    }
    Ok(out)
}

fn holdout_count(value: &Value, n: usize) -> BuiltinResult<usize> {
    if n < 2 {
        return Err(invalid_argument(
            "crossvalind: HoldOut requires at least two observations",
        ));
    }
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|count| *count > 0 && *count < n)
            .ok_or_else(|| {
                invalid_argument(
                    "crossvalind: HoldOut must be a fraction in (0,1) or an integer in [1,n)",
                )
            });
    }
    let raw = scalar_number(value, "HoldOut")?;
    if raw > 0.0 && raw < 1.0 {
        return Ok(((raw * n as f64).round() as usize).clamp(1, n.saturating_sub(1)));
    }
    if raw.fract().abs() <= EPS && raw >= 1.0 && raw < n as f64 {
        return Ok(raw as usize);
    }
    Err(invalid_argument(
        "crossvalind: HoldOut must be a fraction in (0,1) or an integer in [1,n)",
    ))
}

fn validate_class_min(
    classes: &[Option<PartitionLabel>],
    min: usize,
    buckets: usize,
) -> BuiltinResult<()> {
    if min == 0 {
        return Err(invalid_argument("crossvalind: Min must be positive"));
    }
    for indices in grouped_indices(classes).values() {
        if indices.len() < min.saturating_mul(buckets) {
            return Err(invalid_argument(
                "crossvalind: each class must contain enough observations for the requested Min value",
            ));
        }
    }
    Ok(())
}

fn classes_from_value(value: &Value, n: usize) -> BuiltinResult<Vec<Option<PartitionLabel>>> {
    let labels = match value {
        Value::Tensor(tensor) => numeric_labels(tensor)?,
        Value::Num(number) => vec![PartitionLabel::number(*number)],
        Value::Int(integer) => vec![Some(PartitionLabel::integer(integer))],
        Value::LogicalArray(array) => logical_labels(array)?,
        Value::StringArray(array) => string_array_labels(array)?,
        Value::CharArray(chars) => char_row_labels(chars)?,
        Value::String(text) => vec![PartitionLabel::text(text.clone())],
        Value::Bool(flag) => vec![Some(PartitionLabel::Logical(*flag))],
        other => {
            return Err(invalid_argument(format!(
                "crossvalind: Classes must be a vector, got {other:?}"
            )))
        }
    };
    if labels.len() != n {
        return Err(invalid_argument(
            "crossvalind: Classes must have one label per observation",
        ));
    }
    Ok(labels)
}

fn numeric_labels(tensor: &Tensor) -> BuiltinResult<Vec<Option<PartitionLabel>>> {
    ensure_vector(tensor.shape(), tensor.len(), "Classes")?;
    if let Some(storage) = tensor.integer_storage() {
        return Ok(storage
            .exact_values()
            .iter()
            .map(|value| Some(PartitionLabel::integer(value)))
            .collect());
    }
    Ok(tensor::tensor_values_f64_cow(tensor)
        .iter()
        .map(|value| PartitionLabel::number(*value))
        .collect())
}

fn logical_labels(array: &LogicalArray) -> BuiltinResult<Vec<Option<PartitionLabel>>> {
    ensure_vector(&array.shape, array.data.len(), "Classes")?;
    Ok(array
        .data
        .iter()
        .map(|flag| Some(PartitionLabel::Logical(*flag != 0)))
        .collect())
}

fn string_array_labels(array: &StringArray) -> BuiltinResult<Vec<Option<PartitionLabel>>> {
    ensure_vector(&array.shape, array.data().len(), "Classes")?;
    Ok(array
        .data()
        .iter()
        .map(|text| PartitionLabel::text(text.clone()))
        .collect())
}

fn char_row_labels(chars: &CharArray) -> BuiltinResult<Vec<Option<PartitionLabel>>> {
    let mut labels = Vec::with_capacity(chars.rows);
    for row in 0..chars.rows {
        let start = row * chars.cols;
        let label = decode_utf16_code_units(&chars.data()[start..start + chars.cols])
            .map_err(|error| invalid_utf16("class label", error))?;
        labels.push(Some(PartitionLabel::Text(label.trim_end().into())));
    }
    Ok(labels)
}

fn ensure_vector(shape: &[usize], len: usize, label: &str) -> BuiltinResult<()> {
    if len == 0 {
        return Err(invalid_argument(format!(
            "crossvalind: {label} must be nonempty"
        )));
    }
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(format!(
            "crossvalind: {label} must be a vector"
        )));
    }
    Ok(())
}

fn grouped_indices(labels: &[Option<PartitionLabel>]) -> BTreeMap<PartitionLabel, Vec<usize>> {
    let mut groups = BTreeMap::<PartitionLabel, Vec<usize>>::new();
    for (idx, label) in labels.iter().enumerate() {
        if let Some(label) = label {
            groups.entry(label.clone()).or_default().push(idx);
        }
    }
    groups
}

fn positive_integer(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "crossvalind: {label} must be a positive integer scalar"
                ))
            });
    }
    let raw = scalar_number(value, label)?;
    positive_integer_number(raw, label)
}

fn positive_integer_number(raw: f64, label: &str) -> BuiltinResult<usize> {
    PartitionCountGrammar::CrossvalInd
        .positive_from_float(raw)
        .ok_or_else(|| {
            invalid_argument(format!(
                "crossvalind: {label} must be a positive integer scalar"
            ))
        })
}

fn scalar_number(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(number) => Ok(*number),
        Value::Int(integer) => Ok(integer.to_f64()),
        Value::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(if array.data[0] == 0 { 0.0 } else { 1.0 })
        }
        other => Err(invalid_argument(format!(
            "crossvalind: {label} must be a numeric scalar, got {other:?}"
        ))),
    }
}

fn scalar_text<'a>(value: &'a Value, label: &str) -> BuiltinResult<std::borrow::Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(std::borrow::Cow::Borrowed(text.as_str()))
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(std::borrow::Cow::Borrowed(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            ))
        }
        Value::CharArray(chars) => decode_utf16_code_units(chars.data())
            .map(std::borrow::Cow::Owned)
            .map_err(|error| invalid_utf16(label, error)),
        other => Err(invalid_argument(format!(
            "crossvalind: {label} must be a text scalar, got {other:?}"
        ))),
    }
}

fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

fn tensor_value(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(format!("crossvalind: {err}")))
}

fn logical_value(data: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
    LogicalArray::new(data, shape)
        .map(Value::LogicalArray)
        .map_err(|err| internal_error(format!("crossvalind: {err}")))
}

#[cfg(test)]
mod tests;
