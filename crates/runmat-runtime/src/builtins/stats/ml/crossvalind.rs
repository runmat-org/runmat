//! MATLAB-compatible `crossvalind` builtin.

use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, LogicalArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "crossvalind";
const MAX_OBSERVATIONS: usize = 10_000_000;
const EPS: f64 = 1.0e-12;

const PARAM_METHOD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cross-validation method: KFold, HoldOut, LeaveMOut, or Resubstitution.",
};

const PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation count.",
};

const PARAM_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Fold count, holdout fraction/count, or leave-M-out count.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options including Classes and Min.",
};

const OUTPUT_INDICES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "indices",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fold indices for KFold, or a logical training mask for partition methods.",
}];

const OUTPUT_TRAIN_TEST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "train",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical training-set mask.",
    },
    BuiltinParamDescriptor {
        name: "test",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Logical test-set mask.",
    },
];

const INPUTS_METHOD_N_VALUE: [BuiltinParamDescriptor; 3] = [PARAM_METHOD, PARAM_N, PARAM_VALUE];
const INPUTS_METHOD_N_VALUE_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_METHOD, PARAM_N, PARAM_VALUE, PARAM_OPTIONS];
const INPUTS_METHOD_N: [BuiltinParamDescriptor; 2] = [PARAM_METHOD, PARAM_N];

const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "indices = crossvalind('KFold', n, k)",
        inputs: &INPUTS_METHOD_N_VALUE,
        outputs: &OUTPUT_INDICES,
    },
    BuiltinSignatureDescriptor {
        label: "indices = crossvalind('KFold', n, k, 'Classes', group)",
        inputs: &INPUTS_METHOD_N_VALUE_OPTIONS,
        outputs: &OUTPUT_INDICES,
    },
    BuiltinSignatureDescriptor {
        label: "train = crossvalind('HoldOut', n, p)",
        inputs: &INPUTS_METHOD_N_VALUE,
        outputs: &OUTPUT_INDICES,
    },
    BuiltinSignatureDescriptor {
        label: "[train, test] = crossvalind('HoldOut', n, p)",
        inputs: &INPUTS_METHOD_N_VALUE,
        outputs: &OUTPUT_TRAIN_TEST,
    },
    BuiltinSignatureDescriptor {
        label: "[train, test] = crossvalind('LeaveMOut', n, m)",
        inputs: &INPUTS_METHOD_N_VALUE,
        outputs: &OUTPUT_TRAIN_TEST,
    },
    BuiltinSignatureDescriptor {
        label: "[train, test] = crossvalind('Resubstitution', n)",
        inputs: &INPUTS_METHOD_N,
        outputs: &OUTPUT_TRAIN_TEST,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSSVALIND.INVALID_ARGUMENT",
    identifier: Some("RunMat:crossvalind:InvalidArgument"),
    when: "Inputs, method names, partition values, options, or output counts are malformed.",
    message: "crossvalind: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSSVALIND.INTERNAL",
    identifier: Some("RunMat:crossvalind:Internal"),
    when: "RunMat cannot allocate or construct the requested partition output.",
    message: "crossvalind: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn crossvalind_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Union(vec![Type::tensor(), Type::logical()])
}

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
    builtin_error(message, &ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    builtin_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Method {
    KFold,
    HoldOut,
    LeaveMOut,
    Resubstitution,
}

#[derive(Clone, Debug, Default)]
struct Options {
    classes: Option<Vec<Option<String>>>,
    min: Option<usize>,
}

#[derive(Clone, Debug)]
enum PartitionOutput {
    Folds(Value),
    TrainTest { train: Value, test: Value },
}

#[runtime_builtin(
    name = "crossvalind",
    category = "stats/ml",
    summary = "Generate cross-validation indices and train/test partitions.",
    keywords = "crossvalind,cross validation,kfold,holdout,leavemout,resubstitution,statistics,machine learning",
    type_resolver(crossvalind_type),
    descriptor(self::DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::crossvalind"
)]
async fn crossvalind_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_values(args).await?;
    let output = crossvalind_compute(args)?;
    output_value(output)
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| invalid_argument(format!("crossvalind: {err}")))?,
        );
    }
    Ok(out)
}

fn crossvalind_compute(args: Vec<Value>) -> BuiltinResult<PartitionOutput> {
    if args.len() < 2 {
        return Err(invalid_argument(
            "crossvalind: method and observation count are required",
        ));
    }
    let method = parse_method(&scalar_text(&args[0], "method")?)?;
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
    let options = parse_options(&args[option_start..], n)?;
    validate_n(n)?;

    match method {
        Method::KFold => {
            let k = positive_integer(value.expect("KFold value"), "KFold")?;
            let folds = kfold_indices(n, k, options.classes.as_deref(), options.min)?;
            Ok(PartitionOutput::Folds(tensor_value(folds, vec![n, 1])?))
        }
        Method::HoldOut => {
            let test_count = holdout_count(value.expect("HoldOut value"), n)?;
            let test = random_test_mask(n, test_count, options.classes.as_deref(), options.min)?;
            train_test_from_test_mask(test)
        }
        Method::LeaveMOut => {
            let m = positive_integer(value.expect("LeaveMOut value"), "LeaveMOut")?;
            if m >= n {
                return Err(invalid_argument(
                    "crossvalind: LeaveMOut must leave between 1 and n-1 observations out",
                ));
            }
            let test = random_test_mask(n, m, options.classes.as_deref(), options.min)?;
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
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(1) => Ok(Value::OutputList(vec![folds])),
            Some(_) => Err(invalid_argument(
                "crossvalind: KFold returns one output argument",
            )),
            None => Ok(folds),
        },
        PartitionOutput::TrainTest { train, test } => {
            match crate::output_count::current_output_count() {
                Some(0) => Ok(Value::OutputList(Vec::new())),
                Some(1) => Ok(Value::OutputList(vec![train])),
                Some(2) => Ok(Value::OutputList(vec![train, test])),
                Some(_) => Err(invalid_argument(
                    "crossvalind: too many output arguments; maximum is 2",
                )),
                None => Ok(train),
            }
        }
    }
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::KFold => "KFold",
            Self::HoldOut => "HoldOut",
            Self::LeaveMOut => "LeaveMOut",
            Self::Resubstitution => "Resubstitution",
        }
    }
}

fn parse_method(text: &str) -> BuiltinResult<Method> {
    match canonical(text).as_str() {
        "kfold" => Ok(Method::KFold),
        "holdout" => Ok(Method::HoldOut),
        "leavemout" | "leavem" => Ok(Method::LeaveMOut),
        "resubstitution" | "resub" => Ok(Method::Resubstitution),
        other => Err(invalid_argument(format!(
            "crossvalind: unsupported cross-validation method '{other}'"
        ))),
    }
}

fn parse_options(rest: &[Value], n: usize) -> BuiltinResult<Options> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "crossvalind: name-value options must be supplied in pairs",
        ));
    }
    let mut options = Options::default();
    let mut idx = 0usize;
    while idx < rest.len() {
        let name = scalar_text(&rest[idx], "option name")?;
        let value = &rest[idx + 1];
        match canonical(&name).as_str() {
            "classes" => options.classes = Some(classes_from_value(value, n)?),
            "min" => options.min = Some(positive_integer(value, "Min")?),
            other => {
                return Err(invalid_argument(format!(
                    "crossvalind: unsupported option '{other}'"
                )))
            }
        }
        idx += 2;
    }
    Ok(options)
}

fn observation_count(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Num(number) => positive_integer_number(*number, "n"),
        Value::Int(integer) => positive_integer_number(integer.to_f64(), "n"),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            positive_integer_number(tensor.data[0], "n")
        }
        other => Err(invalid_argument(format!(
            "crossvalind: observation count must be a positive scalar, got {other:?}"
        ))),
    }
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
    classes: Option<&[Option<String>]>,
    min: Option<usize>,
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
            let shuffled = shuffled_indices(indices)?;
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
        for (offset, row) in shuffled_indices(&indices)?.into_iter().enumerate() {
            folds[row] = (offset % k + 1) as f64;
        }
    }
    Ok(folds)
}

fn random_test_mask(
    n: usize,
    test_count: usize,
    classes: Option<&[Option<String>]>,
    min: Option<usize>,
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
            for row in shuffled_indices(indices)?.into_iter().take(target) {
                if selected < test_count {
                    test[row] = 1;
                    selected += 1;
                }
            }
        }
        if selected < test_count {
            let candidates: Vec<usize> = (0..n).filter(|row| test[*row] == 0).collect();
            for row in shuffled_indices(&candidates)? {
                if selected >= test_count {
                    break;
                }
                test[row] = 1;
                selected += 1;
            }
        }
    } else {
        let indices: Vec<usize> = (0..n).collect();
        for row in shuffled_indices(&indices)?.into_iter().take(test_count) {
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

fn shuffled_indices(indices: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut out = indices.to_vec();
    if out.len() <= 1 {
        return Ok(out);
    }
    let uniforms = random::generate_uniform(out.len() - 1, NAME)?;
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

fn validate_class_min(classes: &[Option<String>], min: usize, buckets: usize) -> BuiltinResult<()> {
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

fn classes_from_value(value: &Value, n: usize) -> BuiltinResult<Vec<Option<String>>> {
    let labels = match value {
        Value::Tensor(tensor) => numeric_labels(tensor)?,
        Value::LogicalArray(array) => logical_labels(array)?,
        Value::StringArray(array) => string_array_labels(array)?,
        Value::CharArray(chars) => char_row_labels(chars),
        Value::String(text) => vec![Some(text.clone())],
        Value::Bool(flag) => vec![Some(if *flag { "true" } else { "false" }.to_string())],
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

fn numeric_labels(tensor: &Tensor) -> BuiltinResult<Vec<Option<String>>> {
    ensure_vector(&tensor.shape, tensor.data.len(), "Classes")?;
    Ok(tensor
        .data
        .iter()
        .map(|value| {
            if value.is_nan() {
                None
            } else if value.is_infinite() {
                Some(value.to_string())
            } else if value.fract().abs() <= EPS {
                Some(format!("{value:.0}"))
            } else {
                Some(value.to_string())
            }
        })
        .collect())
}

fn logical_labels(array: &LogicalArray) -> BuiltinResult<Vec<Option<String>>> {
    ensure_vector(&array.shape, array.data.len(), "Classes")?;
    Ok(array
        .data
        .iter()
        .map(|flag| Some(if *flag == 0 { "false" } else { "true" }.to_string()))
        .collect())
}

fn string_array_labels(array: &StringArray) -> BuiltinResult<Vec<Option<String>>> {
    ensure_vector(&array.shape, array.data.len(), "Classes")?;
    Ok(array.data.iter().map(|text| Some(text.clone())).collect())
}

fn char_row_labels(chars: &CharArray) -> Vec<Option<String>> {
    let mut labels = Vec::with_capacity(chars.rows);
    for row in 0..chars.rows {
        let mut label = String::with_capacity(chars.cols);
        for col in 0..chars.cols {
            label.push(chars.data[row + col * chars.rows]);
        }
        labels.push(Some(label.trim_end().to_string()));
    }
    labels
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

fn grouped_indices(labels: &[Option<String>]) -> BTreeMap<String, Vec<usize>> {
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (idx, label) in labels.iter().enumerate() {
        if let Some(label) = label {
            groups.entry(label.clone()).or_default().push(idx);
        }
    }
    groups
}

fn positive_integer(value: &Value, label: &str) -> BuiltinResult<usize> {
    let raw = scalar_number(value, label)?;
    positive_integer_number(raw, label)
}

fn positive_integer_number(raw: f64, label: &str) -> BuiltinResult<usize> {
    if raw.is_finite() && raw.fract().abs() <= EPS && raw >= 1.0 && raw <= usize::MAX as f64 {
        Ok(raw as usize)
    } else {
        Err(invalid_argument(format!(
            "crossvalind: {label} must be a positive integer scalar"
        )))
    }
}

fn scalar_number(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(number) => Ok(*number),
        Value::Int(integer) => Ok(integer.to_f64()),
        Value::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(if array.data[0] == 0 { 0.0 } else { 1.0 })
        }
        other => Err(invalid_argument(format!(
            "crossvalind: {label} must be a numeric scalar, got {other:?}"
        ))),
    }
}

fn scalar_text(value: &Value, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
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
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn call(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = crate::output_count::push_output_count(outputs);
        block_on(crossvalind_builtin(args))
    }

    fn logical_data(value: Value) -> Vec<u8> {
        match value {
            Value::LogicalArray(array) => array.data,
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.data,
            Value::Num(number) => vec![number],
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn output_list(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn holdout_default_output_is_training_mask() {
        let _rng_guard = random::test_guard();
        random::set_seed(2026).expect("seed");
        let train = logical_data(
            call(
                vec![Value::from("HoldOut"), Value::Num(10.0), Value::Num(0.3)],
                None,
            )
            .expect("crossvalind"),
        );
        assert_eq!(train.len(), 10);
        assert_eq!(train.iter().filter(|flag| **flag != 0).count(), 7);
    }

    #[test]
    fn holdout_two_outputs_are_complements() {
        let _rng_guard = random::test_guard();
        random::set_seed(7).expect("seed");
        let outputs = output_list(
            call(
                vec![Value::from("holdout"), Value::Num(12.0), Value::Num(4.0)],
                Some(2),
            )
            .expect("crossvalind"),
        );
        let train = logical_data(outputs[0].clone());
        let test = logical_data(outputs[1].clone());
        assert_eq!(test.iter().filter(|flag| **flag != 0).count(), 4);
        for (a, b) in train.iter().zip(test.iter()) {
            assert_ne!(*a != 0, *b != 0);
        }
    }

    #[test]
    fn kfold_returns_fold_indices_covering_all_folds() {
        let _rng_guard = random::test_guard();
        random::set_seed(11).expect("seed");
        let folds = tensor_data(
            call(
                vec![Value::from("KFold"), Value::Num(9.0), Value::Num(3.0)],
                None,
            )
            .expect("crossvalind"),
        );
        assert_eq!(folds.len(), 9);
        for fold in 1..=3 {
            assert!(folds
                .iter()
                .any(|value| (*value - fold as f64).abs() <= EPS));
        }
    }

    #[test]
    fn classes_option_stratifies_holdout() {
        let _rng_guard = random::test_guard();
        random::set_seed(13).expect("seed");
        let classes =
            Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], vec![8, 1]).expect("classes");
        let outputs = output_list(
            call(
                vec![
                    Value::from("HoldOut"),
                    Value::Num(8.0),
                    Value::Num(0.5),
                    Value::from("Classes"),
                    Value::Tensor(classes),
                ],
                Some(2),
            )
            .expect("crossvalind"),
        );
        let test = logical_data(outputs[1].clone());
        assert_eq!(test[..4].iter().filter(|flag| **flag != 0).count(), 2);
        assert_eq!(test[4..].iter().filter(|flag| **flag != 0).count(), 2);
    }

    #[test]
    fn resubstitution_returns_all_true_train_and_test() {
        let outputs = output_list(
            call(
                vec![Value::from("Resubstitution"), Value::Num(4.0)],
                Some(2),
            )
            .expect("crossvalind"),
        );
        assert_eq!(logical_data(outputs[0].clone()), vec![1, 1, 1, 1]);
        assert_eq!(logical_data(outputs[1].clone()), vec![1, 1, 1, 1]);
    }

    #[test]
    fn invalid_forms_are_rejected() {
        let err = call(
            vec![Value::from("KFold"), Value::Num(5.0), Value::Num(2.0)],
            Some(2),
        )
        .expect_err("too many KFold outputs");
        assert_eq!(err.identifier(), Some("RunMat:crossvalind:InvalidArgument"));

        let err = call(
            vec![Value::from("HoldOut"), Value::Num(5.0), Value::Num(5.0)],
            None,
        )
        .expect_err("invalid holdout");
        assert_eq!(err.identifier(), Some("RunMat:crossvalind:InvalidArgument"));
    }
}
