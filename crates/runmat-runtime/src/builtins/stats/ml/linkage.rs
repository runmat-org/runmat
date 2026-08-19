//! Hierarchical agglomerative clustering compatibility surface.

use std::cmp::Ordering;
use std::collections::HashMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "linkage";
const EPS: f64 = 1.0e-12;
const MAX_OBSERVATIONS: usize = 700;
const MAX_CONDENSED_DISTANCES: usize = MAX_OBSERVATIONS * (MAX_OBSERVATIONS - 1) / 2;

const INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "linkage-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "linkage with typed-integer observation or distance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LinkageIntegerDataExtension"),
};

const INTEGER_DISTANCE_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "linkage-integer-distance-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "linkage with typed-integer pdist distance parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LinkageIntegerDistanceParameterExtension"),
    };

const EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "linkage-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "linkage host fallback for explicit gpuArray inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LinkageExplicitGpuInputExtension"),
};

pub const LINKAGE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_DATA_EXTENSION,
    INTEGER_DISTANCE_PARAMETER_EXTENSION,
    EXPLICIT_GPU_INPUT_EXTENSION,
];

const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X or y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer observations or condensed distances are gated before gather and checked for exact binary64 representation.",
    }];

const INTEGER_DISTANCE_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "pdist DistParameter",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer Minkowski exponents, standardized scales, and covariance entries are checked before entering floating distance arithmetic.",
    }];

pub const LINKAGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Z = linkage(integer_X_or_y, ___)",
        inputs: &INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer data crosses a checked binary64 distance boundary. Z remains a homogeneous floating matrix: columns 1 and 2 are integer-valued cluster labels and column 3 is distance. Public documentation does not resolve single/double propagation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Z = linkage(X, method, {metric, integer_DistParameter})",
        inputs: &INTEGER_DISTANCE_PARAMETER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Distance parameters are numeric algorithm inputs, not structural integer controls; lossy integer-to-double conversion rejects.",
    },
];

const OUTPUT_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hierarchical cluster tree with one merge per row.",
}];

const PARAM_X_OR_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X_or_Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix X or condensed distance vector Y.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Linkage method, distance metric, pdist metric parameters, and SaveMemory option.",
};

const INPUTS_BASIC: [BuiltinParamDescriptor; 1] = [PARAM_X_OR_Y];
const INPUTS_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X_OR_Y, PARAM_OPTIONS];

const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X)",
        inputs: &INPUTS_BASIC,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X, method)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X, method, metric)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X, method, metric, 'savememory', value)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X, method, pdist_inputs)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(Y, method)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAGE.INVALID_ARGUMENT",
    identifier: Some("RunMat:linkage:InvalidArgument"),
    when: "Inputs, linkage methods, distance metrics, or name-value options are malformed.",
    message: "linkage: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAGE.INTERNAL",
    identifier: Some("RunMat:linkage:Internal"),
    when: "RunMat cannot allocate or construct the linkage output.",
    message: "linkage: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const LINKAGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn linkage_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![None, Some(3)]),
    }
}

fn linkage_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    linkage_error(message, &ERROR_INVALID_ARGUMENT)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    linkage_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinkageMethod {
    Single,
    Complete,
    Average,
    Weighted,
    Centroid,
    Median,
    Ward,
}

#[derive(Clone, Debug)]
struct LinkageOptions {
    method: LinkageMethod,
    metric_args: Vec<Value>,
    savememory: Option<bool>,
}

#[runtime_builtin(
    name = "linkage",
    category = "stats/ml",
    summary = "Construct a hierarchical agglomerative clustering tree.",
    keywords = "linkage,hierarchical,cluster,clustering,dendrogram,statistics,machine learning",
    type_resolver(linkage_type),
    descriptor(crate::builtins::stats::ml::linkage::LINKAGE_DESCRIPTOR),
    extensions(crate::builtins::stats::ml::linkage::LINKAGE_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::ml::linkage::LINKAGE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::ml::linkage"
)]
async fn linkage_builtin(input: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_extensions(&input, &rest)?;
    let tensor = value_to_tensor(input).await?;
    let options = parse_options(rest)?;
    ensure_exact_metric_integer_boundaries(&options.metric_args).await?;
    let is_condensed_vector = tensor.rows == 1 || tensor.shape.len() == 1;
    let (observations, distances) = if is_condensed_vector {
        if !options.metric_args.is_empty() {
            return Err(invalid(
                "linkage: condensed distance vector input does not accept a distance metric",
            ));
        }
        if options.savememory == Some(true) {
            return Err(invalid(
                "linkage: SaveMemory 'on' is only supported for observation matrix input",
            ));
        }
        let distances = tensor::tensor_values_f64(&tensor);
        let observations = triangular_observation_count(distances.len()).ok_or_else(|| {
            invalid("linkage: condensed distance vector length must be n*(n-1)/2")
        })?;
        validate_distances(&distances)?;
        if options.method.requires_euclidean_distances() {
            validate_euclidean_condensed(&distances, observations)?;
        }
        (observations, distances)
    } else {
        if tensor.shape.len() > 2 || tensor.rows < 2 {
            return Err(invalid(
                "linkage: X must be a numeric matrix with at least two observations in rows",
            ));
        }
        validate_savememory_for_matrix(&options)?;
        validate_observation_count(tensor.rows)?;
        let distances = crate::builtins::stats::ml::distance::condensed_distances_from_metric_args(
            NAME,
            &tensor,
            options.metric_args,
        )
        .await?;
        let distances = distances.materialize_f64();
        validate_distances(&distances)?;
        (tensor.rows, distances)
    };
    let output = compute_linkage(observations, distances, options.method)?;
    Ok(Value::Tensor(output))
}

async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("linkage: {err}")))?;
    let tensor = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid(format!("linkage: {err}")))?;
    if tensor.shape.len() > 2 {
        return Err(invalid(
            "linkage: input must be a numeric vector or 2-D matrix",
        ));
    }
    ensure_exact_integer_tensor(&tensor, "X or y")?;
    Ok(tensor)
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn contains_typed_integer(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data.iter().any(contains_typed_integer),
        Value::Struct(value) => value.fields.values().any(contains_typed_integer),
        Value::Object(value) => value.properties.values().any(contains_typed_integer),
        Value::Closure(value) => value.captures.iter().any(contains_typed_integer),
        Value::OutputList(values) => values.iter().any(contains_typed_integer),
        _ => is_typed_integer(value),
    }
}

fn contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data.iter().any(contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(contains_explicit_gpu),
        Value::Object(value) => value.properties.values().any(contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(contains_explicit_gpu),
        _ => false,
    }
}

fn has_typed_integer_distance_parameter(rest: &[Value]) -> bool {
    if rest
        .iter()
        .any(|value| matches!(value, Value::Cell(_)) && contains_typed_integer(value))
    {
        return true;
    }
    rest.windows(2).any(|pair| {
        keyword_of(&pair[0]).is_some_and(|metric| {
            matches!(
                metric.to_ascii_lowercase().as_str(),
                "minkowski"
                    | "seuclidean"
                    | "standardizedeuclidean"
                    | "standardized euclidean"
                    | "mahalanobis"
                    | "mahal"
            )
        }) && contains_typed_integer(&pair[1])
    })
}

fn ensure_extensions(input: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if is_typed_integer(input) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_DATA_EXTENSION, NAME)?;
    }
    if has_typed_integer_distance_parameter(rest) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_DISTANCE_PARAMETER_EXTENSION,
            NAME,
        )?;
    }
    if contains_explicit_gpu(input) || rest.iter().any(contains_explicit_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPLICIT_GPU_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn ensure_exact_integer_tensor(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    if storage
        .exact_values()
        .iter()
        .any(|integer| !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(integer))
    {
        return Err(invalid(format!(
            "linkage: integer {role} values must be exactly representable as double"
        )));
    }
    Ok(())
}

async fn ensure_exact_metric_integer_boundaries(values: &[Value]) -> BuiltinResult<()> {
    for value in values {
        if !contains_typed_integer(value) {
            continue;
        }
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(|err| invalid(format!("linkage: {err}")))?;
        match gathered {
            Value::Int(integer) => {
                if !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&integer) {
                    return Err(invalid(
                        "linkage: integer distance parameter values must be exactly representable as double",
                    ));
                }
            }
            Value::Tensor(tensor) => {
                ensure_exact_integer_tensor(&tensor, "distance parameter")?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_options(args: Vec<Value>) -> BuiltinResult<LinkageOptions> {
    let mut method = LinkageMethod::Single;
    let mut metric_args = Vec::new();
    let mut savememory = None;
    let mut idx = 0usize;

    if let Some(first) = args.first() {
        if let Some(text) = keyword_of(first) {
            let lower = text.to_ascii_lowercase();
            if lower != "savememory" {
                method = parse_method(&lower)?;
                idx = 1;
            }
        } else {
            return Err(invalid("linkage: method must be text"));
        }
    }

    while idx < args.len() {
        if let Some(keyword) = keyword_of(&args[idx]) {
            if keyword.eq_ignore_ascii_case("savememory") {
                if idx + 1 >= args.len() {
                    return Err(invalid("linkage: SaveMemory must be paired with a value"));
                }
                if savememory.is_some() {
                    return Err(invalid("linkage: SaveMemory can only be specified once"));
                }
                savememory = Some(parse_savememory(&args[idx + 1])?);
                idx += 2;
                continue;
            }
        }

        if let Value::Cell(cell) = &args[idx] {
            if !metric_args.is_empty() {
                return Err(invalid(
                    "linkage: pdist input cell must be the complete distance metric argument",
                ));
            }
            metric_args.extend(cell.data.iter().cloned());
            idx += 1;
        } else {
            metric_args.push(args[idx].clone());
            idx += 1;
        }
    }

    Ok(LinkageOptions {
        method,
        metric_args,
        savememory,
    })
}

fn parse_method(text: &str) -> BuiltinResult<LinkageMethod> {
    match text {
        "single" | "nearest" => Ok(LinkageMethod::Single),
        "complete" | "farthest" => Ok(LinkageMethod::Complete),
        "average" | "upgma" => Ok(LinkageMethod::Average),
        "weighted" | "wpgma" => Ok(LinkageMethod::Weighted),
        "centroid" | "upgmc" => Ok(LinkageMethod::Centroid),
        "median" | "wpgmc" => Ok(LinkageMethod::Median),
        "ward" | "wardlinkage" => Ok(LinkageMethod::Ward),
        other => Err(invalid(format!(
            "linkage: unsupported linkage method '{other}'"
        ))),
    }
}

impl LinkageMethod {
    fn requires_euclidean_distances(self) -> bool {
        matches!(
            self,
            LinkageMethod::Centroid | LinkageMethod::Median | LinkageMethod::Ward
        )
    }
}

fn parse_savememory(value: &Value) -> BuiltinResult<bool> {
    if let Some(text) = keyword_of(value) {
        match text.to_ascii_lowercase().as_str() {
            "on" => return Ok(true),
            "off" => return Ok(false),
            other => {
                return Err(invalid(format!(
                    "linkage: SaveMemory value must be 'on' or 'off', got '{other}'"
                )))
            }
        }
    }
    Err(invalid("linkage: SaveMemory value must be 'on' or 'off'"))
}

fn validate_savememory_for_matrix(options: &LinkageOptions) -> BuiltinResult<()> {
    if options.savememory != Some(true) {
        return Ok(());
    }
    if !options.method.requires_euclidean_distances() {
        return Err(invalid(
            "linkage: SaveMemory 'on' is only valid for centroid, median, or ward linkage",
        ));
    }
    if !metric_is_default_or_euclidean(&options.metric_args) {
        return Err(invalid(
            "linkage: SaveMemory 'on' requires the Euclidean distance metric",
        ));
    }
    Ok(())
}

fn metric_is_default_or_euclidean(args: &[Value]) -> bool {
    if args.is_empty() {
        return true;
    }
    args.len() == 1
        && keyword_of(&args[0])
            .map(|text| text.eq_ignore_ascii_case("euclidean"))
            .unwrap_or(false)
}

fn triangular_observation_count(len: usize) -> Option<usize> {
    let discriminant = len.checked_mul(8)?.checked_add(1)?;
    let root = (discriminant as f64).sqrt() as usize;
    if root.checked_mul(root)? != discriminant || !(root + 1).is_multiple_of(2) {
        return None;
    }
    let n = root.div_ceil(2);
    if n >= 2 {
        Some(n)
    } else {
        None
    }
}

fn validate_observation_count(n: usize) -> BuiltinResult<()> {
    let condensed = n
        .checked_mul(n.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| internal("linkage: distance count overflow"))?;
    if condensed > MAX_CONDENSED_DISTANCES {
        return Err(invalid(format!(
            "linkage: too many observations ({n}) for an in-memory clustering tree"
        )));
    }
    Ok(())
}

fn validate_distances(values: &[f64]) -> BuiltinResult<()> {
    if values.len() > MAX_CONDENSED_DISTANCES {
        return Err(invalid(
            "linkage: condensed distance vector is too large for an in-memory clustering tree",
        ));
    }
    for value in values {
        if !value.is_finite() {
            return Err(invalid("linkage: distances must be finite"));
        }
        if *value < 0.0 {
            return Err(invalid("linkage: distances must be nonnegative"));
        }
    }
    Ok(())
}

fn validate_euclidean_condensed(values: &[f64], observations: usize) -> BuiltinResult<()> {
    let mut squared = vec![0.0; observations * observations];
    let mut offset = 0usize;
    for col in 0..observations {
        for row in (col + 1)..observations {
            let value = values[offset] * values[offset];
            squared[row * observations + col] = value;
            squared[col * observations + row] = value;
            offset += 1;
        }
    }

    let row_means = (0..observations)
        .map(|row| {
            (0..observations)
                .map(|col| squared[row * observations + col])
                .sum::<f64>()
                / observations as f64
        })
        .collect::<Vec<_>>();
    let col_means = (0..observations)
        .map(|col| {
            (0..observations)
                .map(|row| squared[row * observations + col])
                .sum::<f64>()
                / observations as f64
        })
        .collect::<Vec<_>>();
    let grand_mean = row_means.iter().sum::<f64>() / observations as f64;
    let mut gram = vec![0.0; observations * observations];
    for row in 0..observations {
        for col in 0..observations {
            gram[row * observations + col] = -0.5
                * (squared[row * observations + col] - row_means[row] - col_means[col]
                    + grand_mean);
        }
    }
    ensure_psd(&gram, observations).map_err(|_| {
        invalid("linkage: centroid, median, and ward require Euclidean condensed distances")
    })
}

fn ensure_psd(matrix: &[f64], n: usize) -> Result<(), ()> {
    let max_diag = (0..n)
        .map(|i| matrix[i * n + i].abs())
        .fold(0.0_f64, f64::max);
    let tol = 1.0e-10 * max_diag.max(1.0) * n as f64;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum < -tol {
                    return Err(());
                }
                l[i * n + j] = sum.max(0.0).sqrt();
            } else if l[j * n + j] > tol {
                l[i * n + j] = sum / l[j * n + j];
            } else if sum.abs() > tol {
                return Err(());
            }
        }
    }
    Ok(())
}

fn compute_linkage(
    observations: usize,
    condensed: Vec<f64>,
    method: LinkageMethod,
) -> BuiltinResult<Tensor> {
    validate_observation_count(observations)?;
    let expected = observations
        .checked_mul(observations.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| internal("linkage: output size overflow"))?;
    if condensed.len() != expected {
        return Err(invalid(
            "linkage: condensed distance vector has the wrong length",
        ));
    }

    let mut distances = HashMap::new();
    distances
        .try_reserve(expected.saturating_add(observations))
        .map_err(|_| internal("linkage: failed to allocate cluster distance workspace"))?;
    let mut offset = 0usize;
    for col in 0..observations {
        for row in (col + 1)..observations {
            distances.insert(cluster_key(row, col), condensed[offset]);
            offset += 1;
        }
    }

    let mut active = (0..observations).collect::<Vec<_>>();
    let mut sizes = vec![1usize; observations];
    let rows = observations - 1;
    let mut output = vec![0.0; rows * 3];

    for step in 0..rows {
        let (left, right, merge_distance) = find_next_merge(&active, &distances)?;
        let left_size = sizes[left];
        let right_size = sizes[right];
        output[step] = (left.min(right) + 1) as f64;
        output[rows + step] = (left.max(right) + 1) as f64;
        output[2 * rows + step] = merge_distance;

        let new_id = sizes.len();
        let new_size = left_size
            .checked_add(right_size)
            .ok_or_else(|| internal("linkage: cluster size overflow"))?;
        let mut new_distances = Vec::with_capacity(active.len().saturating_sub(2));
        for &other in &active {
            if other == left || other == right {
                continue;
            }
            let d_left = distance_between(&distances, left, other)?;
            let d_right = distance_between(&distances, right, other)?;
            let updated = update_distance(
                method,
                d_left,
                d_right,
                merge_distance,
                left_size,
                right_size,
                sizes[other],
            )?;
            new_distances.push((other, updated));
        }

        sizes.push(new_size);
        for &other in &active {
            if other != left {
                distances.remove(&cluster_key(left, other));
            }
            if other != right {
                distances.remove(&cluster_key(right, other));
            }
        }
        distances.remove(&cluster_key(left, right));
        active.retain(|id| *id != left && *id != right);
        for (other, distance) in new_distances {
            distances.insert(cluster_key(new_id, other), distance);
        }
        active.push(new_id);
        active.sort_unstable();
    }

    Tensor::new(output, vec![rows, 3]).map_err(|err| internal(format!("linkage: {err}")))
}

fn find_next_merge(
    active: &[usize],
    distances: &HashMap<(usize, usize), f64>,
) -> BuiltinResult<(usize, usize, f64)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let left = active[i];
            let right = active[j];
            let distance = distance_between(distances, left, right)?;
            let candidate = (left.min(right), left.max(right), distance);
            if let Some(current) = best {
                if compare_merge(candidate, current) == Ordering::Less {
                    best = Some(candidate);
                }
            } else {
                best = Some(candidate);
            }
        }
    }
    best.ok_or_else(|| internal("linkage: no active cluster pair found"))
}

fn compare_merge(a: (usize, usize, f64), b: (usize, usize, f64)) -> Ordering {
    compare_distance(a.2, b.2)
        .then_with(|| a.0.cmp(&b.0))
        .then_with(|| a.1.cmp(&b.1))
}

fn compare_distance(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

fn update_distance(
    method: LinkageMethod,
    d_left: f64,
    d_right: f64,
    d_join: f64,
    left_size: usize,
    right_size: usize,
    other_size: usize,
) -> BuiltinResult<f64> {
    let nl = left_size as f64;
    let nr = right_size as f64;
    let nk = other_size as f64;
    match method {
        LinkageMethod::Single => {
            if d_left.is_nan() || d_right.is_nan() {
                Ok(f64::NAN)
            } else {
                Ok(d_left.min(d_right))
            }
        }
        LinkageMethod::Complete => {
            if d_left.is_nan() || d_right.is_nan() {
                Ok(f64::NAN)
            } else {
                Ok(d_left.max(d_right))
            }
        }
        LinkageMethod::Average => Ok((nl * d_left + nr * d_right) / (nl + nr)),
        LinkageMethod::Weighted => Ok(0.5 * (d_left + d_right)),
        LinkageMethod::Centroid => nonnegative_sqrt(
            (nl * d_left.powi(2) + nr * d_right.powi(2)) / (nl + nr)
                - (nl * nr * d_join.powi(2)) / (nl + nr).powi(2),
        ),
        LinkageMethod::Median => {
            nonnegative_sqrt(0.5 * d_left.powi(2) + 0.5 * d_right.powi(2) - 0.25 * d_join.powi(2))
        }
        LinkageMethod::Ward => nonnegative_sqrt(
            ((nl + nk) * d_left.powi(2) + (nr + nk) * d_right.powi(2) - nk * d_join.powi(2))
                / (nl + nr + nk),
        ),
    }
}

fn nonnegative_sqrt(value: f64) -> BuiltinResult<f64> {
    if value.is_nan() {
        Ok(f64::NAN)
    } else if value < 0.0 && value.abs() <= EPS {
        Ok(0.0)
    } else if value < 0.0 {
        Err(invalid(
            "linkage: Euclidean linkage update produced an invalid negative distance",
        ))
    } else {
        Ok(value.sqrt())
    }
}

fn distance_between(
    distances: &HashMap<(usize, usize), f64>,
    a: usize,
    b: usize,
) -> BuiltinResult<f64> {
    distances
        .get(&cluster_key(a, b))
        .copied()
        .ok_or_else(|| internal("linkage: missing inter-cluster distance"))
}

fn cluster_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_value::{CellArray, CharArray, IntegerStorage};

    use super::*;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).unwrap()
    }

    fn tensor_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(tensor(data, rows, cols))
    }

    fn typed_integer_tensor_value(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-8,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_matrix_close(tensor: &Tensor, rows: &[[f64; 3]]) {
        assert_eq!(tensor.shape, vec![rows.len(), 3]);
        for (row_idx, expected) in rows.iter().enumerate() {
            for col in 0..3 {
                assert_close(
                    tensor.materialize_f64()[col * rows.len() + row_idx],
                    expected[col],
                );
            }
        }
    }

    #[test]
    fn linkage_complete_accepts_condensed_distance_vector() {
        let y = tensor_value(vec![1.0, 4.0, 6.0, 5.0, 7.0, 2.0], 1, 6);
        let Value::Tensor(z) =
            block_on(linkage_builtin(y, vec![Value::String("complete".into())])).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_matrix_close(&z, &[[1.0, 2.0, 1.0], [3.0, 4.0, 2.0], [5.0, 6.0, 7.0]]);
    }

    #[test]
    fn linkage_condensed_vector_reads_typed_integer_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let y = typed_integer_tensor_value(IntegerStorage::I16(vec![1, 4, 6, 5, 7, 2]), vec![1, 6]);
        let Value::Tensor(z) =
            block_on(linkage_builtin(y, vec![Value::String("complete".into())])).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_matrix_close(&z, &[[1.0, 2.0, 1.0], [3.0, 4.0, 2.0], [5.0, 6.0, 7.0]]);
    }

    #[test]
    fn linkage_single_computes_from_observation_matrix() {
        let x = tensor_value(
            vec![
                0.0, 3.0, 4.0, 0.0, // x column
                0.0, 4.0, 0.0, 2.0, // y column
            ],
            4,
            2,
        );
        let Value::Tensor(z) =
            block_on(linkage_builtin(x, vec![Value::String("single".into())])).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_matrix_close(
            &z,
            &[
                [1.0, 4.0, 2.0],
                [2.0, 5.0, 13.0_f64.sqrt()],
                [3.0, 6.0, 4.0],
            ],
        );
    }

    #[test]
    fn linkage_column_vector_is_observation_matrix_not_condensed_vector() {
        let x = tensor_value(vec![0.0, 1.0, 2.0], 3, 1);
        let Value::Tensor(z) =
            block_on(linkage_builtin(x, vec![Value::String("single".into())])).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_matrix_close(&z, &[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]);
    }

    #[test]
    fn linkage_average_uses_pdist_metric_cell() {
        let x = tensor_value(
            vec![
                0.0, 3.0, 4.0, 0.0, // x column
                0.0, 4.0, 0.0, 2.0, // y column
            ],
            4,
            2,
        );
        let metric = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(CharArray::new_row("minkowski")),
                    Value::Num(3.0),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let Value::Tensor(z) = block_on(linkage_builtin(
            x,
            vec![Value::String("average".into()), metric],
        ))
        .unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(z.shape, vec![3, 3]);
        assert_close(z.materialize_f64()[6], 2.0);
    }

    #[test]
    fn linkage_ward_matches_singleton_euclidean_first_merge() {
        let x = tensor_value(
            vec![
                0.0, 3.0, 4.0, 0.0, // x column
                0.0, 4.0, 0.0, 2.0, // y column
            ],
            4,
            2,
        );
        let Value::Tensor(z) =
            block_on(linkage_builtin(x, vec![Value::String("ward".into())])).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_eq!(z.shape, vec![3, 3]);
        assert_close(z.materialize_f64()[6], 2.0);
    }

    #[test]
    fn linkage_rejects_bad_distance_vector_length() {
        let err =
            block_on(linkage_builtin(tensor_value(vec![1.0, 2.0], 1, 2), vec![])).unwrap_err();
        assert!(err.to_string().contains("condensed distance vector length"));
    }

    #[test]
    fn linkage_rejects_metric_for_condensed_vector() {
        let err = block_on(linkage_builtin(
            tensor_value(vec![1.0, 4.0, 6.0], 1, 3),
            vec![
                Value::String("single".into()),
                Value::String("cityblock".into()),
            ],
        ))
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("does not accept a distance metric"));
    }

    #[test]
    fn linkage_rejects_nonfinite_distances() {
        let err = block_on(linkage_builtin(
            tensor_value(vec![1.0, f64::INFINITY, 2.0], 1, 3),
            vec![],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("distances must be finite"));
    }

    #[test]
    fn linkage_rejects_noneuclidean_condensed_distances_for_ward_family() {
        let err = block_on(linkage_builtin(
            tensor_value(vec![1.0, 1.0, 3.0], 1, 3),
            vec![Value::String("ward".into())],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("require Euclidean"));
    }

    #[test]
    fn linkage_savememory_requires_text_and_euclidean_ward_family() {
        let x = tensor_value(vec![0.0, 1.0, 2.0], 3, 1);
        let err = block_on(linkage_builtin(
            x.clone(),
            vec![
                Value::String("single".into()),
                Value::String("euclidean".into()),
                Value::String("savememory".into()),
                Value::String("on".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("centroid, median, or ward"));

        let err = block_on(linkage_builtin(
            x.clone(),
            vec![
                Value::String("ward".into()),
                Value::String("cityblock".into()),
                Value::String("savememory".into()),
                Value::String("on".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("Euclidean distance metric"));

        let err = block_on(linkage_builtin(
            x,
            vec![
                Value::String("ward".into()),
                Value::String("euclidean".into()),
                Value::String("savememory".into()),
                Value::Bool(true),
            ],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("must be 'on' or 'off'"));
    }

    #[test]
    fn linkage_integer_roles_are_independently_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let data_error = block_on(linkage_builtin(
            typed_integer_tensor_value(IntegerStorage::U16(vec![1, 2, 3]), vec![1, 3]),
            vec![],
        ))
        .unwrap_err();
        assert_eq!(
            data_error.identifier(),
            Some("RunMat:compatibility:LinkageIntegerDataExtension")
        );

        let metric = Value::Cell(
            CellArray::new(
                vec![
                    Value::from("minkowski"),
                    Value::Int(runmat_value::IntValue::U16(3)),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let parameter_error = block_on(linkage_builtin(
            tensor_value(vec![0.0, 1.0, 2.0], 3, 1),
            vec![Value::from("average"), metric],
        ))
        .unwrap_err();
        assert_eq!(
            parameter_error.identifier(),
            Some("RunMat:compatibility:LinkageIntegerDistanceParameterExtension")
        );
    }

    #[test]
    fn linkage_rejects_lossy_integer_data_and_distance_parameters() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let data_error = block_on(linkage_builtin(
            typed_integer_tensor_value(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
            vec![],
        ))
        .unwrap_err();
        assert!(data_error
            .message()
            .contains("exactly representable as double"));

        let metric = Value::Cell(
            CellArray::new(
                vec![
                    Value::from("minkowski"),
                    Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let parameter_error = block_on(linkage_builtin(
            tensor_value(vec![0.0, 1.0, 2.0], 3, 1),
            vec![Value::from("average"), metric],
        ))
        .unwrap_err();
        assert!(parameter_error
            .message()
            .contains("exactly representable as double"));
    }

    #[test]
    fn linkage_explicit_gpu_fallback_is_gated_before_download() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 3],
            device_id: 0,
            buffer_id: 9_426_001,
            descriptor: Default::default(),
        };
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let error =
            block_on(linkage_builtin(Value::GpuTensor(handle.clone()), vec![])).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:LinkageExplicitGpuInputExtension")
        );
        runmat_accelerate_api::clear_handle_metadata(&handle);
    }

    #[test]
    fn linkage_automatic_residency_gathers_transparently() {
        use crate::builtins::common::{gpu_helpers, test_support};

        test_support::with_test_provider(|provider| {
            let input = tensor(vec![0.0, 1.0, 2.0], 3, 1);
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let Value::Tensor(output) = block_on(linkage_builtin(Value::GpuTensor(handle), vec![]))
                .expect("automatic residency may gather transparently")
            else {
                panic!("linkage remains a host implementation");
            };
            assert_matrix_close(&output, &[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]);
        });
    }
}
