//! Data normalization helpers.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric, logical, complex, or gpuArray data.",
};

const PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension to operate along.",
};

const PARAM_METHOD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"zscore\""),
    description: "Normalization method: zscore, norm, scale, range, center, or medianiqr.",
};

const PARAM_METHODTYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "methodtype",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Method subtype or explicit center/scale/range parameter.",
};

const PARAM_CENTER_KEYWORD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "center_keyword",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"center\""),
    description: "Literal center method selector.",
};

const PARAM_CENTER_TYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "centertype",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Centering mode or explicit center values.",
};

const PARAM_SCALE_KEYWORD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scale_keyword",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"scale\""),
    description: "Literal scale method selector.",
};

const PARAM_SCALE_TYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scaletype",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scaling mode or explicit scale values.",
};

const OUTPUT_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Normalized data.",
}];

const OUTPUT_NCS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized data.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Centering values.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Scale values.",
    },
];

const INPUTS_A: [BuiltinParamDescriptor; 1] = [PARAM_A];
const INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_DIM];
const INPUTS_A_METHOD: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_METHOD];
const INPUTS_A_DIM_METHOD: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_DIM, PARAM_METHOD];
const INPUTS_A_METHOD_TYPE: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_METHOD, PARAM_METHODTYPE];
const INPUTS_A_DIM_METHOD_TYPE: [BuiltinParamDescriptor; 4] =
    [PARAM_A, PARAM_DIM, PARAM_METHOD, PARAM_METHODTYPE];
const INPUTS_A_CENTER_SCALE: [BuiltinParamDescriptor; 5] = [
    PARAM_A,
    PARAM_CENTER_KEYWORD,
    PARAM_CENTER_TYPE,
    PARAM_SCALE_KEYWORD,
    PARAM_SCALE_TYPE,
];
const INPUTS_A_DIM_CENTER_SCALE: [BuiltinParamDescriptor; 6] = [
    PARAM_A,
    PARAM_DIM,
    PARAM_CENTER_KEYWORD,
    PARAM_CENTER_TYPE,
    PARAM_SCALE_KEYWORD,
    PARAM_SCALE_TYPE,
];

const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "N = normalize(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, dim)",
        inputs: &INPUTS_A_DIM,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, method)",
        inputs: &INPUTS_A_METHOD,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, dim, method)",
        inputs: &INPUTS_A_DIM_METHOD,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, method, methodtype)",
        inputs: &INPUTS_A_METHOD_TYPE,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, dim, method, methodtype)",
        inputs: &INPUTS_A_DIM_METHOD_TYPE,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, \"center\", centertype, \"scale\", scaletype)",
        inputs: &INPUTS_A_CENTER_SCALE,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, dim, \"center\", centertype, \"scale\", scaletype)",
        inputs: &INPUTS_A_DIM_CENTER_SCALE,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[N, C, S] = normalize(___)",
        inputs: &INPUTS_A,
        outputs: &OUTPUT_NCS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.normalize.INVALID_ARGUMENT",
    identifier: Some("RunMat:normalize:InvalidArgument"),
    when: "Inputs, dimension, method, method type, or name-value options are malformed.",
    message: "normalize: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.normalize.INTERNAL",
    identifier: Some("RunMat:normalize:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "normalize: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone)]
enum NumericInput {
    Real {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
    Complex {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
    },
}

#[derive(Clone)]
enum ParamReal {
    Computed { data: Vec<f64>, shape: Vec<usize> },
    Explicit(Tensor),
}

#[derive(Clone)]
enum ParamComplex {
    Computed {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
    },
    ExplicitReal(Tensor),
    ExplicitComplex(ComplexTensor),
}

#[derive(Clone)]
enum RealPlan {
    CenterScale {
        center: CenterSpec,
        scale: ScaleSpec,
    },
    Range {
        bounds: RangeBounds,
    },
}

#[derive(Clone)]
enum ComplexPlan {
    CenterScale {
        center: ComplexCenterSpec,
        scale: ScaleSpec,
    },
}

#[derive(Clone)]
enum CenterSpec {
    None,
    Mean,
    Median,
    Explicit(Tensor),
}

#[derive(Clone)]
enum ComplexCenterSpec {
    None,
    Mean,
    Median,
    ExplicitReal(Tensor),
    ExplicitComplex(ComplexTensor),
}

#[derive(Clone)]
enum ScaleSpec {
    One,
    Std,
    Mad,
    Iqr,
    First,
    Norm(f64),
    Explicit(Tensor),
}

#[derive(Clone, Copy)]
struct RangeBounds {
    lower: f64,
    upper: f64,
}

struct ParsedArgs {
    dim: Option<usize>,
    real_plan: RealPlan,
    complex_plan: ComplexPlan,
}

struct NormalizeEval {
    n: Value,
    c: Value,
    s: Value,
}

const MAX_NORMALIZE_DIM: usize = 64;

fn normalize_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "normalize",
    category = "stats/summary",
    summary = "Normalize data by centering and scaling along a dimension.",
    keywords = "normalize,zscore,scale,range,norm,center,statistics",
    type_resolver(normalize_type),
    descriptor(crate::builtins::stats::summary::normalize::DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::normalize"
)]
async fn normalize_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let input = value_to_input(value).await?;
    let parsed = parse_args(&input, rest).await?;
    let eval = evaluate(input, parsed)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.n]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.n, eval.c, eval.s],
        ));
    }
    Ok(eval.n)
}

fn normalize_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("normalize")
        .with_identifier("RunMat:normalize:InvalidArgument")
        .build()
}

fn normalize_internal(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("normalize")
        .with_identifier("RunMat:normalize:Internal")
        .build()
}

async fn value_to_input(value: Value) -> BuiltinResult<NumericInput> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| normalize_internal(format!("normalize: {err}")))?;
    match value {
        Value::Tensor(tensor) => {
            let shape = normalize_shape_for(&tensor.shape, tensor::tensor_element_len(&tensor));
            Ok(NumericInput::Real {
                shape,
                data: tensor::tensor_into_values_f64(tensor),
            })
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| normalize_internal(format!("normalize: {err}")))?;
            let len = tensor.len();
            let shape = normalize_shape_for(&tensor.shape, len);
            Ok(NumericInput::Real {
                shape,
                data: tensor::tensor_into_values_f64(tensor),
            })
        }
        Value::Num(n) => Ok(NumericInput::Real {
            data: vec![n],
            shape: vec![1, 1],
        }),
        Value::Int(i) => Ok(NumericInput::Real {
            data: vec![i.to_f64()],
            shape: vec![1, 1],
        }),
        Value::Bool(b) => Ok(NumericInput::Real {
            data: vec![if b { 1.0 } else { 0.0 }],
            shape: vec![1, 1],
        }),
        Value::Complex(re, im) => Ok(NumericInput::Complex {
            data: vec![(re, im)],
            shape: vec![1, 1],
        }),
        Value::ComplexTensor(tensor) => Ok(NumericInput::Complex {
            shape: normalize_shape_for(&tensor.shape, tensor::complex_tensor_element_len(&tensor)),
            data: tensor::complex_tensor_into_values_complex64(tensor)
                .into_iter()
                .map(|value| (value.re, value.im))
                .collect(),
        }),
        other => Err(normalize_error(format!(
            "normalize: unsupported input type {other:?}"
        ))),
    }
}

async fn parse_args(input: &NumericInput, rest: Vec<Value>) -> BuiltinResult<ParsedArgs> {
    let rest = gather_rest(rest).await?;
    let mut dim = None;
    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if keyword_of(first).is_none() {
            if let Some(parsed_dim) = tensor::dimension_from_value_async(first, "normalize", false)
                .await
                .map_err(normalize_error)?
            {
                if parsed_dim > MAX_NORMALIZE_DIM {
                    return Err(normalize_error(format!(
                        "normalize: dimension must be <= {MAX_NORMALIZE_DIM}"
                    )));
                }
                dim = Some(parsed_dim);
                idx = 1;
            }
        }
    }

    let mut real_plan = RealPlan::CenterScale {
        center: CenterSpec::Mean,
        scale: ScaleSpec::Std,
    };
    let mut complex_plan = ComplexPlan::CenterScale {
        center: ComplexCenterSpec::Mean,
        scale: ScaleSpec::Std,
    };
    if idx < rest.len() {
        let keyword = keyword_of(&rest[idx]).ok_or_else(|| {
            normalize_error(format!(
                "normalize: expected method string at argument {}",
                idx + 2
            ))
        })?;
        idx += 1;
        match keyword.as_str() {
            "zscore" => {
                let method_type = optional_keyword(&rest, &mut idx);
                match method_type.as_deref().unwrap_or("std") {
                    "std" => {
                        real_plan = RealPlan::CenterScale {
                            center: CenterSpec::Mean,
                            scale: ScaleSpec::Std,
                        };
                        complex_plan = ComplexPlan::CenterScale {
                            center: ComplexCenterSpec::Mean,
                            scale: ScaleSpec::Std,
                        };
                    }
                    "robust" => {
                        real_plan = RealPlan::CenterScale {
                            center: CenterSpec::Median,
                            scale: ScaleSpec::Mad,
                        };
                        complex_plan = ComplexPlan::CenterScale {
                            center: ComplexCenterSpec::Median,
                            scale: ScaleSpec::Mad,
                        };
                    }
                    other => {
                        return Err(normalize_error(format!(
                            "normalize: unsupported zscore method type '{other}'"
                        )));
                    }
                }
            }
            "norm" => {
                let p = if idx < rest.len() && keyword_of(&rest[idx]).is_none() {
                    let p = scalar_f64(&rest[idx]).await?.ok_or_else(|| {
                        normalize_error("normalize: norm method type must be a scalar")
                    })?;
                    idx += 1;
                    p
                } else {
                    2.0
                };
                if p.is_nan() || p <= 0.0 {
                    return Err(normalize_error("normalize: norm order must be positive"));
                }
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::None,
                    scale: ScaleSpec::Norm(p),
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::None,
                    scale: ScaleSpec::Norm(p),
                };
            }
            "scale" => {
                let spec = parse_scale_spec(&rest, &mut idx).await?;
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::None,
                    scale: spec.clone(),
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::None,
                    scale: spec,
                };
            }
            "center" => {
                let (real_center, complex_center) = parse_center_spec(
                    &rest,
                    &mut idx,
                    matches!(input, NumericInput::Complex { .. }),
                )
                .await?;
                real_plan = RealPlan::CenterScale {
                    center: real_center.clone(),
                    scale: ScaleSpec::One,
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: complex_center.clone(),
                    scale: ScaleSpec::One,
                };
                if idx < rest.len() && keyword_of(&rest[idx]).as_deref() == Some("scale") {
                    idx += 1;
                    let scale = parse_scale_spec(&rest, &mut idx).await?;
                    real_plan = RealPlan::CenterScale {
                        center: real_center,
                        scale: scale.clone(),
                    };
                    complex_plan = ComplexPlan::CenterScale {
                        center: complex_center,
                        scale,
                    };
                }
            }
            "medianiqr" => {
                real_plan = RealPlan::CenterScale {
                    center: CenterSpec::Median,
                    scale: ScaleSpec::Iqr,
                };
                complex_plan = ComplexPlan::CenterScale {
                    center: ComplexCenterSpec::Median,
                    scale: ScaleSpec::Iqr,
                };
            }
            "range" => {
                if matches!(input, NumericInput::Complex { .. }) {
                    return Err(normalize_error(
                        "normalize: range normalization is not supported for complex inputs",
                    ));
                }
                let bounds = parse_range_bounds(&rest, &mut idx).await?;
                real_plan = RealPlan::Range { bounds };
            }
            "datavariables" | "replacevalues" => {
                return Err(normalize_error(
                    "normalize: table name-value options are not supported for array inputs",
                ));
            }
            other => {
                return Err(normalize_error(format!(
                    "normalize: unsupported method '{other}'"
                )));
            }
        }
        if idx < rest.len() {
            return Err(normalize_error(
                "normalize: unexpected arguments after normalization method",
            ));
        }
    }

    Ok(ParsedArgs {
        dim,
        real_plan,
        complex_plan,
    })
}

async fn gather_rest(rest: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(rest.len());
    for value in rest {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| normalize_internal(format!("normalize: {err}")))?,
        );
    }
    Ok(out)
}

fn optional_keyword(rest: &[Value], idx: &mut usize) -> Option<String> {
    let keyword = rest.get(*idx).and_then(keyword_of)?;
    *idx += 1;
    Some(keyword)
}

async fn parse_center_spec(
    rest: &[Value],
    idx: &mut usize,
    input_is_complex: bool,
) -> BuiltinResult<(CenterSpec, ComplexCenterSpec)> {
    if *idx >= rest.len() || keyword_of(&rest[*idx]).as_deref() == Some("scale") {
        return Ok((CenterSpec::Mean, ComplexCenterSpec::Mean));
    }
    if let Some(keyword) = keyword_of(&rest[*idx]) {
        *idx += 1;
        return match keyword.as_str() {
            "mean" => Ok((CenterSpec::Mean, ComplexCenterSpec::Mean)),
            "median" => Ok((CenterSpec::Median, ComplexCenterSpec::Median)),
            other => Err(normalize_error(format!(
                "normalize: unsupported center type '{other}'"
            ))),
        };
    }
    let value = rest[*idx].clone();
    *idx += 1;
    match value {
        Value::Complex(re, im) => {
            if !input_is_complex {
                return Err(normalize_error(
                    "normalize: complex explicit center is not supported for real inputs",
                ));
            }
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|err| normalize_internal(format!("normalize: {err}")))?;
            Ok((
                CenterSpec::Explicit(Tensor::new(vec![re], vec![1, 1]).unwrap()),
                ComplexCenterSpec::ExplicitComplex(tensor),
            ))
        }
        Value::ComplexTensor(tensor) => {
            if !input_is_complex {
                return Err(normalize_error(
                    "normalize: complex explicit center is not supported for real inputs",
                ));
            }
            Ok((
                CenterSpec::Explicit(real_part_tensor(&tensor)?),
                ComplexCenterSpec::ExplicitComplex(tensor),
            ))
        }
        other => {
            let tensor = tensor::value_into_tensor_for("normalize", other)
                .map_err(|err| normalize_error(format!("normalize: {err}")))?;
            Ok((
                CenterSpec::Explicit(tensor.clone()),
                ComplexCenterSpec::ExplicitReal(tensor),
            ))
        }
    }
}

async fn parse_scale_spec(rest: &[Value], idx: &mut usize) -> BuiltinResult<ScaleSpec> {
    if *idx >= rest.len() {
        return Ok(ScaleSpec::Std);
    }
    if let Some(keyword) = keyword_of(&rest[*idx]) {
        *idx += 1;
        return match keyword.as_str() {
            "std" => Ok(ScaleSpec::Std),
            "mad" => Ok(ScaleSpec::Mad),
            "iqr" => Ok(ScaleSpec::Iqr),
            "first" => Ok(ScaleSpec::First),
            other => Err(normalize_error(format!(
                "normalize: unsupported scale type '{other}'"
            ))),
        };
    }
    let tensor = tensor::value_into_tensor_for("normalize", rest[*idx].clone())
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    *idx += 1;
    Ok(ScaleSpec::Explicit(tensor))
}

async fn parse_range_bounds(rest: &[Value], idx: &mut usize) -> BuiltinResult<RangeBounds> {
    if *idx >= rest.len() || keyword_of(&rest[*idx]).is_some() {
        return Ok(RangeBounds {
            lower: 0.0,
            upper: 1.0,
        });
    }
    let tensor = tensor::value_into_tensor_for("normalize", rest[*idx].clone())
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    *idx += 1;
    let values = tensor::tensor_values_f64(&tensor);
    match values.as_slice() {
        [upper] => Ok(RangeBounds {
            lower: 0.0,
            upper: *upper,
        }),
        [lower, upper] if lower < upper => Ok(RangeBounds {
            lower: *lower,
            upper: *upper,
        }),
        _ => Err(normalize_error(
            "normalize: range target must be a scalar or increasing two-element vector",
        )),
    }
}

async fn scalar_f64(value: &Value) -> BuiltinResult<Option<f64>> {
    tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| normalize_error(format!("normalize: {err}")))
}

fn evaluate(input: NumericInput, parsed: ParsedArgs) -> BuiltinResult<NormalizeEval> {
    match input {
        NumericInput::Real { data, shape } => evaluate_real(data, shape, parsed),
        NumericInput::Complex { data, shape } => evaluate_complex(data, shape, parsed),
    }
}

fn evaluate_real(
    data: Vec<f64>,
    shape: Vec<usize>,
    parsed: ParsedArgs,
) -> BuiltinResult<NormalizeEval> {
    let axis = parsed
        .dim
        .unwrap_or_else(|| first_non_singleton(&shape))
        .saturating_sub(1);
    let rank = shape.len().max(axis + 1).max(2);
    let mut padded_shape = shape;
    padded_shape.resize(rank, 1);
    let mut param_shape = padded_shape.clone();
    param_shape[axis] = 1;
    let buckets = buckets_for(&padded_shape, axis);
    match parsed.real_plan {
        RealPlan::CenterScale { center, scale } => {
            let center = compute_real_center(&data, &buckets, &param_shape, &center)?;
            let scale = compute_real_scale(&data, &buckets, &param_shape, &scale)?;
            let n_data = normalize_real_center_scale(&data, &padded_shape, &center, &scale)?;
            Ok(NormalizeEval {
                n: tensor_value(n_data, padded_shape)?,
                c: real_param_value(center)?,
                s: real_param_value(scale)?,
            })
        }
        RealPlan::Range { bounds } => {
            let (center_values, scale_values) = compute_range_params(&data, &buckets, bounds);
            let center = ParamReal::Computed {
                data: center_values,
                shape: param_shape.clone(),
            };
            let scale = ParamReal::Computed {
                data: scale_values,
                shape: param_shape,
            };
            let n_data = normalize_real_center_scale(&data, &padded_shape, &center, &scale)?;
            Ok(NormalizeEval {
                n: tensor_value(n_data, padded_shape)?,
                c: real_param_value(center)?,
                s: real_param_value(scale)?,
            })
        }
    }
}

fn evaluate_complex(
    data: Vec<(f64, f64)>,
    shape: Vec<usize>,
    parsed: ParsedArgs,
) -> BuiltinResult<NormalizeEval> {
    let axis = parsed
        .dim
        .unwrap_or_else(|| first_non_singleton(&shape))
        .saturating_sub(1);
    let rank = shape.len().max(axis + 1).max(2);
    let mut padded_shape = shape;
    padded_shape.resize(rank, 1);
    let mut param_shape = padded_shape.clone();
    param_shape[axis] = 1;
    let buckets = buckets_for(&padded_shape, axis);
    let ComplexPlan::CenterScale { center, scale } = parsed.complex_plan;
    let center = compute_complex_center(&data, &buckets, &param_shape, &center)?;
    let scale = compute_complex_scale(&data, &buckets, &param_shape, &scale)?;
    let n_data = normalize_complex_center_scale(&data, &padded_shape, &center, &scale)?;
    Ok(NormalizeEval {
        n: complex_value(n_data, padded_shape)?,
        c: complex_param_value(center)?,
        s: real_param_value(scale)?,
    })
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|dim| *dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for idx in 1..shape.len() {
        strides[idx] = strides[idx - 1] * shape[idx - 1];
    }
    strides
}

fn buckets_for(shape: &[usize], axis: usize) -> Vec<Vec<usize>> {
    let mut param_shape = shape.to_vec();
    param_shape[axis] = 1;
    let param_len = tensor::element_count(&param_shape);
    let strides = strides_for(shape);
    let param_strides = strides_for(&param_shape);
    let mut buckets = vec![Vec::new(); param_len];
    for linear in 0..tensor::element_count(shape) {
        let mut dst = 0usize;
        for dim in 0..shape.len() {
            let coord = (linear / strides[dim]) % shape[dim];
            if dim != axis {
                dst += coord * param_strides[dim];
            }
        }
        buckets[dst].push(linear);
    }
    buckets
}

fn non_nan(values: impl Iterator<Item = f64>) -> Vec<f64> {
    values.filter(|value| !value.is_nan()).collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn std_sample(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return 0.0;
    }
    let mean = mean(values);
    let m2 = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>();
    (m2 / (values.len() - 1) as f64).sqrt()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let n = values.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn quantile_linear(mut values: Vec<f64>, p: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let pos = p * (values.len() - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let frac = pos - lower as f64;
    values[lower] * (1.0 - frac) + values[upper] * frac
}

fn iqr(values: Vec<f64>) -> f64 {
    quantile_linear(values.clone(), 0.75) - quantile_linear(values, 0.25)
}

fn mad(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let med = median(values.to_vec());
    median(values.iter().map(|value| (value - med).abs()).collect())
}

fn p_norm(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if p.is_infinite() {
        return values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    }
    values
        .iter()
        .map(|value| value.abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

fn compute_real_center(
    data: &[f64],
    buckets: &[Vec<usize>],
    shape: &[usize],
    spec: &CenterSpec,
) -> BuiltinResult<ParamReal> {
    match spec {
        CenterSpec::Explicit(tensor) => Ok(ParamReal::Explicit(tensor.clone())),
        CenterSpec::None => Ok(ParamReal::Computed {
            data: vec![0.0; buckets.len()],
            shape: shape.to_vec(),
        }),
        CenterSpec::Mean => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| mean(&non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
        CenterSpec::Median => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| median(non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
    }
}

fn compute_real_scale(
    data: &[f64],
    buckets: &[Vec<usize>],
    shape: &[usize],
    spec: &ScaleSpec,
) -> BuiltinResult<ParamReal> {
    match spec {
        ScaleSpec::Explicit(tensor) => Ok(ParamReal::Explicit(tensor.clone())),
        ScaleSpec::One => Ok(ParamReal::Computed {
            data: vec![1.0; buckets.len()],
            shape: shape.to_vec(),
        }),
        ScaleSpec::Std => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| std_sample(&non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Mad => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| mad(&non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Iqr => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| iqr(non_nan(indices.iter().map(|&idx| data[idx]))))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::First => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| {
                    indices
                        .iter()
                        .map(|&idx| data[idx])
                        .find(|value| !value.is_nan())
                        .unwrap_or(f64::NAN)
                })
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Norm(p) => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| p_norm(&non_nan(indices.iter().map(|&idx| data[idx])), *p))
                .collect(),
            shape: shape.to_vec(),
        }),
    }
}

fn compute_range_params(
    data: &[f64],
    buckets: &[Vec<usize>],
    bounds: RangeBounds,
) -> (Vec<f64>, Vec<f64>) {
    let mut centers = Vec::with_capacity(buckets.len());
    let mut scales = Vec::with_capacity(buckets.len());
    for indices in buckets {
        let values = non_nan(indices.iter().map(|&idx| data[idx]));
        if values.is_empty() {
            centers.push(f64::NAN);
            scales.push(f64::NAN);
            continue;
        }
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if min == max {
            centers.push(f64::NAN);
            scales.push(f64::NAN);
        } else {
            let scale = (max - min) / (bounds.upper - bounds.lower);
            centers.push(min - bounds.lower * scale);
            scales.push(scale);
        }
    }
    (centers, scales)
}

fn complex_mean(values: &[(f64, f64)]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let (sum_re, sum_im) = values
        .iter()
        .fold((0.0, 0.0), |acc, value| (acc.0 + value.0, acc.1 + value.1));
    (sum_re / values.len() as f64, sum_im / values.len() as f64)
}

fn complex_median(mut values: Vec<(f64, f64)>) -> (f64, f64) {
    values.sort_by(|a, b| {
        a.0.hypot(a.1)
            .partial_cmp(&b.0.hypot(b.1))
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                a.1.atan2(a.0)
                    .partial_cmp(&b.1.atan2(b.0))
                    .unwrap_or(Ordering::Equal)
            })
    });
    if values.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        values[values.len() / 2]
    }
}

fn complex_values(data: &[(f64, f64)], indices: &[usize]) -> Vec<(f64, f64)> {
    indices
        .iter()
        .map(|&idx| data[idx])
        .filter(|(re, im)| !re.is_nan() && !im.is_nan())
        .collect()
}

fn complex_std(values: &[(f64, f64)]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return 0.0;
    }
    let mean = complex_mean(values);
    let m2 = values
        .iter()
        .map(|value| {
            let re = value.0 - mean.0;
            let im = value.1 - mean.1;
            re * re + im * im
        })
        .sum::<f64>();
    (m2 / (values.len() - 1) as f64).sqrt()
}

fn complex_mad(values: &[(f64, f64)]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let med = complex_median(values.to_vec());
    median(
        values
            .iter()
            .map(|value| (value.0 - med.0).hypot(value.1 - med.1))
            .collect(),
    )
}

fn complex_iqr(values: &[(f64, f64)]) -> f64 {
    iqr(values.iter().map(|value| value.0.hypot(value.1)).collect())
}

fn complex_norm(values: &[(f64, f64)], p: f64) -> f64 {
    p_norm(
        &values
            .iter()
            .map(|value| value.0.hypot(value.1))
            .collect::<Vec<_>>(),
        p,
    )
}

fn compute_complex_center(
    data: &[(f64, f64)],
    buckets: &[Vec<usize>],
    shape: &[usize],
    spec: &ComplexCenterSpec,
) -> BuiltinResult<ParamComplex> {
    match spec {
        ComplexCenterSpec::ExplicitReal(tensor) => Ok(ParamComplex::ExplicitReal(tensor.clone())),
        ComplexCenterSpec::ExplicitComplex(tensor) => {
            Ok(ParamComplex::ExplicitComplex(tensor.clone()))
        }
        ComplexCenterSpec::None => Ok(ParamComplex::Computed {
            data: vec![(0.0, 0.0); buckets.len()],
            shape: shape.to_vec(),
        }),
        ComplexCenterSpec::Mean => Ok(ParamComplex::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_mean(&complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
        ComplexCenterSpec::Median => Ok(ParamComplex::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_median(complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
    }
}

fn compute_complex_scale(
    data: &[(f64, f64)],
    buckets: &[Vec<usize>],
    shape: &[usize],
    spec: &ScaleSpec,
) -> BuiltinResult<ParamReal> {
    match spec {
        ScaleSpec::Explicit(tensor) => Ok(ParamReal::Explicit(tensor.clone())),
        ScaleSpec::One => Ok(ParamReal::Computed {
            data: vec![1.0; buckets.len()],
            shape: shape.to_vec(),
        }),
        ScaleSpec::Std => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_std(&complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Mad => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_mad(&complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Iqr => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_iqr(&complex_values(data, indices)))
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::First => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| {
                    indices
                        .iter()
                        .map(|&idx| data[idx])
                        .find(|(re, im)| !re.is_nan() && !im.is_nan())
                        .map(|value| value.0.hypot(value.1))
                        .unwrap_or(f64::NAN)
                })
                .collect(),
            shape: shape.to_vec(),
        }),
        ScaleSpec::Norm(p) => Ok(ParamReal::Computed {
            data: buckets
                .iter()
                .map(|indices| complex_norm(&complex_values(data, indices), *p))
                .collect(),
            shape: shape.to_vec(),
        }),
    }
}

fn normalize_real_center_scale(
    data: &[f64],
    shape: &[usize],
    center: &ParamReal,
    scale: &ParamReal,
) -> BuiltinResult<Vec<f64>> {
    let center_values = real_param_values(center);
    let center_shape = real_param_shape(center, center_values.len());
    let scale_values = real_param_values(scale);
    let scale_shape = real_param_shape(scale, scale_values.len());
    let center_plan = BroadcastPlan::new(shape, &center_shape)
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    if center_plan.output_shape() != shape {
        return Err(normalize_error(
            "normalize: center parameter is not compatible with input shape",
        ));
    }
    let scale_plan = BroadcastPlan::new(shape, &scale_shape)
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    if scale_plan.output_shape() != shape {
        return Err(normalize_error(
            "normalize: scale parameter is not compatible with input shape",
        ));
    }
    Ok(data
        .iter()
        .zip(center_plan.iter().zip(scale_plan.iter()))
        .map(|(value, ((_, _, c_idx), (_, _, s_idx)))| {
            let c = center_values[c_idx];
            let s = scale_values[s_idx];
            normalize_scalar(*value, c, s)
        })
        .collect())
}

fn normalize_complex_center_scale(
    data: &[(f64, f64)],
    shape: &[usize],
    center: &ParamComplex,
    scale: &ParamReal,
) -> BuiltinResult<Vec<(f64, f64)>> {
    let center_values = complex_param_values(center);
    let center_shape = complex_param_shape(center, center_values.len());
    let scale_values = real_param_values(scale);
    let scale_shape = real_param_shape(scale, scale_values.len());
    let center_plan = BroadcastPlan::new(shape, &center_shape)
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    if center_plan.output_shape() != shape {
        return Err(normalize_error(
            "normalize: center parameter is not compatible with input shape",
        ));
    }
    let scale_plan = BroadcastPlan::new(shape, &scale_shape)
        .map_err(|err| normalize_error(format!("normalize: {err}")))?;
    if scale_plan.output_shape() != shape {
        return Err(normalize_error(
            "normalize: scale parameter is not compatible with input shape",
        ));
    }
    Ok(data
        .iter()
        .zip(center_plan.iter().zip(scale_plan.iter()))
        .map(|(value, ((_, _, c_idx), (_, _, s_idx)))| {
            let c = center_values[c_idx];
            let s = scale_values[s_idx];
            let re = normalize_scalar(value.0, c.0, s);
            let im = normalize_scalar(value.1, c.1, s);
            (re, im)
        })
        .collect())
}

fn normalize_scalar(value: f64, center: f64, scale: f64) -> f64 {
    if value.is_nan() || scale == 0.0 || scale.is_nan() {
        f64::NAN
    } else {
        (value - center) / scale
    }
}

fn real_param_values(param: &ParamReal) -> Vec<f64> {
    match param {
        ParamReal::Computed { data, .. } => data.clone(),
        ParamReal::Explicit(tensor) => tensor::tensor_values_f64(tensor),
    }
}

fn real_param_shape(param: &ParamReal, len: usize) -> Vec<usize> {
    match param {
        ParamReal::Computed { shape, .. } => shape.clone(),
        ParamReal::Explicit(tensor) => {
            normalize_shape_for(&tensor.shape, tensor::tensor_element_len(tensor))
        }
    }
    .shape_fallback(len)
}

fn complex_param_values(param: &ParamComplex) -> Vec<(f64, f64)> {
    match param {
        ParamComplex::Computed { data, .. } => data.clone(),
        ParamComplex::ExplicitReal(tensor) => tensor::tensor_values_f64(tensor)
            .into_iter()
            .map(|value| (value, 0.0))
            .collect(),
        ParamComplex::ExplicitComplex(tensor) => tensor::complex_tensor_values_complex64(tensor)
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect(),
    }
}

fn complex_param_shape(param: &ParamComplex, len: usize) -> Vec<usize> {
    match param {
        ParamComplex::Computed { shape, .. } => shape.clone(),
        ParamComplex::ExplicitReal(tensor) => {
            normalize_shape_for(&tensor.shape, tensor::tensor_element_len(tensor))
        }
        ParamComplex::ExplicitComplex(tensor) => {
            normalize_shape_for(&tensor.shape, tensor::complex_tensor_element_len(tensor))
        }
    }
    .shape_fallback(len)
}

trait ShapeFallback {
    fn shape_fallback(self, len: usize) -> Vec<usize>;
}

impl ShapeFallback for Vec<usize> {
    fn shape_fallback(self, len: usize) -> Vec<usize> {
        if self.is_empty() {
            vec![len, 1]
        } else {
            self
        }
    }
}

fn real_param_value(param: ParamReal) -> BuiltinResult<Value> {
    let data = real_param_values(&param);
    let shape = real_param_shape(&param, data.len());
    tensor_value(data, shape)
}

fn complex_param_value(param: ParamComplex) -> BuiltinResult<Value> {
    let data = complex_param_values(&param);
    let shape = complex_param_shape(&param, data.len());
    complex_value(data, shape)
}

fn tensor_value(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| normalize_internal(format!("normalize: {err}")))
}

fn complex_value(data: Vec<(f64, f64)>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 && normalize_shape_for(&shape, data.len()) == vec![1, 1] {
        Ok(Value::Complex(data[0].0, data[0].1))
    } else {
        ComplexTensor::new(data, shape)
            .map(Value::ComplexTensor)
            .map_err(|err| normalize_internal(format!("normalize: {err}")))
    }
}

fn normalize_shape_for(shape: &[usize], len: usize) -> Vec<usize> {
    if shape.is_empty() {
        tensor::default_shape_for(shape, len)
    } else {
        shape.to_vec()
    }
}

fn real_part_tensor(tensor: &ComplexTensor) -> BuiltinResult<Tensor> {
    Tensor::new(
        tensor::complex_tensor_values_complex64(tensor)
            .into_iter()
            .map(|value| value.re)
            .collect(),
        tensor.shape.clone(),
    )
    .map_err(|err| normalize_internal(format!("normalize: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage};

    fn call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(normalize_builtin(value, rest))
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor = Tensor::new_integer(storage, shape).unwrap();
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor = Tensor::new_integer(storage, shape).unwrap();
        tensor.data.clear();
        Value::Tensor(tensor)
    }

    fn complex_int_tensor(real: IntegerStorage, imag: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor =
            ComplexTensor::new_integer(IntegerComplexStorage::new(real, imag).unwrap(), shape)
                .unwrap();
        tensor.data.clear();
        Value::ComplexTensor(tensor)
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn default_zscore_normalizes_first_non_singleton_dimension() {
        let out = expect_tensor(call(tensor(vec![1., 2., 3.], vec![3, 1]), vec![]).unwrap());
        assert_close(out.data[0], -1.0);
        assert_close(out.data[1], 0.0);
        assert_close(out.data[2], 1.0);
    }

    #[test]
    fn normalize_reads_typed_integer_input_storage_exactly() {
        let out = expect_tensor(
            call(
                int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
                vec![],
            )
            .unwrap(),
        );
        assert_close(out.data[0], -1.0);
        assert_close(out.data[1], 0.0);
        assert_close(out.data[2], 1.0);
    }

    #[test]
    fn normalize_real_input_shape_uses_typed_integer_storage_not_mirror() {
        let out = expect_tensor(
            call(
                mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
                vec![],
            )
            .unwrap(),
        );
        assert_eq!(out.shape, vec![3, 1]);
        assert_close(out.data[0], -1.0);
        assert_close(out.data[1], 0.0);
        assert_close(out.data[2], 1.0);
    }

    #[test]
    fn dim_argument_normalizes_rows_independently() {
        let out = expect_tensor(
            call(
                tensor(vec![1., 2., 3., 5., 5., 9.], vec![3, 2]),
                vec![Value::Num(2.0)],
            )
            .unwrap(),
        );
        assert_eq!(out.shape, vec![3, 2]);
        assert_close(out.data[0], -std::f64::consts::FRAC_1_SQRT_2);
        assert_close(out.data[3], std::f64::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn range_method_maps_to_requested_bounds() {
        let out = expect_tensor(
            call(
                tensor(vec![2., 4., 6.], vec![3, 1]),
                vec![
                    Value::from("range"),
                    Value::Tensor(Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap()),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.data, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn normalize_reads_typed_integer_range_bounds_exactly() {
        let out = expect_tensor(
            call(
                tensor(vec![2., 4., 6.], vec![3, 1]),
                vec![
                    Value::from("range"),
                    int_tensor(IntegerStorage::I16(vec![-1, 1]), vec![1, 2]),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.data, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn norm_method_accepts_p_norm() {
        let out = expect_tensor(
            call(
                tensor(vec![3., 4.], vec![2, 1]),
                vec![Value::from("norm"), Value::Num(2.0)],
            )
            .unwrap(),
        );
        assert_close(out.data[0], 0.6);
        assert_close(out.data[1], 0.8);
    }

    #[test]
    fn center_scale_outputs_can_be_reused() {
        let values = {
            let _guard = crate::output_count::push_output_count(Some(3));
            let result = call(
                tensor(vec![1., 2., 3., 5.], vec![2, 2]),
                vec![
                    Value::Num(1.0),
                    Value::from("center"),
                    Value::from("mean"),
                    Value::from("scale"),
                    Value::from("std"),
                ],
            )
            .unwrap();
            let Value::OutputList(values) = result else {
                panic!("expected outputs");
            };
            values
        };
        assert_eq!(values.len(), 3);
        let n = expect_tensor(values[0].clone());
        let c = expect_tensor(values[1].clone());
        let s = expect_tensor(values[2].clone());
        assert_eq!(n.shape, vec![2, 2]);
        assert_eq!(c.shape, vec![1, 2]);
        assert_eq!(s.shape, vec![1, 2]);
        let reused = expect_tensor(
            call(
                tensor(vec![1., 2., 3., 5.], vec![2, 2]),
                vec![
                    Value::from("center"),
                    values[1].clone(),
                    Value::from("scale"),
                    values[2].clone(),
                ],
            )
            .unwrap(),
        );
        assert_eq!(n.data, reused.data);
    }

    #[test]
    fn normalize_reads_typed_integer_explicit_center_and_scale_exactly() {
        let out = expect_tensor(
            call(
                int_tensor(IntegerStorage::I16(vec![2, 4, 6]), vec![3, 1]),
                vec![
                    Value::from("center"),
                    int_tensor(IntegerStorage::I16(vec![2]), vec![1, 1]),
                    Value::from("scale"),
                    int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.data, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn normalize_explicit_param_shape_uses_typed_integer_storage_not_mirror() {
        let out = expect_tensor(
            call(
                mirrorless_int_tensor(IntegerStorage::I16(vec![2, 4, 6]), vec![3, 1]),
                vec![
                    Value::from("center"),
                    mirrorless_int_tensor(IntegerStorage::I16(vec![2, 2, 2]), vec![3, 1]),
                    Value::from("scale"),
                    mirrorless_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.data, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn zero_scale_output_can_be_reused() {
        let values = {
            let _guard = crate::output_count::push_output_count(Some(3));
            let result = call(tensor(vec![4., 4.], vec![2, 1]), vec![]).unwrap();
            let Value::OutputList(values) = result else {
                panic!("expected outputs");
            };
            values
        };
        let s = expect_tensor(values[2].clone());
        assert_eq!(s.data, vec![0.0]);

        let reused = expect_tensor(
            call(
                tensor(vec![4., 4.], vec![2, 1]),
                vec![
                    Value::from("center"),
                    values[1].clone(),
                    Value::from("scale"),
                    values[2].clone(),
                ],
            )
            .unwrap(),
        );
        assert!(reused.data.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn explicit_nan_scale_is_accepted_and_propagates() {
        let out = expect_tensor(
            call(
                tensor(vec![1., 2.], vec![2, 1]),
                vec![Value::from("scale"), Value::Num(f64::NAN)],
            )
            .unwrap(),
        );
        assert!(out.data.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn empty_inputs_preserve_empty_shapes() {
        let values = {
            let _guard = crate::output_count::push_output_count(Some(3));
            let result = call(tensor(vec![], vec![0, 3]), vec![]).unwrap();
            let Value::OutputList(values) = result else {
                panic!("expected outputs");
            };
            values
        };
        let n = expect_tensor(values[0].clone());
        let c = expect_tensor(values[1].clone());
        let s = expect_tensor(values[2].clone());
        assert_eq!(n.shape, vec![0, 3]);
        assert_eq!(c.shape, vec![0, 1]);
        assert_eq!(s.shape, vec![0, 1]);
        assert!(n.data.is_empty());
        assert!(c.data.is_empty());
        assert!(s.data.is_empty());
    }

    #[test]
    fn nan_values_are_omitted_from_parameters_and_preserved_in_output() {
        let out = expect_tensor(
            call(
                tensor(vec![1., f64::NAN, 3.], vec![3, 1]),
                vec![Value::from("center"), Value::from("mean")],
            )
            .unwrap(),
        );
        assert_eq!(out.data[0], -1.0);
        assert!(out.data[1].is_nan());
        assert_eq!(out.data[2], 1.0);
    }

    #[test]
    fn complex_norm_uses_magnitudes_but_preserves_phase() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(3.0, 4.0), (0.0, 12.0)], vec![2, 1]).unwrap(),
        );
        let Value::ComplexTensor(out) =
            call(input, vec![Value::from("norm"), Value::Num(2.0)]).unwrap()
        else {
            panic!("expected complex output");
        };
        assert_close(out.data[0].0, 3.0 / 13.0);
        assert_close(out.data[0].1, 4.0 / 13.0);
        assert_close(out.data[1].1, 12.0 / 13.0);
    }

    #[test]
    fn normalize_reads_typed_complex_integer_storage_exactly() {
        let input = complex_int_tensor(
            IntegerStorage::I16(vec![3, 0]),
            IntegerStorage::I16(vec![4, 12]),
            vec![2, 1],
        );
        let Value::ComplexTensor(out) =
            call(input, vec![Value::from("norm"), Value::Num(2.0)]).unwrap()
        else {
            panic!("expected complex output");
        };
        assert_close(out.data[0].0, 3.0 / 13.0);
        assert_close(out.data[0].1, 4.0 / 13.0);
        assert_close(out.data[1].0, 0.0);
        assert_close(out.data[1].1, 12.0 / 13.0);
    }

    #[test]
    fn normalize_reuses_typed_complex_integer_center_from_storage() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(3.0, 4.0), (5.0, 8.0)], vec![2, 1]).unwrap(),
        );
        let center = complex_int_tensor(
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I16(vec![2]),
            vec![1, 1],
        );
        let Value::ComplexTensor(out) = call(
            input,
            vec![
                Value::from("center"),
                center,
                Value::from("scale"),
                Value::Num(2.0),
            ],
        )
        .unwrap() else {
            panic!("expected complex output");
        };
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.data, vec![(1.0, 1.0), (2.0, 3.0)]);
    }

    #[test]
    fn rejects_multiple_top_level_methods() {
        let err = call(
            tensor(vec![1., 2., 3.], vec![3, 1]),
            vec![
                Value::from("center"),
                Value::from("mean"),
                Value::from("range"),
            ],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:normalize:InvalidArgument"));
    }

    #[test]
    fn rejects_unbounded_dimension() {
        let err = call(
            tensor(vec![1., 2.], vec![2, 1]),
            vec![Value::Num((MAX_NORMALIZE_DIM + 1) as f64)],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:normalize:InvalidArgument"));
    }

    #[test]
    fn rejects_complex_explicit_center_for_real_input() {
        let err = call(
            tensor(vec![1., 2.], vec![2, 1]),
            vec![
                Value::from("center"),
                Value::Complex(1.0, 1.0),
                Value::from("scale"),
                Value::Num(1.0),
            ],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:normalize:InvalidArgument"));
    }

    #[test]
    fn rejects_table_only_name_value_options_for_arrays() {
        let err = call(
            tensor(vec![1., 2.], vec![2, 1]),
            vec![Value::from("DataVariables"), Value::from("x")],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:normalize:InvalidArgument"));
    }
}
