//! Multiple linear regression compatibility surface.

use nalgebra::{DMatrix, DVector};
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

use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math::{regularized_beta, student_t_inv};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "regress";
const EPS: f64 = 1.0e-12;
const MAX_REGRESS_CELLS: usize = 50_000_000;

const OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Least-squares coefficient estimates.",
}];

const OUTPUT_B_BINT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "bint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for coefficient estimates.",
    },
];

const OUTPUT_B_BINT_R: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "bint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Regression residuals for complete observations.",
    },
];

const OUTPUT_B_BINT_R_RINT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "bint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Regression residuals for complete observations.",
    },
    BuiltinParamDescriptor {
        name: "rint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for residuals.",
    },
];

const OUTPUT_B_BINT_R_RINT_STATS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "bint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Regression residuals for complete observations.",
    },
    BuiltinParamDescriptor {
        name: "rint",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Confidence intervals for residuals.",
    },
    BuiltinParamDescriptor {
        name: "stats",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row vector [R2 F p errorVariance].",
    },
];

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response vector.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Design matrix with observations in rows and model terms in columns.",
};

const PARAM_ALPHA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "alpha",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.05"),
    description: "Significance level for coefficient and residual intervals.",
};

const INPUTS_Y_X: [BuiltinParamDescriptor; 2] = [PARAM_Y, PARAM_X];
const INPUTS_Y_X_ALPHA: [BuiltinParamDescriptor; 3] = [PARAM_Y, PARAM_X, PARAM_ALPHA];

const SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "b = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "b = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint] = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B_BINT,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint] = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B_BINT,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r] = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B_BINT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r] = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B_BINT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r, rint] = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B_BINT_R_RINT,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r, rint] = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B_BINT_R_RINT,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r, rint, stats] = regress(y, X)",
        inputs: &INPUTS_Y_X,
        outputs: &OUTPUT_B_BINT_R_RINT_STATS,
    },
    BuiltinSignatureDescriptor {
        label: "[b, bint, r, rint, stats] = regress(y, X, alpha)",
        inputs: &INPUTS_Y_X_ALPHA,
        outputs: &OUTPUT_B_BINT_R_RINT_STATS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGRESS.INVALID_ARGUMENT",
    identifier: Some("RunMat:regress:InvalidArgument"),
    when: "Inputs, dimensions, alpha, or requested output counts are malformed.",
    message: "regress: invalid argument",
};

const ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGRESS.NUMERICAL",
    identifier: Some("RunMat:regress:Numerical"),
    when: "The regression design cannot be solved numerically.",
    message: "regress: numerical failure",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGRESS.INTERNAL",
    identifier: Some("RunMat:regress:Internal"),
    when: "RunMat cannot allocate or construct regression outputs.",
    message: "regress: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_NUMERICAL, ERROR_INTERNAL];

pub const REGRESS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const REGRESS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "regress-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "regress accepts typed-integer response and design data as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RegressIntegerDataExtension"),
};
const REGRESS_INTEGER_ALPHA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "regress-integer-alpha",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "regress accepts a typed-integer alpha as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RegressIntegerAlphaExtension"),
};
pub const REGRESS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    REGRESS_INTEGER_DATA_EXTENSION,
    REGRESS_INTEGER_ALPHA_EXTENSION,
];
const REGRESS_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double response data; typed integers cross a checked binary64 regression boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double design data; typed integers cross the same checked boundary independently of y.",
    },
];
const REGRESS_INTEGER_ALPHA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "alpha",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double alpha; a typed integer is admitted only when exact in binary64 and within the open unit interval.",
    }];
pub const REGRESS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "regress(integer_y, integer_X [, alpha])",
        inputs: &REGRESS_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact authoritative integer values are checked before transparent gather and then intentionally enter the binary64 least-squares domain; outputs are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "regress(y, X, integer_alpha)",
        inputs: &REGRESS_INTEGER_ALPHA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The typed alpha extension is gated and checked before provider access; the statistical computation and outputs are double.",
    },
];

fn regress_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn regress_error(
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
    regress_error(message, &ERROR_INVALID_ARGUMENT)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    regress_error(message, &ERROR_NUMERICAL)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    regress_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Debug)]
struct PreparedData {
    design: DMatrix<f64>,
    y: DVector<f64>,
}

#[derive(Clone, Debug)]
struct CoefficientSolve {
    beta: Vec<f64>,
    fitted: DVector<f64>,
    inv_xtx_active: Option<DMatrix<f64>>,
    active_columns: Vec<usize>,
    rank: usize,
}

#[runtime_builtin(
    name = "regress",
    category = "stats/ml",
    summary = "Fit a multiple linear regression by ordinary least squares.",
    keywords = "regress,linear regression,least squares,statistics,machine learning",
    type_resolver(regress_type),
    descriptor(crate::builtins::stats::ml::regress::REGRESS_DESCRIPTOR),
    extensions(crate::builtins::stats::ml::regress::REGRESS_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::ml::regress::REGRESS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::ml::regress"
)]
async fn regress_builtin(y: Value, x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid("regress: accepts at most one alpha argument"));
    }
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &y,
        &REGRESS_INTEGER_DATA_EXTENSION,
        NAME,
        "response",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &x,
        &REGRESS_INTEGER_DATA_EXTENSION,
        NAME,
        "design",
    )
    .await?;
    if let Some(value) = rest.first() {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            value,
            &REGRESS_INTEGER_ALPHA_EXTENSION,
            NAME,
            "alpha",
        )
        .await?;
    }
    let y = gather_tensor(y).await?;
    let x = gather_tensor(x).await?;
    let alpha = if let Some(value) = rest.first() {
        parse_alpha(value).await?
    } else {
        0.05
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(out_count) if out_count > 5 => Err(invalid("regress: too many output arguments")),
        Some(out_count) => {
            let outputs = regress_compute(y, x, alpha, out_count)?;
            Ok(crate::output_count::output_list_with_padding(
                out_count, outputs,
            ))
        }
        None => Ok(regress_compute(y, x, alpha, 1)?
            .into_iter()
            .next()
            .expect("regress always returns b for scalar-output calls")),
    }
}

async fn gather_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("regress: {err}")))?;
    tensor::value_into_tensor_for(NAME, gathered).map_err(|err| invalid(format!("regress: {err}")))
}

async fn parse_alpha(value: &Value) -> BuiltinResult<f64> {
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|err| invalid(format!("regress: {err}")))?;
    let alpha = match gathered {
        Value::Num(value) => value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64(&tensor);
            if values.len() != 1 {
                return Err(invalid(format!(
                    "regress: alpha must be a numeric scalar, got {:?}",
                    Value::Tensor(tensor)
                )));
            }
            values[0]
        }
        other => {
            return Err(invalid(format!(
                "regress: alpha must be a numeric scalar, got {other:?}"
            )));
        }
    };
    if !(0.0..1.0).contains(&alpha) {
        return Err(invalid("regress: alpha must be between 0 and 1"));
    }
    Ok(alpha)
}

fn regress_compute(
    y: Tensor,
    x: Tensor,
    alpha: f64,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let y_values = vector_values(&y)?;
    if x.shape.len() > 2 {
        return Err(invalid("regress: X must be a 2-D numeric matrix"));
    }
    if x.rows != y_values.len() {
        return Err(invalid(
            "regress: length(y) must match the number of rows in X",
        ));
    }
    if x.cols == 0 {
        return Err(invalid("regress: X must contain at least one column"));
    }
    ensure_input_budget(x.rows, x.cols)?;
    let prepared = complete_rows(&x, &y_values)?;
    let rows = prepared.y.len();
    let cols = x.cols;
    if rows == 0 {
        return Err(invalid(
            "regress: at least one complete observation is required",
        ));
    }
    ensure_output_budget(rows, cols, requested_outputs)?;
    let active_columns = independent_columns(&prepared.design)?;
    ensure_diagnostic_budget(active_columns.len(), requested_outputs)?;

    let solve = solve_coefficients(&prepared, cols, active_columns, requested_outputs > 1)?;
    let residuals = &prepared.y - &solve.fitted;
    let sse = residuals.iter().map(|value| value * value).sum::<f64>();
    let y_mean = prepared.y.iter().sum::<f64>() / rows as f64;
    let sst = prepared
        .y
        .iter()
        .map(|value| {
            let diff = value - y_mean;
            diff * diff
        })
        .sum::<f64>();
    let dfe = rows as f64 - solve.rank as f64;
    let mse = if dfe > 0.0 { sse / dfe } else { f64::NAN };

    let mut outputs = Vec::with_capacity(requested_outputs.max(1));
    outputs.push(tensor_value(solve.beta.clone(), vec![cols, 1], "b")?);
    if requested_outputs == 1 {
        return Ok(outputs);
    }

    let inv_xtx_active = solve
        .inv_xtx_active
        .as_ref()
        .expect("diagnostic output requested inverse covariance");
    let tcrit = if dfe > 0.0 {
        student_t_inv(1.0 - alpha / 2.0, dfe)
    } else {
        f64::NAN
    };
    outputs.push(tensor_value(
        coefficient_intervals(
            &solve.beta,
            inv_xtx_active,
            &solve.active_columns,
            mse,
            tcrit,
        ),
        vec![cols, 2],
        "bint",
    )?);
    if requested_outputs == 2 {
        return Ok(outputs);
    }

    let residual_values = residuals.iter().copied().collect::<Vec<_>>();
    outputs.push(tensor_value(residual_values.clone(), vec![rows, 1], "r")?);
    if requested_outputs == 3 {
        return Ok(outputs);
    }

    outputs.push(tensor_value(
        residual_intervals(
            &prepared.design,
            inv_xtx_active,
            &solve.active_columns,
            &residual_values,
            sse,
            dfe,
            alpha,
        ),
        vec![rows, 2],
        "rint",
    )?);
    if requested_outputs == 4 {
        return Ok(outputs);
    }

    outputs.push(tensor_value(
        stats_values(sse, sst, mse, rows, solve.rank),
        vec![1, 4],
        "stats",
    )?);
    Ok(outputs)
}

fn solve_coefficients(
    prepared: &PreparedData,
    total_cols: usize,
    active_columns: Vec<usize>,
    need_covariance: bool,
) -> BuiltinResult<CoefficientSolve> {
    let rows = prepared.design.nrows();
    let rank = active_columns.len();
    if rank == 0 {
        return Ok(CoefficientSolve {
            beta: vec![0.0; total_cols],
            fitted: DVector::zeros(rows),
            inv_xtx_active: need_covariance.then(|| DMatrix::zeros(0, 0)),
            active_columns,
            rank,
        });
    }
    let active_design = active_design(&prepared.design, &active_columns);
    let svd = active_design.clone().svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| numerical("regress: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| numerical("regress: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.iter().copied().collect::<Vec<_>>();
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (rows.max(rank) as f64) * f64::EPSILON * largest.max(1.0);
    let mut beta_active = DVector::zeros(rank);
    let mut inv_xtx_active = need_covariance.then(|| DMatrix::<f64>::zeros(rank, rank));
    for (index, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value.abs() <= tolerance {
            continue;
        }
        let uty = u.column(index).dot(&prepared.y);
        let contribution = uty / singular_value;
        for row in 0..rank {
            beta_active[row] += v_t[(index, row)] * contribution;
        }
        if let Some(inv_xtx) = inv_xtx_active.as_mut() {
            let inv_s2 = 1.0 / (singular_value * singular_value);
            for row in 0..rank {
                for col in 0..rank {
                    inv_xtx[(row, col)] += v_t[(index, row)] * v_t[(index, col)] * inv_s2;
                }
            }
        }
    }
    let fitted = &active_design * &beta_active;
    let mut beta = vec![0.0; total_cols];
    for (idx, col) in active_columns.iter().copied().enumerate() {
        beta[col] = beta_active[idx];
    }
    Ok(CoefficientSolve {
        beta,
        fitted,
        inv_xtx_active,
        active_columns,
        rank,
    })
}

fn ensure_input_budget(rows: usize, cols: usize) -> BuiltinResult<()> {
    let input_cells = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("regress: input is too large"))?;
    if input_cells > MAX_REGRESS_CELLS {
        return Err(invalid("regress: input is too large"));
    }
    Ok(())
}

fn ensure_output_budget(rows: usize, cols: usize, requested_outputs: usize) -> BuiltinResult<()> {
    let mut output_cells = cols;
    if requested_outputs >= 2 {
        output_cells = output_cells
            .checked_add(
                cols.checked_mul(2)
                    .ok_or_else(|| invalid("regress: output is too large"))?,
            )
            .ok_or_else(|| invalid("regress: output is too large"))?;
    }
    if requested_outputs >= 3 {
        output_cells = output_cells
            .checked_add(rows)
            .ok_or_else(|| invalid("regress: output is too large"))?;
    }
    if requested_outputs >= 4 {
        output_cells = output_cells
            .checked_add(
                rows.checked_mul(2)
                    .ok_or_else(|| invalid("regress: output is too large"))?,
            )
            .ok_or_else(|| invalid("regress: output is too large"))?;
    }
    if requested_outputs >= 5 {
        output_cells = output_cells
            .checked_add(4)
            .ok_or_else(|| invalid("regress: output is too large"))?;
    }
    if output_cells > MAX_REGRESS_CELLS {
        return Err(invalid("regress: output is too large"));
    }
    Ok(())
}

fn ensure_diagnostic_budget(active_cols: usize, requested_outputs: usize) -> BuiltinResult<()> {
    if requested_outputs <= 1 {
        return Ok(());
    }
    let covariance_cells = active_cols
        .checked_mul(active_cols)
        .ok_or_else(|| invalid("regress: diagnostic work array is too large"))?;
    if covariance_cells > MAX_REGRESS_CELLS {
        return Err(invalid("regress: diagnostic work array is too large"));
    }
    Ok(())
}

fn vector_values(tensor: &Tensor) -> BuiltinResult<Vec<f64>> {
    if tensor.shape.len() > 2 || !(tensor.rows == 1 || tensor.cols == 1) {
        return Err(invalid("regress: y must be a vector"));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

fn complete_rows(x: &Tensor, y: &[f64]) -> BuiltinResult<PreparedData> {
    let x_values = tensor::tensor_values_f64_cow(x);
    let mut rows = Vec::new();
    let mut y_values = Vec::new();
    for row in 0..x.rows {
        let y_value = y[row];
        let mut has_nan = y_value.is_nan();
        let mut has_inf = y_value.is_infinite();
        let start = rows.len();
        for col in 0..x.cols {
            let value = matrix_value(&x_values, x.rows, row, col);
            has_nan |= value.is_nan();
            has_inf |= value.is_infinite();
            rows.push(value);
        }
        if has_inf {
            return Err(invalid("regress: X and y must not contain Inf values"));
        }
        if has_nan {
            rows.truncate(start);
            continue;
        }
        y_values.push(y_value);
    }
    let observations = y_values.len();
    Ok(PreparedData {
        design: DMatrix::from_row_slice(observations, x.cols, &rows),
        y: DVector::from_column_slice(&y_values),
    })
}

fn matrix_value(values: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    values[row + col * rows]
}

fn independent_columns(design: &DMatrix<f64>) -> BuiltinResult<Vec<usize>> {
    let rows = design.nrows();
    let cols = design.ncols();
    let mut largest_norm = 0.0_f64;
    for col in 0..cols {
        let norm = column_norm(design, col);
        largest_norm = largest_norm.max(norm);
    }
    let tolerance = (rows.max(cols) as f64) * f64::EPSILON * largest_norm.max(1.0);
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut active = Vec::new();
    let mut selected = vec![false; cols];

    loop {
        let mut best: Option<(usize, Vec<f64>, f64)> = None;
        for (col, is_selected) in selected.iter().copied().enumerate() {
            if is_selected {
                continue;
            }
            let residual = column_residual(design, col, &basis);
            let norm = residual
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if best
                .as_ref()
                .is_none_or(|(_, _, best_norm)| norm > *best_norm)
            {
                best = Some((col, residual, norm));
            }
        }
        let Some((col, mut residual, norm)) = best else {
            break;
        };
        if norm <= tolerance {
            break;
        }
        for value in &mut residual {
            *value /= norm;
        }
        selected[col] = true;
        basis.push(residual);
        active.push(col);
    }
    Ok(active)
}

fn column_residual(design: &DMatrix<f64>, col: usize, basis: &[Vec<f64>]) -> Vec<f64> {
    let rows = design.nrows();
    let mut residual = (0..rows).map(|row| design[(row, col)]).collect::<Vec<_>>();
    for vector in basis {
        let projection = residual
            .iter()
            .zip(vector)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        for row in 0..rows {
            residual[row] -= projection * vector[row];
        }
    }
    residual
}

fn column_norm(design: &DMatrix<f64>, col: usize) -> f64 {
    (0..design.nrows())
        .map(|row| {
            let value = design[(row, col)];
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

fn active_design(design: &DMatrix<f64>, active_columns: &[usize]) -> DMatrix<f64> {
    let rows = design.nrows();
    let cols = active_columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in active_columns {
            data.push(design[(row, *col)]);
        }
    }
    DMatrix::from_row_slice(rows, cols, &data)
}

fn coefficient_intervals(
    beta: &[f64],
    inv_xtx_active: &DMatrix<f64>,
    active_columns: &[usize],
    mse: f64,
    tcrit: f64,
) -> Vec<f64> {
    let cols = beta.len();
    let mut out = vec![0.0; cols * 2];
    for (active_idx, col) in active_columns.iter().copied().enumerate() {
        let variance = mse * inv_xtx_active[(active_idx, active_idx)];
        let half_width = if variance >= 0.0 {
            tcrit * variance.sqrt()
        } else {
            f64::NAN
        };
        out[col] = beta[col] - half_width;
        out[col + cols] = beta[col] + half_width;
    }
    out
}

fn residual_intervals(
    design: &DMatrix<f64>,
    inv_xtx_active: &DMatrix<f64>,
    active_columns: &[usize],
    residuals: &[f64],
    sse: f64,
    dfe: f64,
    alpha: f64,
) -> Vec<f64> {
    let rows = design.nrows();
    let mut out = vec![f64::NAN; rows * 2];
    let residual_df = dfe - 1.0;
    let tcrit = if residual_df > 0.0 {
        student_t_inv(1.0 - alpha / 2.0, residual_df)
    } else {
        f64::NAN
    };
    for row in 0..rows {
        let mut leverage = 0.0;
        for (left_idx, left_col) in active_columns.iter().copied().enumerate() {
            for (right_idx, right_col) in active_columns.iter().copied().enumerate() {
                leverage += design[(row, left_col)]
                    * inv_xtx_active[(left_idx, right_idx)]
                    * design[(row, right_col)];
            }
        }
        let one_minus_h = 1.0 - leverage.clamp(0.0, 1.0);
        let deleted_sse = if one_minus_h > EPS {
            sse - residuals[row] * residuals[row] / one_minus_h
        } else {
            f64::NAN
        };
        let deleted_mse = if residual_df > 0.0 {
            (deleted_sse / residual_df).max(0.0)
        } else {
            f64::NAN
        };
        let variance = deleted_mse * one_minus_h;
        let half_width = if variance >= 0.0 && tcrit.is_finite() {
            tcrit * variance.sqrt()
        } else {
            f64::NAN
        };
        out[row] = residuals[row] - half_width;
        out[row + rows] = residuals[row] + half_width;
    }
    out
}

fn stats_values(sse: f64, sst: f64, mse: f64, rows: usize, rank: usize) -> Vec<f64> {
    let r_squared = if sst > EPS {
        1.0 - sse / sst
    } else if sse <= EPS {
        1.0
    } else {
        f64::NAN
    };
    let dfe = rows as f64 - rank as f64;
    let df_model = rank.saturating_sub(1) as f64;
    let ssr = if sst.is_finite() && sse.is_finite() {
        sst - sse
    } else {
        f64::NAN
    };
    let f_stat = if df_model > 0.0 && dfe > 0.0 && mse > 0.0 {
        (ssr / df_model) / mse
    } else if df_model > 0.0 && dfe > 0.0 && mse == 0.0 && ssr > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let p_value = if f_stat.is_finite() && f_stat >= 0.0 && df_model > 0.0 && dfe > 0.0 {
        f_upper_tail(f_stat, df_model, dfe)
    } else if f_stat == f64::INFINITY {
        0.0
    } else {
        f64::NAN
    };
    vec![r_squared, f_stat, p_value, mse]
}

fn f_upper_tail(f: f64, df1: f64, df2: f64) -> f64 {
    if f < 0.0 || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if f == f64::INFINITY {
        return 0.0;
    }
    let x = (df1 * f) / (df1 * f + df2);
    (1.0 - regularized_beta(x, df1 / 2.0, df2 / 2.0)).clamp(0.0, 1.0)
}

fn tensor_value(data: Vec<f64>, shape: Vec<usize>, label: &str) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| internal(format!("regress: failed to construct {label}: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn poisoned_int_tensor(
        storage: IntegerStorage,
        rows: usize,
        cols: usize,
        _poison: f64,
    ) -> Value {
        let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        Value::Tensor(tensor)
    }

    fn outputs(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn tensor_ref(value: &Value) -> &Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn assert_close(left: f64, right: f64) {
        assert_close_tol(left, right, 1.0e-10);
    }

    fn assert_close_tol(left: f64, right: f64, tol: f64) {
        assert!(
            (left - right).abs() < tol,
            "{left:?} not close to {right:?}"
        );
    }

    #[test]
    fn regress_fits_coefficients_and_requested_outputs() {
        let _guard = crate::output_count::push_output_count(Some(5));
        let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
        let x = tensor(vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0], 4, 2);
        let out = outputs(block_on(regress_builtin(y, x, Vec::new())).unwrap());
        let b = tensor_ref(&out[0]);
        assert_eq!(b.shape, vec![2, 1]);
        assert_close(b.materialize_f64()[0], 1.0);
        assert_close(b.materialize_f64()[1], 2.0);
        assert_eq!(tensor_ref(&out[1]).shape, vec![2, 2]);
        assert_eq!(tensor_ref(&out[2]).shape, vec![4, 1]);
        assert_eq!(tensor_ref(&out[3]).shape, vec![4, 2]);
        let stats = tensor_ref(&out[4]);
        assert_eq!(stats.shape, vec![1, 4]);
        assert_close(stats.materialize_f64()[0], 1.0);
        assert_close(stats.materialize_f64()[3], 0.0);
    }

    #[test]
    fn regress_accepts_typed_integer_design_and_response() {
        let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = crate::output_count::push_output_count(Some(1));
        let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), 4, 1, f64::NAN);
        let x = poisoned_int_tensor(
            IntegerStorage::I16(vec![1, 1, 1, 1, 0, 1, 2, 3]),
            4,
            2,
            f64::NAN,
        );
        let out = outputs(block_on(regress_builtin(y, x, Vec::new())).unwrap());
        let b = tensor_ref(&out[0]);
        assert_eq!(b.shape, vec![2, 1]);
        assert_close(b.materialize_f64()[0], 1.0);
        assert_close(b.materialize_f64()[1], 2.0);
    }

    #[test]
    fn regress_rejects_typed_integer_alpha_boundaries() {
        let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
        let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), 4, 1, 0.0);
        let x = poisoned_int_tensor(IntegerStorage::I16(vec![1, 1, 1, 1, 0, 1, 2, 3]), 4, 2, 0.0);
        let alpha = poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1, 0.5);
        let err = block_on(regress_builtin(y, x, vec![alpha])).unwrap_err();
        assert!(err.message.contains("alpha must be between 0 and 1"));
    }

    #[test]
    fn regress_omits_nan_rows_and_uses_alpha() {
        let _guard = crate::output_count::push_output_count(Some(5));
        let y = tensor(vec![1.0, f64::NAN, 5.0, 7.0], 4, 1);
        let x = tensor(vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0], 4, 2);
        let out = outputs(block_on(regress_builtin(y, x, vec![Value::Num(0.1)])).unwrap());
        let b = tensor_ref(&out[0]);
        assert_close(b.materialize_f64()[0], 1.0);
        assert_close(b.materialize_f64()[1], 2.0);
        assert_eq!(tensor_ref(&out[2]).shape, vec![3, 1]);
        assert_eq!(tensor_ref(&out[3]).shape, vec![3, 2]);
    }

    #[test]
    fn regress_supports_partial_outputs() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
        let x = tensor(vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0], 4, 2);
        let out = outputs(block_on(regress_builtin(y, x, Vec::new())).unwrap());
        assert_eq!(out.len(), 3);
        assert_eq!(tensor_ref(&out[0]).shape, vec![2, 1]);
        assert_eq!(tensor_ref(&out[1]).shape, vec![2, 2]);
        assert_eq!(tensor_ref(&out[2]).shape, vec![4, 1]);
    }

    #[test]
    fn regress_reports_nonzero_intervals_and_stats() {
        let _guard = crate::output_count::push_output_count(Some(5));
        let y = tensor(vec![1.0, 2.1, 2.9, 4.2, 4.8, 6.1], 6, 1);
        let x = tensor(
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // intercept
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, // predictor
            ],
            6,
            2,
        );
        let out = outputs(block_on(regress_builtin(y, x, Vec::new())).unwrap());

        let b = tensor_ref(&out[0]);
        assert_close_tol(b.materialize_f64()[0], 1.023809523809523, 1.0e-12);
        assert_close_tol(b.materialize_f64()[1], 0.9971428571428576, 1.0e-12);

        let bint = tensor_ref(&out[1]);
        let expected_bint = [
            0.6933332335194298,
            0.8879901308900255,
            1.3542858140996163,
            1.1062955833956896,
        ];
        for (actual, expected) in bint.materialize_f64().iter().zip(expected_bint) {
            assert_close_tol(*actual, expected, 1.0e-9);
        }

        let rint = tensor_ref(&out[3]);
        let expected_rint = [
            -0.43855599446509025,
            -0.40707908392230313,
            -0.6201647380283666,
            -0.24408652319230167,
            -0.5366307280868483,
            -0.2920062540275898,
            0.39093694684604413,
            0.565174322017542,
            0.38397426183788963,
            0.6136103327161107,
            0.1118688233249418,
            0.4729586349799683,
        ];
        for (actual, expected) in rint.materialize_f64().iter().zip(expected_rint) {
            assert_close_tol(*actual, expected, 1.0e-9);
        }

        let stats = tensor_ref(&out[4]);
        let expected_stats = [
            0.9938206296321479,
            643.3151408450694,
            1.4348829360466553e-05,
            0.027047619047619084,
        ];
        for (actual, expected) in stats.materialize_f64().iter().zip(expected_stats) {
            assert_close_tol(*actual, expected, 1.0e-9);
        }
    }

    #[test]
    fn regress_zeroes_dependent_columns_and_intervals() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
        let x = tensor(
            vec![
                1.0, 1.0, 1.0, 1.0, // intercept
                0.0, 1.0, 2.0, 3.0, // independent slope
                0.0, 2.0, 4.0, 6.0, // dependent on slope
            ],
            4,
            3,
        );
        let out = outputs(block_on(regress_builtin(y, x, Vec::new())).unwrap());
        let b = tensor_ref(&out[0]);
        assert_close(b.materialize_f64()[0], 1.0);
        assert_close(b.materialize_f64()[1], 0.0);
        assert_close(b.materialize_f64()[2], 1.0);
        let bint = tensor_ref(&out[1]);
        assert_close(bint.materialize_f64()[1], 0.0);
        assert_close(bint.materialize_f64()[4], 0.0);
    }

    #[test]
    fn regress_budgets_only_requested_outputs() {
        ensure_output_budget(MAX_REGRESS_CELLS, 1, 2).unwrap();
        assert!(ensure_output_budget(MAX_REGRESS_CELLS, 1, 3)
            .unwrap_err()
            .message
            .contains("output is too large"));
    }

    #[test]
    fn regress_stats_preserve_negative_model_sum() {
        let stats = stats_values(10.0, 5.0, 2.0, 6, 2);
        assert_close(stats[0], -1.0);
        assert_close(stats[1], -2.5);
        assert!(stats[2].is_nan());
        assert_close(stats[3], 2.0);
    }

    #[test]
    fn regress_resource_guards_cover_input_and_diagnostics() {
        assert!(ensure_input_budget(MAX_REGRESS_CELLS + 1, 1)
            .unwrap_err()
            .message
            .contains("input is too large"));
        assert!(ensure_diagnostic_budget(1_000_000, 2)
            .unwrap_err()
            .message
            .contains("diagnostic work array"));
    }

    #[test]
    fn regress_rejects_bad_dimensions_and_alpha() {
        let err = block_on(regress_builtin(
            tensor(vec![1.0, 2.0], 2, 1),
            tensor(vec![1.0, 1.0, 1.0], 3, 1),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("length(y)"));

        let err = block_on(regress_builtin(
            tensor(vec![1.0, 2.0], 2, 1),
            tensor(vec![1.0, 1.0], 2, 1),
            vec![Value::Num(1.0)],
        ))
        .unwrap_err();
        assert!(err.message.contains("alpha"));
    }
}
