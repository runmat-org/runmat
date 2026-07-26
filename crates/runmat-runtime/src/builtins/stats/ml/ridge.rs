//! Ridge regression compatibility surface.

use nalgebra::{DMatrix, DVector};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "ridge";
const EPS: f64 = 1.0e-12;

const OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Coefficient estimates, one column per ridge parameter.",
}];

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response vector with one value per observation.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predictor matrix with observations in rows and predictors in columns.",
};

const PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative ridge parameters.",
};

const PARAM_SCALED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scaled",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Scaling flag: 1 returns standardized coefficients, 0 restores original scale and intercept.",
};

const INPUTS_Y_X_K: [BuiltinParamDescriptor; 3] = [PARAM_Y, PARAM_X, PARAM_K];
const INPUTS_Y_X_K_SCALED: [BuiltinParamDescriptor; 4] = [PARAM_Y, PARAM_X, PARAM_K, PARAM_SCALED];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = ridge(y, X, k)",
        inputs: &INPUTS_Y_X_K,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = ridge(y, X, k, scaled)",
        inputs: &INPUTS_Y_X_K_SCALED,
        outputs: &OUTPUT_B,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIDGE.INVALID_ARGUMENT",
    identifier: Some("RunMat:ridge:InvalidArgument"),
    when: "Inputs, dimensions, ridge parameters, or scaling flag are malformed.",
    message: "ridge: invalid argument",
};

const ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIDGE.NUMERICAL",
    identifier: Some("RunMat:ridge:Numerical"),
    when: "The regularized normal equations cannot be solved numerically.",
    message: "ridge: numerical solve failed",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIDGE.INTERNAL",
    identifier: Some("RunMat:ridge:Internal"),
    when: "RunMat cannot allocate or construct ridge outputs.",
    message: "ridge: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_NUMERICAL, ERROR_INTERNAL];

pub const RIDGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn ridge_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![None, None]),
    }
}

fn ridge_error(
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
    ridge_error(message, &ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    ridge_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Debug)]
struct PreparedRidge {
    singular_values: Vec<f64>,
    right_singular_vectors: DMatrix<f64>,
    u_transpose_y: Vec<f64>,
    x_means: Vec<f64>,
    x_stds: Vec<f64>,
    y_mean: f64,
    sv_tolerance: f64,
    predictors: usize,
}

#[runtime_builtin(
    name = "ridge",
    category = "stats/ml",
    summary = "Fit ridge regression coefficients for one or more regularization parameters.",
    keywords = "ridge,regression,regularization,statistics,machine learning",
    type_resolver(ridge_type),
    descriptor(crate::builtins::stats::ml::ridge::RIDGE_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::ridge"
)]
async fn ridge_builtin(y: Value, x: Value, k: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid_argument(
            "ridge: accepts at most one scaled argument",
        ));
    }
    let y = value_to_tensor(y).await?;
    let x = value_to_tensor(x).await?;
    let k = value_to_tensor(k).await?;
    let scaled = if let Some(value) = rest.first() {
        parse_scaled(value).await?
    } else {
        true
    };
    ridge_compute(y, x, k, scaled).map(Value::Tensor)
}

async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("ridge: {err}")))?;
    tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid_argument(format!("ridge: {err}")))
}

async fn parse_scaled(value: &Value) -> BuiltinResult<bool> {
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|err| invalid_argument(format!("ridge: {err}")))?;
    let raw = match gathered {
        Value::Bool(value) => return Ok(value),
        Value::Num(value) => value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64(&tensor);
            if values.len() != 1 {
                return Err(invalid_argument(format!(
                    "ridge: scaled must be 0, 1, false, or true, got {:?}",
                    Value::Tensor(tensor)
                )));
            }
            values[0]
        }
        other => {
            return Err(invalid_argument(format!(
                "ridge: scaled must be 0, 1, false, or true, got {other:?}"
            )));
        }
    };
    if (raw - 0.0).abs() <= EPS {
        Ok(false)
    } else if (raw - 1.0).abs() <= EPS {
        Ok(true)
    } else {
        Err(invalid_argument(
            "ridge: scaled must be 0, 1, false, or true",
        ))
    }
}

fn ridge_compute(y: Tensor, x: Tensor, k: Tensor, scaled: bool) -> BuiltinResult<Tensor> {
    let y_values = vector_values(&y, "y")?;
    if x.shape.len() > 2 {
        return Err(invalid_argument("ridge: X must be a 2-D numeric matrix"));
    }
    if x.rows != y_values.len() {
        return Err(invalid_argument(
            "ridge: length(y) must match the number of rows in X",
        ));
    }
    let lambdas = ridge_parameters(&k)?;
    let prepared = prepare_data(&x, &y_values)?;
    let rows = if scaled {
        prepared.predictors
    } else {
        prepared.predictors + 1
    };
    let cols = lambdas.len();
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| internal_error("ridge: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error("ridge: output allocation failed"))?;
    for lambda in lambdas {
        let beta_scaled = solve_ridge(&prepared, lambda)?;
        if scaled {
            out.extend(beta_scaled.iter().copied());
        } else {
            let beta_original = beta_scaled
                .iter()
                .zip(prepared.x_stds.iter())
                .map(|(coef, std)| if *std > EPS { coef / std } else { 0.0 })
                .collect::<Vec<_>>();
            let intercept = prepared.y_mean
                - prepared
                    .x_means
                    .iter()
                    .zip(beta_original.iter())
                    .map(|(mean, coef)| mean * coef)
                    .sum::<f64>();
            out.push(intercept);
            out.extend(beta_original);
        }
    }
    Tensor::new(out, vec![rows, cols]).map_err(|err| internal_error(format!("ridge: {err}")))
}

fn vector_values(tensor: &Tensor, label: &str) -> BuiltinResult<Vec<f64>> {
    if tensor.shape.len() > 2 || !(tensor.rows == 1 || tensor.cols == 1) {
        return Err(invalid_argument(format!("ridge: {label} must be a vector")));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

fn ridge_parameters(tensor: &Tensor) -> BuiltinResult<Vec<f64>> {
    let values = vector_values(tensor, "k")?;
    if values.is_empty() {
        return Err(invalid_argument(
            "ridge: k must contain at least one ridge parameter",
        ));
    }
    for value in &values {
        if !value.is_finite() || *value < 0.0 {
            return Err(invalid_argument(
                "ridge: k values must be nonnegative finite scalars",
            ));
        }
    }
    Ok(values)
}

fn prepare_data(x: &Tensor, y: &[f64]) -> BuiltinResult<PreparedRidge> {
    let rows = x.rows;
    let cols = x.cols;
    let x_values = tensor::tensor_values_f64_cow(x);
    let mut clean_rows = Vec::with_capacity(rows);
    for row in 0..rows {
        let y_value = y[row];
        let mut has_nan = y_value.is_nan();
        let mut has_nonfinite = y_value.is_infinite();
        for col in 0..cols {
            let value = x_value(&x_values, rows, row, col);
            has_nan |= value.is_nan();
            has_nonfinite |= value.is_infinite();
        }
        if has_nonfinite {
            return Err(invalid_argument(
                "ridge: X and y must not contain Inf values",
            ));
        }
        if !has_nan {
            clean_rows.push(row);
        }
    }
    if clean_rows.len() < 2 {
        return Err(invalid_argument(
            "ridge: at least two complete observations are required",
        ));
    }
    let n = clean_rows.len();
    let mut x_means = vec![0.0; cols];
    let mut y_mean = 0.0;
    for &row in &clean_rows {
        y_mean += y[row];
        for (col, mean) in x_means.iter_mut().enumerate().take(cols) {
            *mean += x_value(&x_values, rows, row, col);
        }
    }
    y_mean /= n as f64;
    for mean in &mut x_means {
        *mean /= n as f64;
    }
    let mut x_stds = vec![0.0; cols];
    for &row in &clean_rows {
        for (col, std) in x_stds.iter_mut().enumerate().take(cols) {
            let diff = x_value(&x_values, rows, row, col) - x_means[col];
            *std += diff * diff;
        }
    }
    for std in &mut x_stds {
        *std = (*std / (n - 1) as f64).sqrt();
        if *std <= EPS {
            *std = 1.0;
        }
    }
    let mut z_data = Vec::new();
    z_data
        .try_reserve(
            n.checked_mul(cols)
                .ok_or_else(|| internal_error("ridge: standardized design matrix size overflow"))?,
        )
        .map_err(|_| internal_error("ridge: standardized design matrix allocation failed"))?;
    for col in 0..cols {
        for &row in &clean_rows {
            z_data.push((x_value(&x_values, rows, row, col) - x_means[col]) / x_stds[col]);
        }
    }
    let y_centered = clean_rows
        .iter()
        .map(|&row| y[row] - y_mean)
        .collect::<Vec<_>>();
    let y_centered = DVector::from_column_slice(&y_centered);
    let z = DMatrix::from_column_slice(n, cols, &z_data);
    let svd = z.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| internal_error("ridge: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| internal_error("ridge: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.iter().copied().collect::<Vec<_>>();
    let largest_sv = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let sv_tolerance = (n.max(cols) as f64) * f64::EPSILON * largest_sv.max(1.0);
    let u_transpose_y = (0..singular_values.len())
        .map(|index| u.column(index).dot(&y_centered))
        .collect::<Vec<_>>();
    Ok(PreparedRidge {
        singular_values,
        right_singular_vectors: v_t.transpose(),
        u_transpose_y,
        x_means,
        x_stds,
        y_mean,
        sv_tolerance,
        predictors: cols,
    })
}

fn solve_ridge(data: &PreparedRidge, lambda: f64) -> BuiltinResult<DVector<f64>> {
    let mut beta = DVector::zeros(data.predictors);
    for (index, singular_value) in data.singular_values.iter().copied().enumerate() {
        let denominator = singular_value.mul_add(singular_value, lambda);
        if denominator <= data.sv_tolerance {
            continue;
        }
        let contribution = (singular_value / denominator) * data.u_transpose_y[index];
        for row in 0..data.predictors {
            beta[row] += data.right_singular_vectors[(row, index)] * contribution;
        }
    }
    Ok(beta)
}

fn x_value(values: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    values[col * rows + row]
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn poisoned_int_tensor(
        storage: IntegerStorage,
        rows: usize,
        cols: usize,
        poison: f64,
    ) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        tensor.data.fill(poison);
        Value::Tensor(tensor)
    }

    fn tensor_out(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn scaled_false_restores_intercept_and_original_scale() {
        let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
        let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let out = block_on(ridge_builtin(y, x, Value::Num(0.0), vec![Value::Num(0.0)])).unwrap();
        let out = tensor_out(out);
        assert_eq!(out.shape, vec![2, 1]);
        assert!((out.data[0] - 1.0).abs() < 1.0e-10);
        assert!((out.data[1] - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn ridge_accepts_typed_integer_response_design_and_k() {
        let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), 4, 1, f64::NAN);
        let x = poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1, f64::NAN);
        let k = poisoned_int_tensor(IntegerStorage::U8(vec![0, 1]), 1, 2, f64::NAN);
        let scaled = poisoned_int_tensor(IntegerStorage::U8(vec![0]), 1, 1, 1.0);
        let out = block_on(ridge_builtin(y, x, k, vec![scaled])).unwrap();
        let out = tensor_out(out);
        assert_eq!(out.shape, vec![2, 2]);
        assert!((out.data[0] - 1.0).abs() < 1.0e-10);
        assert!((out.data[1] - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn scaled_true_returns_standardized_coefficients_for_each_k() {
        let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
        let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let out = block_on(ridge_builtin(
            y,
            x,
            tensor(vec![0.0, 1.0], 1, 2),
            Vec::new(),
        ))
        .unwrap();
        let out = tensor_out(out);
        assert_eq!(out.shape, vec![1, 2]);
        assert!(out.data[0] > out.data[1]);
        assert!((out.data[0] - 2.581_988_897).abs() < 1.0e-8);
    }

    #[test]
    fn rows_with_nan_are_omitted() {
        let y = tensor(vec![1.0, f64::NAN, 5.0, 7.0], 4, 1);
        let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let out = block_on(ridge_builtin(y, x, Value::Num(0.0), vec![Value::Num(0.0)])).unwrap();
        let out = tensor_out(out);
        assert!((out.data[0] - 1.0).abs() < 1.0e-10);
        assert!((out.data[1] - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let y = tensor(vec![1.0, 2.0], 2, 1);
        let x = tensor(vec![1.0, 2.0], 2, 1);
        assert!(block_on(ridge_builtin(
            y.clone(),
            x.clone(),
            Value::Num(-1.0),
            Vec::new()
        ))
        .is_err());
        assert!(block_on(ridge_builtin(y, x, Value::Num(0.0), vec![Value::Num(2.0)])).is_err());
    }

    #[test]
    fn rank_deficient_k_zero_uses_minimum_norm_solution() {
        let y = tensor(vec![2.0, 4.0, 6.0], 3, 1);
        let x = tensor(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 3, 2);
        let out = block_on(ridge_builtin(y, x, Value::Num(0.0), vec![Value::Num(0.0)])).unwrap();
        let out = tensor_out(out);
        assert_eq!(out.shape, vec![3, 1]);
        assert!(out.data[0].abs() < 1.0e-10);
        assert!((out.data[1] - 1.0).abs() < 1.0e-10);
        assert!((out.data[2] - 0.5).abs() < 1.0e-10);
    }
}
