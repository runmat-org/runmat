//! User Jacobians share the same callback evaluation and numeric boundary as F.
use super::*;

impl FsolveEvaluator<'_> {
    pub(super) async fn residual_with_jacobian(&mut self, x: &[f64]) -> BuiltinResult<Vec<f64>> {
        self.cooperation.completed(0).await?;
        let arg = crate::builtins::math::optim::common::output_callback::vector(
            x,
            &self.shape,
            self.scalar,
            self.cooperation,
        )
        .await?;
        let result =
            call_function_in_context(self.cooperation.runtime(), self.function, [arg], 2).await?;
        result
            .with_owned_async(|value| async move {
                let Value::OutputList(outputs) = value else {
                    return Err(fsolve_error_with_detail(
                        &FSOLVE_ERROR_INVALID_INPUT,
                        "SpecifyObjectiveGradient requires [F,J] from the objective",
                    ));
                };
                if outputs.len() != 2 {
                    return Err(fsolve_error_with_detail(
                        &FSOLVE_ERROR_INVALID_INPUT,
                        "objective must return exactly F and J",
                    ));
                }
                let outputs = outputs.try_into_owned().map_err(|_| {
                    crate::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?;
                let mut outputs = outputs.into_iter();
                let residual = self
                    .prepare_numeric(outputs.next().expect("two outputs"), "function residual")
                    .await?;
                let (shape, scalar) = match &residual {
                    Value::Num(_) | Value::Int(_) | Value::Bool(_) => (vec![1, 1], true),
                    Value::Tensor(value) => (value.shape().to_vec(), false),
                    Value::LogicalArray(value) => (value.shape.to_vec(), false),
                    _ => {
                        return Err(fsolve_error_with_detail(
                            &FSOLVE_ERROR_INVALID_INPUT,
                            "objective residual must be real numeric",
                        ))
                    }
                };
                let residual =
                    value_to_real_vector_with_cooperation(NAME, residual, self.cooperation).await?;
                if residual.is_empty() {
                    return Err(fsolve_error_with_detail(
                        &FSOLVE_ERROR_INVALID_INPUT,
                        "function value must not be empty",
                    ));
                }
                self.accept_residual_shape(shape, scalar)?;
                let jacobian = outputs.next().expect("two outputs");
                let m = residual.len();
                let n = x.len();
                let row_major = if let Value::SparseTensor(sparse) = jacobian {
                    super::sparse_jacobian::row_major(&sparse, m, n, self.cooperation).await?
                } else {
                    let jacobian = self.prepare_numeric(jacobian, "objective Jacobian").await?;
                    let valid_shape = match &jacobian {
                        Value::Num(_) | Value::Int(_) | Value::Bool(_) => m == 1 && n == 1,
                        Value::Tensor(value) => value.shape() == [m, n],
                        Value::LogicalArray(value) => value.shape == [m, n],
                        _ => false,
                    };
                    if !valid_shape {
                        return Err(fsolve_error_with_detail(
                            &FSOLVE_ERROR_INVALID_INPUT,
                            "Jacobian must have one row per residual and one column per variable",
                        ));
                    }
                    let count =
                        crate::builtins::math::optim::least_squares::checked_least_squares_work(
                            NAME, m, n,
                        )?;
                    let mut values = runmat_value::exact_capacity_vec(count).map_err(|source| {
                        crate::build_runtime_error("Jacobian allocation failed")
                            .with_source(source)
                            .build()
                    })?;
                    for first in (0..count).step_by(1024) {
                        let end = first.saturating_add(1024).min(count);
                        for index in first..end {
                            let row = index / n;
                            let col = index % n;
                            let value = match &jacobian {
                                Value::Num(value) => *value,
                                Value::Int(value) => value.to_f64(),
                                Value::Bool(value) => {
                                    if *value {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                }
                                Value::Tensor(value) => {
                                    crate::builtins::common::tensor::tensor_value_f64(
                                        value,
                                        col * m + row,
                                    )
                                }
                                Value::LogicalArray(value) => {
                                    if value.data[col * m + row] == 0 {
                                        0.0
                                    } else {
                                        1.0
                                    }
                                }
                                _ => unreachable!("admitted real Jacobian"),
                            };
                            if !value.is_finite() {
                                return Err(fsolve_error_with_detail(
                                    &FSOLVE_ERROR_INVALID_INPUT,
                                    "Jacobian must contain finite values",
                                ));
                            }
                            values.push(value);
                        }
                        self.cooperation.completed((end - first) as u64).await?;
                    }
                    values
                };
                self.jacobian = Some(row_major);
                Ok(residual)
            })
            .await
    }

    async fn prepare_numeric(&mut self, value: Value, role: &str) -> BuiltinResult<Value> {
        crate::builtins::math::optim::common::prepare_floating_value_with_cooperation(
            NAME,
            value,
            &FSOLVE_CALLBACK_NUMERIC_EXTENSION,
            &FSOLVE_RESIDENT_EXTENSION,
            role,
            self.cooperation,
        )
        .await
    }
}
