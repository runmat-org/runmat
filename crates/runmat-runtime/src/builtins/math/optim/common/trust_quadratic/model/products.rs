//! One scoped product view per accepted model generation. The existing
//! borrowed typed operator carries canonical topology validity; the owned
//! Model remains outside this scope and cannot move while it is borrowed.
use super::*;
use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator;

pub(crate) enum Products<'a> {
    Matrix {
        source: &'a Value,
        operator: TypedLinearOperator<'a, f64>,
        policy: Policy,
    },
    Multiply {
        info: &'a mut Option<RootedValue>,
        function: &'a Value,
        policy: Policy,
    },
}

impl Products<'_> {
    /// Only explicit normalized coefficients provide this norm bound. An
    /// opaque multiply callback is not certified by sampling basis products.
    pub(crate) async fn magnitude(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>> {
        let Self::Matrix { source, .. } = self else {
            return Ok(None);
        };
        let source = match *source {
            Value::Num(value) => return Ok(value.is_finite().then_some(value.abs())),
            Value::Tensor(value) => value.as_f64_slice(),
            Value::SparseTensor(value) => value.as_f64_slice(),
            _ => None,
        };
        let Some(source) = source else {
            return Ok(None);
        };
        let mut scale = 0.0_f64;
        for chunk in source.chunks(1024) {
            for &value in chunk {
                if !value.is_finite() {
                    return Ok(None);
                }
                scale = scale.max(value.abs());
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        Ok(Some(scale))
    }
    /// Read a retained explicit matrix's requested band directly. An opaque
    /// carrier has no such structural authority and must use real products.
    pub(crate) async fn copy_band(
        &self,
        n: usize,
        bandwidth: usize,
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let policy = self.policy();
        let Self::Matrix { source, .. } = self else {
            return Ok(false);
        };
        let stride = bandwidth + 1;
        match *source {
            Value::Num(value) => output[0] = *value,
            Value::Tensor(value) => {
                let values = value
                    .as_f64_slice()
                    .ok_or_else(|| policy.invalid("normalized Hessian has no double backing"))?;
                for row in 0..n {
                    for column in row.saturating_sub(bandwidth)..=row {
                        output[row * stride + row - column] = values[column * n + row];
                        cooperation.completed(1).await?;
                    }
                }
            }
            Value::SparseTensor(value) => {
                let values = value.as_f64_slice().ok_or_else(|| {
                    policy.invalid("normalized sparse Hessian has no double backing")
                })?;
                for column in 0..n {
                    for index in value.col_ptrs[column]..value.col_ptrs[column + 1] {
                        let row = value.row_indices[index];
                        if row >= column && row - column <= bandwidth {
                            output[row * stride + row - column] = values[index];
                        }
                        cooperation.completed(1).await?;
                    }
                    cooperation.completed(1).await?;
                }
            }
            _ => return Err(policy.invalid("invalid normalized Hessian storage")),
        }
        Ok(true)
    }

    /// Direct mode owns one dense numerical input, filled from explicit
    /// storage without n redundant matrix products or a second real matrix.
    pub(crate) async fn copy_matrix(
        &self,
        n: usize,
        output: &mut Vec<num_complex::Complex64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        use num_complex::Complex64;
        let policy = self.policy();
        let Self::Matrix { source, .. } = self else {
            return Ok(false);
        };
        match *source {
            Value::Num(value) => output.push(Complex64::new(*value, 0.0)),
            Value::Tensor(value) => {
                let values = value
                    .as_f64_slice()
                    .ok_or_else(|| policy.invalid("normalized Hessian has no double backing"))?;
                for chunk in values.chunks(1024) {
                    output.extend(chunk.iter().map(|&value| Complex64::new(value, 0.0)));
                    cooperation.completed(chunk.len() as u64).await?;
                }
            }
            Value::SparseTensor(value) => {
                let length = n
                    .checked_mul(n)
                    .ok_or_else(|| policy.invalid("direct Hessian extent overflow"))?;
                while output.len() < length {
                    let end = output.len().saturating_add(1024).min(length);
                    let count = end - output.len();
                    output.resize(end, Complex64::default());
                    cooperation.completed(count as u64).await?;
                }
                let values = value.as_f64_slice().ok_or_else(|| {
                    policy.invalid("normalized sparse Hessian has no double backing")
                })?;
                for column in 0..n {
                    for index in value.col_ptrs[column]..value.col_ptrs[column + 1] {
                        output[column * n + value.row_indices[index]] =
                            Complex64::new(values[index], 0.0);
                        cooperation.completed(1).await?;
                    }
                    cooperation.completed(1).await?;
                }
            }
            _ => return Err(policy.invalid("invalid normalized Hessian storage")),
        }
        Ok(true)
    }

    pub async fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.apply_values(input, output, cooperation, false)
            .await
            .map(|_| ())
    }
    /// Pure explicit-matrix arithmetic can be numerically undefined. Opaque
    /// callback, admission, cancellation and service failures stay outer errors.
    pub(crate) async fn apply_numerical(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        self.apply_values(input, output, cooperation, true).await
    }
    async fn apply_values(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
        numerical: bool,
    ) -> BuiltinResult<Option<()>> {
        let policy = self.policy();
        let explicit = matches!(self, Self::Matrix { .. });
        match self {
            Self::Matrix { operator, .. } => {
                let (runtime, budget) = cooperation.runtime_budget();
                match (runtime, budget) {
                    (Some(runtime), Some(budget)) => operator
                        .apply_cooperatively(input, output, runtime, budget)
                        .await
                        .map_err(|source| operator_error(source, Some(runtime), policy))?,
                    (runtime, _) => {
                        operator
                            .apply_into_unbounded(input, output)
                            .map_err(|source| {
                                operator_error(
                                    CooperativeOperatorError::Operator(source),
                                    runtime,
                                    policy,
                                )
                            })?
                    }
                }
            }
            Self::Multiply { info, function, .. } => {
                let info = &mut **info;
                let function = *function;
                let runtime = cooperation.runtime().cloned();
                let y =
                    common::output_callback::vector(input, &[input.len(), 1], false, cooperation)
                        .await
                        .map_err(|error| policy.admission(error))?;
                let original = info
                    .take()
                    .expect("Hinfo owner is present outside one exclusive product");
                let product_cooperation = &mut *cooperation;
                let (values, restored) = original
                    .with_owned_async(|info| async move {
                        let args = [info, y];
                        let result =
                            common::call_function_in_context(runtime.as_ref(), function, &args, 1)
                                .await?;
                        let values = result
                            .with_owned_async(|value| async move {
                                if let Some(preflight) = policy.value_source()?.product_preflight {
                                    preflight(&value, input.len(), product_cooperation.runtime())?;
                                }
                                vector_geometry(&value, input.len(), policy)?;
                                common::value_to_real_vector_with_extensions(
                                    policy.identity.name,
                                    value,
                                    policy.value_source()?.numeric,
                                    policy.value_source()?.resident,
                                    "Hessian product",
                                    product_cooperation,
                                )
                                .await
                            })
                            .await
                            .map_err(|error| policy.admission(error))?;
                        let [info, _] = args;
                        // Successor registration precedes retirement of the old
                        // rooted argument generation; there is no unrooted gap.
                        let restored = RootedValue::in_optional_context(info, runtime.as_ref())
                            .map_err(|error| policy.admission(error))?;
                        Ok::<_, RuntimeError>((values, restored))
                    })
                    .await?;
                *info = Some(restored);
                if values.len() != output.len() {
                    return Err(policy.invalid("Hessian product changed its output extent"));
                }
                for (to, from) in output.chunks_mut(1024).zip(values.chunks(1024)) {
                    to.copy_from_slice(from);
                    cooperation.completed(to.len() as u64).await?;
                }
            }
        }
        for chunk in output.chunks(1024) {
            if chunk.iter().any(|value| !value.is_finite()) {
                if numerical && explicit {
                    return Ok(None);
                }
                return Err(policy.invalid("Hessian product contains a nonfinite value"));
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        Ok(Some(()))
    }
}

impl Operator for Products<'_> {
    fn policy(&self) -> Policy {
        match self {
            Self::Matrix { policy, .. } | Self::Multiply { policy, .. } => *policy,
        }
    }
    async fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        Products::apply(self, input, output, cooperation)
            .await
            .map(Some)
    }
    async fn copy_band(
        &self,
        n: usize,
        bandwidth: usize,
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Materialization> {
        Products::copy_band(self, n, bandwidth, output, cooperation)
            .await
            .map(|copied| {
                if copied {
                    Materialization::Complete
                } else {
                    Materialization::Unavailable
                }
            })
    }
    async fn copy_matrix(
        &mut self,
        n: usize,
        output: &mut Vec<num_complex::Complex64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Materialization> {
        Products::copy_matrix(self, n, output, cooperation)
            .await
            .map(|copied| {
                if copied {
                    Materialization::Complete
                } else {
                    Materialization::Unavailable
                }
            })
    }
}
