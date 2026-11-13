//! Controls consumed only after the effective algorithm is trust-region.
use super::*;
use crate::builtins::math::optim::common::{self, ScalarOptionFields};
use runmat_builtins::{OptimHessianSource, OptimOptionField as Field, OptimSubproblemAlgorithm};

#[derive(Clone, Debug)]
pub(in crate::builtins::math::optim::fminunc) enum ModelSource {
    GradientDifferences { pattern: Option<Value> },
    Objective,
    Multiply { function: Value },
}

#[derive(Clone, Debug)]
pub(in crate::builtins::math::optim::fminunc) struct Controls {
    pub source: ModelSource,
    pub maximum_cg_iterations: usize,
    pub cg_tolerance: f64,
    pub bandwidth: usize,
    pub subproblem: OptimSubproblemAlgorithm,
}

impl Controls {
    pub fn from_fields(fields: &ScalarOptionFields<'_>, n: usize) -> BuiltinResult<Self> {
        let source = match fields.exact(Field::HessianFcn) {
            None => OptimHessianSource::FiniteDifference,
            Some(value) => {
                let text = common::option_string_value(Some(value), "HessianFcn", "")?;
                match OptimHessianSource::parse(&text) {
                    Some(OptimHessianSource::Objective) => OptimHessianSource::Objective,
                    _ => return Err(invalid("HessianFcn must be [] or 'objective'")),
                }
            }
        };
        // The provided Hessian takes precedence. Inactive callbacks/patterns
        // are not inspected, gathered, invoked or classified by value.
        let source = if source == OptimHessianSource::Objective {
            ModelSource::Objective
        } else if let Some(function) = fields.exact(Field::HessianMultiplyFcn) {
            if !matches!(
                function,
                Value::ScopedFunctionHandle(_)
                    | Value::FunctionHandle(_)
                    | Value::ExternalFunctionHandle(_)
                    | Value::MethodFunctionHandle(_)
                    | Value::BoundFunctionHandle(_)
                    | Value::Closure(_)
            ) {
                return Err(invalid(
                    "HessianMultiplyFcn must be a function handle or []",
                ));
            }
            ModelSource::Multiply {
                function: function.clone(),
            }
        } else {
            ModelSource::GradientDifferences {
                pattern: fields.exact(Field::HessPattern).cloned(),
            }
        };
        let requested = common::option_string_value(
            fields.exact(Field::SubproblemAlgorithm),
            "SubproblemAlgorithm",
            "cg",
        )?;
        let mut subproblem = OptimSubproblemAlgorithm::parse(&requested)
            .ok_or_else(|| invalid("SubproblemAlgorithm must be 'cg' or 'factorization'"))?;
        let mut bandwidth = 0;
        if subproblem == OptimSubproblemAlgorithm::Cg {
            if let Some(value) = fields.exact(Field::PrecondBandWidth) {
                let direct = match value {
                    Value::Num(value) => *value == f64::INFINITY,
                    Value::Tensor(value)
                        if value.len() == 1 && value.integer_storage().is_none() =>
                    {
                        crate::builtins::common::tensor::tensor_value_f64(value, 0) == f64::INFINITY
                    }
                    _ => false,
                };
                if direct {
                    subproblem = OptimSubproblemAlgorithm::Factorization;
                } else {
                    bandwidth = count(value, "PrecondBandWidth", false)?;
                }
            }
        }
        let (maximum_cg_iterations, cg_tolerance) = if subproblem == OptimSubproblemAlgorithm::Cg {
            let maximum = fields
                .exact(Field::MaxPCGIter)
                .map(|value| count(value, "MaxPCGIter", true))
                .transpose()?
                .unwrap_or((n / 2).max(1));
            let tolerance =
                common::option_f64_value(NAME, fields.exact(Field::TolPCG), "TolPCG", 0.1)?;
            if tolerance <= 0.0 {
                return Err(invalid("TolPCG must be positive"));
            }
            (maximum, tolerance)
        } else {
            // PCG controls have no numerical role in the direct algorithm.
            (0, 0.0)
        };
        Ok(Self {
            source,
            maximum_cg_iterations,
            cg_tolerance,
            bandwidth,
            subproblem,
        })
    }
}

fn count(value: &Value, field: &str, positive: bool) -> BuiltinResult<usize> {
    let parsed = if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value)
    {
        integer.try_to_usize().ok_or_else(|| {
            invalid(format!(
                "{field} must be a representable nonnegative integer"
            ))
        })?
    } else {
        let value = common::option_f64_value(NAME, Some(value), field, 0.0)?;
        if value < 0.0 || value.fract() != 0.0 || value >= 2f64.powi(usize::BITS as i32) {
            return Err(invalid(format!(
                "{field} must be a representable nonnegative integer"
            )));
        }
        value as usize
    };
    if positive && parsed == 0 {
        return Err(invalid(format!("{field} must be positive")));
    }
    Ok(parsed)
}
