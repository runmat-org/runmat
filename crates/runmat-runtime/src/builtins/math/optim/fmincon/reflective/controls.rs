//! Only the selected reflective model's live controls are interpreted.
use super::super::invalid;
use super::restrictions::Mode;
use crate::builtins::math::optim::common::{self, ScalarOptionFields};
use crate::BuiltinResult;
use runmat_builtins::{OptimHessianSource, OptimOptionField as Field, OptimSubproblemAlgorithm};
use runmat_value::Value;

pub(super) enum Source {
    Objective,
    Multiply { function: Value },
    GradientDifferences { pattern: Option<Value> },
}
pub(in crate::builtins::math::optim::fmincon) struct Controls {
    pub(super) source: Source,
    pub(super) subproblem: OptimSubproblemAlgorithm,
    pub(super) bandwidth: usize,
    pub(super) maximum_cg_iterations: usize,
    pub(super) cg_tolerance: f64,
    pub(in crate::builtins::math::optim::fmincon) function_tolerance: f64,
}
impl Controls {
    pub(in crate::builtins::math::optim::fmincon) fn parse(
        fields: &ScalarOptionFields<'_>,
        n: usize,
        mode: Mode,
    ) -> BuiltinResult<Self> {
        let supplied = fields
            .exact(Field::HessianFcn)
            .map(|value| common::option_string_value(Some(value), "HessianFcn", ""))
            .transpose()?
            .map(|text| {
                OptimHessianSource::parse(&text)
                    .ok_or_else(|| invalid("HessianFcn must be [] or 'objective'"))
            })
            .transpose()?
            .unwrap_or(OptimHessianSource::FiniteDifference);
        let source = if supplied == OptimHessianSource::Objective {
            Source::Objective
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
                    "HessianMultiplyFcn must be a single function handle or []",
                ));
            }
            Source::Multiply {
                function: function.clone(),
            }
        } else {
            Source::GradientDifferences {
                pattern: fields.exact(Field::HessPattern).cloned(),
            }
        };
        let subproblem =
            OptimSubproblemAlgorithm::parse(&fields.text(Field::SubproblemAlgorithm, "cg")?)
                .ok_or_else(|| invalid("SubproblemAlgorithm must be 'cg' or 'factorization'"))?;
        let mut direct = subproblem == OptimSubproblemAlgorithm::Factorization;
        let mut bandwidth = 0;
        if !direct {
            if let Some(value) = fields.exact(Field::PrecondBandWidth) {
                direct = match value {
                    Value::Num(value) => *value == f64::INFINITY,
                    Value::Tensor(value)
                        if value.len() == 1 && value.integer_storage().is_none() =>
                    {
                        crate::builtins::common::tensor::tensor_value_f64(value, 0) == f64::INFINITY
                    }
                    _ => false,
                };
                if !direct {
                    bandwidth = fields.count("fmincon", Field::PrecondBandWidth, 0)?;
                }
            }
        }
        if direct && matches!(source, Source::Multiply { .. }) {
            return Err(invalid("trust-region-reflective HessianMultiplyFcn requires SubproblemAlgorithm='cg' without infinite PrecondBandWidth"));
        }
        let (maximum_cg_iterations, cg_tolerance) = if direct {
            (0, 0.0)
        } else {
            let iterations = fields.count(
                "fmincon",
                Field::MaxPCGIter,
                match mode {
                    Mode::Bounds => (n / 2).max(1),
                    Mode::Equalities => n,
                },
            )?;
            let tolerance = fields.real("fmincon", Field::TolPCG, 0.1)?;
            if iterations == 0 || tolerance <= 0.0 {
                return Err(invalid("MaxPCGIter and TolPCG must be positive"));
            }
            (iterations, tolerance)
        };
        let modern = fields.exact(Field::FunctionTolerance);
        let legacy = fields.exact(Field::TolFun);
        if modern.is_some() && legacy.is_some() {
            return Err(invalid(
                "conflicting legacy TolFun and FunctionTolerance controls",
            ));
        }
        let function_tolerance =
            common::option_f64_value("fmincon", modern.or(legacy), "FunctionTolerance", 1e-6)?;
        if function_tolerance < 0.0 {
            return Err(invalid("FunctionTolerance must be nonnegative"));
        }
        Ok(Self {
            source,
            subproblem: if direct {
                OptimSubproblemAlgorithm::Factorization
            } else {
                OptimSubproblemAlgorithm::Cg
            },
            bandwidth,
            maximum_cg_iterations,
            cg_tolerance,
            function_tolerance,
        })
    }
}
