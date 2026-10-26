//! One typed public control owner; no accepted algorithm aliases or ignored keys.
use super::derivatives::Difference;
use crate::builtins::math::optim::common::{self, ScalarCooperation, ScalarOptionFields};
use crate::BuiltinResult;
use runmat_builtins::{OptimAlgorithm, OptimDisplay, OptimOptionField as Field, OptimSolver};
use runmat_value::Value;

pub(super) struct Controls {
    pub algorithm: OptimAlgorithm,
    pub max_iterations: usize,
    pub point_limit: Option<usize>,
    pub feasibility: f64,
    pub optimality: f64,
    pub step: f64,
    pub objective_gradient: bool,
    pub constraint_gradient: bool,
    pub display: OptimDisplay,
    pub differences: Difference,
    pub scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling,
    prepared: Option<common::PreparedSolverOptions>,
}

impl Controls {
    pub(super) fn prepared_value(&self) -> Value {
        Value::Struct(
            self.prepared
                .as_ref()
                .map(|value| value.structure().clone())
                .unwrap_or_else(|| runmat_value::UnboundedStructValueBuilder::new().finish()),
        )
    }
    pub(super) fn selected_algorithm(value: &Value) -> BuiltinResult<OptimAlgorithm> {
        let structure = match value {
            Value::Struct(value) => Some(value),
            Value::Tensor(value) if value.is_empty() => None,
            _ => return Err(super::invalid("options must be [] or a scalar structure")),
        };
        let fields = ScalarOptionFields::from_struct(structure);
        OptimAlgorithm::parse(&fields.text(Field::Algorithm, "interior-point")?)
            .filter(|algorithm| {
                matches!(
                    algorithm,
                    OptimAlgorithm::InteriorPoint
                        | OptimAlgorithm::Sqp
                        | OptimAlgorithm::ActiveSet
                        | OptimAlgorithm::TrustRegionReflective
                )
            })
            .ok_or_else(|| {
                super::invalid("unsupported fmincon algorithm; sqp-legacy is not implemented")
            })
    }
    pub fn callback(&self) -> Option<&Value> {
        ScalarOptionFields::from_struct(self.prepared.as_ref().map(|value| value.structure()))
            .exact(Field::OutputFcn)
    }
    pub(super) fn reflective(
        &self,
        n: usize,
        mode: super::reflective::Mode,
    ) -> BuiltinResult<super::reflective::Controls> {
        super::reflective::Controls::parse(
            &ScalarOptionFields::from_struct(self.prepared.as_ref().map(|value| value.structure())),
            n,
            mode,
        )
    }
    pub async fn parse(
        value: Value,
        nonlinear_constraints: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let algorithm = Self::selected_algorithm(&value)?;
        let options = match value {
            Value::Struct(value) => Some(value),
            Value::Tensor(value) if value.is_empty() => None,
            _ => return Err(super::invalid("options must be [] or a scalar structure")),
        };
        if let Some(options) = &options {
            for (name, value) in &options.fields {
                if name.eq_ignore_ascii_case("Solver") {
                    if OptimSolver::parse(&common::field_name(value)?) != Some(OptimSolver::Fmincon)
                    {
                        return Err(super::invalid("options belong to another solver"));
                    }
                } else if Field::parse_for_solver(name, OptimSolver::Fmincon).is_none() {
                    return Err(super::invalid(format!("unsupported fmincon option {name}")));
                }
            }
        }
        // Algorithm is scalar text. Select the real mechanism before any
        // option/data readback or objective/constraint invocation.
        if algorithm != OptimAlgorithm::TrustRegionReflective {
            let fields = ScalarOptionFields::from_struct(options.as_ref());
            for field in [
                Field::FunctionTolerance,
                Field::HessianFcn,
                Field::HessianMultiplyFcn,
                Field::HessPattern,
                Field::MaxPCGIter,
                Field::TolPCG,
                Field::PrecondBandWidth,
                Field::SubproblemAlgorithm,
            ] {
                if fields.exact(field).is_some()
                    && (algorithm == OptimAlgorithm::InteriorPoint
                        || field == Field::FunctionTolerance)
                {
                    return Err(super::invalid(format!(
                        "{} is not implemented for the selected {} algorithm",
                        field.name(),
                        algorithm.name()
                    )));
                }
            }
        }
        let runtime = cooperation.runtime().cloned();
        let prepared = common::prepare_fmincon_options(
            options,
            nonlinear_constraints,
            Some(&runmat_builtins::FMINCON_INTEGER_OPTION_EXTENSION),
            Some(&runmat_builtins::FMINCON_RESIDENT_EXTENSION),
            runtime.as_ref(),
            cooperation,
        )
        .await?;
        let fields =
            ScalarOptionFields::from_struct(prepared.as_ref().map(|value| value.structure()));
        let modern = fields.exact(Field::OptimalityTolerance);
        let legacy = (algorithm != OptimAlgorithm::TrustRegionReflective)
            .then(|| fields.exact(Field::TolFun))
            .flatten();
        if modern.is_some() && legacy.is_some() {
            return Err(super::invalid(
                "conflicting legacy TolFun and OptimalityTolerance controls",
            ));
        }
        let optimality =
            common::option_f64_value("fmincon", modern.or(legacy), "OptimalityTolerance", 1e-6)?;
        let feasibility = fields.real("fmincon", Field::TolCon, 1e-6)?;
        let step = fields.real(
            "fmincon",
            Field::TolX,
            if algorithm == OptimAlgorithm::InteriorPoint {
                1e-10
            } else {
                1e-6
            },
        )?;
        if optimality < 0.0 || feasibility < 0.0 || step < 0.0 {
            return Err(super::invalid(
                "optimality, constraint and step tolerances must be nonnegative",
            ));
        }
        let point_limit = fields
            .exact(Field::MaxFunEvals)
            .map(|_| fields.count("fmincon", Field::MaxFunEvals, 0))
            .transpose()?
            .or_else(|| (algorithm == OptimAlgorithm::InteriorPoint).then_some(3000));
        let max_iterations = fields.count(
            "fmincon",
            Field::MaxIter,
            if algorithm == OptimAlgorithm::InteriorPoint {
                1000
            } else {
                400
            },
        )?;
        let objective_gradient = fields.switch("fmincon", Field::SpecifyObjectiveGradient)?;
        let constraint_gradient = nonlinear_constraints
            && objective_gradient
            && fields.switch("fmincon", Field::SpecifyConstraintGradient)?;
        let display = OptimDisplay::parse(&fields.text(Field::Display, "final")?)
            .filter(|value| OptimSolver::Fmincon.accepts_display(*value))
            .ok_or_else(|| super::invalid("unsupported Display value"))?;
        Ok(Self {
            algorithm,
            max_iterations,
            point_limit,
            feasibility,
            optimality,
            step,
            objective_gradient,
            constraint_gradient,
            display,
            differences: runmat_builtins::OptimDifferenceType::parse(
                &fields.text(Field::FiniteDifferenceType, "forward")?,
            )
            .ok_or_else(|| super::invalid("FiniteDifferenceType must be forward or central"))?,
            scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling::from_fields("fmincon", &fields, !objective_gradient || (nonlinear_constraints && !constraint_gradient))?,
            prepared,
        })
    }
}
