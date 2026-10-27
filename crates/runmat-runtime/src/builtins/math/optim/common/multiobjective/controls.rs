//! Selected multiobjective controls, using the canonical spelling, residency,
//! exact-integer and prepared-root owners rather than a second option grammar.
use super::{invalid, Kind};
use crate::builtins::math::optim::common::{self, ScalarCooperation, ScalarOptionFields};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinExtensionDescriptor, OptimDisplay, OptimOptionField as Field};
use runmat_value::Value;

pub(crate) struct Controls {
    pub max_iterations: usize,
    pub point_limit: Option<usize>,
    pub feasibility: f64,
    pub optimality: f64,
    pub function: f64,
    pub step: f64,
    pub objective_gradient: bool,
    pub constraint_gradient: bool,
    pub display: OptimDisplay,
    pub differences: runmat_builtins::OptimDifferenceType,
    pub scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling,
    pub first_objectives: usize,
    pub merit: runmat_builtins::OptimMeritFunction,
    prepared: Option<common::PreparedSolverOptions>,
}
impl Controls {
    pub fn callback(&self) -> Option<&Value> {
        ScalarOptionFields::from_struct(self.prepared.as_ref().map(|value| value.structure()))
            .exact(Field::OutputFcn)
    }

    pub(crate) async fn parse(
        kind: Kind,
        value: Value,
        nonlinear: bool,
        numeric: &'static BuiltinExtensionDescriptor,
        resident: Option<&'static BuiltinExtensionDescriptor>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let options = match value {
            Value::Struct(value) => Some(value),
            Value::Tensor(value) if value.is_empty() => None,
            _ => return Err(invalid(kind, "options must be [] or a scalar structure")),
        };
        if let Some(options) = &options {
            for (name, value) in &options.fields {
                if name.eq_ignore_ascii_case("Solver") {
                    let name = common::field_name(value)?;
                    if runmat_builtins::OptimSolver::parse(&name) != Some(kind.solver()) {
                        return Err(invalid(kind, "options belong to another solver"));
                    }
                } else if Field::parse_for_solver(name, kind.solver()).is_none() {
                    return Err(invalid(kind, format!("option {name} is not implemented")));
                }
            }
        }
        let runtime = cooperation.runtime().cloned();
        let prepared = common::prepare_multiobjective_options(
            kind.solver(),
            options,
            nonlinear,
            Some(numeric),
            resident,
            runtime.as_ref(),
            cooperation,
        )
        .await?;
        let fields =
            ScalarOptionFields::from_struct(prepared.as_ref().map(|value| value.structure()));
        let legacy = fields.exact(Field::TolFun);
        let optimality = tolerance(kind, &fields, Field::OptimalityTolerance, legacy)?;
        let function = tolerance(kind, &fields, Field::FunctionTolerance, legacy)?;
        let feasibility = fields.real(kind.name(), Field::TolCon, 1e-6)?;
        let step = fields.real(kind.name(), Field::TolX, 1e-6)?;
        if feasibility < 0.0 || step < 0.0 {
            return Err(invalid(
                kind,
                "constraint and step tolerances must be nonnegative",
            ));
        }
        let objective_requested = fields.switch(kind.name(), Field::SpecifyObjectiveGradient)?;
        let constraint_gradient = nonlinear
            && objective_requested
            && fields.switch(kind.name(), Field::SpecifyConstraintGradient)?;
        let objective_gradient = objective_requested && (!nonlinear || constraint_gradient);
        let count = match kind {
            Kind::Minimax => Field::AbsoluteMaxObjectiveCount,
            Kind::GoalAttain => Field::EqualityGoalCount,
        };
        let first_objectives = fields.count(kind.name(), count, 0)?;
        let display = OptimDisplay::parse(&fields.text(Field::Display, "final")?)
            .filter(|value| kind.solver().accepts_display(*value))
            .ok_or_else(|| invalid(kind, "unsupported Display value"))?;
        let differences = runmat_builtins::OptimDifferenceType::parse(
            &fields.text(Field::FiniteDifferenceType, "forward")?,
        )
        .ok_or_else(|| invalid(kind, "FiniteDifferenceType must be forward or central"))?;
        let merit = runmat_builtins::OptimMeritFunction::parse(
            &fields.text(Field::MeritFunction, "multiobj")?,
        )
        .ok_or_else(|| invalid(kind, "MeritFunction must be multiobj or singleobj"))?;
        let point_limit = fields
            .exact(Field::MaxFunEvals)
            .map(|_| fields.count(kind.name(), Field::MaxFunEvals, 0))
            .transpose()?;
        Ok(Self {
            max_iterations: fields.count(kind.name(), Field::MaxIter, 400)?,
            point_limit,
            feasibility,
            optimality,
            function,
            step,
            objective_gradient,
            constraint_gradient,
            display,
            differences,
            first_objectives,
            merit,
            scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling::from_fields(kind.name(),&fields,!objective_gradient || (nonlinear&&!constraint_gradient))?,
            prepared,
        })
    }
}

fn tolerance(
    kind: Kind,
    fields: &ScalarOptionFields<'_>,
    field: Field,
    legacy: Option<&Value>,
) -> BuiltinResult<f64> {
    let modern = fields.exact(field);
    if let (Some(modern), Some(legacy)) = (modern, legacy) {
        let modern = common::option_f64_value(kind.name(), Some(modern), field.name(), 1e-6)?;
        let legacy = common::option_f64_value(kind.name(), Some(legacy), "TolFun", 1e-6)?;
        if modern != legacy {
            return Err(invalid(
                kind,
                format!("conflicting TolFun and {} controls", field.name()),
            ));
        }
    }
    let value = common::option_f64_value(kind.name(), modern.or(legacy), field.name(), 1e-6)?;
    if value < 0.0 {
        return Err(invalid(
            kind,
            "function and optimality tolerances must be nonnegative",
        ));
    }
    Ok(value)
}
