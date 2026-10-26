//! Selected canonical controls retain their original rooted generation.
use super::{bernstein::invalid, controls::Controls};
use crate::builtins::math::optim::common::{self, ScalarCooperation, ScalarOptionFields};
use crate::BuiltinResult;
use runmat_builtins::{OptimOptionField as Field, OptimSolver};
use runmat_value::Value;
pub(super) struct Options {
    pub controls: Controls,
    pub max_iterations: usize,
    pub point_limit: Option<usize>,
    pub optimality: f64,
    pub function: f64,
    pub step: f64,
    pub gradient: bool,
    pub display: runmat_builtins::OptimDisplay,
    pub differences: runmat_builtins::OptimDifferenceType,
    pub scaling: common::derivative_scaling::DerivativeScaling,
    prepared: Option<common::PreparedSolverOptions>,
}
impl Options {
    fn fields(&self) -> ScalarOptionFields<'_> {
        ScalarOptionFields::from_struct(self.prepared.as_ref().map(|p| p.structure()))
    }
    pub fn callback(&self) -> Option<&Value> {
        self.fields().exact(Field::OutputFcn)
    }
    pub fn plots(&self) -> Option<&Value> {
        self.fields().exact(Field::PlotFcn)
    }
    pub async fn parse(value: Value, cooperation: &mut ScalarCooperation) -> BuiltinResult<Self> {
        let options = match value {
            Value::Struct(value) => Some(value),
            Value::Tensor(value) if value.is_empty() => None,
            _ => return Err(invalid("options must be [] or a scalar structure")),
        };
        if let Some(options) = &options {
            for (name, value) in &options.fields {
                if name.eq_ignore_ascii_case("Solver") {
                    if OptimSolver::parse(&common::field_name(value)?) != Some(OptimSolver::Fseminf)
                    {
                        return Err(invalid("options belong to another solver"));
                    }
                } else if Field::parse_for_solver(name, OptimSolver::Fseminf).is_none()
                    && !matches!(value,Value::Tensor(v) if v.is_empty())
                {
                    return Err(invalid(format!("unsupported fseminf option {name}")));
                }
            }
        }
        let runtime = cooperation.runtime().cloned();
        let prepared = common::prepare_solver_options(
            OptimSolver::Fseminf,
            options,
            Some(&runmat_builtins::FSEMINF_INTEGER_OPTION_EXTENSION),
            Some(&runmat_builtins::FSEMINF_RESIDENT_EXTENSION),
            runtime.as_ref(),
            cooperation,
        )
        .await?;
        let fields = ScalarOptionFields::from_struct(prepared.as_ref().map(|p| p.structure()));
        let legacy = fields.exact(Field::TolFun);
        let tolerance = |field: Field, default: f64| -> BuiltinResult<f64> {
            let modern = fields.exact(field);
            let value =
                common::option_f64_value("fseminf", modern.or(legacy), field.name(), default)?;
            if let (Some(modern), Some(legacy)) = (modern, legacy) {
                if common::option_f64_value("fseminf", Some(modern), field.name(), default)?
                    != common::option_f64_value("fseminf", Some(legacy), "TolFun", default)?
                {
                    return Err(invalid(format!("conflicting TolFun and {}", field.name())));
                }
            }
            if value < 0.0 {
                return Err(invalid("tolerances must be nonnegative"));
            }
            Ok(value)
        };
        let (display, detailed) = match fields.exact(Field::Display) {
            Some(v) => common::semi_options::display(v)?,
            None => (runmat_builtins::OptimDisplay::Final, false),
        };
        let number = |field: Field, default: f64| -> BuiltinResult<f64> {
            fields
                .exact(field)
                .map_or(Ok(default), |v| common::semi_options::number(v, field))
        };
        let switch = |field: Field| -> BuiltinResult<bool> {
            fields
                .exact(field)
                .map_or(Ok(false), |v| common::semi_options::on_off(v, field))
        };
        let count = |field: Field| -> BuiltinResult<Option<usize>> {
            fields
                .exact(field)
                .map(|_| fields.count("fseminf", field, 0))
                .transpose()
        };
        let controls = Controls {
            mesh_tolerance: fields.real("fseminf", Field::TolCon, 1e-6)?,
            check_values: switch(Field::FunValCheck)?,
            diagnostics: switch(Field::Diagnostics)?,
            detailed_display: detailed,
            diff_min_change: number(Field::DiffMinChange, 0.0)?,
            diff_max_change: number(Field::DiffMaxChange, f64::INFINITY)?,
            inner_iterations: count(Field::MaxSQPIter)?,
            inner_tolerance: number(Field::TolConSQP, 1e-6)?,
            relative_line_bound: fields
                .exact(Field::RelLineSrchBnd)
                .map(|v| common::semi_options::number(v, Field::RelLineSrchBnd))
                .transpose()?,
            relative_line_duration: fields.count("fseminf", Field::RelLineSrchBndDuration, 1)?,
        };
        controls.validate()?;
        let scaling = controls.scaling(
            common::derivative_scaling::DerivativeScaling::from_fields("fseminf", &fields, true)?,
        )?;
        Ok(Self {
            max_iterations: fields.count("fseminf", Field::MaxIter, 400)?,
            point_limit: count(Field::MaxFunEvals)?,
            optimality: tolerance(Field::OptimalityTolerance, 1e-6)?,
            function: tolerance(Field::FunctionTolerance, 1e-4)?,
            step: fields.real("fseminf", Field::TolX, 1e-4)?,
            gradient: fields.switch("fseminf", Field::SpecifyObjectiveGradient)?,
            differences: runmat_builtins::OptimDifferenceType::parse(
                &fields.text(Field::FiniteDifferenceType, "forward")?,
            )
            .ok_or_else(|| invalid("FiniteDifferenceType must be forward or central"))?,
            controls,
            display,
            scaling,
            prepared,
        })
    }
}
