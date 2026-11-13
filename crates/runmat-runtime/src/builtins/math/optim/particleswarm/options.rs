//! Effective controls, selected once through the canonical field owner.
use super::{creation, fitness::Fitness, invalid, population::capacity, ScalarCooperation};
use crate::builtins::math::optim::common::{
    self,
    swarm_options::{self, Field, Owner, Prepared},
    ScalarOptionFields,
};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};
pub(super) struct Options {
    pub prepared: Prepared,
    pub members: usize,
    pub iterations: usize,
    pub stall_iterations: usize,
    pub tolerance: f64,
    pub objective_limit: f64,
    pub maximum: swarm_options::TimeLimit,
    pub stall_time: swarm_options::TimeLimit,
    pub display: runmat_builtins::OptimDisplay,
    pub display_interval: usize,
    pub vectorized: bool,
    pub check_values: bool,
    pub parallel: crate::execution::CallbackParallelMode,
    pub parameters: super::dynamics::Parameters,
    pub span: Vec<f64>,
    pub initial: Option<creation::Initial>,
    pub creation: creation::Source,
    pub hybrid: Option<swarm_options::Hybrid>,
    pub output: Option<Value>,
    pub plot: Option<Value>,
}
impl Options {
    pub async fn parse(
        value: Value,
        n: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let prepared = swarm_options::prepare(
            value,
            n,
            &runmat_builtins::PARTICLESWARM_INTEGER_OPTIONS_EXTENSION,
            &runmat_builtins::PARTICLESWARM_RESIDENT_OPTIONS_EXTENSION,
            cooperation,
        )
        .await?;
        let fields = ScalarOptionFields::from_struct(Some(prepared.structure()));
        let count = |field, default| fields.count("particleswarm", field, default);
        let real = |field, default| fields.real("particleswarm", field, default);
        let default_members = n.saturating_mul(10).min(100);
        let members = count(Field::SwarmSize, default_members)?;
        members
            .checked_mul(n)
            .ok_or_else(|| invalid("swarm coordinate extent overflow"))?;
        let iterations = count(
            Field::MaxIter,
            n.checked_mul(200)
                .ok_or_else(|| invalid("default iteration limit overflow"))?,
        )?;
        let stall_iterations = count(Field::MaxStallIterations, 20)?;
        let tolerance = real(Field::FunctionTolerance, 1e-6)?;
        let objective_limit = fields
            .exact(Field::ObjectiveLimit)
            .map(|v| common::quasi_newton_controls::objective_limit("particleswarm", v))
            .transpose()?
            .unwrap_or(f64::NEG_INFINITY);
        let maximum = fields
            .exact(Field::MaxTime)
            .map(|v| swarm_options::time_limit(Owner::Solver, v))
            .transpose()?
            .unwrap_or_default();
        let stall_time = fields
            .exact(Field::MaxStallTime)
            .map(|v| swarm_options::time_limit(Owner::Solver, v))
            .transpose()?
            .unwrap_or_default();
        let display = fields
            .exact(Field::Display)
            .map(|v| swarm_options::text(Owner::Solver, v, Field::Display))
            .transpose()?
            .as_deref()
            .and_then(runmat_builtins::OptimDisplay::parse)
            .unwrap_or(runmat_builtins::OptimDisplay::Final);
        let display_interval = count(Field::DisplayInterval, 1)?;
        let vectorized = fields
            .exact(Field::UseVectorized)
            .map(|v| swarm_options::switch(Owner::Solver, v, Field::UseVectorized))
            .transpose()?
            .unwrap_or(false);
        let parallel = if vectorized {
            crate::execution::CallbackParallelMode::Off
        } else {
            fields
                .exact(Field::UseParallel)
                .map(|v| common::quasi_newton_controls::parallel_mode("particleswarm", v))
                .transpose()?
                .unwrap_or(crate::execution::CallbackParallelMode::Off)
        };
        let check_values = fields
            .exact(Field::FunValCheck)
            .map(|v| swarm_options::text(Owner::Solver, v, Field::FunValCheck))
            .transpose()?
            .as_deref()
            .and_then(runmat_builtins::OptimSwitch::parse)
            .is_some_and(|v| v.enabled());
        let inertia = if let Some(value) = fields.exact(Field::InertiaRange) {
            let v = swarm_options::floating(Owner::Solver, value)?;
            [v.get(0), v.get(1)]
        } else {
            [0.1, 1.1]
        };
        let parameters = super::dynamics::Parameters {
            inertia,
            neighbors_fraction: real(Field::MinNeighborsFraction, 0.25)?,
            personal_weight: real(Field::SelfAdjustmentWeight, 1.49)?,
            social_weight: real(Field::SocialAdjustmentWeight, 1.49)?,
        };
        parameters.validate()?;
        let mut span = capacity(n, cooperation)?;
        if let Some(value) = fields.exact(Field::InitialSwarmSpan) {
            let values = swarm_options::floating(Owner::Solver, value)?;
            if values.len() != 1 && values.len() != n {
                return Err(invalid(
                    "InitialSwarmSpan must be scalar or have nvars entries",
                ));
            }
            for column in 0..n {
                span.push(values.get(if values.len() == 1 { 0 } else { column }));
                cooperation.completed(1).await?;
            }
        } else {
            for _ in 0..n {
                span.push(2000.0);
                cooperation.completed(1).await?;
            }
        }
        let initial = match fields.exact(Field::InitialPoints) {
            Some(value) => Some(initial(value, n, check_values, cooperation).await?),
            None => None,
        };
        let creation = match fields.exact(Field::CreationFcn) {
            None => creation::Source::Uniform,
            Some(value)
                if crate::builtins::common::validation::value_text_scalar_string(value)
                    .is_some() =>
            {
                creation::Source::Uniform
            }
            Some(value) => creation::Source::Custom(value.clone()),
        };
        let hybrid = fields
            .exact(Field::HybridFcn)
            .map(|v| swarm_options::hybrid(Owner::Solver, v))
            .transpose()?
            .flatten();
        let output = fields.exact(Field::OutputFcn).cloned();
        let plot = fields.exact(Field::PlotFcn).cloned();
        Ok(Self {
            prepared,
            members,
            iterations,
            stall_iterations,
            tolerance,
            objective_limit,
            maximum,
            stall_time,
            display,
            display_interval,
            vectorized,
            check_values,
            parallel,
            parameters,
            span,
            initial,
            creation,
            hybrid,
            output,
            plot,
        })
    }
    pub async fn creation_options(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        // Dimension-resolved defaults are public CreationFcn inputs, not an
        // alternate mutable copy of the numerical swarm or callback results.
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        let admitted = ScalarOptionFields::from_struct(Some(self.prepared.structure()));
        fields.insert("Solver", Value::from("particleswarm"));
        for field in Field::ALL {
            if let Some(value) = admitted.exact(field) {
                fields.insert(
                    field.name_for_solver(runmat_builtins::OptimSolver::Particleswarm),
                    value.clone(),
                );
                cooperation.completed(1).await?;
            }
        }
        fields.insert("SwarmSize", Value::Num(self.members as f64));
        fields.insert("MaxIterations", Value::Num(self.iterations as f64));
        fields.insert(
            "MaxStallIterations",
            Value::Num(self.stall_iterations as f64),
        );
        fields.insert("FunctionTolerance", Value::Num(self.tolerance));
        fields.insert(
            "InitialSwarmSpan",
            common::output_callback::vector(&self.span, &[1, self.span.len()], false, cooperation)
                .await?,
        );
        fields.insert(
            "InertiaRange",
            Value::Tensor(
                Tensor::new(self.parameters.inertia.to_vec(), vec![1, 2]).map_err(invalid)?,
            ),
        );
        fields.insert(
            "MinNeighborsFraction",
            Value::Num(self.parameters.neighbors_fraction),
        );
        fields.insert(
            "SelfAdjustmentWeight",
            Value::Num(self.parameters.personal_weight),
        );
        fields.insert(
            "SocialAdjustmentWeight",
            Value::Num(self.parameters.social_weight),
        );
        fields.insert("UseVectorized", Value::Bool(self.vectorized));
        fields.insert(
            "FunValCheck",
            Value::from(if self.check_values { "on" } else { "off" }),
        );
        for (name, value) in [
            ("MaxTime", Value::Num(f64::INFINITY)),
            ("MaxStallTime", Value::Num(f64::INFINITY)),
            ("ObjectiveLimit", Value::Num(self.objective_limit)),
            ("Display", Value::from(self.display.name())),
            ("DisplayInterval", Value::Num(self.display_interval as f64)),
            ("UseParallel", Value::from("off")),
            ("CreationFcn", Value::from("pswcreationuniform")),
            ("InitialPoints", super::inputs::empty()),
            ("HybridFcn", super::inputs::empty()),
            ("OutputFcn", super::inputs::empty()),
            ("PlotFcn", super::inputs::empty()),
        ] {
            if !fields.fields.contains_key(name) {
                fields.insert(name, value);
            }
        }
        crate::execution::RootedValue::in_optional_context(
            Value::Struct(fields.finish()),
            cooperation.runtime(),
        )
    }
}
async fn initial(
    value: &Value,
    n: usize,
    check: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<creation::Initial> {
    let (x, f) = if let Value::Struct(points) = value {
        (&points.fields["X"], points.fields.get("Fval"))
    } else {
        (value, None)
    };
    let positions = match x {
        Value::Num(v) => Tensor::new(vec![*v], vec![1, 1]).map_err(invalid)?,
        Value::Tensor(v) => v.clone(),
        _ => return Err(invalid("InitialPoints.X must be a full double matrix")),
    };
    if positions.shape().len() != 2 || positions.shape()[1] != n {
        return Err(invalid("InitialPoints.X must have nvars columns"));
    }
    let rows = positions.shape()[0];
    let fitness = if let Some(value) = f {
        let len = match value {
            Value::Num(_) => 1,
            Value::Tensor(v) => v.len(),
            _ => return Err(invalid("InitialPoints.Fval must be real double")),
        };
        if len > rows {
            return Err(invalid(
                "InitialPoints.Fval cannot have more entries than InitialPoints.X rows",
            ));
        }
        if len == 0 {
            None
        } else {
            Some(Fitness::decode(super::POLICY, value.clone(), len, check, cooperation).await?)
        }
    } else {
        None
    };
    Ok(creation::Initial { positions, fitness })
}
