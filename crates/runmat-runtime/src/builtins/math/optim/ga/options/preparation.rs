//! Effective-field class/intent preflight precedes the first provider read. Dormant
//! nonlinear InitialScores and custom/bitstring constraint roles are not read.
use super::{admission, common, Field, Owner, ScalarCooperation};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::{OptimGeneticPopulation, OptimSolver};
use runmat_value::{StructValue, Value};
#[derive(Clone, Copy, Eq, PartialEq)]
pub(in super::super) enum Activity {
    Linear,
    Nonlinear,
    IntegerLinear,
    IntegerNonlinear,
}
impl Activity {
    pub fn from_raw(
        raw: &super::super::inputs::raw::Raw,
        population: OptimGeneticPopulation,
    ) -> Self {
        let nonlinear = population == OptimGeneticPopulation::DoubleVector
            && !super::super::inputs::raw::unspecified(&raw.nonlinear);
        let integer = !super::super::inputs::raw::unspecified(&raw.integers)
            && crate::value_fact::value_fact(&raw.integers)
                .shape
                .element_count()
                != Some(0);
        Self::new(nonlinear, integer)
    }
    pub fn from_problem(problem: &super::super::inputs::Problem) -> Self {
        Self::new(problem.nonlinear.is_some(), problem.mixed_integer)
    }
    fn new(nonlinear: bool, integer: bool) -> Self {
        match (nonlinear, integer) {
            (false, false) => Self::Linear,
            (true, false) => Self::Nonlinear,
            (false, true) => Self::IntegerLinear,
            (true, true) => Self::IntegerNonlinear,
        }
    }
    fn has_nonlinear(self) -> bool {
        matches!(self, Self::Nonlinear | Self::IntegerNonlinear)
    }
}

pub(in super::super) struct Prepared {
    pub population: OptimGeneticPopulation,
    root: RootedValue,
    pub nonlinear_creation: Option<super::nonlinear_creation::Prepared>,
}
impl Prepared {
    pub fn structure(&self) -> &StructValue {
        let Value::Struct(value) = self.root.value() else {
            unreachable!("prepared GA controls")
        };
        value
    }
    pub fn value(&self) -> &Value {
        self.root.value()
    }
    pub fn population(value: &Value) -> BuiltinResult<OptimGeneticPopulation> {
        let Value::Struct(options) = value else {
            return if super::super::inputs::raw::unspecified(value) {
                Ok(OptimGeneticPopulation::DoubleVector)
            } else {
                Err(Owner::Solver.invalid("options must be a scalar struct or []"))
            };
        };
        let fields = common::ScalarOptionFields::from_struct_for_solver(
            Some(options),
            runmat_builtins::OptimSolver::Ga,
        );
        fields
            .exact(Field::PopulationType)
            .map(|v| {
                super::text(v).and_then(|name| {
                    OptimGeneticPopulation::parse(&name)
                        .ok_or_else(|| Owner::Solver.invalid("unknown PopulationType"))
                })
            })
            .transpose()
            .map(|value| value.unwrap_or(OptimGeneticPopulation::DoubleVector))
    }
    pub fn new<'a>(
        value: Value,
        activity: Activity,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<Self>> + 'a {
        Self::prepare(value, activity, true, cooperation)
    }
    pub fn continuation<'a>(
        value: Value,
        activity: Activity,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<Self>> + 'a {
        Self::prepare(value, activity, false, cooperation)
    }
    async fn prepare(
        value: Value,
        activity: Activity,
        initial: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let population = Self::population(&value)?;
        let value = if super::super::inputs::raw::unspecified(&value) {
            Value::Struct(runmat_value::UnboundedStructValueBuilder::new().finish())
        } else {
            value
        };
        let source = RootedValue::in_optional_context(value, cooperation.runtime())?;
        let Value::Struct(options) = source.value() else {
            unreachable!("checked options carrier")
        };
        let active_scores =
            !activity.has_nonlinear() || population != OptimGeneticPopulation::DoubleVector;
        let selected_fields =
            common::ScalarOptionFields::from_struct_for_solver(Some(options), OptimSolver::Ga);
        let augmented = activity == Activity::Nonlinear
            && selected_fields
                .exact(Field::NonlinearConstraintAlgorithm)
                .map(|value| {
                    super::text(value).and_then(|name| {
                        runmat_builtins::OptimGeneticConstraint::parse(&name).ok_or_else(|| {
                            Owner::Solver.invalid("unknown NonlinearConstraintAlgorithm")
                        })
                    })
                })
                .transpose()?
                .unwrap_or(runmat_builtins::OptimGeneticConstraint::AugmentedLagrangian)
                == runmat_builtins::OptimGeneticConstraint::AugmentedLagrangian;
        for (name, value) in &options.fields {
            if name.eq_ignore_ascii_case("Solver") {
                if OptimSolver::parse(&super::text(value)?) != Some(OptimSolver::Ga) {
                    return Err(Owner::Solver.invalid("options belong to another solver"));
                }
                continue;
            }
            let field = Field::parse_for_solver(name, OptimSolver::Ga)
                .ok_or_else(|| Owner::Solver.invalid(format!("unsupported ga option {name}")))?;
            if !preflight_field(
                field,
                value,
                activity,
                population,
                initial,
                augmented,
                cooperation.runtime(),
            )? {
                continue;
            }
            cooperation.completed(1).await?;
        }
        let fields = common::ScalarOptionFields::from_struct_for_solver(
            Some(options),
            runmat_builtins::OptimSolver::Ga,
        );
        let nonlinear_creation = if initial {
            super::nonlinear_creation::prepare(fields.exact(Field::CreationFcn), cooperation)
                .await?
        } else {
            None
        };
        let vectorized = if let Some(value) = fields.exact(Field::UseVectorized) {
            let value = materialize(Field::UseVectorized, value, cooperation).await?;
            admission::validate(Owner::Solver, Field::UseVectorized, &value)?;
            Some(value)
        } else {
            None
        };
        let active_parallel = !vectorized
            .as_ref()
            .map(|v| super::switch(Owner::Solver, v))
            .transpose()?
            .unwrap_or(false);
        if active_parallel {
            if let Some(value) = fields.exact(Field::UseParallel) {
                admission::metadata(
                    Owner::Solver,
                    Field::UseParallel,
                    value,
                    cooperation.runtime(),
                )?;
            }
        }
        let mut replacements =
            super::super::POLICY.capacity(Field::ALL.len(), cooperation.runtime())?;
        for field in Field::ALL {
            let active = !(field == Field::UseParallel && !active_parallel
                || matches!(field, Field::InitialPenalty | Field::PenaltyFactor) && !augmented
                || field == Field::InitialScoresMatrix && (!initial || !active_scores)
                || field == Field::InitialPopulationMatrix && !initial);
            let replacement = if active {
                if let Some(value) = fields.exact(field) {
                    let value = if field == Field::UseVectorized {
                        vectorized.clone().expect("selected flag")
                    } else if field == Field::InitialPopulationMatrix
                        && population == OptimGeneticPopulation::Custom
                    {
                        value.clone()
                    } else {
                        materialize(field, value, cooperation).await?
                    };
                    admission::validate_cooperative(Owner::Solver, field, &value, cooperation)
                        .await?;
                    Some((field, value))
                } else {
                    None
                }
            } else {
                None
            };
            replacements.push(replacement);
            cooperation.completed(1).await?;
        }
        let fields = options.fields.iter().map(|(name, source)| {
            let value = Field::parse_for_solver(name, OptimSolver::Ga)
                .and_then(|field| {
                    replacements
                        .iter()
                        .flatten()
                        .find_map(|(candidate, value)| (*candidate == field).then_some(value))
                })
                .unwrap_or(source);
            (name, value)
        });
        let root = crate::execution::construct_struct_from_fields_in_optional_context(
            cooperation.runtime(),
            fields,
        )?;
        Ok(Self {
            population,
            root,
            nonlinear_creation,
        })
    }
}

pub(in crate::builtins::math::optim::ga) fn preflight_field(
    field: Field,
    value: &Value,
    activity: Activity,
    population: OptimGeneticPopulation,
    initial: bool,
    augmented: bool,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<bool> {
    let active_scores =
        !activity.has_nonlinear() || population != OptimGeneticPopulation::DoubleVector;
    if field == Field::UseParallel
        || matches!(field, Field::InitialPenalty | Field::PenaltyFactor) && !augmented
        || field == Field::InitialScoresMatrix && (!initial || !active_scores)
        || field == Field::InitialPopulationMatrix && !initial
    {
        return Ok(false);
    }
    admission::metadata(Owner::Solver, field, value, runtime)?;
    if (field == Field::InitialPopulationMatrix && population != OptimGeneticPopulation::Custom)
        || field == Field::InitialScoresMatrix
    {
        data_metadata(value, field, runtime)?;
    }
    Ok(true)
}
fn data_metadata(
    value: &Value,
    field: Field,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if super::super::inputs::raw::unspecified(value) {
        return Ok(());
    }
    let geometry = if field == Field::InitialScoresMatrix {
        common::constrained_numeric::Geometry::Vector
    } else {
        common::constrained_numeric::Geometry::MatrixAny
    };
    super::super::numeric::preflight(value, field.name(), geometry, runtime)?;
    if crate::value_fact::value_fact(value).storage == runmat_types::StorageFact::Sparse {
        return Err(Owner::Solver.invalid("initial population and scores must use full arrays"));
    }
    Ok(())
}
async fn materialize(
    field: Field,
    value: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    if let Some(value) = super::parameters::materialize(field, value, cooperation).await? {
        return Ok(value);
    }
    if matches!(admission::role(field), admission::Role::Opaque)
        || crate::builtins::duration::is_duration_object(value)
    {
        return Ok(value.clone());
    }
    crate::dispatcher::gather_if_needed_in_context_async(value, cooperation.runtime()).await
}
