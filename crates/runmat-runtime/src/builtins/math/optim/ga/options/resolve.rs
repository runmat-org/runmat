//! Dimension- and constraint-resolved public GA controls.
use super::super::{inputs::Problem, population::layout::Layout};
use super::{common, Field, Operator, Owner, Prepared, ScalarCooperation};
use crate::BuiltinResult;
use common::global_time::TimeLimit;
use runmat_builtins::*;
use runmat_value::Value;

pub(in super::super) struct Options {
    pub prepared: Prepared,
    pub layout: Layout,
    pub elites: Vec<usize>,
    pub generations: usize,
    pub stall_generations: usize,
    pub function_tolerance: f64,
    pub constraint_tolerance: f64,
    pub fitness_limit: f64,
    pub maximum: TimeLimit,
    pub stall_time: TimeLimit,
    pub display: OptimDisplay,
    pub display_interval: usize,
    pub plot_interval: usize,
    pub vectorized: bool,
    pub parallel: crate::execution::CallbackParallelMode,
    pub check_values: bool,
    pub constraint: OptimGeneticConstraint,
    pub population: OptimGeneticPopulation,
    pub penalty: f64,
    pub penalty_factor: f64,
    pub crossover_fraction: f64,
    pub migration: OptimGeneticMigration,
    pub migration_fraction: f64,
    pub migration_interval: usize,
    pub stall_test: OptimGeneticStall,
    pub creation: Operator<OptimGeneticCreation>,
    pub crossover: Operator<OptimGeneticCrossover>,
    pub crossover_ratio: Vec<f64>,
    pub mutation: Operator<OptimGeneticMutation>,
    pub selection: Operator<OptimGeneticSelection>,
    pub scaling: Operator<OptimGeneticScaling>,
    pub ranges: Vec<(f64, f64)>,
    pub tournament_size: usize,
}
impl Options {
    pub fn bind_generation_callbacks(
        &mut self,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<()> {
        if self.generations == 0 {
            return Ok(());
        }
        self.crossover.bind(cooperation)?;
        self.mutation.bind(cooperation)?;
        self.selection.bind(cooperation)?;
        self.scaling.bind(cooperation)
    }
    pub async fn resolve(
        prepared: Prepared,
        problem: &Problem,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let linear = problem
            .linear
            .as_ref()
            .ok_or_else(|| Owner::Solver.invalid("initial options require original geometry"))?;
        Self::with_bounds(
            prepared,
            problem,
            (&linear.lower, &linear.upper),
            cooperation,
        )
        .await
    }
    pub async fn with_bounds(
        prepared: Prepared,
        problem: &Problem,
        bounds: (&[f64], &[f64]),
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let fields = common::ScalarOptionFields::from_struct_for_solver(
            Some(prepared.structure()),
            runmat_builtins::OptimSolver::Ga,
        );
        let n = problem.variables;
        if bounds.0.len() != n || bounds.1.len() != n {
            return Err(Owner::Solver.invalid("original bound extent changed"));
        }
        let counts = if let Some(value) = fields.exact(Field::PopulationSize) {
            counts(value, cooperation).await?
        } else {
            let mut values = super::super::POLICY.capacity(1, cooperation.runtime())?;
            values.push(if problem.mixed_integer {
                n.saturating_mul(10).clamp(40, 100)
            } else if n <= 5 {
                50
            } else {
                200
            });
            values
        };
        let layout = Layout::new(&counts, cooperation).await?;
        layout
            .total()
            .checked_mul(n)
            .ok_or_else(|| Owner::Solver.invalid("population extent overflow"))?;
        let mut elites = super::super::POLICY.capacity(counts.len(), cooperation.runtime())?;
        for count in counts {
            let elite = fields
                .exact(Field::EliteCount)
                .map(|value| common::option_usize_value("ga", Some(value), "EliteCount", 0))
                .transpose()?
                .unwrap_or(count / 20 + usize::from(count % 20 != 0));
            if elite > count {
                return Err(Owner::Solver.invalid("EliteCount cannot exceed a subpopulation size"));
            }
            elites.push(elite);
            cooperation.completed(1).await?;
        }
        let generations = fields.count(
            "ga",
            Field::MaxIter,
            n.checked_mul(100)
                .ok_or_else(|| Owner::Solver.invalid("default generation limit overflow"))?,
        )?;
        let population = prepared.population;
        let constraint = selected(
            &fields,
            Field::NonlinearConstraintAlgorithm,
            OptimGeneticConstraint::parse,
            if problem.mixed_integer {
                OptimGeneticConstraint::Penalty
            } else {
                OptimGeneticConstraint::AugmentedLagrangian
            },
        )?;
        if problem.mixed_integer && constraint != OptimGeneticConstraint::Penalty {
            return Err(Owner::Solver
                .invalid("integer nonlinear constraints require the penalty algorithm"));
        }
        let creation = operator(
            &fields,
            Field::CreationFcn,
            OptimGeneticCreation::parse,
            if problem.mixed_integer {
                OptimGeneticCreation::UniformInteger
            } else if problem.has_linear {
                OptimGeneticCreation::LinearFeasible
            } else if problem.nonlinear.is_some() && constraint == OptimGeneticConstraint::Penalty {
                OptimGeneticCreation::NonlinearFeasible
            } else {
                OptimGeneticCreation::Uniform
            },
        )?;
        let crossover = operator(
            &fields,
            Field::CrossoverFcn,
            OptimGeneticCrossover::parse,
            if problem.mixed_integer {
                OptimGeneticCrossover::Laplace
            } else if problem.has_linear {
                OptimGeneticCrossover::Intermediate
            } else {
                OptimGeneticCrossover::Scattered
            },
        )?;
        let mutation = operator(
            &fields,
            Field::MutationFcn,
            OptimGeneticMutation::parse,
            if population == OptimGeneticPopulation::BitString {
                OptimGeneticMutation::Uniform
            } else if problem.mixed_integer {
                OptimGeneticMutation::Power
            } else if problem.has_linear || problem.has_bounds {
                OptimGeneticMutation::AdaptiveFeasible
            } else {
                OptimGeneticMutation::Gaussian
            },
        )?;
        // The nonlinear penalty controller owns tournament selection; its
        // documented rule overrides a supplied scaling/selection policy.
        let penalty_selection =
            problem.nonlinear.is_some() && constraint == OptimGeneticConstraint::Penalty;
        let selection = if penalty_selection {
            Operator::default(OptimGeneticSelection::Tournament)
        } else {
            operator(
                &fields,
                Field::SelectionFcn,
                OptimGeneticSelection::parse,
                if problem.mixed_integer
                    || problem.nonlinear.is_some() && constraint == OptimGeneticConstraint::Penalty
                {
                    OptimGeneticSelection::Tournament
                } else {
                    OptimGeneticSelection::StochasticUniform
                },
            )?
        };
        let scaling = operator(
            &fields,
            Field::FitnessScalingFcn,
            OptimGeneticScaling::parse,
            OptimGeneticScaling::Rank,
        )?;
        if population == OptimGeneticPopulation::Custom {
            if crossover.custom().is_none() || mutation.custom().is_none() {
                return Err(Owner::Solver.invalid(
                    "custom populations require custom crossover and mutation functions",
                ));
            }
            if creation.custom().is_none() && fields.exact(Field::InitialPopulationMatrix).is_none()
            {
                return Err(Owner::Solver.invalid(
                    "custom population requires custom creation or a complete initial population",
                ));
            }
        }
        if population == OptimGeneticPopulation::BitString
            && (creation
                .kind()
                .is_some_and(|v| v != OptimGeneticCreation::Uniform)
                || mutation
                    .kind()
                    .is_some_and(|v| v != OptimGeneticMutation::Uniform)
                || crossover.kind().is_some_and(|v| {
                    !matches!(
                        v,
                        OptimGeneticCrossover::Scattered
                            | OptimGeneticCrossover::SinglePoint
                            | OptimGeneticCrossover::TwoPoint
                    )
                }))
        {
            return Err(Owner::Solver.invalid("bitstring requires uniform/custom creation and mutation, and scattered/single-point/two-point/custom crossover"));
        }
        if fields.exact(Field::HybridFcn).is_some()
            && (problem.mixed_integer || population != OptimGeneticPopulation::DoubleVector)
        {
            return Err(Owner::Solver.invalid(
                "HybridFcn is unavailable for integer, custom and bitstring populations",
            ));
        }
        let tournament_size = if penalty_selection {
            2
        } else if let Some(value) = selection
            .extra()
            .first()
            .filter(|_| selection.kind() == Some(OptimGeneticSelection::Tournament))
        {
            common::option_usize_value("ga", Some(value), "tournament size", 0)?
        } else if problem.nonlinear.is_some() {
            2
        } else {
            4
        };
        if tournament_size < 2 {
            return Err(Owner::Solver.invalid("tournament size must be at least two"));
        }
        let vectorized = fields
            .exact(Field::UseVectorized)
            .map(|v| super::switch(Owner::Solver, v))
            .transpose()?
            .unwrap_or(false);
        let parallel = if vectorized {
            crate::execution::CallbackParallelMode::Off
        } else {
            fields
                .exact(Field::UseParallel)
                .map(|v| common::quasi_newton_controls::parallel_mode("ga", v))
                .transpose()?
                .unwrap_or(crate::execution::CallbackParallelMode::Off)
        };
        let ranges = super::ranges::resolve(
            if creation.kind() == Some(OptimGeneticCreation::LinearFeasible) {
                None
            } else {
                fields.exact(Field::InitialPopulationRange)
            },
            bounds.0,
            bounds.1,
            problem.mixed_integer,
            population == OptimGeneticPopulation::BitString,
            cooperation,
        )
        .await?;
        let crossover_ratio =
            super::parameters::crossover_ratio(&crossover, n, cooperation).await?;
        let result = Self {
            layout,
            elites,
            generations,
            population,
            constraint,
            creation,
            crossover,
            crossover_ratio,
            mutation,
            selection,
            scaling,
            ranges,
            tournament_size,
            vectorized,
            parallel,
            stall_generations: fields.count("ga", Field::MaxStallIterations, 50)?,
            function_tolerance: fields.real("ga", Field::FunctionTolerance, 1e-6)?,
            constraint_tolerance: fields.real("ga", Field::TolCon, 1e-3)?,
            fitness_limit: fields
                .exact(Field::ObjectiveLimit)
                .map(|v| super::real(Owner::Solver, v))
                .transpose()?
                .unwrap_or(f64::NEG_INFINITY),
            maximum: fields
                .exact(Field::MaxTime)
                .map(|v| common::global_time::parse(Owner::Solver.policy(), v))
                .transpose()?
                .unwrap_or_default(),
            stall_time: fields
                .exact(Field::MaxStallTime)
                .map(|v| common::global_time::parse(Owner::Solver.policy(), v))
                .transpose()?
                .unwrap_or_default(),
            display: selected(
                &fields,
                Field::Display,
                OptimDisplay::parse,
                OptimDisplay::Final,
            )?,
            display_interval: fields.count("ga", Field::DisplayInterval, 1)?,
            plot_interval: fields.count("ga", Field::PlotInterval, 1)?,
            check_values: selected(
                &fields,
                Field::FunValCheck,
                OptimSwitch::parse,
                OptimSwitch::Off,
            )?
            .enabled(),
            penalty: if problem.nonlinear.is_some()
                && !problem.mixed_integer
                && constraint == OptimGeneticConstraint::AugmentedLagrangian
            {
                fields.real("ga", Field::InitialPenalty, 10.)?
            } else {
                10.
            },
            penalty_factor: if problem.nonlinear.is_some()
                && !problem.mixed_integer
                && constraint == OptimGeneticConstraint::AugmentedLagrangian
            {
                fields.real("ga", Field::PenaltyFactor, 100.)?
            } else {
                100.
            },
            crossover_fraction: fields.real("ga", Field::CrossoverFraction, 0.8)?,
            migration: selected(
                &fields,
                Field::MigrationDirection,
                OptimGeneticMigration::parse,
                OptimGeneticMigration::Forward,
            )?,
            migration_fraction: fields.real("ga", Field::MigrationFraction, 0.2)?,
            migration_interval: fields.count("ga", Field::MigrationInterval, 20)?,
            stall_test: selected(
                &fields,
                Field::StallTest,
                OptimGeneticStall::parse,
                OptimGeneticStall::AverageChange,
            )?,
            prepared,
        };
        Ok(result)
    }
    pub fn field(&self, field: Field) -> Option<&Value> {
        common::ScalarOptionFields::from_struct_for_solver(
            Some(self.prepared.structure()),
            runmat_builtins::OptimSolver::Ga,
        )
        .exact(field)
    }
}
fn selected<T>(
    fields: &common::ScalarOptionFields,
    field: Field,
    parse: fn(&str) -> Option<T>,
    default: T,
) -> BuiltinResult<T> {
    fields
        .exact(field)
        .map(|v| {
            super::text(v).and_then(|v| {
                parse(&v).ok_or_else(|| Owner::Solver.invalid("invalid selected option"))
            })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}
fn operator<T: Copy>(
    fields: &common::ScalarOptionFields,
    field: Field,
    parse: fn(&str) -> Option<T>,
    default: T,
) -> BuiltinResult<Operator<T>> {
    fields
        .exact(field)
        .map(|v| Operator::parse(v, parse))
        .transpose()
        .map(|v| v.unwrap_or_else(|| Operator::default(default)))
}
async fn counts(value: &Value, cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<usize>> {
    let len = match value {
        Value::Tensor(values) => values.len(),
        _ => 1,
    };
    let mut counts = super::super::POLICY.capacity(len, cooperation.runtime())?;
    for index in 0..len {
        let scalar = match value {
            Value::Tensor(values) => {
                if let Some(storage) = values.integer_storage() {
                    Value::Int(storage.value_at(index).expect("count element"))
                } else {
                    Value::Num(crate::builtins::common::tensor::tensor_value_f64(
                        values, index,
                    ))
                }
            }
            value => value.clone(),
        };
        counts.push(common::option_usize_value(
            "ga",
            Some(&scalar),
            "PopulationSize",
            0,
        )?);
        cooperation.completed(1).await?;
    }
    Ok(counts)
}
