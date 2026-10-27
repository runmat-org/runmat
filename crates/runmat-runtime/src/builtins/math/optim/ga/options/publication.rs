//! Dimension-resolved controls passed to ordinary genetic callbacks.
use super::{common, Field, ScalarCooperation};
use crate::BuiltinResult;
use runmat_builtins::*;
use runmat_value::Value;

impl super::resolve::Options {
    pub async fn public(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        let defaults = super::super::super::optimoptions::default_options_value(
            cooperation.runtime(),
            OptimSolver::Ga,
        )?;
        let Value::Struct(defaults) = defaults.value() else {
            return Err(super::super::invalid(
                "canonical GA defaults are not a structure",
            ));
        };
        let mut builder =
            runmat_value::UnboundedStructValueBuilder::from_structure(defaults.clone());
        let fields = common::ScalarOptionFields::from_struct_for_solver(
            Some(self.prepared.structure()),
            runmat_builtins::OptimSolver::Ga,
        );
        for field in Field::ALL {
            if let Some(value) = fields.exact(field) {
                builder.insert(field.name_for_solver(OptimSolver::Ga), value.clone());
            }
            cooperation.completed(1).await?;
        }
        builder.insert("Solver", Value::from("ga"));
        if fields.exact(Field::MaxIter).is_none() {
            builder.insert("MaxGenerations", Value::Num(self.generations as f64));
        }
        if fields.exact(Field::PopulationSize).is_none() {
            let mut counts =
                super::super::POLICY.capacity(self.layout.groups(), cooperation.runtime())?;
            for group in 0..self.layout.groups() {
                counts.push(self.layout.range(group).len() as f64);
                cooperation.completed(1).await?;
            }
            builder.insert(
                "PopulationSize",
                Value::Tensor(
                    runmat_value::Tensor::new(counts, vec![1, self.layout.groups()])
                        .map_err(super::super::invalid)?,
                ),
            );
        }
        for (field, value) in [
            (Field::CrossoverFraction, self.crossover_fraction),
            (Field::FunctionTolerance, self.function_tolerance),
            (Field::TolCon, self.constraint_tolerance),
            (Field::ObjectiveLimit, self.fitness_limit),
            (Field::InitialPenalty, self.penalty),
            (Field::PenaltyFactor, self.penalty_factor),
            (Field::MigrationFraction, self.migration_fraction),
            (Field::MaxStallIterations, self.stall_generations as f64),
            (Field::DisplayInterval, self.display_interval as f64),
            (Field::PlotInterval, self.plot_interval as f64),
            (Field::MigrationInterval, self.migration_interval as f64),
        ] {
            if fields.exact(field).is_none() {
                builder.insert(field.name_for_solver(OptimSolver::Ga), Value::Num(value));
            }
        }
        for (field, name) in [
            (Field::PopulationType, self.population.name()),
            (Field::NonlinearConstraintAlgorithm, self.constraint.name()),
            (Field::StallTest, self.stall_test.name()),
            (Field::MigrationDirection, self.migration.name()),
            (Field::Display, self.display.name()),
            (
                Field::FunValCheck,
                if self.check_values { "on" } else { "off" },
            ),
        ] {
            builder.insert(field.name_for_solver(OptimSolver::Ga), Value::from(name));
        }
        for (field, value) in [
            (
                Field::CreationFcn,
                self.creation.public_value(OptimGeneticCreation::name),
            ),
            (
                Field::CrossoverFcn,
                self.crossover.public_value(OptimGeneticCrossover::name),
            ),
            (
                Field::MutationFcn,
                self.mutation.public_value(OptimGeneticMutation::name),
            ),
            (
                Field::SelectionFcn,
                self.selection.public_value(OptimGeneticSelection::name),
            ),
            (
                Field::FitnessScalingFcn,
                self.scaling.public_value(OptimGeneticScaling::name),
            ),
        ] {
            builder.insert(field.name_for_solver(OptimSolver::Ga), value);
        }
        builder.insert("UseVectorized", Value::Bool(self.vectorized));
        if fields.exact(Field::MaxTime).is_none() {
            builder.insert("MaxTime", Value::Num(f64::INFINITY));
        }
        if fields.exact(Field::MaxStallTime).is_none() {
            builder.insert("MaxStallTime", Value::Num(f64::INFINITY));
        }
        crate::execution::RootedValue::in_optional_context(
            Value::Struct(builder.finish()),
            cooperation.runtime(),
        )
    }
}
