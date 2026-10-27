//! Migration moves existing observations with their exact pre-migration rows.
//! It makes no fitness/constraint calls and never relabels an inherited score.
use super::super::{
    assessment::Assessment,
    common::population::{
        projection::{Argument, Arguments},
        Fitness,
    },
    lineage::Lineage,
    population::{migration, movement::Schedule, Generation},
    ScalarCooperation, POLICY,
};
use super::search::{self, Record, Search};
use crate::{execution::RootedValue, BuiltinResult};

impl<G: super::super::linear::Geometry> Search<G> {
    pub async fn migrate<I: super::super::callbacks::Invocation>(
        &self,
        generation: &mut Generation,
        evaluated: &mut super::super::evaluation::generation::Evaluated,
        assessment: &mut Assessment,
        record: &mut Record,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<std::ops::ControlFlow<I::Stop, ()>> {
        cooperation.completed(0).await?;
        if self.options.maximum.reached(self.timing.elapsed()) {
            return Ok(std::ops::ControlFlow::Continue(()));
        }
        let ordinal = self.progress.following_migration_generation()?;
        if ordinal == 0 || ordinal % self.options.migration_interval != 0 {
            return Ok(std::ops::ControlFlow::Continue(()));
        }
        let copies = migration::plan(
            &self.options.layout,
            &assessment.order,
            self.options.migration_fraction,
            self.options.migration,
            cooperation,
        )
        .await?;
        if copies.is_empty() {
            return Ok(std::ops::ControlFlow::Continue(()));
        }
        let schedule = Schedule::new(self.options.layout.total(), &copies, cooperation).await?;
        match generation {
            Generation::Numeric(population) => schedule.apply(population, cooperation).await?,
            custom @ Generation::Custom { .. } => {
                let mut sources = POLICY.capacity(copies.len(), cooperation.runtime())?;
                let mut targets = POLICY.capacity(copies.len(), cooperation.runtime())?;
                for copy in &copies {
                    sources.push(copy.source);
                    targets.push(copy.target);
                    cooperation.completed(1).await?;
                }
                let prepared = custom.prepare_projection(&sources, cooperation).await?;
                let projected = match invocation.projection(prepared, cooperation).await? {
                    std::ops::ControlFlow::Continue(value) => value,
                    std::ops::ControlFlow::Break(stop) => {
                        return Ok(std::ops::ControlFlow::Break(stop))
                    }
                };
                let children = match projected {
                    Argument::Rooted(value) => value,
                    Argument::Source(value) => {
                        RootedValue::in_optional_context(value, cooperation.runtime())?
                    }
                };
                let children = Generation::from_result_with(
                    children,
                    copies.len(),
                    self.problem.variables,
                    self.options.population,
                    invocation,
                    cooperation,
                )
                .await?;
                let children = match children {
                    std::ops::ControlFlow::Continue(value) => value,
                    std::ops::ControlFlow::Break(stop) => {
                        return Ok(std::ops::ControlFlow::Break(stop))
                    }
                };
                if let std::ops::ControlFlow::Break(stop) = custom
                    .replace_with(&targets, children, invocation, cooperation)
                    .await?
                {
                    return Ok(std::ops::ControlFlow::Break(stop));
                }
            }
        }
        let mut measurements =
            POLICY.capacity(self.options.layout.total(), cooperation.runtime())?;
        for row in 0..self.options.layout.total() {
            measurements.push(evaluated.fitness.values.measurement(row));
            cooperation.completed(1).await?;
        }
        schedule.records(&mut measurements, cooperation).await?;
        schedule
            .records(&mut evaluated.fitness.origins, cooperation)
            .await?;
        evaluated.fitness.values =
            Fitness::from_measurements(POLICY, measurements, cooperation).await?;
        schedule
            .tensor(&mut evaluated.constraints.inequalities, cooperation)
            .await?;
        schedule
            .tensor(&mut evaluated.constraints.equalities, cooperation)
            .await?;
        schedule
            .records(&mut evaluated.violations, cooperation)
            .await?;
        *assessment = Assessment::new(
            evaluated,
            search::assessment_mode(&self.problem, &self.options, self.merit.as_ref()),
            self.options.constraint_tolerance,
            cooperation,
        )
        .await?;
        if !record.lineage.is_empty() {
            schedule.records(&mut record.ancestors, cooperation).await?;
            for copy in &copies {
                record.lineage[copy.target] = Lineage::Migration;
                cooperation.completed(1).await?;
            }
        }
        // Expectations were computed for the pre-migration mating pools.
        record.expectation = None;
        Ok(std::ops::ControlFlow::Continue(()))
    }
}
