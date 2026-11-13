//! A completed fixed-merit epoch is not interchangeable with an inner step.
use super::super::{
    assessment::{self, Assessment},
    invalid,
    merit::Update,
    outcome::Termination,
    reproduction, ScalarCooperation,
};
use super::search::Search;
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticAction;

impl<G: super::super::linear::Geometry> Search<G> {
    pub fn complete_subproblem(&mut self, solved: bool) -> BuiltinResult<Option<Termination>> {
        self.progress.complete_subproblem()?;
        let Some(merit) = self.merit.as_ref() else {
            return Ok(None);
        };
        let Some(fitness) = self.best.score() else {
            return Ok(None);
        };
        let Some(best_merit) = merit.score(fitness, &self.best.inequalities, &self.best.equalities)
        else {
            return Ok(None);
        };
        let Some(comparison) = self.assessment.best_comparison() else {
            return Ok(None);
        };
        let scale = best_merit.abs().max(comparison.abs()).max(1.);
        let same_subproblem_solution =
            best_merit / scale - comparison / scale <= self.options.function_tolerance;
        if solved
            && same_subproblem_solution
            && self.best.feasible(self.options.constraint_tolerance)
            && merit
                .complementarity(&self.best.inequalities)
                .is_some_and(|value| value <= self.options.constraint_tolerance.sqrt())
        {
            Ok(Some(Termination::Augmented))
        } else {
            Ok(None)
        }
    }
    pub async fn update_merit(
        &mut self,
        solved: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Termination>> {
        let Some(merit) = self.merit.as_mut() else {
            return Ok(None);
        };
        let row = *self
            .assessment
            .order
            .first()
            .ok_or_else(|| invalid("AL population has no comparison row"))?;
        let scratch = self
            .merit_row
            .as_mut()
            .ok_or_else(|| invalid("AL constraint workspace is missing"))?;
        scratch.capture(&self.evaluated, row, cooperation).await?;
        let solved = solved && self.assessment.best_comparison().is_some();
        let update = merit
            .update(
                &scratch.inequalities,
                &scratch.equalities,
                solved,
                cooperation,
            )
            .await?;
        self.action = Some(match update {
            Update::Multipliers => OptimGeneticAction::UpdateMultipliers,
            Update::Penalty => OptimGeneticAction::IncreasePenalty,
            Update::Unrepresentable => return Ok(Some(Termination::Unverified)),
        });
        self.assessment = Assessment::new(
            &self.evaluated,
            assessment::Mode::Augmented(merit),
            self.options.constraint_tolerance,
            cooperation,
        )
        .await?;
        self.progress.reset_merit(&self.options);
        self.progress
            .observed(&self.assessment, false, false, &self.timing, cooperation)
            .await?;
        self.mutation = reproduction::Mutation::new(&self.options, cooperation).await?;
        Ok(None)
    }
}
