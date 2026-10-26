//! Comparison/merit, raw observations and public score storage are distinct.
use super::{
    evaluation::generation::Evaluated,
    invalid,
    merit::AugmentedLagrangian,
    operators::ranking::{self, RankOrder, Score},
    ScalarCooperation, POLICY,
};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

pub(super) enum Mode<'a> {
    Raw,
    Penalty,
    Augmented(&'a AugmentedLagrangian),
}
pub(super) struct Assessment {
    pub comparison: Vec<Score>,
    pub public_scores: Value,
    pub order: RankOrder,
    pub best_feasible: Option<usize>,
}
impl Assessment {
    pub async fn new(
        evaluated: &Evaluated,
        mode: Mode<'_>,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let rows = evaluated.constraints.rows;
        if rows != evaluated.fitness.values.len()
            || rows != evaluated.violations.len()
            || !tolerance.is_finite()
            || tolerance < 0.
        {
            return Err(invalid(
                "generation assessment geometry or tolerance changed",
            ));
        }
        let (comparison, public_scores) = match mode {
            Mode::Raw => {
                let mut comparison = POLICY.capacity(rows, cooperation.runtime())?;
                for row in 0..rows {
                    comparison.push(
                        evaluated
                            .fitness
                            .values
                            .measurement(row)
                            .score()
                            .map_or(Score::Undefined, Score::Defined),
                    );
                    cooperation.completed(1).await?;
                }
                (comparison, evaluated.fitness.values.value())
            }
            Mode::Penalty => {
                let scores = evaluated.penalty_scores(tolerance, cooperation).await?;
                let mut comparison = POLICY.capacity(rows, cooperation.runtime())?;
                let mut values = POLICY.capacity(rows, cooperation.runtime())?;
                for score in scores {
                    comparison.push(score.map_or(Score::Undefined, Score::Defined));
                    values.push(score.unwrap_or(f64::NAN));
                    cooperation.completed(1).await?;
                }
                (
                    comparison,
                    Value::Tensor(Tensor::new(values, vec![rows, 1]).map_err(invalid)?),
                )
            }
            Mode::Augmented(merit) => {
                let comparison = evaluated.augmented_scores(merit, cooperation).await?;
                let mut values = POLICY.capacity(rows, cooperation.runtime())?;
                for &score in &comparison {
                    // Domain-restoration distances remain comparison evidence,
                    // never user fitness or a finite logarithmic-barrier score.
                    values.push(match score {
                        Score::Defined(value) => value,
                        _ => f64::NAN,
                    });
                    cooperation.completed(1).await?;
                }
                (
                    comparison,
                    Value::Tensor(Tensor::new(values, vec![rows, 1]).map_err(invalid)?),
                )
            }
        };
        let mut best_feasible = None;
        let mut best_fitness = f64::INFINITY;
        for row in 0..rows {
            if evaluated.violations[row].admitted(tolerance) {
                if let Some(fitness) = evaluated.fitness.values.measurement(row).score() {
                    if best_feasible.is_none() || fitness < best_fitness {
                        best_feasible = Some(row);
                        best_fitness = fitness;
                    }
                }
            }
            cooperation.completed(1).await?;
        }
        let order = ranking::order(&comparison, cooperation).await?;
        Ok(Self {
            comparison,
            public_scores,
            order,
            best_feasible,
        })
    }
    pub fn best_comparison(&self) -> Option<f64> {
        self.order
            .first()
            .and_then(|&row| match self.comparison[row] {
                Score::Defined(value) => Some(value),
                _ => None,
            })
    }
    /// A callback may replace selection scores, not the independently retained
    /// objective and physical-constraint observations used for publication.
    pub async fn override_scores(
        &mut self,
        scores: super::common::population::Fitness,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if scores.len() != self.comparison.len() {
            return Err(invalid("observer selection score extent changed"));
        }
        self.comparison.clear();
        for row in 0..scores.len() {
            self.comparison.push(
                scores
                    .measurement(row)
                    .score()
                    .map_or(Score::Undefined, Score::Defined),
            );
            cooperation.completed(1).await?;
        }
        self.order = ranking::order(&self.comparison, cooperation).await?;
        self.public_scores = scores.value();
        Ok(())
    }
}
