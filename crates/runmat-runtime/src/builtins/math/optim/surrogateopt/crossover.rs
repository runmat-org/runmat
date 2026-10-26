//! Four-player tournament projection over actual measured history. Only row
//! indices are selected; no population payload or rescaled score mirror.
use super::{
    domain::BoxView,
    history::{History, Point, TrialIndex},
    incumbent, invalid,
    merit::Phase,
    observation::ObjectiveKind,
    NAME,
};
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use crate::BuiltinResult;

pub(super) struct Parents<'a> {
    history: &'a History,
    rows: Vec<TrialIndex>,
}
pub(super) struct Pair<'a> {
    pub first: &'a Point,
    pub second: &'a Point,
    pub weight: f64,
}
impl<'a> Parents<'a> {
    pub(super) async fn new(
        history: &'a History,
        phase: Phase,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        cooperation.completed(0).await?;
        let Some(schema) = history.schema() else {
            return Ok(None);
        };
        let start = usize::from(schema.objective == ObjectiveKind::Present);
        let objective = match phase {
            Phase::Feasibility { inequality_start } => {
                if inequality_start != start || schema.inequalities == 0 {
                    return Err(invalid("crossover feasibility lane schema mismatch"));
                }
                false
            }
            Phase::Objective {
                objective_lane,
                inequality_start,
            } => {
                if schema.objective != ObjectiveKind::Present
                    || objective_lane != 0
                    || inequality_start != start
                {
                    return Err(invalid("crossover objective lane schema mismatch"));
                }
                true
            }
        };
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            NAME,
            "finite host-memory execution has no complete crossover metadata bound",
        )?;
        let mut rows = runmat_value::exact_capacity_vec(history.len()).map_err(|source| {
            crate::build_runtime_error("surrogateopt: crossover row metadata allocation failed")
                .with_builtin(NAME)
                .with_source(source)
                .build()
        })?;
        // All measured phases form the crossover population. Rejected NaN
        // rows cannot supply a rank, and Phase2 excludes infeasible rows.
        for index in 0..history.len() {
            let row = TrialIndex(index);
            if history
                .trial(row)
                .observation
                .measurement()
                .is_some_and(|value| !objective || value.nonlinear.violated == 0)
            {
                rows.push(row);
            }
            cooperation.completed(1).await?;
        }
        Ok((!rows.is_empty()).then_some(Self { history, rows }))
    }
    pub(super) async fn select(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Pair<'a>> {
        cooperation.completed(0).await?;
        let mut turn = random::acquire_original_turn(cooperation.runtime(), NAME).await?;
        let first = self.tournament(&mut turn, cooperation).await?;
        let second = self.tournament(&mut turn, cooperation).await?;
        let weight = {
            let (runtime, budget) = cooperation.runtime_budget();
            let values = match budget {
                Some(budget) => {
                    random::generate_uniform_cooperative(&mut turn, 1, NAME, runtime, budget)
                        .await?
                }
                None => random::generate_uniform(&mut turn, 1, NAME)?,
            };
            values[0]
        };
        drop(turn);
        Ok(Pair {
            first: &self.history.trial(first).point,
            second: &self.history.trial(second).point,
            weight,
        })
    }
    async fn tournament(
        &self,
        turn: &mut random::RandomTurn,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<TrialIndex> {
        let mut players = [0u64; 4];
        random::fill_integer_offsets(turn, self.rows.len() as u128, &mut players, NAME)?;
        let mut best = self.rows[players[0] as usize];
        for &player in &players[1..] {
            let row = self.rows[player as usize];
            let candidate = self
                .history
                .trial(row)
                .observation
                .measurement()
                .expect("finite parent");
            let incumbent = self
                .history
                .trial(best)
                .observation
                .measurement()
                .expect("finite parent");
            if incumbent::preferred(candidate, incumbent)? {
                best = row;
            }
            cooperation.completed(1).await?;
        }
        Ok(best)
    }
}
impl Pair<'_> {
    pub(super) async fn point(
        &self,
        domain: &BoxView<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        cooperation.completed(0).await?;
        if self.first.variables() != domain.variables()
            || self.second.variables() != domain.variables()
            || !self.weight.is_finite()
            || !(0.0..=1.0).contains(&self.weight)
        {
            return Err(invalid("invalid arithmetic crossover geometry"));
        }
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            NAME,
            "finite host-memory execution has no complete crossover point bound",
        )?;
        let mut point = runmat_value::exact_capacity_vec(domain.variables()).map_err(|source| {
            crate::build_runtime_error("surrogateopt: crossover point allocation failed")
                .with_builtin(NAME)
                .with_source(source)
                .build()
        })?;
        for column in 0..domain.variables() {
            let (lower, upper) = domain.effective_bounds(column);
            let first = self.first.coordinate(column);
            let second = self.second.coordinate(column);
            if first < lower || first > upper || second < lower || second > upper {
                return Err(invalid(
                    "crossover parent is outside its original effective bounds",
                ));
            }
            let Some(mut value) =
                crate::builtins::math::optim::common::affine_coordinate::interpolate(
                    first,
                    second,
                    self.weight,
                )
            else {
                return Ok(None);
            };
            if domain.integers().binary_search(&column).is_ok() {
                value = value.round();
            }
            if value < lower || value > upper {
                return Ok(None);
            }
            point.push(value);
            cooperation.completed(1).await?;
        }
        // Integer rounding may disturb linear feasibility. The actual
        // original-domain proposal owner still classifies/repairs this point;
        // no parent objective value is transferred to the new coordinates.
        Ok(Some(point))
    }
}
