//! The current swarm, individual bests and bounded best-value stopping window.
use super::{
    dynamics::Dynamics,
    fitness::Fitness,
    invalid,
    population::{capacity, Population},
    ScalarCooperation,
};
use crate::BuiltinResult;
use std::collections::VecDeque;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Termination {
    Function,
    Iterations,
    Observer,
    InconsistentBounds,
    Objective,
    StallTime,
    Time,
}
impl Termination {
    pub fn flag(self) -> i32 {
        match self {
            Self::Function => 1,
            Self::Iterations => 0,
            Self::Observer => -1,
            Self::InconsistentBounds => -2,
            Self::Objective => -3,
            Self::StallTime => -4,
            Self::Time => -5,
        }
    }
    pub fn message(self) -> &'static str {
        match self{
            Self::Function=>"The best objective's relative change over MaxStallIterations meets FunctionTolerance; no global minimum is certified.",
            Self::Iterations=>"MaxIterations reached.",
            Self::Observer=>"The output or plot function requested termination.",
            Self::InconsistentBounds=>"No feasible swarm exists because supplied bounds are inconsistent.",
            Self::Objective=>"The actual best objective is below ObjectiveLimit.",
            Self::StallTime=>"No best-objective improvement occurred within MaxStallTime.",
            Self::Time=>"The original admitted Clock reached MaxTime.",
        }
    }
}
pub(super) struct State {
    pub population: Population,
    pub fitness: Fitness,
    pub dynamics: Dynamics,
    pub iteration: usize,
    pub best_member: usize,
    pub best_fitness: f64,
    pub stall_iterations: usize,
    window: VecDeque<f64>,
}
impl State {
    pub async fn new(
        population: Population,
        fitness: Fitness,
        dynamics: Dynamics,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut best = None;
        for member in 0..population.members {
            if let Some(value) = fitness.measurement(member).score() {
                if best.is_none_or(|(_, old)| value < old) {
                    best = Some((member, value));
                }
            }
            cooperation.completed(1).await?;
        }
        let (best_member, best_fitness) =
            best.ok_or_else(|| invalid("no finite real initial objective value was found"))?;
        let mut window = VecDeque::new();
        window.try_reserve(1).map_err(|_| {
            crate::context::host_allocation::host_allocation_runtime_error(
                cooperation.runtime(),
                "RunMat:HostAllocation:ParticleBestWindow",
                "particle-swarm stopping window",
                crate::context::host_allocation::HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })?;
        window.push_back(best_fitness);
        Ok(Self {
            population,
            fitness,
            dynamics,
            iteration: 0,
            best_member,
            best_fitness,
            stall_iterations: 0,
            window,
        })
    }
    pub async fn accept(
        &mut self,
        fitness: Fitness,
        parameters: &super::dynamics::Parameters,
        stall_window: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let previous = self.best_fitness;
        let scores = fitness.scores(cooperation).await?;
        for (member, &value) in scores.iter().enumerate() {
            if let Some(value) = value {
                if value < self.best_fitness {
                    self.best_fitness = value;
                    self.best_member = member;
                }
            }
            cooperation.completed(1).await?;
        }
        let improved = self.best_fitness < previous;
        self.dynamics
            .accept(&self.population, &scores, improved, parameters, cooperation)
            .await?;
        self.fitness = fitness;
        self.iteration = self
            .iteration
            .checked_add(1)
            .ok_or_else(|| invalid("iteration counter overflow"))?;
        self.stall_iterations = if improved {
            0
        } else {
            self.stall_iterations
                .checked_add(1)
                .ok_or_else(|| invalid("stall iteration overflow"))?
        };
        let capacity = stall_window
            .checked_add(1)
            .ok_or_else(|| invalid("stall window extent overflow"))?;
        if self.window.len() == capacity {
            self.window.pop_front();
        }
        self.window.try_reserve(1).map_err(|_| {
            crate::context::host_allocation::host_allocation_runtime_error(
                cooperation.runtime(),
                "RunMat:HostAllocation:ParticleBestWindow",
                "particle-swarm stopping window",
                crate::context::host_allocation::HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })?;
        self.window.push_back(self.best_fitness);
        Ok(improved)
    }
    pub fn function_small(&self, window: usize, tolerance: f64) -> bool {
        if self.window.len() <= window {
            return false;
        }
        let old = *self.window.front().expect("nonempty best-value window");
        let denominator = old.abs().max(1.0);
        let difference = (old - self.best_fitness).abs();
        let relative = if difference.is_finite() {
            difference / denominator
        } else {
            (old / denominator - self.best_fitness / denominator).abs()
        };
        relative < tolerance
    }
    pub async fn best_point(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<runmat_value::Value> {
        self.dynamics
            .personal
            .row(self.best_member, cooperation)
            .await
    }
    pub async fn unsupplied(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<Option<super::fitness::Measurement>>> {
        let mut values = capacity(self.population.members, cooperation)?;
        for _ in 0..self.population.members {
            values.push(None);
            cooperation.completed(1).await?;
        }
        Ok(values)
    }
}
