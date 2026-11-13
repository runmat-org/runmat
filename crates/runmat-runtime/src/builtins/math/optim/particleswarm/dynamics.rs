//! Neighborhood-adaptive particle velocities, not a local optimizer alias.
use super::{
    invalid,
    population::{capacity, Population},
    random, ScalarCooperation,
};
use crate::BuiltinResult;
pub(super) struct Dynamics {
    pub velocity: Vec<f64>,
    pub personal: Population,
    pub personal_fitness: Vec<Option<f64>>,
    pub inertia: f64,
    pub neighborhood: usize,
    minimum_neighborhood: usize,
    stall_counter: usize,
}
pub(super) struct Parameters {
    pub inertia: [f64; 2],
    pub neighbors_fraction: f64,
    pub personal_weight: f64,
    pub social_weight: f64,
}
impl Parameters {
    pub fn validate(&self) -> BuiltinResult<()> {
        let [low, high] = self.inertia;
        if !low.is_finite()
            || !high.is_finite()
            || low > high
            || low * high < 0.0
            || !self.neighbors_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.neighbors_fraction)
            || !self.personal_weight.is_finite()
            || !self.social_weight.is_finite()
        {
            return Err(invalid(
                "invalid inertia, neighborhood or adjustment-weight controls",
            ));
        }
        Ok(())
    }
}
impl Dynamics {
    pub fn new(
        velocity: Vec<f64>,
        population: &Population,
        fitness: Vec<Option<f64>>,
        parameters: &Parameters,
    ) -> BuiltinResult<Self> {
        parameters.validate()?;
        if population.members < 2
            || fitness.len() != population.members
            || velocity.len() != population.positions.len()
        {
            return Err(invalid("swarm dynamics extent mismatch"));
        }
        let minimum = ((population.members as f64 * parameters.neighbors_fraction).floor()
            as usize)
            .max(2)
            .min(population.members);
        Ok(Self {
            velocity,
            personal: Population::new(
                super::POLICY,
                population.positions.clone(),
                population.members,
                population.variables,
            )?,
            personal_fitness: fitness,
            inertia: if parameters.inertia[1] < 0.0 {
                parameters.inertia[0]
            } else {
                parameters.inertia[1]
            },
            neighborhood: minimum,
            minimum_neighborhood: minimum,
            stall_counter: 0,
        })
    }
    pub async fn propose(
        &mut self,
        population: &mut Population,
        parameters: &Parameters,
        lower: &[f64],
        upper: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let members = population.members;
        let variables = population.variables;
        if lower.len() != variables || upper.len() != variables {
            return Err(invalid("bound extent mismatch"));
        }
        population.writable(cooperation).await?;
        let count = variables
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.neighborhood))
            .ok_or_else(|| invalid("swarm random range overflow"))?;
        // One ordered stream turn, but only O(nvars+neighborhood) draw scratch.
        // This scope invokes no user callback.
        let mut turn = random::turn(cooperation).await?;
        let mut candidates = capacity(members.saturating_sub(1), cooperation)?;
        for member in 0..members {
            let draws = random::uniform_in_turn(&mut turn, count, cooperation).await?;
            let mut draw = draws.into_iter();
            candidates.clear();
            for other in 0..members {
                if other != member {
                    candidates.push(other);
                }
                cooperation.completed(1).await?;
            }
            let mut neighbor = if self.neighborhood == members {
                Some(member)
            } else {
                None
            };
            for index in 0..self.neighborhood {
                let u = draw.next().expect("admitted neighborhood draw");
                if index >= candidates.len() {
                    continue;
                }
                let selected = index + ((candidates.len() - index) as f64 * u).floor() as usize;
                let last = candidates.len() - 1;
                candidates.swap(index, selected.min(last));
                let other = candidates[index];
                if neighbor.is_none_or(|old| {
                    better(self.personal_fitness[other], self.personal_fitness[old])
                }) {
                    neighbor = Some(other);
                }
                cooperation.completed(1).await?;
            }
            let neighbor = neighbor.expect("nonempty neighborhood");
            for column in 0..variables {
                let a = draw.next().expect("admitted personal draw");
                let b = draw.next().expect("admitted social draw");
                let index = member + members * column;
                let x = population.get(member, column);
                let p = if self.personal_fitness[member].is_some() {
                    self.personal.get(member, column)
                } else {
                    x
                };
                let g = if self.personal_fitness[neighbor].is_some() {
                    self.personal.get(neighbor, column)
                } else {
                    x
                };
                let mut velocity = self.inertia * self.velocity[index]
                    + parameters.personal_weight * a * (p - x)
                    + parameters.social_weight * b * (g - x);
                let next = x + velocity;
                if !velocity.is_finite() || !next.is_finite() {
                    return Err(invalid("particle displacement exceeds finite arithmetic"));
                }
                let bounded = if next < lower[column] {
                    if velocity < 0.0 {
                        velocity = 0.0;
                    }
                    lower[column]
                } else if next > upper[column] {
                    if velocity > 0.0 {
                        velocity = 0.0;
                    }
                    upper[column]
                } else {
                    next
                };
                self.velocity[index] = velocity;
                population
                    .positions
                    .try_as_f64_slice_mut_unique()
                    .expect("writable generation")[index] = bounded;
                cooperation.completed(1).await?;
            }
        }
        Ok(())
    }
    pub async fn accept(
        &mut self,
        population: &Population,
        fitness: &[Option<f64>],
        improved: bool,
        parameters: &Parameters,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if fitness.len() != population.members {
            return Err(invalid("fitness extent mismatch"));
        }
        self.personal.writable(cooperation).await?;
        for (member, &value) in fitness.iter().enumerate() {
            if better(value, self.personal_fitness[member]) {
                self.personal_fitness[member] = value;
                for column in 0..population.variables {
                    self.personal
                        .positions
                        .try_as_f64_slice_mut_unique()
                        .expect("writable personal generation")
                        [member + population.members * column] = population.get(member, column);
                    cooperation.completed(1).await?;
                }
            }
            cooperation.completed(1).await?;
        }
        if improved {
            self.stall_counter = self.stall_counter.saturating_sub(1);
            self.neighborhood = self.minimum_neighborhood;
            if self.stall_counter < 2 {
                self.inertia *= 2.0;
            } else if self.stall_counter > 5 {
                self.inertia *= 0.5;
            }
            self.inertia = self
                .inertia
                .clamp(parameters.inertia[0], parameters.inertia[1]);
        } else {
            self.stall_counter = self
                .stall_counter
                .checked_add(1)
                .ok_or_else(|| invalid("stall counter overflow"))?;
            self.neighborhood = self
                .neighborhood
                .saturating_add(self.minimum_neighborhood)
                .min(population.members);
        }
        Ok(())
    }
}
fn better(candidate: Option<f64>, previous: Option<f64>) -> bool {
    match (candidate, previous) {
        (Some(a), Some(b)) => a < b,
        (Some(_), None) => true,
        _ => false,
    }
}
