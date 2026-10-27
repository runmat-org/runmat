//! Raw fitness provenance is separate from penalty and augmented scores.
use super::super::{common, invalid, ScalarCooperation, POLICY};
use crate::{execution::CallbackBatch, BuiltinResult};
use common::callback_source::Callable;
use common::population::projection::Arguments;
use common::population::{Counts, Fitness, Measurement, Wave};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::optim) enum Origin {
    Measured,
    SuppliedInitial,
    ObserverSupplied,
    HybridReported,
    ConstraintRejected,
}
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Supplied {
    pub value: Measurement,
    pub origin: Origin,
}
pub(in super::super) struct RawFitness {
    pub values: Fitness,
    pub origins: Vec<Origin>,
}
impl RawFitness {
    pub fn supplied(&self, member: usize) -> Supplied {
        Supplied {
            value: self.values.measurement(member),
            origin: self.origins[member],
        }
    }
}

pub(in super::super) struct Objective<'a> {
    pub function: Callable<'a>,
    pub vectorized: bool,
    pub check_values: bool,
    pub parallel: Option<&'a CallbackBatch>,
    pub counts: &'a mut Counts,
    pub aliases: Option<&'a super::duplicates::Aliases<'a>>,
    pub cooperation: &'a mut ScalarCooperation,
}
use std::ops::ControlFlow;

impl Objective<'_> {
    #[cfg(test)]
    pub async fn evaluate(
        &mut self,
        population: &super::super::population::Generation,
        supplied: Vec<Option<Supplied>>,
    ) -> BuiltinResult<RawFitness> {
        self.evaluate_with(population, supplied, &mut super::source::Direct)
            .await
            .map(super::super::callbacks::take_direct)
    }
    pub async fn evaluate_with<S: super::source::Source>(
        &mut self,
        population: &super::super::population::Generation,
        mut supplied: Vec<Option<Supplied>>,
        source: &mut S,
    ) -> BuiltinResult<ControlFlow<S::Stop, RawFitness>> {
        if supplied.len() != population.members() {
            return Err(invalid("supplied fitness extent differs from population"));
        }
        if let Some(aliases) = self.aliases {
            aliases.validate(population)?;
            // Explicit supplied rows are never overwritten. A missing exact
            // duplicate may reuse their authority with the same provenance.
            for member in 0..population.members() {
                let representative = aliases.representative(member);
                if supplied[representative].is_none() && supplied[member].is_some() {
                    supplied[representative] = supplied[member];
                }
                self.cooperation.completed(1).await?;
            }
        }
        let mut missing = POLICY.capacity(population.members(), self.cooperation.runtime())?;
        let mut origins = POLICY.capacity(population.members(), self.cooperation.runtime())?;
        for (member, value) in supplied.iter().enumerate() {
            if self.check_values
                && value.is_some_and(|value| {
                    value.origin != Origin::ConstraintRejected && value.value.score().is_none()
                })
            {
                return Err(invalid(
                    "FunValCheck requires finite real supplied fitness values",
                ));
            }
            if value.is_none()
                && self
                    .aliases
                    .is_none_or(|aliases| aliases.representative(member) == member)
            {
                missing.push(member);
            }
            origins.push(value.map_or(Origin::Measured, |value| value.origin));
            self.cooperation.completed(1).await?;
        }
        let mut start = 0;
        while start < missing.len() {
            let request = super::source::Request {
                policy: POLICY,
                function: self.function,
                vectorized: self.vectorized,
                parallel: self.parallel,
                counts: self.counts,
                population,
                selected: &missing,
                start,
                outputs: 1,
            };
            let wave = match source.next(request, self.cooperation).await? {
                ControlFlow::Continue(wave) => wave,
                ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
            };
            let (first, end) = wave.range();
            if first != start || end <= start || end > missing.len() {
                return Err(invalid(
                    "fitness source wave does not advance its selected rows",
                ));
            }
            start = end;
            match wave {
                super::source::Wave::Dispatched(Wave::Vectorized {
                    start,
                    end,
                    returned,
                }) => {
                    let returned = {
                        let _scope = self
                            .cooperation
                            .runtime()
                            .map(crate::context::RuntimeContext::enter);
                        returned.try_into_single()?
                    };
                    let result = returned
                        .with_owned_async(|value| async {
                            Fitness::decode(
                                POLICY,
                                value,
                                end - start,
                                self.check_values,
                                self.cooperation,
                            )
                            .await
                        })
                        .await?;
                    if end - start == population.members() {
                        return Ok(ControlFlow::Continue(RawFitness {
                            values: result,
                            origins,
                        }));
                    }
                    for (index, &member) in missing[start..end].iter().enumerate() {
                        supplied[member] = Some(Supplied {
                            value: result.measurement(index),
                            origin: Origin::Measured,
                        });
                        self.cooperation.completed(1).await?;
                    }
                }
                super::source::Wave::Dispatched(Wave::Rows {
                    start,
                    end,
                    returned,
                }) => {
                    for (&member, returned) in missing[start..end].iter().zip(returned) {
                        let returned = {
                            let _scope = self
                                .cooperation
                                .runtime()
                                .map(crate::context::RuntimeContext::enter);
                            returned.try_into_single()?
                        };
                        let result = returned
                            .with_owned_async(|value| async {
                                Fitness::decode(
                                    POLICY,
                                    value,
                                    1,
                                    self.check_values,
                                    self.cooperation,
                                )
                                .await
                            })
                            .await?;
                        supplied[member] = Some(Supplied {
                            value: result.measurement(0),
                            origin: Origin::Measured,
                        });
                        self.cooperation.completed(1).await?;
                    }
                }
                super::source::Wave::Measured { start, end, values } => {
                    if start >= end || end > missing.len() || values.len() != end - start {
                        return Err(invalid("borrowed fitness wave extent changed"));
                    }
                    for (index, &member) in missing[start..end].iter().enumerate() {
                        let value = values.measurement(index);
                        if self.check_values && value.score().is_none() {
                            return Err(invalid("FunValCheck requires finite real fitness values"));
                        }
                        supplied[member] = Some(Supplied {
                            value,
                            origin: Origin::Measured,
                        });
                        self.cooperation.completed(1).await?;
                    }
                }
            }
        }
        if let Some(aliases) = self.aliases {
            for member in 0..population.members() {
                if supplied[member].is_none() {
                    supplied[member] = supplied[aliases.representative(member)];
                }
                self.cooperation.completed(1).await?;
            }
        }
        let mut values = POLICY.capacity(population.members(), self.cooperation.runtime())?;
        origins.clear();
        for value in supplied {
            let value =
                value.ok_or_else(|| invalid("an objective row was never supplied or measured"))?;
            values.push(value.value);
            origins.push(value.origin);
            self.cooperation.completed(1).await?;
        }
        Ok(ControlFlow::Continue(RawFitness {
            values: Fitness::from_measurements(POLICY, values, self.cooperation).await?,
            origins,
        }))
    }
}
