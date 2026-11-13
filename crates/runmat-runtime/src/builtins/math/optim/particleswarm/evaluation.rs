//! Particle-specific missing-value selection and decoding over shared actual waves.
pub(super) use super::common::population::Counts;
use super::{
    common::population::{Invocation, Wave},
    fitness::{Fitness, Measurement},
    invalid,
    population::{capacity, Population},
    ScalarCooperation, POLICY,
};
use crate::BuiltinResult;
use runmat_value::Value;
pub(super) struct Evaluator<'a> {
    pub function: &'a Value,
    pub cooperation: &'a mut ScalarCooperation,
    pub vectorized: bool,
    pub check_values: bool,
    pub parallel: Option<crate::execution::CallbackBatch>,
    pub counts: Counts,
}
impl Evaluator<'_> {
    pub async fn evaluate(
        &mut self,
        population: &Population,
        supplied: Vec<Option<Measurement>>,
    ) -> BuiltinResult<Fitness> {
        if supplied.len() != population.members {
            return Err(invalid("initial fitness extent mismatch"));
        }
        let mut missing = capacity(population.members, self.cooperation)?;
        let mut values = supplied;
        for (row, value) in values.iter().enumerate() {
            if let Some(value) = value {
                if self.check_values && value.score().is_none() {
                    return Err(invalid("FunValCheck rejected initial values"));
                }
            } else {
                missing.push(row);
            }
            self.cooperation.completed(1).await?;
        }
        let mut start = 0;
        while start < missing.len() {
            let wave = Invocation {
                policy: POLICY,
                function: super::common::callback_source::Callable::Dynamic(self.function),
                cooperation: self.cooperation,
                vectorized: self.vectorized,
                parallel: self.parallel.as_ref(),
                counts: &mut self.counts,
            }
            .next(population, &missing, start, 1)
            .await?;
            start = wave.end();
            match wave {
                Wave::Vectorized {
                    start,
                    end,
                    returned,
                } => {
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
                    if end - start == population.members {
                        return Ok(result);
                    }
                    for (ordinal, &row) in missing[start..end].iter().enumerate() {
                        values[row] = Some(result.measurement(ordinal));
                    }
                }
                Wave::Rows {
                    start,
                    end,
                    returned,
                } => {
                    for (&row, returned) in missing[start..end].iter().zip(returned) {
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
                        values[row] = Some(result.measurement(0));
                    }
                }
            }
        }
        let mut measured = capacity(population.members, self.cooperation)?;
        for value in values {
            measured.push(value.ok_or_else(|| invalid("missing actual particle measurement"))?);
            self.cooperation.completed(1).await?;
        }
        Fitness::from_measurements(POLICY, measured, self.cooperation).await
    }
}
