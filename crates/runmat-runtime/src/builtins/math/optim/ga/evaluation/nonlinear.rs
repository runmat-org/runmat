//! Real multi-output nonlinear calls, using the same bounded worker-wave owner.
use super::super::{common, invalid, ScalarCooperation, POLICY};
use super::constraints::{self, Constraints};
use crate::{execution::CallbackBatch, BuiltinResult};
use common::callback_source::Callable;
use common::population::projection::Arguments;
use common::population::{Counts, Invocation, Wave};

pub(in super::super) struct Nonlinear<'a> {
    pub function: Option<Callable<'a>>,
    pub vectorized: bool,
    pub parallel: Option<&'a CallbackBatch>,
    pub counts: &'a mut Counts,
    pub extents: &'a mut Option<(usize, usize)>,
    pub mixed_integer: bool,
    pub aliases: Option<&'a super::duplicates::Aliases<'a>>,
    pub cooperation: &'a mut ScalarCooperation,
}
impl Nonlinear<'_> {
    pub async fn evaluate(
        &mut self,
        population: &super::super::population::Generation,
    ) -> BuiltinResult<Constraints> {
        let Some(function) = self.function else {
            return Constraints::empty(population.members());
        };
        if let Some(aliases) = self.aliases {
            aliases.validate(population)?;
        }
        let mut all = POLICY.capacity(
            if self.aliases.is_none() {
                population.members()
            } else {
                0
            },
            self.cooperation.runtime(),
        )?;
        let selected = if let Some(aliases) = self.aliases {
            aliases.selected()
        } else {
            for member in 0..population.members() {
                all.push(member);
                self.cooperation.completed(1).await?;
            }
            &all
        };
        let mut rows = POLICY.capacity(selected.len(), self.cooperation.runtime())?;
        let mut start = 0;
        while start < selected.len() {
            let wave = Invocation {
                policy: POLICY,
                function,
                vectorized: self.vectorized,
                parallel: self.parallel,
                counts: self.counts,
                cooperation: self.cooperation,
            }
            .next(population, selected, start, 2)
            .await?;
            start = wave.end();
            match wave {
                Wave::Vectorized {
                    start,
                    end,
                    returned,
                } => {
                    let result = constraints::decode(
                        returned,
                        end - start,
                        *self.extents,
                        self.mixed_integer,
                        self.cooperation,
                    )
                    .await?;
                    *self.extents = Some(result.extents());
                    if start != 0 || end != selected.len() {
                        return Err(invalid(
                            "full nonlinear vectorized wave lost its original row extent",
                        ));
                    }
                    return if let Some(aliases) = self.aliases {
                        result.reorder(aliases.positions(), self.cooperation).await
                    } else {
                        Ok(result)
                    };
                }
                Wave::Rows { returned, .. } => {
                    for returned in returned {
                        let result = constraints::decode(
                            returned,
                            1,
                            *self.extents,
                            self.mixed_integer,
                            self.cooperation,
                        )
                        .await?;
                        *self.extents = Some(result.extents());
                        rows.push(result);
                        self.cooperation.completed(1).await?;
                    }
                }
            }
        }
        if rows.len() != selected.len() {
            return Err(invalid("nonlinear wave cardinality changed"));
        }
        constraints::assemble(
            rows,
            self.aliases.map(|aliases| aliases.positions()),
            self.cooperation,
        )
        .await
    }
}
