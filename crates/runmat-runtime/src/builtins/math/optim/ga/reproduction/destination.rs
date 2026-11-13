//! Partial offspring stay private until the entire generation is admitted.
use super::super::callbacks::Invocation;
use super::super::{creation::Draft, invalid, population::Generation, ScalarCooperation};
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticPopulation;
use std::ops::ControlFlow;

pub(super) enum Destination {
    Numeric(Draft),
    Custom(Generation),
}
impl Destination {
    pub async fn new(
        current: &Generation,
        kind: OptimGeneticPopulation,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        use super::super::common::population::projection::Arguments;
        match current {
            Generation::Numeric(_) => Ok(Self::Numeric(
                Draft::new(
                    None,
                    current.members(),
                    current.variables(),
                    kind == OptimGeneticPopulation::BitString,
                    cooperation,
                )
                .await?,
            )),
            Generation::Custom { value, variables } => Ok(Self::Custom(Generation::Custom {
                value: value.fork(cooperation)?,
                variables: *variables,
            })),
        }
    }
    pub async fn copy<I: Invocation>(
        &mut self,
        destinations: &[usize],
        sources: &[usize],
        current: &Generation,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop, ()>> {
        if destinations.len() != sources.len() {
            return Err(invalid("copied offspring extent changed"));
        }
        if destinations.is_empty() {
            return Ok(ControlFlow::Continue(()));
        }
        match (self, current) {
            (Self::Numeric(next), Generation::Numeric(current)) => {
                for (&destination, &source) in destinations.iter().zip(sources) {
                    let point =
                        super::super::population::coordinates(current, source, cooperation).await?;
                    next.set(destination, &point, false, cooperation).await?;
                }
                Ok(ControlFlow::Continue(()))
            }
            (
                Self::Custom(Generation::Custom { value: next, .. }),
                Generation::Custom { value: current, .. },
            ) => {
                use super::super::common::population::projection::{Argument, Arguments};
                let prepared = current.prepare_projection(sources, cooperation).await?;
                let projected = match invocation.projection(prepared, cooperation).await? {
                    ControlFlow::Continue(value) => value,
                    ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
                };
                let projected = match projected {
                    Argument::Rooted(value) => value,
                    Argument::Source(value) => crate::execution::RootedValue::in_optional_context(
                        value,
                        cooperation.runtime(),
                    )?,
                };
                let children = super::super::population::custom::Custom::from_result(
                    projected,
                    sources.len(),
                    invocation,
                    cooperation,
                )
                .await?;
                let children = match children {
                    ControlFlow::Continue(value) => value,
                    ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
                };
                next.replace_rows(destinations, &children, invocation, cooperation)
                    .await
            }
            _ => Err(invalid(
                "offspring destination changed population representation",
            )),
        }
    }
    pub async fn point(
        &mut self,
        row: usize,
        point: &[f64],
        bitstring: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        match self {
            Self::Numeric(next) => next
                .set(row, point, bitstring, cooperation)
                .await
                .map(|_| ()),
            Self::Custom(_) => Err(invalid("numeric operator cannot install a custom genome")),
        }
    }
    pub async fn children<I: Invocation>(
        &mut self,
        rows: &[usize],
        children: Generation,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop, ()>> {
        use super::super::common::population::projection::Arguments;
        match (self, children) {
            (Self::Numeric(next), Generation::Numeric(children)) => {
                if rows.len() != children.members() {
                    return Err(invalid("custom offspring count changed"));
                }
                for (source, &destination) in rows.iter().enumerate() {
                    let point =
                        super::super::population::coordinates(&children, source, cooperation)
                            .await?;
                    next.set(destination, &point, false, cooperation).await?;
                }
                Ok(ControlFlow::Continue(()))
            }
            (Self::Custom(next), children @ Generation::Custom { .. }) => {
                next.replace_with(rows, children, invocation, cooperation)
                    .await
            }
            _ => Err(invalid("custom operator changed population representation")),
        }
    }
    pub fn finish(self) -> Generation {
        match self {
            Self::Numeric(next) => Generation::Numeric(next.finish()),
            Self::Custom(next) => next,
        }
    }
}
