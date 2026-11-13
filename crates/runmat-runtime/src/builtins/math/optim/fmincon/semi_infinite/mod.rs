//! Semi-infinite mesh and accepted-generation ownership, not a public wrapper.
//! Public grammar, selected controls and original-state publication join this
//! same accepted-generation numerical owner.
mod acceptance;
mod entry;
mod inputs;
mod options;
mod publication;
pub(in crate::builtins::math::optim) use entry::execute;
mod controls;
pub(super) use controls::Controls;
mod bernstein;
mod generation;
mod mesh;
mod observation;
mod oracle;
use generation::{FrozenRows, State};
use mesh::Samples;
pub(super) use observation::{description, exitflag, observe};
use std::sync::Arc;
pub(super) struct Domain {
    state: State,
    pub controls: Controls,
    plots: Option<crate::execution::RootedValue>,
    #[cfg(feature = "plot-core")]
    layout: Option<crate::builtins::plotting::state::OptimizationPlotLayout>,
}
impl Domain {
    pub fn new(families: usize, controls: Controls) -> crate::BuiltinResult<Self> {
        controls.validate()?;
        Ok(Self {
            state: State::new(families)?,
            controls,
            plots: None,
            #[cfg(feature = "plot-core")]
            layout: None,
        })
    }
}
pub(super) struct Payload {
    rows: FrozenRows,
    samples: Vec<Arc<Samples>>,
    finite: generation::FiniteExtents,
}
#[derive(Clone)]
pub(super) struct GenerationStamp(FrozenRows);
pub(super) fn stamp(point: &super::evaluation::Point) -> Option<GenerationStamp> {
    point
        .semi_infinite
        .as_ref()
        .map(|p| GenerationStamp(p.rows.clone()))
}
pub(super) fn require_current(
    evaluator: &super::evaluation::Evaluator<'_>,
    point: &super::evaluation::Point,
    model: &super::model::Linearization,
) -> crate::BuiltinResult<()> {
    match (
        &evaluator.semi_infinite,
        &point.semi_infinite,
        &model.semi_infinite,
    ) {
        (None, None, None) => Ok(()),
        (Some(domain), Some(payload), Some(stamp)) => {
            domain.state.require_current(&payload.rows, &stamp.0)
        }
        _ => Err(bernstein::invalid(
            "accepted point and linearization have inconsistent mesh ownership",
        )),
    }
}
pub(super) use acceptance::{refresh, Refresh};
pub(super) use oracle::{admit_double, point, probe};

#[cfg(test)]
mod tests;
