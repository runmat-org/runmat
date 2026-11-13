//! Pure projection of an admitted graphics selection into an owned figure.
//!
//! Runtime code resolves public handles and closes the selected hierarchy.
//! This module owns the private `Figure` layout and copies only the selected
//! subgraph. It does not assign public handles or mutate the plot registry.

mod build;
mod plan;
mod types;

#[cfg(test)]
mod tests;

pub use types::{
    FigureAnnotationRemap, FigureAxesContainerSelection, FigureAxesPersistenceSelection,
    FigureAxesRemap, FigureAxesSelectionKind, FigureDecoration, FigureDecorationRemap,
    FigurePersistenceProjection, FigurePersistenceProjectionError, FigurePersistenceSelection,
    FigurePlotRemap, FigureProjectionRemap,
};

use super::Figure;

/// Project an already admitted hierarchy selection without registry effects.
pub fn project_figure_for_persistence(
    source: &Figure,
    selection: &FigurePersistenceSelection,
) -> Result<FigurePersistenceProjection, FigurePersistenceProjectionError> {
    let plan = plan::ProjectionPlan::prepare(source, selection)?;
    build::project(source, plan)
}
