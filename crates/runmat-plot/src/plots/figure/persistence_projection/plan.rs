use std::collections::{BTreeMap, BTreeSet};

use super::super::Figure;
use super::types::{
    FigureAxesContainerSelection, FigureAxesSelectionKind, FigureDecoration,
    FigurePersistenceProjectionError, FigurePersistenceSelection,
};

pub(super) struct ProjectionPlan {
    pub(super) full_figure: bool,
    pub(super) include_super_title: bool,
    pub(super) axes: Vec<AxesPlan>,
}

pub(super) struct AxesPlan {
    pub(super) source: usize,
    pub(super) destination: usize,
    pub(super) full: bool,
    pub(super) decorations: Vec<FigureDecoration>,
    pub(super) annotations: Vec<usize>,
    pub(super) plots: Vec<usize>,
}

#[derive(Default)]
struct AxesAccumulator {
    full: bool,
    decorations: BTreeSet<FigureDecoration>,
    annotations: BTreeSet<usize>,
    plots: BTreeSet<usize>,
}

impl ProjectionPlan {
    pub(super) fn prepare(
        source: &Figure,
        selection: &FigurePersistenceSelection,
    ) -> Result<Self, FigurePersistenceProjectionError> {
        let (include_super_title, selected_axes) = match selection {
            FigurePersistenceSelection::FullFigure => return Ok(full_figure_plan(source)),
            FigurePersistenceSelection::Hierarchy {
                include_super_title,
                axes,
            } => (*include_super_title, axes.as_slice()),
        };
        if selected_axes.is_empty() && !include_super_title {
            return Err(FigurePersistenceProjectionError::EmptySelection);
        }

        let mut axes = BTreeMap::<usize, AxesAccumulator>::new();
        for selected in selected_axes {
            let source_axes = selected.source_axes_index;
            let metadata = source
                .axes_metadata
                .get(source_axes)
                .ok_or(FigurePersistenceProjectionError::AxesOutOfBounds { axes: source_axes })?;
            let accumulated = axes.entry(source_axes).or_default();
            match &selected.kind {
                FigureAxesSelectionKind::FullAxes => accumulated.full = true,
                FigureAxesSelectionKind::Container(container) => {
                    merge_container(accumulated, container);
                }
            }
            if let Some(parent) = metadata.overlay_parent {
                // Validation after the merge permits the parent to appear later.
                if parent >= source.axes_metadata.len() {
                    return Err(FigurePersistenceProjectionError::AxesOutOfBounds { axes: parent });
                }
            }
        }

        for (&source_axes, accumulated) in &axes {
            let metadata = &source.axes_metadata[source_axes];
            if let Some(parent) = metadata.overlay_parent {
                if !axes.contains_key(&parent) {
                    return Err(FigurePersistenceProjectionError::MissingOverlayParent {
                        axes: source_axes,
                        parent,
                    });
                }
            }
            validate_members(source, source_axes, accumulated)?;
        }

        let mut plots_by_axes = plots_by_axes(source);
        let axes = axes
            .into_iter()
            .enumerate()
            .map(|(destination, (source_axes, accumulated))| {
                let plots = if accumulated.full {
                    std::mem::take(&mut plots_by_axes[source_axes])
                } else {
                    accumulated.plots.into_iter().collect()
                };
                let annotations = if accumulated.full {
                    (0..source.axes_metadata[source_axes]
                        .world_text_annotations
                        .len())
                        .collect()
                } else {
                    accumulated.annotations.into_iter().collect()
                };
                let decorations = if accumulated.full {
                    all_decorations()
                } else {
                    accumulated.decorations.into_iter().collect()
                };
                AxesPlan {
                    source: source_axes,
                    destination,
                    full: accumulated.full,
                    decorations,
                    annotations,
                    plots,
                }
            })
            .collect();

        Ok(Self {
            full_figure: false,
            include_super_title,
            axes,
        })
    }
}

fn full_figure_plan(source: &Figure) -> ProjectionPlan {
    let plots_by_axes = plots_by_axes(source);
    let mut axes = Vec::with_capacity(source.axes_metadata.len());
    for (source_axes, plots) in plots_by_axes.into_iter().enumerate() {
        axes.push(AxesPlan {
            source: source_axes,
            destination: source_axes,
            full: true,
            decorations: all_decorations(),
            annotations: (0..source.axes_metadata[source_axes]
                .world_text_annotations
                .len())
                .collect(),
            plots,
        });
    }
    ProjectionPlan {
        full_figure: true,
        include_super_title: true,
        axes,
    }
}

fn plots_by_axes(source: &Figure) -> Vec<Vec<usize>> {
    let mut grouped = vec![Vec::new(); source.axes_metadata.len()];
    for (plot, &axes) in source.plot_axes_indices.iter().enumerate() {
        if let Some(plots) = grouped.get_mut(axes) {
            plots.push(plot);
        }
    }
    grouped
}

fn merge_container(target: &mut AxesAccumulator, selected: &FigureAxesContainerSelection) {
    target
        .decorations
        .extend(selected.decorations.iter().copied());
    target
        .annotations
        .extend(selected.annotation_indices.iter().copied());
    target.plots.extend(selected.plot_indices.iter().copied());
}

fn validate_members(
    source: &Figure,
    source_axes: usize,
    selected: &AxesAccumulator,
) -> Result<(), FigurePersistenceProjectionError> {
    if selected.full {
        return Ok(());
    }
    for &plot in &selected.plots {
        let actual_axes = source
            .plot_axes_indices
            .get(plot)
            .copied()
            .ok_or(FigurePersistenceProjectionError::PlotOutOfBounds { plot })?;
        if actual_axes != source_axes {
            return Err(FigurePersistenceProjectionError::PlotOwnedByDifferentAxes {
                plot,
                selected_axes: source_axes,
                actual_axes,
            });
        }
    }
    let annotation_count = source.axes_metadata[source_axes]
        .world_text_annotations
        .len();
    for &annotation in &selected.annotations {
        if annotation >= annotation_count {
            return Err(FigurePersistenceProjectionError::AnnotationOutOfBounds {
                axes: source_axes,
                annotation,
            });
        }
    }
    Ok(())
}

fn all_decorations() -> Vec<FigureDecoration> {
    vec![
        FigureDecoration::Title,
        FigureDecoration::Subtitle,
        FigureDecoration::XLabel,
        FigureDecoration::YLabel,
        FigureDecoration::ZLabel,
        FigureDecoration::Legend,
        FigureDecoration::ColorBar,
        FigureDecoration::XRuler,
        FigureDecoration::YRuler,
    ]
}
