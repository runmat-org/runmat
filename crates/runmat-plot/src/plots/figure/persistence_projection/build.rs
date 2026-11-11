use std::collections::HashMap;

use super::super::{AxesGridCell, AxesMetadata, Figure, TextStyle};
use super::plan::{AxesPlan, ProjectionPlan};
use super::types::{
    FigureAnnotationRemap, FigureAxesRemap, FigureDecoration, FigureDecorationRemap,
    FigurePersistenceProjection, FigurePersistenceProjectionError, FigurePlotRemap,
    FigureProjectionRemap,
};

pub(super) fn project(
    source: &Figure,
    plan: ProjectionPlan,
) -> Result<FigurePersistenceProjection, FigurePersistenceProjectionError> {
    if plan.full_figure {
        return Ok(FigurePersistenceProjection {
            figure: source.clone(),
            remap: full_remap(source),
        });
    }

    let axes_remap = plan
        .axes
        .iter()
        .map(|axes| (axes.source, axes.destination))
        .collect::<HashMap<_, _>>();
    let mut remap = FigureProjectionRemap::default();
    let mut axes_metadata = Vec::with_capacity(plan.axes.len().max(1));
    let mut axes_grid_cells = Vec::with_capacity(plan.axes.len().max(1));
    let mut selected_plots = HashMap::new();

    for axes in &plan.axes {
        remap.axes.push(FigureAxesRemap {
            source: axes.source,
            destination: axes.destination,
        });
        axes_metadata.push(project_axes_metadata(
            source,
            axes,
            &axes_remap,
            &mut remap,
        )?);
        axes_grid_cells.push(
            (!source.axes_metadata[axes.source].position_explicit)
                .then(|| source_grid_cell(source, axes.source))
                .transpose()?,
        );
        for &source_plot in &axes.plots {
            selected_plots.insert(source_plot, axes.destination);
        }
    }

    // Preserve the source figure's global render order across selected axes.
    let mut plots = Vec::with_capacity(selected_plots.len());
    let mut plot_axes_indices = Vec::with_capacity(selected_plots.len());
    for (source_plot, plot) in source.plots.iter().enumerate() {
        let Some(&destination_axes) = selected_plots.get(&source_plot) else {
            continue;
        };
        let destination = plots.len();
        plots.push(plot.clone());
        plot_axes_indices.push(destination_axes);
        remap.plots.push(FigurePlotRemap {
            source: source_plot,
            destination,
        });
    }

    if axes_metadata.is_empty() {
        axes_metadata.push(Figure::default_axes_metadata());
        axes_grid_cells.push(None);
    }
    let active_axes_index = axes_remap
        .get(&source.active_axes_index)
        .copied()
        .unwrap_or(0);
    let base_axes = plan
        .axes
        .iter()
        .filter(|axes| source.axes_metadata[axes.source].overlay_parent.is_none())
        .count()
        .max(1);
    // An imported geometry scene stays only when the projection keeps every plot it drew.
    let geometry_source = (plots.len() == source.plots.len())
        .then(|| source.geometry_source.clone())
        .flatten();
    let mut figure = Figure {
        plots,
        name: source.name.clone(),
        number_title: source.number_title,
        visible: source.visible,
        position: source.position,
        title: None,
        sg_title: plan
            .include_super_title
            .then(|| source.sg_title.clone())
            .flatten(),
        x_label: None,
        y_label: None,
        z_label: None,
        legend_enabled: false,
        grid_enabled: false,
        minor_grid_enabled: false,
        box_enabled: false,
        background_color: source.background_color,
        x_limits: None,
        y_limits: None,
        z_limits: None,
        x_log: false,
        y_log: false,
        axis_equal: false,
        colormap: source.colormap.clone(),
        default_colormap: source.default_colormap.clone(),
        default_colormap_length: source.default_colormap_length,
        colorbar_enabled: false,
        color_limits: None,
        bounds: None,
        dirty: true,
        axes_rows: 1,
        axes_cols: base_axes,
        plot_axes_indices,
        axes_grid_cells,
        active_axes_index,
        axes_metadata,
        sg_title_style: if plan.include_super_title {
            source.sg_title_style.clone()
        } else {
            TextStyle::default()
        },
        geometry_source,
    };
    figure.sync_legacy_fields_from_active_axes();
    Ok(FigurePersistenceProjection { figure, remap })
}

fn project_axes_metadata(
    figure: &Figure,
    plan: &AxesPlan,
    axes_remap: &HashMap<usize, usize>,
    remap: &mut FigureProjectionRemap,
) -> Result<AxesMetadata, FigurePersistenceProjectionError> {
    let source = &figure.axes_metadata[plan.source];
    if plan.full {
        add_full_axes_remap(source, plan, remap);
        let mut cloned = source.clone();
        cloned.overlay_parent = remap_overlay(source, plan, axes_remap)?;
        validate_source_layout(figure, plan.source, &cloned)?;
        return Ok(cloned);
    }

    let selected = |decoration| plan.decorations.contains(&decoration);
    for &decoration in &plan.decorations {
        remap.decorations.push(FigureDecorationRemap {
            source_axes: plan.source,
            destination_axes: plan.destination,
            decoration,
        });
    }
    let mut annotations = Vec::with_capacity(plan.annotations.len());
    for &source_annotation in &plan.annotations {
        let destination_annotation = annotations.len();
        annotations.push(source.world_text_annotations[source_annotation].clone());
        remap.annotations.push(FigureAnnotationRemap {
            source_axes: plan.source,
            source_annotation,
            destination_axes: plan.destination,
            destination_annotation,
        });
    }

    let projected = AxesMetadata {
        colormap_length: source.colormap_length,
        y_direction: source.y_direction,
        axes_kind: source.axes_kind,
        overlay_parent: remap_overlay(source, plan, axes_remap)?,
        y_axis_location: source.y_axis_location.clone(),
        position: source.position,
        position_explicit: source.position_explicit,
        units: source.units.clone(),
        title: selected(FigureDecoration::Title)
            .then(|| source.title.clone())
            .flatten(),
        subtitle: selected(FigureDecoration::Subtitle)
            .then(|| source.subtitle.clone())
            .flatten(),
        x_label: selected(FigureDecoration::XLabel)
            .then(|| source.x_label.clone())
            .flatten(),
        y_label: selected(FigureDecoration::YLabel)
            .then(|| source.y_label.clone())
            .flatten(),
        z_label: selected(FigureDecoration::ZLabel)
            .then(|| source.z_label.clone())
            .flatten(),
        // Ruler geometry belongs to the implicit parent container and is
        // retained even when the selected public root is a plot or label.
        x_ticks: source.x_ticks.clone(),
        y_ticks: source.y_ticks.clone(),
        x_tick_labels: source.x_tick_labels.clone(),
        y_tick_labels: source.y_tick_labels.clone(),
        x_tick_format: source.x_tick_format.clone(),
        y_tick_format: source.y_tick_format.clone(),
        x_tick_label_rotation: source.x_tick_label_rotation,
        y_tick_label_rotation: source.y_tick_label_rotation,
        x_limits: source.x_limits,
        y_limits: source.y_limits,
        z_limits: source.z_limits,
        x_limit_method: source.x_limit_method,
        y_limit_method: source.y_limit_method,
        z_limit_method: source.z_limit_method,
        x_log: source.x_log,
        y_log: source.y_log,
        view_azimuth_deg: source.view_azimuth_deg,
        view_elevation_deg: source.view_elevation_deg,
        view_revision: source.view_revision,
        grid_enabled: source.grid_enabled,
        minor_grid_enabled: source.minor_grid_enabled,
        minor_grid_explicit: source.minor_grid_explicit,
        hidden_line_removal: source.hidden_line_removal,
        box_enabled: source.box_enabled,
        axis_equal: source.axis_equal,
        data_aspect_ratio: source.data_aspect_ratio,
        data_aspect_ratio_mode: source.data_aspect_ratio_mode.clone(),
        legend_enabled: selected(FigureDecoration::Legend) && source.legend_enabled,
        colorbar_enabled: selected(FigureDecoration::ColorBar) && source.colorbar_enabled,
        colormap: source.colormap.clone(),
        color_order: source.color_order.clone(),
        color_limits: source.color_limits,
        axes_style: source.axes_style.clone(),
        title_style: if selected(FigureDecoration::Title) {
            source.title_style.clone()
        } else {
            Default::default()
        },
        subtitle_style: if selected(FigureDecoration::Subtitle) {
            source.subtitle_style.clone()
        } else {
            Default::default()
        },
        x_label_style: if selected(FigureDecoration::XLabel) {
            source.x_label_style.clone()
        } else {
            Default::default()
        },
        y_label_style: if selected(FigureDecoration::YLabel) {
            source.y_label_style.clone()
        } else {
            Default::default()
        },
        z_label_style: if selected(FigureDecoration::ZLabel) {
            source.z_label_style.clone()
        } else {
            Default::default()
        },
        legend_style: if selected(FigureDecoration::Legend) {
            source.legend_style.clone()
        } else {
            Default::default()
        },
        world_text_annotations: annotations,
    };
    validate_source_layout(figure, plan.source, &projected)?;
    Ok(projected)
}

fn validate_source_layout(
    figure: &Figure,
    source_axes: usize,
    projected: &AxesMetadata,
) -> Result<(), FigurePersistenceProjectionError> {
    if projected.position_explicit {
        return Ok(());
    }
    let _ = source_grid_cell(figure, source_axes)?;
    Ok(())
}

fn source_grid_cell(
    figure: &Figure,
    source_axes: usize,
) -> Result<AxesGridCell, FigurePersistenceProjectionError> {
    if let Some(cell) = figure.axes_grid_cell(source_axes) {
        return Ok(cell);
    }
    let mut grid_axes = source_axes;
    while let Some(parent) = figure.axes_metadata[grid_axes].overlay_parent {
        grid_axes = parent;
    }
    if grid_axes >= figure.axes_rows.saturating_mul(figure.axes_cols) {
        return Err(FigurePersistenceProjectionError::MissingGridPosition {
            axes: source_axes,
            rows: figure.axes_rows,
            columns: figure.axes_cols,
        });
    }
    Ok(AxesGridCell {
        rows: figure.axes_rows.try_into().map_err(|_| {
            FigurePersistenceProjectionError::MissingGridPosition {
                axes: source_axes,
                rows: figure.axes_rows,
                columns: figure.axes_cols,
            }
        })?,
        columns: figure.axes_cols.try_into().map_err(|_| {
            FigurePersistenceProjectionError::MissingGridPosition {
                axes: source_axes,
                rows: figure.axes_rows,
                columns: figure.axes_cols,
            }
        })?,
        index: grid_axes.try_into().map_err(|_| {
            FigurePersistenceProjectionError::MissingGridPosition {
                axes: source_axes,
                rows: figure.axes_rows,
                columns: figure.axes_cols,
            }
        })?,
    })
}

fn remap_overlay(
    source: &AxesMetadata,
    plan: &AxesPlan,
    axes_remap: &HashMap<usize, usize>,
) -> Result<Option<usize>, FigurePersistenceProjectionError> {
    source
        .overlay_parent
        .map(|parent| {
            axes_remap.get(&parent).copied().ok_or(
                FigurePersistenceProjectionError::MissingOverlayParent {
                    axes: plan.source,
                    parent,
                },
            )
        })
        .transpose()
}

fn add_full_axes_remap(source: &AxesMetadata, plan: &AxesPlan, remap: &mut FigureProjectionRemap) {
    remap.decorations.extend(
        plan.decorations
            .iter()
            .map(|&decoration| FigureDecorationRemap {
                source_axes: plan.source,
                destination_axes: plan.destination,
                decoration,
            }),
    );
    remap
        .annotations
        .extend(
            source
                .world_text_annotations
                .iter()
                .enumerate()
                .map(|(annotation, _)| FigureAnnotationRemap {
                    source_axes: plan.source,
                    source_annotation: annotation,
                    destination_axes: plan.destination,
                    destination_annotation: annotation,
                }),
        );
}

fn full_remap(source: &Figure) -> FigureProjectionRemap {
    let mut remap = FigureProjectionRemap::default();
    for axes in 0..source.axes_metadata.len() {
        remap.axes.push(FigureAxesRemap {
            source: axes,
            destination: axes,
        });
        remap.decorations.extend([
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::Title,
            },
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::Subtitle,
            },
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::XLabel,
            },
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::YLabel,
            },
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::ZLabel,
            },
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::Legend,
            },
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::ColorBar,
            },
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::XRuler,
            },
            FigureDecorationRemap {
                source_axes: axes,
                destination_axes: axes,
                decoration: FigureDecoration::YRuler,
            },
        ]);
        remap.annotations.extend(
            source.axes_metadata[axes]
                .world_text_annotations
                .iter()
                .enumerate()
                .map(|(annotation, _)| FigureAnnotationRemap {
                    source_axes: axes,
                    source_annotation: annotation,
                    destination_axes: axes,
                    destination_annotation: annotation,
                }),
        );
    }
    remap
        .plots
        .extend((0..source.plots.len()).map(|plot| FigurePlotRemap {
            source: plot,
            destination: plot,
        }));
    remap
}
