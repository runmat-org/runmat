use super::super::AxesGridCell;
use super::*;
use crate::plots::figure::{LegendStyle, TextStyle};
use crate::plots::LinePlot;
use glam::Vec3;

fn line(y: f64) -> LinePlot {
    LinePlot::new(vec![0.0, 1.0], vec![y, y + 1.0]).expect("line")
}

#[test]
fn leaf_plot_keeps_container_geometry_without_sibling_decorations() {
    let mut figure = Figure::new();
    figure.set_axes_title(0, "sibling title");
    figure.set_axes_subtitle(0, "sibling subtitle");
    figure.set_axes_xlabel(0, "sibling x label");
    figure.set_axes_legend_enabled(0, true);
    figure.set_axes_legend_style(
        0,
        LegendStyle {
            location: Some("north".into()),
            ..Default::default()
        },
    );
    figure.set_axes_limits(0, Some((-4.0, 9.0)), Some((2.0, 12.0)));
    figure.set_axes_ticks(0, Some(vec![0.0, 3.0]), Some(vec![2.0, 6.0]));
    figure.add_axes_text_annotation(
        0,
        Vec3::new(1.0, 2.0, 0.0),
        "sibling annotation",
        TextStyle::default(),
    );
    let selected_plot = figure.add_line_plot_on_axes(line(1.0), 0);
    figure.add_line_plot_on_axes(line(10.0), 0);

    let projected = project_figure_for_persistence(
        &figure,
        &FigurePersistenceSelection::Hierarchy {
            include_super_title: false,
            axes: vec![FigureAxesPersistenceSelection {
                source_axes_index: 0,
                kind: FigureAxesSelectionKind::Container(FigureAxesContainerSelection {
                    plot_indices: vec![selected_plot],
                    ..Default::default()
                }),
            }],
        },
    )
    .expect("leaf projection");

    let axes = projected.figure.axes_metadata(0).expect("projected axes");
    assert_eq!(axes.x_limits, Some((-4.0, 9.0)));
    assert_eq!(axes.y_limits, Some((2.0, 12.0)));
    assert_eq!(axes.x_ticks.as_deref(), Some([0.0, 3.0].as_slice()));
    assert_eq!(axes.y_ticks.as_deref(), Some([2.0, 6.0].as_slice()));
    assert_eq!(axes.title, None);
    assert_eq!(axes.subtitle, None);
    assert_eq!(axes.x_label, None);
    assert!(!axes.legend_enabled);
    assert_eq!(axes.legend_style.location, None);
    assert!(axes.world_text_annotations.is_empty());
    assert_eq!(projected.figure.len(), 1);
    assert_eq!(projected.remap.plots[0].source, selected_plot);
}

#[test]
fn selected_decoration_and_annotation_do_not_pull_siblings() {
    let mut figure = Figure::new();
    figure.set_axes_title(0, "selected title");
    figure.set_axes_subtitle(0, "not selected subtitle");
    figure.set_axes_legend_enabled(0, true);
    figure.add_axes_text_annotation(0, Vec3::ZERO, "first", TextStyle::default());
    figure.add_axes_text_annotation(0, Vec3::ONE, "second", TextStyle::default());

    let projected = project_figure_for_persistence(
        &figure,
        &FigurePersistenceSelection::Hierarchy {
            include_super_title: false,
            axes: vec![FigureAxesPersistenceSelection {
                source_axes_index: 0,
                kind: FigureAxesSelectionKind::Container(FigureAxesContainerSelection {
                    decorations: vec![FigureDecoration::Title],
                    annotation_indices: vec![1],
                    plot_indices: Vec::new(),
                }),
            }],
        },
    )
    .expect("decoration projection");

    let axes = projected.figure.axes_metadata(0).expect("projected axes");
    assert_eq!(axes.title.as_deref(), Some("selected title"));
    assert_eq!(axes.subtitle, None);
    assert!(!axes.legend_enabled);
    assert_eq!(axes.world_text_annotations.len(), 1);
    assert_eq!(axes.world_text_annotations[0].text, "second");
    assert_eq!(projected.remap.annotations[0].source_annotation, 1);
}

#[test]
fn single_compacted_leaf_keeps_its_nonexplicit_source_grid_cell() {
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    let plot = figure.add_line_plot_on_axes(line(4.0), 1);
    let projected = project_figure_for_persistence(
        &figure,
        &FigurePersistenceSelection::Hierarchy {
            include_super_title: false,
            axes: vec![FigureAxesPersistenceSelection {
                source_axes_index: 1,
                kind: FigureAxesSelectionKind::Container(FigureAxesContainerSelection {
                    plot_indices: vec![plot],
                    ..Default::default()
                }),
            }],
        },
    )
    .expect("single leaf projection");

    assert_eq!(projected.figure.axes_count(), 1);
    assert_eq!(
        crate::core::axes_layout::figure_viewports_px(&projected.figure, 401, 5),
        vec![(200, 0, 201, 5)]
    );
}

#[test]
fn stacked_axes_selection_remaps_overlay_closure() {
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    let overlay = figure.ensure_overlay_axes(1);
    // Deliberately interleave the global render order against axes order.
    let overlay_plot = figure.add_line_plot_on_axes(line(3.0), overlay);
    let base_plot = figure.add_line_plot_on_axes(line(1.0), 1);

    let projected = project_figure_for_persistence(
        &figure,
        &FigurePersistenceSelection::Hierarchy {
            include_super_title: false,
            axes: vec![
                FigureAxesPersistenceSelection {
                    source_axes_index: overlay,
                    kind: FigureAxesSelectionKind::Container(FigureAxesContainerSelection {
                        plot_indices: vec![overlay_plot],
                        ..Default::default()
                    }),
                },
                FigureAxesPersistenceSelection {
                    source_axes_index: 1,
                    kind: FigureAxesSelectionKind::Container(FigureAxesContainerSelection {
                        plot_indices: vec![base_plot],
                        ..Default::default()
                    }),
                },
            ],
        },
    )
    .expect("stacked axes projection");

    assert_eq!(projected.remap.axes.len(), 2);
    assert_eq!(projected.remap.axes[0].source, 1);
    assert_eq!(projected.remap.axes[0].destination, 0);
    assert_eq!(projected.remap.axes[1].source, overlay);
    assert_eq!(projected.remap.axes[1].destination, 1);
    assert_eq!(projected.figure.axes_overlay_parent(1), Some(0));
    assert_eq!(projected.figure.plot_axes_indices(), &[1, 0]);
    let base = projected.figure.axes_metadata(0).expect("base axes");
    let overlay = projected.figure.axes_metadata(1).expect("overlay axes");
    assert!(!base.position_explicit);
    assert!(!overlay.position_explicit);
    assert_eq!(
        projected.figure.axes_grid_cell(0),
        Some(AxesGridCell {
            rows: 1,
            columns: 2,
            index: 1,
        })
    );
    assert_eq!(
        projected.figure.axes_grid_cell(1),
        projected.figure.axes_grid_cell(0)
    );
    assert_eq!(
        crate::core::axes_layout::figure_viewports_px(&projected.figure, 401, 5),
        vec![(200, 0, 201, 5), (200, 0, 201, 5)]
    );
}

#[test]
fn many_full_axes_keep_one_global_plot_order() {
    const AXES: usize = 64;
    let mut figure = Figure::new();
    figure.set_subplot_grid(8, 8);
    for axes in (0..AXES).rev() {
        figure.add_line_plot_on_axes(line(axes as f64), axes);
    }
    let projected = project_figure_for_persistence(
        &figure,
        &FigurePersistenceSelection::Hierarchy {
            include_super_title: false,
            axes: (0..AXES)
                .map(|source_axes_index| FigureAxesPersistenceSelection {
                    source_axes_index,
                    kind: FigureAxesSelectionKind::FullAxes,
                })
                .collect(),
        },
    )
    .expect("many-axes projection");

    assert_eq!(projected.figure.len(), AXES);
    let expected_axes = (0..AXES).rev().collect::<Vec<_>>();
    assert_eq!(
        projected.figure.plot_axes_indices(),
        expected_axes.as_slice()
    );
    assert_eq!(
        projected
            .remap
            .plots
            .iter()
            .map(|entry| entry.source)
            .collect::<Vec<_>>(),
        (0..AXES).collect::<Vec<_>>()
    );

    let full = project_figure_for_persistence(&figure, &FigurePersistenceSelection::FullFigure)
        .expect("full-figure projection");
    assert_eq!(full.figure.plot_axes_indices(), expected_axes.as_slice());
    assert_eq!(
        full.remap
            .plots
            .iter()
            .map(|entry| entry.source)
            .collect::<Vec<_>>(),
        (0..AXES).collect::<Vec<_>>()
    );
}

#[test]
fn nonexplicit_grid_and_explicit_positions_preserve_effective_geometry() {
    let mut figure = Figure::new();
    figure.set_subplot_grid(2, 2);
    let grid_plot = figure.add_line_plot_on_axes(line(1.0), 2);
    figure.set_axes_position(3, [0.2, 0.3, 0.4, 0.5]);
    figure.set_axes_units(3, "normalized");
    let positioned_plot = figure.add_line_plot_on_axes(line(2.0), 3);

    let projected = project_figure_for_persistence(
        &figure,
        &FigurePersistenceSelection::Hierarchy {
            include_super_title: false,
            axes: vec![
                FigureAxesPersistenceSelection {
                    source_axes_index: 2,
                    kind: FigureAxesSelectionKind::Container(FigureAxesContainerSelection {
                        plot_indices: vec![grid_plot],
                        ..Default::default()
                    }),
                },
                FigureAxesPersistenceSelection {
                    source_axes_index: 3,
                    kind: FigureAxesSelectionKind::Container(FigureAxesContainerSelection {
                        plot_indices: vec![positioned_plot],
                        ..Default::default()
                    }),
                },
            ],
        },
    )
    .expect("layout projection");

    let grid = projected.figure.axes_metadata(0).expect("grid axes");
    assert!(!grid.position_explicit);
    assert_eq!(
        projected.figure.axes_grid_cell(0),
        Some(AxesGridCell {
            rows: 2,
            columns: 2,
            index: 2,
        })
    );
    let positioned = projected.figure.axes_metadata(1).expect("positioned axes");
    assert!(positioned.position_explicit);
    assert_eq!(positioned.position, [0.2, 0.3, 0.4, 0.5]);
    assert_eq!(projected.figure.axes_grid_cell(1), None);
    let viewports = crate::core::axes_layout::figure_viewports_px(&projected.figure, 401, 5);
    assert_eq!(viewports[0], (0, 2, 200, 3));
    assert_eq!(viewports[1], (80, 1, 160, 3));

    let scene = crate::event::FigureScene::capture(&projected.figure);
    let mut stale = scene.clone();
    stale.schema_version = 4;
    assert!(stale.into_figure().is_err());
    let rebuilt = scene.into_figure().expect("layout identity round trip");
    assert_eq!(
        rebuilt.axes_grid_cell(0),
        projected.figure.axes_grid_cell(0)
    );
    assert_eq!(rebuilt.axes_grid_cell(1), None);
}

#[test]
fn overlay_without_selected_parent_is_rejected_before_projection() {
    let mut figure = Figure::new();
    let overlay = figure.ensure_overlay_axes(0);
    let error = project_figure_for_persistence(
        &figure,
        &FigurePersistenceSelection::Hierarchy {
            include_super_title: false,
            axes: vec![FigureAxesPersistenceSelection {
                source_axes_index: overlay,
                kind: FigureAxesSelectionKind::FullAxes,
            }],
        },
    )
    .expect_err("overlay closure is required");
    assert_eq!(
        error,
        FigurePersistenceProjectionError::MissingOverlayParent {
            axes: overlay,
            parent: 0,
        }
    );
}

#[test]
fn scene_grid_cells_are_revisioned_and_validated() {
    let mut legacy_json = serde_json::to_value(crate::event::FigureScene::capture(&Figure::new()))
        .expect("serialize scene");
    legacy_json["schemaVersion"] = serde_json::json!(4);
    legacy_json["layout"]
        .as_object_mut()
        .expect("layout object")
        .remove("axesGridCells");
    let legacy: crate::event::FigureScene =
        serde_json::from_value(legacy_json).expect("deserialize v4 scene");
    assert!(legacy.layout.axes_grid_cells.is_empty());
    legacy.into_figure().expect("v4 scene without grid cells");

    let mut base = crate::event::FigureScene::capture(&Figure::new());
    for invalid in [
        AxesGridCell {
            rows: 0,
            columns: 1,
            index: 0,
        },
        AxesGridCell {
            rows: u32::MAX,
            columns: 2,
            index: 0,
        },
        AxesGridCell {
            rows: 1,
            columns: 1,
            index: 1,
        },
    ] {
        base.layout.axes_grid_cells = vec![Some(invalid)];
        assert!(base.clone().into_figure().is_err());
    }

    base.layout.axes_grid_cells = vec![None, None];
    assert!(base.into_figure().is_err());
}
