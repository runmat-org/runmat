use glam::{Vec3, Vec4};
use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_analysis_fea::contracts::{
    FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
    FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
};
use runmat_geometry_core::UnitSystem;
use runmat_plot::plots::{
    BarChart, Figure, LinePlot, MeshDeformation, MeshEdgeMode, MeshFieldLocation, MeshPlot,
    MeshScalarField, MeshVectorField, PlotElement,
};

use super::contracts::{
    AnalysisFieldDescriptor, AnalysisFieldLocation, AnalysisRenderTopology,
    AnalysisResultsCompareData, AnalysisResultsCompareQuery, AnalysisRunKind, AnalysisRunResult,
    AnalysisStudySpec, AnalysisTrendsData, AnalysisTrendsQuery,
};
use super::{analysis_results_compare_op, analysis_trends_op, collect_analysis_result_fields};
use super::{run_kind, storage};
use crate::geometry::{geometry_preview_figure, GeometryPreviewFigureOptions};
use crate::operations::OperationContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisGeneratedFigureKind {
    MeshResult,
    Summary,
    Convergence,
    Modal,
    Electromagnetic,
    Comparison,
    Trend,
}

#[derive(Debug, Clone)]
pub struct AnalysisGeneratedFigure {
    pub kind: AnalysisGeneratedFigureKind,
    pub title: String,
    pub field_ids: Vec<String>,
    pub topology_ids: Vec<String>,
    pub warnings: Vec<String>,
    pub figure: Figure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisFigureMeshSource {
    Auto,
    Solver,
    Cad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisFigureGenerationOptions {
    pub max_overlay_values: usize,
    pub max_vector_glyphs: usize,
    pub max_mesh_result_figures: usize,
    pub max_mesh_geometry_bytes: usize,
    pub edge_overlay_triangle_limit: usize,
    pub mesh_source: AnalysisFigureMeshSource,
    pub show_solver_mesh_edges: bool,
    pub apply_deformation_overlay: bool,
    pub include_comparison: bool,
    pub include_trends: bool,
}

impl Default for AnalysisFigureGenerationOptions {
    fn default() -> Self {
        Self {
            max_overlay_values: 1_500_000,
            max_vector_glyphs: 40_000,
            max_mesh_result_figures: 4,
            max_mesh_geometry_bytes: 256 * 1024 * 1024,
            edge_overlay_triangle_limit: 250_000,
            mesh_source: AnalysisFigureMeshSource::Auto,
            show_solver_mesh_edges: false,
            apply_deformation_overlay: true,
            include_comparison: true,
            include_trends: true,
        }
    }
}

#[derive(Debug, Clone)]
struct MeshCounts {
    plot_index: usize,
    vertices: usize,
    triangles: usize,
    vertex_volume_node_indices: Vec<Option<usize>>,
    triangle_volume_element_indices: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
struct ScalarOverlay {
    field_id: String,
    label: String,
    location: MeshFieldLocation,
    chunks: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct VectorOverlay {
    field_id: String,
    label: String,
    location: MeshFieldLocation,
    chunks: Vec<Vec<Vec3>>,
    stride: usize,
}

#[derive(Debug, Clone)]
struct DeformationOverlay {
    field_id: String,
    label: String,
    chunks: Vec<Vec<Vec3>>,
    scale: f32,
}

pub fn analysis_generate_study_run_figures(
    study: &AnalysisStudySpec,
    run_id: &str,
    options: AnalysisFigureGenerationOptions,
) -> Result<Vec<AnalysisGeneratedFigure>, String> {
    let current = storage::load_run_result(run_id)?
        .ok_or_else(|| format!("FEA run_id '{run_id}' was not found"))?;
    let mut figures = generate_run_figures(&study.geometry, &current, options);

    if options.include_comparison {
        if let Some(previous) = previous_run_of_kind(&current)? {
            let query = AnalysisResultsCompareQuery {
                baseline_run_id: previous.run_id.clone(),
                candidate_run_id: current.run_id.clone(),
            };
            if let Ok(envelope) =
                analysis_results_compare_op(query, OperationContext::new(None, None))
            {
                if let Some(figure) = comparison_figure(&envelope.data) {
                    figures.push(figure);
                }
            }
        }
    }

    if options.include_trends {
        if let Ok(envelope) = analysis_trends_op(
            AnalysisTrendsQuery::default(),
            OperationContext::new(None, None),
        ) {
            figures.extend(trend_figures(&envelope.data));
        }
    }

    Ok(figures)
}

fn generate_run_figures(
    geometry: &runmat_geometry_core::GeometryAsset,
    run: &AnalysisRunResult,
    options: AnalysisFigureGenerationOptions,
) -> Vec<AnalysisGeneratedFigure> {
    let mut figures = Vec::new();
    figures.extend(mesh_result_figures(geometry, run, options));
    figures.extend(summary_figures(run));
    figures.extend(convergence_figures(run));
    figures
}

fn mesh_result_figures(
    geometry: &runmat_geometry_core::GeometryAsset,
    run: &AnalysisRunResult,
    options: AnalysisFigureGenerationOptions,
) -> Vec<AnalysisGeneratedFigure> {
    let render_topology = run
        .render_topology
        .as_ref()
        .filter(|topology| render_topology_has_meshes(topology));
    if (render_topology.is_none() && geometry.surface_meshes.is_empty())
        || options.max_mesh_result_figures == 0
    {
        return Vec::new();
    }

    let estimated_geometry_bytes = render_topology
        .map(render_topology_mesh_bytes)
        .unwrap_or_else(|| geometry_surface_mesh_bytes(geometry));
    let mut per_run_mesh_figure_limit = options.max_mesh_result_figures;
    let mut shared_warnings = Vec::new();
    if estimated_geometry_bytes > options.max_mesh_geometry_bytes {
        per_run_mesh_figure_limit = 1;
        shared_warnings.push(format!(
            "mesh result figure count capped to 1 because the render mesh is approximately {} bytes",
            estimated_geometry_bytes
        ));
    }

    let fields = collect_analysis_result_fields(run);
    let probe =
        match base_mesh_figure_for_run_source(geometry, render_topology, "FEA result", options) {
            Some(figure) => figure,
            None => {
                return vec![warning_line_figure(
                    AnalysisGeneratedFigureKind::MeshResult,
                    "FEA result visualization",
                    "Solver render topology and geometry preview are unavailable".to_string(),
                )];
            }
        };
    let mesh_counts = collect_mesh_counts_with_topology(&probe, render_topology);
    if mesh_counts.is_empty() {
        return Vec::new();
    }

    let deformation = if options.apply_deformation_overlay {
        fields
            .iter()
            .filter(|field| is_deformation_candidate(&field.field_id))
            .find_map(|field| deformation_overlay(field, &mesh_counts, &probe, options))
    } else {
        None
    };

    let mut figures = Vec::new();
    if let Some(deformation) = deformation.as_ref() {
        if figures.len() < per_run_mesh_figure_limit {
            if let Some(mut figure) = base_mesh_figure(
                geometry,
                render_topology,
                format!("FEA deformed shape: {}", deformation.field_id),
                options,
            ) {
                let mut warnings = shared_warnings.clone();
                append_deformed_mesh_overlay(&mut figure, deformation, &mesh_counts, &mut warnings);
                figures.push(AnalysisGeneratedFigure {
                    kind: AnalysisGeneratedFigureKind::MeshResult,
                    title: format!("FEA deformed shape: {}", deformation.field_id),
                    field_ids: vec![deformation.field_id.clone()],
                    topology_ids: topology_ids_for_field_ids([deformation.field_id.as_str()]),
                    warnings,
                    figure,
                });
            }
        }
    }

    let mut topology_warnings = Vec::new();
    for field in &fields {
        if figures.len() >= per_run_mesh_figure_limit {
            break;
        }
        let Some(scalar) = scalar_overlay(field, &mesh_counts, options) else {
            if let Some(warning) = field_topology_mismatch_warning(field, &mesh_counts) {
                topology_warnings.push(warning);
            }
            continue;
        };
        let title = format!("FEA scalar field: {}", scalar.field_id);
        let Some(mut figure) = base_mesh_figure(geometry, render_topology, title.clone(), options)
        else {
            continue;
        };
        let mut warnings = shared_warnings.clone();
        apply_scalar_overlay(&mut figure, &scalar, &mesh_counts, &mut warnings);
        if let Some(deformation) = deformation.as_ref() {
            apply_deformation_to_existing_meshes(
                &mut figure,
                deformation,
                &mesh_counts,
                &mut warnings,
            );
        }
        figure.colorbar_enabled = true;
        figures.push(AnalysisGeneratedFigure {
            kind: AnalysisGeneratedFigureKind::MeshResult,
            title,
            field_ids: vec![scalar.field_id],
            topology_ids: topology_ids_for_fields(std::iter::once(field)),
            warnings,
            figure,
        });
    }

    for field in &fields {
        if figures.len() >= per_run_mesh_figure_limit {
            break;
        }
        let Some(vector) = vector_overlay(field, &mesh_counts, options) else {
            if let Some(warning) = field_topology_mismatch_warning(field, &mesh_counts) {
                topology_warnings.push(warning);
            }
            continue;
        };
        let title = format!("FEA vector field: {}", vector.field_id);
        let Some(mut figure) = base_mesh_figure(geometry, render_topology, title.clone(), options)
        else {
            continue;
        };
        let mut warnings = shared_warnings.clone();
        apply_vector_overlay(&mut figure, &vector, &mesh_counts, &mut warnings);
        if let Some(deformation) = deformation.as_ref() {
            apply_deformation_to_existing_meshes(
                &mut figure,
                deformation,
                &mesh_counts,
                &mut warnings,
            );
        }
        figures.push(AnalysisGeneratedFigure {
            kind: AnalysisGeneratedFigureKind::MeshResult,
            title,
            field_ids: vec![vector.field_id],
            topology_ids: topology_ids_for_fields(std::iter::once(field)),
            warnings,
            figure,
        });
    }

    if figures.is_empty() {
        if let Some(warning) = topology_warnings.first() {
            figures.push(warning_line_figure(
                AnalysisGeneratedFigureKind::MeshResult,
                "FEA field topology mismatch",
                warning.clone(),
            ));
            return figures;
        }
        if let Some(figure) = base_mesh_figure(
            geometry,
            render_topology,
            format!("FEA geometry result: {}", run.run_id),
            options,
        ) {
            figures.push(AnalysisGeneratedFigure {
                kind: AnalysisGeneratedFigureKind::MeshResult,
                title: format!("FEA geometry result: {}", run.run_id),
                field_ids: Vec::new(),
                topology_ids: Vec::new(),
                warnings: shared_warnings,
                figure,
            });
        }
    }

    figures
}

fn summary_figures(run: &AnalysisRunResult) -> Vec<AnalysisGeneratedFigure> {
    let fields = collect_analysis_result_fields(run);
    let Some(figure) = structural_result_summary_figure(&fields) else {
        return Vec::new();
    };
    vec![figure]
}

fn structural_result_summary_figure(fields: &[AnalysisField]) -> Option<AnalysisGeneratedFigure> {
    let mut labels = Vec::new();
    let mut values = Vec::new();
    let mut field_ids = Vec::new();

    if let Some(value) = vector_field_total_magnitude(fields, FEA_FIELD_STRUCTURAL_REACTION_FORCE) {
        labels.push("Reaction force norm".to_string());
        values.push(value);
        field_ids.push(FEA_FIELD_STRUCTURAL_REACTION_FORCE.to_string());
    }
    if let Some(value) = vector_field_total_magnitude(fields, FEA_FIELD_STRUCTURAL_REACTION_MOMENT)
    {
        labels.push("Reaction moment norm".to_string());
        values.push(value);
        field_ids.push(FEA_FIELD_STRUCTURAL_REACTION_MOMENT.to_string());
    }
    if let Some(value) = scalar_field_value(fields, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY) {
        labels.push("Total strain energy".to_string());
        values.push(value);
        field_ids.push(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY.to_string());
    }
    if let Some(value) = scalar_field_value(fields, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM) {
        labels.push("Residual norm".to_string());
        values.push(value);
        field_ids.push(FEA_FIELD_STRUCTURAL_RESIDUAL_NORM.to_string());
    }

    if labels.is_empty() {
        return None;
    }
    let mut chart = BarChart::new(labels, values).ok()?;
    chart.label = Some("Structural summary".to_string());
    chart.color = Vec4::new(0.30, 0.64, 0.58, 1.0);
    let mut figure = Figure::new()
        .with_title("FEA structural result summary")
        .with_labels("Metric", "Value")
        .with_grid(true);
    figure.add_bar_chart(chart);
    Some(AnalysisGeneratedFigure {
        kind: AnalysisGeneratedFigureKind::Summary,
        title: "FEA structural result summary".to_string(),
        field_ids,
        topology_ids: Vec::new(),
        warnings: Vec::new(),
        figure,
    })
}

fn convergence_figures(run: &AnalysisRunResult) -> Vec<AnalysisGeneratedFigure> {
    let mut figures = Vec::new();
    if let Some(modal) = run.modal_results.as_ref() {
        if !modal.eigenvalues_hz.is_empty() {
            let labels = (1..=modal.eigenvalues_hz.len())
                .map(|idx| format!("Mode {idx}"))
                .collect::<Vec<_>>();
            if let Ok(mut chart) = BarChart::new(labels, modal.eigenvalues_hz.clone()) {
                chart.label = Some("Frequency".to_string());
                chart.color = Vec4::new(0.33, 0.66, 0.96, 1.0);
                let mut figure = Figure::new()
                    .with_title("FEA modal frequencies")
                    .with_labels("Mode", "Frequency (Hz)")
                    .with_grid(true);
                figure.add_bar_chart(chart);
                figures.push(AnalysisGeneratedFigure {
                    kind: AnalysisGeneratedFigureKind::Modal,
                    title: "FEA modal frequencies".to_string(),
                    field_ids: modal
                        .mode_shapes
                        .iter()
                        .map(|field| field.field_id.clone())
                        .collect(),
                    topology_ids: topology_ids_for_fields(modal.mode_shapes.iter()),
                    warnings: Vec::new(),
                    figure,
                });
            }
        }
        if !modal.residual_norms.is_empty() {
            figures.push(line_figure(
                AnalysisGeneratedFigureKind::Convergence,
                "FEA modal residuals",
                "Mode",
                "Residual norm",
                vec![(
                    "Residual".to_string(),
                    index_axis(modal.residual_norms.len(), 1.0),
                    modal.residual_norms.clone(),
                    Vec4::new(0.93, 0.48, 0.26, 1.0),
                )],
                Vec::new(),
                true,
            ));
        }
    }

    if let Some(thermal) = run.thermal_results.as_ref() {
        if !thermal.residual_norms.is_empty() {
            figures.push(line_figure(
                AnalysisGeneratedFigureKind::Convergence,
                "FEA thermal convergence",
                "Time (s)",
                "Residual norm",
                vec![(
                    "Thermal residual".to_string(),
                    axis_or_index(&thermal.time_points_s, thermal.residual_norms.len()),
                    thermal.residual_norms.clone(),
                    Vec4::new(0.92, 0.38, 0.31, 1.0),
                )],
                thermal
                    .temperature_snapshots
                    .iter()
                    .map(|field| field.field_id.clone())
                    .collect(),
                true,
            ));
        }
    }

    if let Some(transient) = run.transient_results.as_ref() {
        if !transient.residual_norms.is_empty() {
            figures.push(line_figure(
                AnalysisGeneratedFigureKind::Convergence,
                "FEA transient convergence",
                "Time (s)",
                "Residual norm",
                vec![(
                    "Transient residual".to_string(),
                    axis_or_index(&transient.time_points_s, transient.residual_norms.len()),
                    transient.residual_norms.clone(),
                    Vec4::new(0.35, 0.72, 0.88, 1.0),
                )],
                transient
                    .displacement_snapshots
                    .iter()
                    .map(|field| field.field_id.clone())
                    .collect(),
                true,
            ));
        }
    }

    if let Some(nonlinear) = run.nonlinear_results.as_ref() {
        if !nonlinear.residual_norms.is_empty() {
            figures.push(line_figure(
                AnalysisGeneratedFigureKind::Convergence,
                "FEA nonlinear convergence",
                "Load factor",
                "Norm",
                vec![
                    (
                        "Residual".to_string(),
                        axis_or_index(&nonlinear.load_factors, nonlinear.residual_norms.len()),
                        nonlinear.residual_norms.clone(),
                        Vec4::new(0.92, 0.38, 0.31, 1.0),
                    ),
                    (
                        "Increment".to_string(),
                        axis_or_index(&nonlinear.load_factors, nonlinear.increment_norms.len()),
                        nonlinear.increment_norms.clone(),
                        Vec4::new(0.33, 0.66, 0.96, 1.0),
                    ),
                ],
                nonlinear
                    .displacement_snapshots
                    .iter()
                    .map(|field| field.field_id.clone())
                    .collect(),
                true,
            ));
        }
        if !nonlinear.iteration_counts.is_empty() {
            figures.push(line_figure(
                AnalysisGeneratedFigureKind::Convergence,
                "FEA nonlinear iterations",
                "Load factor",
                "Iterations",
                vec![(
                    "Iterations".to_string(),
                    axis_or_index(&nonlinear.load_factors, nonlinear.iteration_counts.len()),
                    nonlinear
                        .iteration_counts
                        .iter()
                        .map(|value| *value as f64)
                        .collect(),
                    Vec4::new(0.73, 0.62, 0.95, 1.0),
                )],
                Vec::new(),
                false,
            ));
        }
    }

    if let Some(em) = run.electromagnetic_results.as_ref() {
        if !em.sweep_frequency_hz.is_empty() && !em.sweep_peak_flux_density.is_empty() {
            figures.push(line_figure(
                AnalysisGeneratedFigureKind::Electromagnetic,
                "FEA electromagnetic sweep",
                "Frequency (Hz)",
                "Peak flux density",
                vec![(
                    "Peak flux".to_string(),
                    axis_or_index(&em.sweep_frequency_hz, em.sweep_peak_flux_density.len()),
                    em.sweep_peak_flux_density.clone(),
                    Vec4::new(0.28, 0.74, 0.57, 1.0),
                )],
                vec![
                    em.vector_potential_real.field_id.clone(),
                    em.magnetic_flux_density_magnitude.field_id.clone(),
                ],
                false,
            ));
        }
        if !em.sweep_frequency_hz.is_empty() && !em.sweep_solve_quality.is_empty() {
            figures.push(line_figure(
                AnalysisGeneratedFigureKind::Electromagnetic,
                "FEA electromagnetic solve quality",
                "Frequency (Hz)",
                "Solve quality",
                vec![(
                    "Solve quality".to_string(),
                    axis_or_index(&em.sweep_frequency_hz, em.sweep_solve_quality.len()),
                    em.sweep_solve_quality.clone(),
                    Vec4::new(0.92, 0.68, 0.28, 1.0),
                )],
                vec![
                    em.vector_potential_real.field_id.clone(),
                    em.magnetic_flux_density_magnitude.field_id.clone(),
                ],
                false,
            ));
        }
    }

    figures
}

fn comparison_figure(data: &AnalysisResultsCompareData) -> Option<AnalysisGeneratedFigure> {
    let mut labels = Vec::new();
    let mut values = Vec::new();
    push_bar_value(
        &mut labels,
        &mut values,
        "Quality reasons",
        Some(data.quality_reason_count_delta as f64),
    );
    push_bar_value(&mut labels, &mut values, "Solve ms", data.solve_ms_delta);
    push_bar_value(
        &mut labels,
        &mut values,
        "Failed increments",
        data.failed_increment_delta.map(|value| value as f64),
    );
    push_bar_value(
        &mut labels,
        &mut values,
        "Max iterations",
        data.max_iteration_delta.map(|value| value as f64),
    );
    push_bar_value(
        &mut labels,
        &mut values,
        "Spikes",
        data.nonlinear_spike_count_delta.map(|value| value as f64),
    );
    push_bar_value(
        &mut labels,
        &mut values,
        "Stalls",
        data.nonlinear_stall_count_delta.map(|value| value as f64),
    );
    push_bar_value(
        &mut labels,
        &mut values,
        "Publishable changed",
        Some(if data.publishable_changed { 1.0 } else { 0.0 }),
    );
    push_bar_value(
        &mut labels,
        &mut values,
        "Status changed",
        Some(if data.run_status_changed { 1.0 } else { 0.0 }),
    );
    if labels.is_empty() {
        return None;
    }
    let mut chart = BarChart::new(labels, values).ok()?;
    chart.label = Some("Delta".to_string());
    chart.color = Vec4::new(0.77, 0.58, 0.95, 1.0);
    let mut figure = Figure::new()
        .with_title("FEA run comparison")
        .with_labels("Metric", "Candidate minus baseline")
        .with_grid(true);
    figure.add_bar_chart(chart);
    Some(AnalysisGeneratedFigure {
        kind: AnalysisGeneratedFigureKind::Comparison,
        title: "FEA run comparison".to_string(),
        field_ids: Vec::new(),
        topology_ids: Vec::new(),
        warnings: Vec::new(),
        figure,
    })
}

fn trend_figures(data: &AnalysisTrendsData) -> Vec<AnalysisGeneratedFigure> {
    let mut figures = Vec::new();
    let labels = data
        .summaries
        .iter()
        .map(|summary| run_kind_label(summary.run_kind).to_string())
        .collect::<Vec<_>>();
    if labels.is_empty() {
        return figures;
    }

    let solve_values = data
        .summaries
        .iter()
        .map(|summary| summary.median_solve_ms.unwrap_or(0.0))
        .collect::<Vec<_>>();
    if solve_values.iter().any(|value| *value != 0.0) {
        if let Ok(mut chart) = BarChart::new(labels.clone(), solve_values) {
            chart.label = Some("Median solve".to_string());
            chart.color = Vec4::new(0.31, 0.62, 0.91, 1.0);
            let mut figure = Figure::new()
                .with_title("FEA solve time trends")
                .with_labels("Run family", "Median solve (ms)")
                .with_grid(true);
            figure.add_bar_chart(chart);
            figures.push(AnalysisGeneratedFigure {
                kind: AnalysisGeneratedFigureKind::Trend,
                title: "FEA solve time trends".to_string(),
                field_ids: Vec::new(),
                topology_ids: Vec::new(),
                warnings: Vec::new(),
                figure,
            });
        }
    }

    let publishable_values = data
        .summaries
        .iter()
        .map(|summary| summary.publishable_rate * 100.0)
        .collect::<Vec<_>>();
    if let Ok(mut chart) = BarChart::new(labels, publishable_values) {
        chart.label = Some("Publishable rate".to_string());
        chart.color = Vec4::new(0.32, 0.74, 0.56, 1.0);
        let mut figure = Figure::new()
            .with_title("FEA publishable result trends")
            .with_labels("Run family", "Publishable (%)")
            .with_grid(true);
        figure.add_bar_chart(chart);
        figures.push(AnalysisGeneratedFigure {
            kind: AnalysisGeneratedFigureKind::Trend,
            title: "FEA publishable result trends".to_string(),
            field_ids: Vec::new(),
            topology_ids: Vec::new(),
            warnings: Vec::new(),
            figure,
        });
    }

    figures
}

fn base_mesh_figure(
    geometry: &runmat_geometry_core::GeometryAsset,
    render_topology: Option<&AnalysisRenderTopology>,
    title: impl Into<String>,
    options: AnalysisFigureGenerationOptions,
) -> Option<Figure> {
    base_mesh_figure_for_run_source(geometry, render_topology, title, options)
}

fn base_mesh_figure_for_run_source(
    geometry: &runmat_geometry_core::GeometryAsset,
    render_topology: Option<&AnalysisRenderTopology>,
    title: impl Into<String>,
    options: AnalysisFigureGenerationOptions,
) -> Option<Figure> {
    let title = title.into();
    match options.mesh_source {
        AnalysisFigureMeshSource::Auto => {
            if let Ok(mut figure) = geometry_preview_figure(
                geometry,
                title.clone(),
                GeometryPreviewFigureOptions {
                    edge_overlay_triangle_limit: options.edge_overlay_triangle_limit,
                    presentation: crate::geometry::GeometryPreviewPresentation::Cad,
                    ..GeometryPreviewFigureOptions::default()
                },
            ) {
                normalize_geometry_meshes_to_solver_units(&mut figure, geometry.units);
                if let Some(topology) = render_topology {
                    if let Ok(solver) = render_topology_figure(topology, title.clone(), options) {
                        append_mesh_plots(&mut figure, &solver);
                    }
                }
                return Some(figure);
            }
            if let Some(topology) = render_topology {
                if let Ok(figure) = render_topology_figure(topology, title.clone(), options) {
                    return Some(figure);
                }
            }
            geometry_preview_figure(
                geometry,
                title,
                GeometryPreviewFigureOptions {
                    edge_overlay_triangle_limit: options.edge_overlay_triangle_limit,
                    ..GeometryPreviewFigureOptions::default()
                },
            )
            .ok()
        }
        AnalysisFigureMeshSource::Solver => render_topology
            .and_then(|topology| render_topology_figure(topology, title, options).ok()),
        AnalysisFigureMeshSource::Cad => geometry_preview_figure(
            geometry,
            title,
            GeometryPreviewFigureOptions {
                edge_overlay_triangle_limit: options.edge_overlay_triangle_limit,
                presentation: crate::geometry::GeometryPreviewPresentation::Cad,
                ..GeometryPreviewFigureOptions::default()
            },
        )
        .ok(),
    }
}

fn normalize_geometry_meshes_to_solver_units(figure: &mut Figure, units: UnitSystem) {
    let scale = geometry_unit_scale_to_meters(units);
    if (scale - 1.0).abs() <= f32::EPSILON {
        return;
    }
    for index in 0..figure.plots().count() {
        let Some(PlotElement::Mesh(mesh)) = figure.get_plot_mut(index) else {
            continue;
        };
        let vertices = mesh
            .vertices()
            .iter()
            .map(|vertex| *vertex * scale)
            .collect::<Vec<_>>();
        let _ = mesh.set_vertices(vertices);
    }
}

fn geometry_unit_scale_to_meters(units: UnitSystem) -> f32 {
    match units {
        UnitSystem::Unspecified | UnitSystem::Meter => 1.0,
        UnitSystem::Millimeter => 0.001,
        UnitSystem::Inch => 0.0254,
    }
}

fn append_mesh_plots(target: &mut Figure, source: &Figure) {
    for plot in source.plots() {
        if let PlotElement::Mesh(mesh) = plot {
            target.add_mesh_plot((**mesh).clone());
        }
    }
}

fn render_topology_figure(
    topology: &AnalysisRenderTopology,
    title: impl Into<String>,
    options: AnalysisFigureGenerationOptions,
) -> Result<Figure, String> {
    if !render_topology_has_meshes(topology) {
        return Err("solver render topology does not contain renderable meshes".to_string());
    }
    let mut figure = Figure::new()
        .with_title(title)
        .with_labels("X", "Y")
        .with_grid(true)
        .with_axis_equal(true);
    figure.z_label = Some("Z".to_string());

    for mesh in &topology.meshes {
        if mesh.vertices.is_empty() || mesh.triangles.is_empty() {
            continue;
        }
        let vertices = mesh
            .vertices
            .iter()
            .map(|vertex| {
                Ok(Vec3::new(
                    f64_to_f32(vertex[0]).ok_or_else(|| {
                        "solver render topology contains a non-renderable X coordinate".to_string()
                    })?,
                    f64_to_f32(vertex[1]).ok_or_else(|| {
                        "solver render topology contains a non-renderable Y coordinate".to_string()
                    })?,
                    f64_to_f32(vertex[2]).ok_or_else(|| {
                        "solver render topology contains a non-renderable Z coordinate".to_string()
                    })?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut plot = MeshPlot::new(vertices, mesh.triangles.clone())?;
        plot.set_mesh_id(Some(mesh.mesh_id.clone()));
        plot.set_label(Some(format!(
            "{}: {} solver triangles",
            mesh.mesh_id,
            mesh.triangles.len()
        )));
        plot.set_face_color(Vec4::new(0.34, 0.57, 0.82, 1.0));
        plot.set_edge_color(Vec4::new(0.88, 0.93, 0.98, 0.82));
        plot.set_face_alpha(0.94);
        if options.show_solver_mesh_edges
            && mesh.triangles.len() <= options.edge_overlay_triangle_limit
        {
            plot.set_edge_mode(MeshEdgeMode::All);
            plot.set_edge_width(0.28);
        } else {
            plot.set_edge_mode(MeshEdgeMode::None);
            plot.set_edge_width(0.0);
        }
        figure.add_mesh_plot(plot);
    }

    if collect_mesh_counts(&figure).is_empty() {
        Err("solver render topology did not produce any mesh plots".to_string())
    } else {
        Ok(figure)
    }
}

fn collect_mesh_counts(figure: &Figure) -> Vec<MeshCounts> {
    collect_mesh_counts_with_topology(figure, None)
}

fn collect_mesh_counts_with_topology(
    figure: &Figure,
    topology: Option<&AnalysisRenderTopology>,
) -> Vec<MeshCounts> {
    let mut counts = Vec::new();
    let mut mesh_ordinal = 0usize;
    for (plot_index, plot) in figure.plots().enumerate() {
        if let PlotElement::Mesh(mesh) = plot {
            let topology_mesh = topology.and_then(|topology| {
                topology
                    .meshes
                    .iter()
                    .find(|render_mesh| mesh.mesh_id() == Some(render_mesh.mesh_id.as_str()))
                    .or_else(|| {
                        if mesh.mesh_id().is_none() {
                            topology.meshes.get(mesh_ordinal)
                        } else {
                            None
                        }
                    })
            });
            if topology.is_some() && topology_mesh.is_none() {
                continue;
            }
            let triangle_volume_element_indices = topology_mesh
                .filter(|render_mesh| {
                    render_mesh.triangle_volume_element_indices.len() == mesh.triangles().len()
                })
                .map(|render_mesh| render_mesh.triangle_volume_element_indices.clone())
                .unwrap_or_default();
            let vertex_volume_node_indices = topology_mesh
                .filter(|render_mesh| {
                    render_mesh.vertex_volume_node_indices.len() == mesh.vertices().len()
                })
                .map(|render_mesh| render_mesh.vertex_volume_node_indices.clone())
                .unwrap_or_default();
            counts.push(MeshCounts {
                plot_index,
                vertices: mesh.vertices().len(),
                triangles: mesh.triangles().len(),
                vertex_volume_node_indices,
                triangle_volume_element_indices,
            });
            if topology_mesh.is_some() {
                mesh_ordinal += 1;
            }
        }
    }
    counts
}

fn field_topology_mismatch_warning(field: &AnalysisField, meshes: &[MeshCounts]) -> Option<String> {
    let values = host_values(field)?;
    if values.is_empty() || meshes.is_empty() {
        return None;
    }
    let descriptor = AnalysisFieldDescriptor::from_field(field);
    let topology_id = descriptor.topology_id.as_deref()?;
    let actual_entities = field_entity_count(field, &descriptor, values.len());
    if topology_id == "analysis_mesh" {
        match descriptor.location {
            AnalysisFieldLocation::Element => {
                if element_field_maps_to_render_triangles(meshes, actual_entities) {
                    return None;
                }
            }
            AnalysisFieldLocation::Node => {
                if node_field_maps_to_render_vertices(meshes, actual_entities) {
                    return None;
                }
            }
            _ => {}
        }
    }
    let total_vertices = meshes.iter().map(|mesh| mesh.vertices).sum::<usize>();
    let total_triangles = meshes.iter().map(|mesh| mesh.triangles).sum::<usize>();
    let (location, expected_entities) = match descriptor.location {
        AnalysisFieldLocation::Node => ("node", total_vertices),
        AnalysisFieldLocation::Element | AnalysisFieldLocation::BoundaryFace => {
            ("element", total_triangles)
        }
        AnalysisFieldLocation::Edge
        | AnalysisFieldLocation::InterfaceFace
        | AnalysisFieldLocation::Mode
        | AnalysisFieldLocation::Global
        | AnalysisFieldLocation::Unknown => return None,
    };
    if actual_entities == expected_entities && topology_id != "analysis_mesh" {
        return None;
    }
    Some(format!(
        "field '{}' uses topology_id={} location={} element_kind={} value_count={} render_vertex_count={} render_triangle_count={}; cannot map field to the current render mesh",
        field.field_id,
        topology_id,
        location,
        descriptor.element_kind.as_deref().unwrap_or("none"),
        actual_entities,
        total_vertices,
        total_triangles
    ))
}

fn element_field_maps_to_render_triangles(meshes: &[MeshCounts], element_count: usize) -> bool {
    if element_count == 0 || meshes.is_empty() {
        return false;
    }
    meshes.iter().all(|mesh| {
        mesh.triangle_volume_element_indices.len() == mesh.triangles
            && mesh
                .triangle_volume_element_indices
                .iter()
                .all(|index| index.is_some_and(|index| index < element_count))
    })
}

fn node_field_maps_to_render_vertices(meshes: &[MeshCounts], node_count: usize) -> bool {
    if node_count == 0 || meshes.is_empty() {
        return false;
    }
    meshes.iter().all(|mesh| {
        mesh.vertex_volume_node_indices.len() == mesh.vertices
            && mesh
                .vertex_volume_node_indices
                .iter()
                .all(|index| index.is_some_and(|index| index < node_count))
    })
}

fn field_entity_count(
    field: &AnalysisField,
    descriptor: &AnalysisFieldDescriptor,
    value_count: usize,
) -> usize {
    if matches!(
        descriptor.location,
        AnalysisFieldLocation::Global | AnalysisFieldLocation::Mode
    ) {
        return value_count;
    }
    if let Some(first_dim) = field.shape.first().copied() {
        if field.shape.len() > 1 || descriptor.component_count.is_some() {
            return first_dim;
        }
    }
    value_count
}

fn scalar_overlay(
    field: &AnalysisField,
    meshes: &[MeshCounts],
    options: AnalysisFigureGenerationOptions,
) -> Option<ScalarOverlay> {
    let values = host_values(field)?;
    let total_vertices = meshes.iter().map(|mesh| mesh.vertices).sum::<usize>();
    let total_triangles = meshes.iter().map(|mesh| mesh.triangles).sum::<usize>();
    if values.len() == total_vertices {
        return scalar_overlay_from_values(
            field,
            MeshFieldLocation::Vertex,
            meshes.iter().map(|mesh| mesh.vertices),
            options,
        );
    }
    if let Some(overlay) = scalar_overlay_from_node_values(field, meshes, options) {
        return Some(overlay);
    }
    if values.len() == total_triangles {
        return scalar_overlay_from_values(
            field,
            MeshFieldLocation::Triangle,
            meshes.iter().map(|mesh| mesh.triangles),
            options,
        );
    }
    if let Some(overlay) = scalar_overlay_from_element_values(field, meshes, options) {
        return Some(overlay);
    }
    if let Some(vectors) = vectors_for_count(field, total_vertices) {
        if total_vertices <= options.max_overlay_values {
            let magnitudes = vectors
                .iter()
                .map(|vector| vector.length())
                .collect::<Vec<_>>();
            return Some(ScalarOverlay {
                field_id: format!("{}.magnitude", field.field_id),
                label: format!("{} magnitude", field.field_id),
                location: MeshFieldLocation::Vertex,
                chunks: split_f32(&magnitudes, meshes.iter().map(|mesh| mesh.vertices))?,
            });
        }
    }
    if let Some(vectors) = vectors_for_count(field, total_triangles) {
        if total_triangles <= options.max_overlay_values {
            let magnitudes = vectors
                .iter()
                .map(|vector| vector.length())
                .collect::<Vec<_>>();
            return Some(ScalarOverlay {
                field_id: format!("{}.magnitude", field.field_id),
                label: format!("{} magnitude", field.field_id),
                location: MeshFieldLocation::Triangle,
                chunks: split_f32(&magnitudes, meshes.iter().map(|mesh| mesh.triangles))?,
            });
        }
    }
    None
}

fn scalar_overlay_from_element_values(
    field: &AnalysisField,
    meshes: &[MeshCounts],
    options: AnalysisFigureGenerationOptions,
) -> Option<ScalarOverlay> {
    let values = host_values(field)?;
    let descriptor = AnalysisFieldDescriptor::from_field(field);
    if descriptor.location != AnalysisFieldLocation::Element
        || descriptor.topology_id.as_deref() != Some("analysis_mesh")
        || field_entity_count(field, &descriptor, values.len()) != values.len()
    {
        return None;
    }
    let total_triangles = meshes.iter().map(|mesh| mesh.triangles).sum::<usize>();
    if total_triangles > options.max_overlay_values {
        return None;
    }
    let values = values
        .iter()
        .copied()
        .map(f64_to_f32)
        .collect::<Option<Vec<_>>>()?;
    let mut chunks = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        if mesh.triangle_volume_element_indices.len() != mesh.triangles {
            return None;
        }
        let mut chunk = Vec::with_capacity(mesh.triangles);
        for element_index in &mesh.triangle_volume_element_indices {
            let element_index = (*element_index)?;
            chunk.push(*values.get(element_index)?);
        }
        chunks.push(chunk);
    }
    Some(ScalarOverlay {
        field_id: field.field_id.clone(),
        label: format!("{} boundary projection", field.field_id),
        location: MeshFieldLocation::Triangle,
        chunks,
    })
}

fn scalar_overlay_from_node_values(
    field: &AnalysisField,
    meshes: &[MeshCounts],
    options: AnalysisFigureGenerationOptions,
) -> Option<ScalarOverlay> {
    let values = host_values(field)?;
    let descriptor = AnalysisFieldDescriptor::from_field(field);
    if descriptor.location != AnalysisFieldLocation::Node
        || descriptor.topology_id.as_deref() != Some("analysis_mesh")
        || field_entity_count(field, &descriptor, values.len()) != values.len()
    {
        return None;
    }
    let total_vertices = meshes.iter().map(|mesh| mesh.vertices).sum::<usize>();
    if total_vertices > options.max_overlay_values {
        return None;
    }
    let values = values
        .iter()
        .copied()
        .map(f64_to_f32)
        .collect::<Option<Vec<_>>>()?;
    let mut chunks = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        if mesh.vertex_volume_node_indices.len() != mesh.vertices {
            return None;
        }
        let mut chunk = Vec::with_capacity(mesh.vertices);
        for node_index in &mesh.vertex_volume_node_indices {
            let node_index = (*node_index)?;
            chunk.push(*values.get(node_index)?);
        }
        chunks.push(chunk);
    }
    Some(ScalarOverlay {
        field_id: field.field_id.clone(),
        label: format!("{} boundary projection", field.field_id),
        location: MeshFieldLocation::Vertex,
        chunks,
    })
}

fn scalar_overlay_from_values<I>(
    field: &AnalysisField,
    location: MeshFieldLocation,
    chunk_lengths: I,
    options: AnalysisFigureGenerationOptions,
) -> Option<ScalarOverlay>
where
    I: Iterator<Item = usize>,
{
    let values = host_values(field)?;
    if values.len() > options.max_overlay_values {
        return None;
    }
    let values = values
        .iter()
        .copied()
        .map(f64_to_f32)
        .collect::<Option<Vec<_>>>()?;
    Some(ScalarOverlay {
        field_id: field.field_id.clone(),
        label: field.field_id.clone(),
        location,
        chunks: split_f32(&values, chunk_lengths)?,
    })
}

fn vector_overlay(
    field: &AnalysisField,
    meshes: &[MeshCounts],
    options: AnalysisFigureGenerationOptions,
) -> Option<VectorOverlay> {
    let total_vertices = meshes.iter().map(|mesh| mesh.vertices).sum::<usize>();
    if let Some(vectors) = vectors_for_count(field, total_vertices) {
        if total_vertices <= options.max_overlay_values {
            let stride = glyph_stride(total_vertices, options.max_vector_glyphs);
            return Some(VectorOverlay {
                field_id: field.field_id.clone(),
                label: field.field_id.clone(),
                location: MeshFieldLocation::Vertex,
                chunks: split_vec3(&vectors, meshes.iter().map(|mesh| mesh.vertices))?,
                stride,
            });
        }
    }
    if let Some(overlay) = vector_overlay_from_node_values(field, meshes, options) {
        return Some(overlay);
    }

    let total_triangles = meshes.iter().map(|mesh| mesh.triangles).sum::<usize>();
    if let Some(vectors) = vectors_for_count(field, total_triangles) {
        if total_triangles <= options.max_overlay_values {
            let stride = glyph_stride(total_triangles, options.max_vector_glyphs);
            return Some(VectorOverlay {
                field_id: field.field_id.clone(),
                label: field.field_id.clone(),
                location: MeshFieldLocation::Triangle,
                chunks: split_vec3(&vectors, meshes.iter().map(|mesh| mesh.triangles))?,
                stride,
            });
        }
    }
    None
}

fn vector_overlay_from_node_values(
    field: &AnalysisField,
    meshes: &[MeshCounts],
    options: AnalysisFigureGenerationOptions,
) -> Option<VectorOverlay> {
    let descriptor = AnalysisFieldDescriptor::from_field(field);
    if descriptor.location != AnalysisFieldLocation::Node
        || descriptor.topology_id.as_deref() != Some("analysis_mesh")
    {
        return None;
    }
    let entity_count = field.shape.first().copied()?;
    let vectors = vectors_for_count(field, entity_count)?;
    let total_vertices = meshes.iter().map(|mesh| mesh.vertices).sum::<usize>();
    if total_vertices > options.max_overlay_values {
        return None;
    }
    let mut chunks = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        if mesh.vertex_volume_node_indices.len() != mesh.vertices {
            return None;
        }
        let mut chunk = Vec::with_capacity(mesh.vertices);
        for node_index in &mesh.vertex_volume_node_indices {
            let node_index = (*node_index)?;
            chunk.push(*vectors.get(node_index)?);
        }
        chunks.push(chunk);
    }
    Some(VectorOverlay {
        field_id: field.field_id.clone(),
        label: format!("{} boundary projection", field.field_id),
        location: MeshFieldLocation::Vertex,
        chunks,
        stride: glyph_stride(total_vertices, options.max_vector_glyphs),
    })
}

fn deformation_overlay(
    field: &AnalysisField,
    meshes: &[MeshCounts],
    figure: &Figure,
    options: AnalysisFigureGenerationOptions,
) -> Option<DeformationOverlay> {
    let total_vertices = meshes.iter().map(|mesh| mesh.vertices).sum::<usize>();
    if total_vertices > options.max_overlay_values {
        return None;
    }
    if let Some(vectors) = vectors_for_count(field, total_vertices) {
        let scale = deformation_scale(&vectors, figure);
        return Some(DeformationOverlay {
            field_id: field.field_id.clone(),
            label: field.field_id.clone(),
            chunks: split_vec3(&vectors, meshes.iter().map(|mesh| mesh.vertices))?,
            scale,
        });
    }
    let descriptor = AnalysisFieldDescriptor::from_field(field);
    if descriptor.location != AnalysisFieldLocation::Node
        || descriptor.topology_id.as_deref() != Some("analysis_mesh")
    {
        return None;
    }
    let entity_count = field.shape.first().copied()?;
    let vectors = vectors_for_count(field, entity_count)?;
    let mut projected = Vec::with_capacity(total_vertices);
    let mut chunks = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        if mesh.vertex_volume_node_indices.len() != mesh.vertices {
            return None;
        }
        let mut chunk = Vec::with_capacity(mesh.vertices);
        for node_index in &mesh.vertex_volume_node_indices {
            let node_index = (*node_index)?;
            let vector = *vectors.get(node_index)?;
            projected.push(vector);
            chunk.push(vector);
        }
        chunks.push(chunk);
    }
    let scale = deformation_scale(&projected, figure);
    Some(DeformationOverlay {
        field_id: field.field_id.clone(),
        label: format!("{} boundary projection", field.field_id),
        chunks,
        scale,
    })
}

fn apply_scalar_overlay(
    figure: &mut Figure,
    overlay: &ScalarOverlay,
    meshes: &[MeshCounts],
    warnings: &mut Vec<String>,
) {
    for (mesh, values) in meshes.iter().zip(&overlay.chunks) {
        let Some(PlotElement::Mesh(plot)) = figure.get_plot_mut(mesh.plot_index) else {
            continue;
        };
        let mut field =
            MeshScalarField::new(overlay.field_id.clone(), overlay.location, values.clone());
        field.label = Some(overlay.label.clone());
        field.alpha = 0.92;
        if let Some(limits) = finite_limits(values) {
            field.color_limits = Some(limits);
        }
        if let Err(err) = plot.set_scalar_field(Some(field)) {
            warnings.push(format!(
                "failed to attach scalar field '{}' to mesh: {err}",
                overlay.field_id
            ));
        }
    }
}

fn apply_vector_overlay(
    figure: &mut Figure,
    overlay: &VectorOverlay,
    meshes: &[MeshCounts],
    warnings: &mut Vec<String>,
) {
    for (mesh, vectors) in meshes.iter().zip(&overlay.chunks) {
        let Some(PlotElement::Mesh(plot)) = figure.get_plot_mut(mesh.plot_index) else {
            continue;
        };
        let mut field =
            MeshVectorField::new(overlay.field_id.clone(), overlay.location, vectors.clone());
        field.label = Some(overlay.label.clone());
        field.stride = overlay.stride.max(1);
        field.scale = vector_scale(vectors);
        if let Err(err) = plot.set_vector_field(Some(field)) {
            warnings.push(format!(
                "failed to attach vector field '{}' to mesh: {err}",
                overlay.field_id
            ));
        }
    }
}

fn apply_deformation_to_existing_meshes(
    figure: &mut Figure,
    overlay: &DeformationOverlay,
    meshes: &[MeshCounts],
    warnings: &mut Vec<String>,
) {
    for (mesh, displacements) in meshes.iter().zip(&overlay.chunks) {
        let Some(PlotElement::Mesh(plot)) = figure.get_plot_mut(mesh.plot_index) else {
            continue;
        };
        let mut deformation = MeshDeformation::new(overlay.field_id.clone(), displacements.clone());
        deformation.label = Some(overlay.label.clone());
        deformation.scale = overlay.scale;
        if let Err(err) = plot.set_deformation(Some(deformation)) {
            warnings.push(format!(
                "failed to attach deformation field '{}' to mesh: {err}",
                overlay.field_id
            ));
        }
    }
}

fn append_deformed_mesh_overlay(
    figure: &mut Figure,
    overlay: &DeformationOverlay,
    meshes: &[MeshCounts],
    warnings: &mut Vec<String>,
) {
    let clones = meshes
        .iter()
        .filter_map(|mesh| match figure.plots().nth(mesh.plot_index) {
            Some(PlotElement::Mesh(plot)) => Some(plot.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    for mesh in meshes {
        if let Some(PlotElement::Mesh(plot)) = figure.get_plot_mut(mesh.plot_index) {
            plot.set_face_alpha(0.14);
            plot.set_edge_alpha(0.72);
            plot.set_edge_width(plot.edge_width().max(0.28));
        }
    }

    for (mut plot, displacements) in clones.into_iter().zip(&overlay.chunks) {
        plot.set_face_alpha(0.72);
        plot.set_edge_alpha(0.45);
        plot.set_face_color(Vec4::new(0.33, 0.66, 0.96, 1.0));
        plot.set_edge_color(Vec4::new(0.90, 0.95, 1.0, 0.55));
        let mut deformation = MeshDeformation::new(overlay.field_id.clone(), displacements.clone());
        deformation.label = Some(overlay.label.clone());
        deformation.scale = overlay.scale;
        if let Err(err) = plot.set_deformation(Some(deformation)) {
            warnings.push(format!(
                "failed to attach deformation field '{}' to mesh: {err}",
                overlay.field_id
            ));
            continue;
        }
        figure.add_mesh_plot(*plot);
    }
}

fn line_figure(
    kind: AnalysisGeneratedFigureKind,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: Vec<(String, Vec<f64>, Vec<f64>, Vec4)>,
    field_ids: Vec<String>,
    y_log: bool,
) -> AnalysisGeneratedFigure {
    let mut figure = Figure::new()
        .with_title(title)
        .with_labels(x_label, y_label)
        .with_grid(true);
    if y_log {
        figure = figure.with_ylog(true);
    }
    let mut warnings = Vec::new();
    for (label, x, y, color) in series {
        if x.is_empty() || y.is_empty() || x.len() != y.len() {
            continue;
        }
        match LinePlot::new(x, y) {
            Ok(mut line) => {
                line.label = Some(label);
                line.color = color;
                line.line_width = 1.8;
                figure.add_line_plot(line);
            }
            Err(err) => warnings.push(format!("failed to create line series: {err}")),
        }
    }
    AnalysisGeneratedFigure {
        kind,
        title: title.to_string(),
        topology_ids: topology_ids_for_field_ids(field_ids.iter().map(String::as_str)),
        field_ids,
        warnings,
        figure,
    }
}

fn warning_line_figure(
    kind: AnalysisGeneratedFigureKind,
    title: &str,
    warning: String,
) -> AnalysisGeneratedFigure {
    let mut figure = Figure::new()
        .with_title(title)
        .with_labels("Step", "Value")
        .with_grid(true);
    if let Ok(mut line) = LinePlot::new(vec![0.0, 1.0], vec![0.0, 0.0]) {
        line.label = Some("No renderable mesh".to_string());
        figure.add_line_plot(line);
    }
    AnalysisGeneratedFigure {
        kind,
        title: title.to_string(),
        field_ids: Vec::new(),
        topology_ids: Vec::new(),
        warnings: vec![warning],
        figure,
    }
}

fn topology_ids_for_fields<'a>(fields: impl IntoIterator<Item = &'a AnalysisField>) -> Vec<String> {
    let mut ids = Vec::new();
    for field in fields {
        if let Some(topology_id) = AnalysisFieldDescriptor::from_field(field).topology_id {
            if !ids.iter().any(|existing| existing == &topology_id) {
                ids.push(topology_id);
            }
        }
    }
    ids
}

fn topology_ids_for_field_ids<'a>(field_ids: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let fields = field_ids
        .into_iter()
        .map(|field_id| AnalysisField::host_f64(field_id, vec![1], vec![0.0]))
        .collect::<Vec<_>>();
    topology_ids_for_fields(fields.iter())
}

fn previous_run_of_kind(current: &AnalysisRunResult) -> Result<Option<AnalysisRunResult>, String> {
    let current_kind = run_kind(current);
    let mut candidates = storage::list_run_results()?
        .into_iter()
        .filter(|run| run.run_id != current.run_id && run_kind(run) == current_kind)
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.run_id.cmp(&a.run_id));
    Ok(candidates.into_iter().next())
}

fn host_values(field: &AnalysisField) -> Option<&[f64]> {
    match &field.values {
        AnalysisFieldValues::HostF64(values) => Some(values.as_slice()),
        AnalysisFieldValues::DeviceRef(_) => None,
    }
}

fn field_by_id<'a>(fields: &'a [AnalysisField], field_id: &str) -> Option<&'a AnalysisField> {
    fields.iter().find(|field| field.field_id == field_id)
}

fn scalar_field_value(fields: &[AnalysisField], field_id: &str) -> Option<f64> {
    let field = field_by_id(fields, field_id)?;
    let values = host_values(field)?;
    if values.len() != 1 {
        return None;
    }
    values.first().copied().filter(|value| value.is_finite())
}

fn vector_field_total_magnitude(fields: &[AnalysisField], field_id: &str) -> Option<f64> {
    let field = field_by_id(fields, field_id)?;
    let values = host_values(field)?;
    if values.is_empty() || !values.len().is_multiple_of(3) {
        return None;
    }
    let mut total = [0.0_f64; 3];
    for chunk in values.chunks_exact(3) {
        total[0] += chunk[0];
        total[1] += chunk[1];
        total[2] += chunk[2];
    }
    let magnitude = (total[0] * total[0] + total[1] * total[1] + total[2] * total[2]).sqrt();
    magnitude.is_finite().then_some(magnitude)
}

fn vectors_for_count(field: &AnalysisField, count: usize) -> Option<Vec<Vec3>> {
    if count == 0 {
        return None;
    }
    let values = host_values(field)?;
    if values.len() == count * 3 {
        return values
            .chunks_exact(3)
            .map(|chunk| {
                Some(Vec3::new(
                    f64_to_f32(chunk[0])?,
                    f64_to_f32(chunk[1])?,
                    f64_to_f32(chunk[2])?,
                ))
            })
            .collect::<Option<Vec<_>>>();
    }
    match field.shape.as_slice() {
        [rows, cols] if *rows == count && *cols == 2 && values.len() == count * 2 => values
            .chunks_exact(2)
            .map(|chunk| Some(Vec3::new(f64_to_f32(chunk[0])?, f64_to_f32(chunk[1])?, 0.0)))
            .collect::<Option<Vec<_>>>(),
        [rows, cols] if *rows == 2 && *cols == count && values.len() == count * 2 => {
            let mut vectors = Vec::with_capacity(count);
            for idx in 0..count {
                vectors.push(Vec3::new(
                    f64_to_f32(values[idx])?,
                    f64_to_f32(values[count + idx])?,
                    0.0,
                ));
            }
            Some(vectors)
        }
        [rows, cols] if *rows == 3 && *cols == count && values.len() == count * 3 => {
            let mut vectors = Vec::with_capacity(count);
            for idx in 0..count {
                vectors.push(Vec3::new(
                    f64_to_f32(values[idx])?,
                    f64_to_f32(values[count + idx])?,
                    f64_to_f32(values[count * 2 + idx])?,
                ));
            }
            Some(vectors)
        }
        _ => None,
    }
}

fn split_f32<I>(values: &[f32], lengths: I) -> Option<Vec<Vec<f32>>>
where
    I: Iterator<Item = usize>,
{
    let mut offset = 0usize;
    let mut chunks = Vec::new();
    for len in lengths {
        let end = offset.checked_add(len)?;
        chunks.push(values.get(offset..end)?.to_vec());
        offset = end;
    }
    if offset == values.len() {
        Some(chunks)
    } else {
        None
    }
}

fn split_vec3<I>(values: &[Vec3], lengths: I) -> Option<Vec<Vec<Vec3>>>
where
    I: Iterator<Item = usize>,
{
    let mut offset = 0usize;
    let mut chunks = Vec::new();
    for len in lengths {
        let end = offset.checked_add(len)?;
        chunks.push(values.get(offset..end)?.to_vec());
        offset = end;
    }
    if offset == values.len() {
        Some(chunks)
    } else {
        None
    }
}

fn finite_limits(values: &[f32]) -> Option<[f32; 2]> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        min = min.min(value);
        max = max.max(value);
    }
    if min.is_finite() && max.is_finite() {
        Some([min, max])
    } else {
        None
    }
}

fn f64_to_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value > f32::MAX as f64 || value < f32::MIN as f64 {
        None
    } else {
        Some(value as f32)
    }
}

fn is_deformation_candidate(field_id: &str) -> bool {
    let normalized = field_id.to_ascii_lowercase();
    normalized.contains("displacement") || normalized.contains("mode_shape")
}

fn deformation_scale(vectors: &[Vec3], figure: &Figure) -> f32 {
    let max_displacement = vectors
        .iter()
        .map(|vector| vector.length())
        .fold(0.0_f32, f32::max);
    if !max_displacement.is_finite() || max_displacement <= f32::EPSILON {
        return 1.0;
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for plot in figure.plots() {
        if let PlotElement::Mesh(mesh) = plot {
            for vertex in mesh.vertices() {
                min = min.min(*vertex);
                max = max.max(*vertex);
            }
        }
    }
    let diagonal = (max - min).length();
    if !diagonal.is_finite() || diagonal <= f32::EPSILON {
        return 1.0;
    }
    ((diagonal * 0.08) / max_displacement).clamp(0.1, 1.0e6)
}

fn vector_scale(vectors: &[Vec3]) -> f32 {
    let max_vector = vectors
        .iter()
        .map(|vector| vector.length())
        .fold(0.0_f32, f32::max);
    if max_vector.is_finite() && max_vector > f32::EPSILON {
        (1.0 / max_vector).clamp(0.001, 1.0e6)
    } else {
        1.0
    }
}

fn glyph_stride(count: usize, max_glyphs: usize) -> usize {
    if max_glyphs == 0 || count <= max_glyphs {
        1
    } else {
        count.div_ceil(max_glyphs)
    }
}

fn axis_or_index(axis: &[f64], count: usize) -> Vec<f64> {
    if axis.len() >= count {
        axis.iter().copied().take(count).collect()
    } else {
        index_axis(count, 1.0)
    }
}

fn index_axis(count: usize, start: f64) -> Vec<f64> {
    (0..count).map(|idx| start + idx as f64).collect()
}

fn push_bar_value(
    labels: &mut Vec<String>,
    values: &mut Vec<f64>,
    label: &str,
    value: Option<f64>,
) {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        labels.push(label.to_string());
        values.push(value);
    }
}

fn geometry_surface_mesh_bytes(geometry: &runmat_geometry_core::GeometryAsset) -> usize {
    geometry
        .surface_meshes
        .iter()
        .map(|mesh| {
            mesh.vertices.len() * 3 * std::mem::size_of::<f32>()
                + mesh.triangles.len() * 3 * std::mem::size_of::<u32>()
        })
        .sum()
}

fn render_topology_has_meshes(topology: &AnalysisRenderTopology) -> bool {
    topology
        .meshes
        .iter()
        .any(|mesh| !mesh.vertices.is_empty() && !mesh.triangles.is_empty())
}

fn render_topology_mesh_bytes(topology: &AnalysisRenderTopology) -> usize {
    topology
        .meshes
        .iter()
        .map(|mesh| {
            mesh.vertices.len() * 3 * std::mem::size_of::<f32>()
                + mesh.triangles.len() * 3 * std::mem::size_of::<u32>()
        })
        .sum()
}

fn run_kind_label(kind: AnalysisRunKind) -> &'static str {
    match kind {
        AnalysisRunKind::LinearStatic => "Linear static",
        AnalysisRunKind::Modal => "Modal",
        AnalysisRunKind::Acoustic => "Acoustic",
        AnalysisRunKind::Thermal => "Thermal",
        AnalysisRunKind::Transient => "Transient",
        AnalysisRunKind::Cfd => "CFD",
        AnalysisRunKind::Cht => "CHT",
        AnalysisRunKind::Fsi => "FSI",
        AnalysisRunKind::Nonlinear => "Nonlinear",
        AnalysisRunKind::Electromagnetic => "Electromagnetic",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_geometry_asset() -> runmat_geometry_core::GeometryAsset {
        simple_geometry_asset_with_units(runmat_geometry_core::UnitSystem::Meter)
    }

    fn simple_geometry_asset_with_units(
        units: runmat_geometry_core::UnitSystem,
    ) -> runmat_geometry_core::GeometryAsset {
        runmat_geometry_core::GeometryAsset {
            geometry_id: "geometry".to_string(),
            source: runmat_geometry_core::GeometrySource {
                path: "/tmp/generic.step".to_string(),
                sha256: "hash".to_string(),
                importer_version: "test/v1".to_string(),
            },
            source_geometry: runmat_geometry_core::SourceGeometry {
                kind: runmat_geometry_core::SourceGeometryKind::Mesh,
                assembly: None,
                material_evidence: Vec::new(),
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: runmat_geometry_core::TessellationProfile::default(),
            units,
            revision: 1,
            meshes: vec![runmat_geometry_core::MeshDescriptor {
                mesh_id: "cad_surface".to_string(),
                kind: runmat_geometry_core::MeshKind::Surface,
                vertex_count: 3,
                element_count: 1,
            }],
            surface_meshes: vec![runmat_geometry_core::SurfaceMesh::new(
                "cad_surface",
                vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                vec![[0, 1, 2]],
            )],
            regions: Vec::new(),
            region_entity_mappings: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn simple_render_topology() -> AnalysisRenderTopology {
        AnalysisRenderTopology {
            schema_version: "analysis_render_topology/v1".to_string(),
            source: crate::analysis::contracts::AnalysisRenderTopologySource::SolverPrep,
            meshes: vec![crate::analysis::contracts::AnalysisRenderMesh {
                mesh_id: "analysis_mesh".to_string(),
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: vec![[0, 1, 2]],
                vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
                triangle_volume_element_indices: Vec::new(),
            }],
        }
    }

    fn first_mesh_plot(figure: &Figure) -> &MeshPlot {
        figure
            .plots()
            .find_map(|plot| match plot {
                PlotElement::Mesh(mesh) => Some(mesh),
                _ => None,
            })
            .expect("figure should include a mesh plot")
    }

    #[test]
    fn render_topology_edges_are_disabled_by_default() {
        let figure = render_topology_figure(
            &simple_render_topology(),
            "solver mesh",
            AnalysisFigureGenerationOptions::default(),
        )
        .expect("solver topology should render");

        let plot = first_mesh_plot(&figure);
        assert_eq!(plot.edge_mode(), MeshEdgeMode::None);
        assert_eq!(plot.edge_width(), 0.0);
    }

    #[test]
    fn render_topology_edges_can_be_enabled() {
        let figure = render_topology_figure(
            &simple_render_topology(),
            "solver mesh",
            AnalysisFigureGenerationOptions {
                show_solver_mesh_edges: true,
                ..AnalysisFigureGenerationOptions::default()
            },
        )
        .expect("solver topology should render");

        let plot = first_mesh_plot(&figure);
        assert_eq!(plot.edge_mode(), MeshEdgeMode::All);
        assert!(plot.edge_width() > 0.0);
    }

    #[test]
    fn base_mesh_figure_can_force_solver_render_topology() {
        let geometry = simple_geometry_asset();
        let topology = simple_render_topology();
        let figure = base_mesh_figure_for_run_source(
            &geometry,
            Some(&topology),
            "solver source",
            AnalysisFigureGenerationOptions {
                mesh_source: AnalysisFigureMeshSource::Solver,
                ..AnalysisFigureGenerationOptions::default()
            },
        )
        .expect("solver source should render from topology");

        let plot = first_mesh_plot(&figure);
        assert_eq!(plot.mesh_id(), Some("analysis_mesh"));
    }

    #[test]
    fn base_mesh_figure_can_force_cad_geometry_source() {
        let geometry = simple_geometry_asset();
        let topology = simple_render_topology();
        let figure = base_mesh_figure_for_run_source(
            &geometry,
            Some(&topology),
            "cad source",
            AnalysisFigureGenerationOptions {
                mesh_source: AnalysisFigureMeshSource::Cad,
                ..AnalysisFigureGenerationOptions::default()
            },
        )
        .expect("CAD source should render from geometry");

        let plot = first_mesh_plot(&figure);
        assert_eq!(plot.mesh_id(), Some("cad_surface"));
    }

    #[test]
    fn base_mesh_figure_auto_layers_cad_context_and_solver_topology() {
        let geometry = simple_geometry_asset();
        let topology = simple_render_topology();
        let figure = base_mesh_figure_for_run_source(
            &geometry,
            Some(&topology),
            "layered result",
            AnalysisFigureGenerationOptions::default(),
        )
        .expect("auto source should render layered CAD and solver topology");

        let mesh_ids = figure
            .plots()
            .filter_map(|plot| match plot {
                PlotElement::Mesh(mesh) => mesh.mesh_id(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(mesh_ids, vec!["cad_surface", "analysis_mesh"]);
    }

    #[test]
    fn auto_layered_result_scales_geometry_context_to_solver_meters() {
        let mut geometry =
            simple_geometry_asset_with_units(runmat_geometry_core::UnitSystem::Millimeter);
        geometry.surface_meshes[0].vertices =
            vec![[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]];
        let topology = simple_render_topology();

        let figure = base_mesh_figure_for_run_source(
            &geometry,
            Some(&topology),
            "layered result",
            AnalysisFigureGenerationOptions::default(),
        )
        .expect("auto source should render layered CAD and solver topology");

        let cad = figure
            .plots()
            .find_map(|plot| match plot {
                PlotElement::Mesh(mesh) if mesh.mesh_id() == Some("cad_surface") => Some(mesh),
                _ => None,
            })
            .expect("CAD context mesh should be present");
        assert_eq!(cad.vertices()[1], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(cad.vertices()[2], Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn topology_mesh_counts_ignore_cad_context_meshes() {
        let geometry = simple_geometry_asset();
        let topology = simple_render_topology();
        let figure = base_mesh_figure_for_run_source(
            &geometry,
            Some(&topology),
            "layered result",
            AnalysisFigureGenerationOptions::default(),
        )
        .expect("auto source should render layered CAD and solver topology");

        let counts = collect_mesh_counts_with_topology(&figure, Some(&topology));

        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0].vertices, topology.meshes[0].vertices.len());
        assert_eq!(counts[0].triangles, topology.meshes[0].triangles.len());
    }

    #[test]
    fn field_topology_warning_reports_solver_mesh_mismatch() {
        let field = AnalysisField::host_f64("structural.von_mises", vec![1], vec![42.0]);
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 4,
            triangles: 12,
            vertex_volume_node_indices: Vec::new(),
            triangle_volume_element_indices: Vec::new(),
        }];

        let warning = field_topology_mismatch_warning(&field, &meshes)
            .expect("mismatched Tet4 element field should produce a warning");

        assert!(warning.contains("structural.von_mises"));
        assert!(warning.contains("topology_id=analysis_mesh"));
        assert!(warning.contains("element_kind=tet4"));
        assert!(warning.contains("value_count=1"));
        assert!(warning.contains("render_triangle_count=12"));
    }

    #[test]
    fn field_topology_warning_accepts_mapped_solver_element_fields() {
        let field = AnalysisField::host_f64("structural.von_mises", vec![2], vec![10.0, 42.0]);
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 5,
            triangles: 3,
            vertex_volume_node_indices: Vec::new(),
            triangle_volume_element_indices: vec![Some(0), Some(1), Some(1)],
        }];

        assert_eq!(field_topology_mismatch_warning(&field, &meshes), None);
    }

    #[test]
    fn field_topology_warning_accepts_mapped_solver_node_fields() {
        let field = AnalysisField::host_f64(
            "structural.displacement",
            vec![3, 3],
            vec![
                1.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 3.0,
            ],
        );
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 2,
            triangles: 1,
            vertex_volume_node_indices: vec![Some(0), Some(2)],
            triangle_volume_element_indices: Vec::new(),
        }];

        assert_eq!(field_topology_mismatch_warning(&field, &meshes), None);
    }

    #[test]
    fn field_topology_warning_rejects_unmapped_solver_vertices() {
        let field = AnalysisField::host_f64(
            "structural.displacement",
            vec![3, 3],
            vec![
                1.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 3.0,
            ],
        );
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 3,
            triangles: 1,
            vertex_volume_node_indices: vec![Some(0), None, Some(2)],
            triangle_volume_element_indices: Vec::new(),
        }];

        let warning = field_topology_mismatch_warning(&field, &meshes)
            .expect("unmapped solver vertex should warn");

        assert!(warning.contains("structural.displacement"));
        assert!(warning.contains("render_vertex_count=3"));
    }

    #[test]
    fn field_topology_warning_rejects_stale_solver_vertex_indices() {
        let field = AnalysisField::host_f64(
            "structural.displacement",
            vec![3, 3],
            vec![
                1.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 3.0,
            ],
        );
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 3,
            triangles: 1,
            vertex_volume_node_indices: vec![Some(0), Some(3), Some(2)],
            triangle_volume_element_indices: Vec::new(),
        }];

        let warning = field_topology_mismatch_warning(&field, &meshes)
            .expect("stale solver vertex index should warn");

        assert!(warning.contains("structural.displacement"));
        assert!(warning.contains("render_vertex_count=3"));
    }

    #[test]
    fn field_topology_warning_rejects_unmapped_solver_triangles() {
        let field = AnalysisField::host_f64("structural.von_mises", vec![2], vec![10.0, 42.0]);
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 5,
            triangles: 3,
            vertex_volume_node_indices: Vec::new(),
            triangle_volume_element_indices: vec![Some(0), None, Some(1)],
        }];

        let warning = field_topology_mismatch_warning(&field, &meshes)
            .expect("unmapped solver triangle should warn");

        assert!(warning.contains("structural.von_mises"));
        assert!(warning.contains("render_triangle_count=3"));
    }

    #[test]
    fn scalar_overlay_projects_element_values_to_boundary_triangles() {
        let field = AnalysisField::host_f64("structural.von_mises", vec![2], vec![10.0, 42.0]);
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 5,
            triangles: 3,
            vertex_volume_node_indices: Vec::new(),
            triangle_volume_element_indices: vec![Some(0), Some(1), Some(1)],
        }];

        let overlay = scalar_overlay(&field, &meshes, AnalysisFigureGenerationOptions::default())
            .expect("element scalar field should project to boundary triangles");

        assert_eq!(overlay.location, MeshFieldLocation::Triangle);
        assert_eq!(overlay.chunks, vec![vec![10.0, 42.0, 42.0]]);
    }

    #[test]
    fn structural_summary_figure_reports_reactions_and_metrics() {
        let fields = vec![
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_REACTION_FORCE,
                vec![2, 3],
                vec![3.0, 4.0, 0.0, 0.0, 0.0, 12.0],
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
                vec![1, 3],
                vec![0.0, 0.0, 2.0],
            ),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, vec![1], vec![7.5]),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, vec![1], vec![0.001]),
        ];

        assert_eq!(
            vector_field_total_magnitude(&fields, FEA_FIELD_STRUCTURAL_REACTION_FORCE),
            Some(13.0)
        );
        let figure = structural_result_summary_figure(&fields)
            .expect("structural summary should be generated");

        assert_eq!(figure.kind, AnalysisGeneratedFigureKind::Summary);
        assert_eq!(figure.title, "FEA structural result summary");
        assert!(figure
            .field_ids
            .iter()
            .any(|field_id| field_id == FEA_FIELD_STRUCTURAL_REACTION_FORCE));
        assert!(figure
            .field_ids
            .iter()
            .any(|field_id| field_id == FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY));
        assert!(figure
            .figure
            .plots()
            .any(|plot| matches!(plot, PlotElement::Bar(_))));
    }

    #[test]
    fn scalar_overlay_projects_node_values_to_render_vertices() {
        let field = AnalysisField::host_f64(
            "structural.nodal_von_mises",
            vec![4],
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 3,
            triangles: 1,
            vertex_volume_node_indices: vec![Some(0), Some(3), Some(1)],
            triangle_volume_element_indices: Vec::new(),
        }];

        let overlay = scalar_overlay(&field, &meshes, AnalysisFigureGenerationOptions::default())
            .expect("node scalar field should project to render vertices");

        assert_eq!(overlay.location, MeshFieldLocation::Vertex);
        assert_eq!(overlay.chunks, vec![vec![1.0, 4.0, 2.0]]);
    }

    #[test]
    fn vector_overlay_projects_node_values_to_render_vertices() {
        let field = AnalysisField::host_f64(
            "structural.displacement",
            vec![4, 3],
            vec![
                1.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 3.0, //
                4.0, 0.0, 0.0,
            ],
        );
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 3,
            triangles: 1,
            vertex_volume_node_indices: vec![Some(0), Some(3), Some(1)],
            triangle_volume_element_indices: Vec::new(),
        }];

        let overlay = vector_overlay(&field, &meshes, AnalysisFigureGenerationOptions::default())
            .expect("node vector field should project to render vertices");

        assert_eq!(overlay.location, MeshFieldLocation::Vertex);
        assert_eq!(
            overlay.chunks,
            vec![vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(4.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0)
            ]]
        );
    }

    #[test]
    fn deformation_overlay_projects_node_values_to_render_vertices() {
        let field = AnalysisField::host_f64(
            "structural.displacement",
            vec![4, 3],
            vec![
                1.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 3.0, //
                4.0, 0.0, 0.0,
            ],
        );
        let meshes = vec![MeshCounts {
            plot_index: 0,
            vertices: 3,
            triangles: 1,
            vertex_volume_node_indices: vec![Some(0), Some(3), Some(1)],
            triangle_volume_element_indices: Vec::new(),
        }];
        let figure = render_topology_figure(
            &simple_render_topology(),
            "solver mesh",
            AnalysisFigureGenerationOptions::default(),
        )
        .expect("solver topology should render");

        let overlay = deformation_overlay(
            &field,
            &meshes,
            &figure,
            AnalysisFigureGenerationOptions::default(),
        )
        .expect("node vector field should project to render deformation");

        assert_eq!(
            overlay.chunks,
            vec![vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(4.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0)
            ]]
        );
    }
}
