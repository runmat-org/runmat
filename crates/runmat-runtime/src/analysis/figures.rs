use glam::{Vec3, Vec4};
use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_plot::plots::{
    BarChart, Figure, LinePlot, MeshDeformation, MeshEdgeMode, MeshFieldLocation, MeshPlot,
    MeshScalarField, MeshVectorField, PlotElement,
};

use super::contracts::{
    AnalysisRenderTopology, AnalysisResultsCompareData, AnalysisResultsCompareQuery,
    AnalysisRunKind, AnalysisRunResult, AnalysisStudySpec, AnalysisTrendsData, AnalysisTrendsQuery,
};
use super::{analysis_results_compare_op, analysis_trends_op, collect_analysis_result_fields};
use super::{run_kind, storage};
use crate::geometry::{geometry_preview_figure, GeometryPreviewFigureOptions};
use crate::operations::OperationContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisGeneratedFigureKind {
    MeshResult,
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
    pub warnings: Vec<String>,
    pub figure: Figure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisFigureGenerationOptions {
    pub max_overlay_values: usize,
    pub max_vector_glyphs: usize,
    pub max_mesh_result_figures: usize,
    pub max_mesh_geometry_bytes: usize,
    pub edge_overlay_triangle_limit: usize,
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
                )]
            }
        };
    let mesh_counts = collect_mesh_counts(&probe);
    if mesh_counts.is_empty() {
        return Vec::new();
    }

    let deformation = fields
        .iter()
        .filter(|field| is_deformation_candidate(&field.field_id))
        .find_map(|field| deformation_overlay(field, &mesh_counts, &probe, options));

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
                    warnings,
                    figure,
                });
            }
        }
    }

    for field in &fields {
        if figures.len() >= per_run_mesh_figure_limit {
            break;
        }
        let Some(scalar) = scalar_overlay(field, &mesh_counts, options) else {
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
            warnings,
            figure,
        });
    }

    for field in &fields {
        if figures.len() >= per_run_mesh_figure_limit {
            break;
        }
        let Some(vector) = vector_overlay(field, &mesh_counts, options) else {
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
            warnings,
            figure,
        });
    }

    if figures.is_empty() {
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
                warnings: shared_warnings,
                figure,
            });
        }
    }

    figures
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
        if mesh.triangles.len() > options.edge_overlay_triangle_limit {
            plot.set_edge_mode(MeshEdgeMode::None);
            plot.set_edge_width(0.0);
        } else {
            plot.set_edge_mode(MeshEdgeMode::All);
            plot.set_edge_width(0.28);
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
    figure
        .plots()
        .enumerate()
        .filter_map(|(plot_index, plot)| match plot {
            PlotElement::Mesh(mesh) => Some(MeshCounts {
                plot_index,
                vertices: mesh.vertices().len(),
                triangles: mesh.triangles().len(),
            }),
            _ => None,
        })
        .collect()
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
    if values.len() == total_triangles {
        return scalar_overlay_from_values(
            field,
            MeshFieldLocation::Triangle,
            meshes.iter().map(|mesh| mesh.triangles),
            options,
        );
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
    let vectors = vectors_for_count(field, total_vertices)?;
    let scale = deformation_scale(&vectors, figure);
    Some(DeformationOverlay {
        field_id: field.field_id.clone(),
        label: field.field_id.clone(),
        chunks: split_vec3(&vectors, meshes.iter().map(|mesh| mesh.vertices))?,
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
        warnings: vec![warning],
        figure,
    }
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
