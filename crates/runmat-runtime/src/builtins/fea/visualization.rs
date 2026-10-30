use runmat_value::Value;

#[cfg(feature = "plot-core")]
use super::ERROR_INPUT;
use super::{builtin_error, BuiltinResult, ERROR_OPERATION, MESH_NAME, PLOT_NAME};

#[cfg(feature = "plot-core")]
mod request;
#[cfg(feature = "plot-core")]
pub(super) use request::{mesh_request_from_args, plot_request_from_args};
#[cfg(feature = "plot-core")]
use request::{reject_requested_device_field, FeaPlotRequest};

pub(super) async fn create_plot_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(feature = "plot-core")]
    {
        let request = plot_request_from_args(&args)?;
        let mut figures = super::artifact_access::read(PLOT_NAME, || {
            reject_requested_device_field(&request)?;
            generate_plot_figures(&request)
        })
        .await?;
        let figure = select_generated_figure(&mut figures, &request.field_id)?;
        let handle = import_generated_figure(figure);
        Ok(Value::Num(f64::from(handle)))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = args;
        Err(builtin_error(
            PLOT_NAME,
            &ERROR_OPERATION,
            "plot for FEA objects requires the plot-core runtime feature",
        ))
    }
}

pub(super) async fn create_mesh_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(feature = "plot-core")]
    {
        let request = mesh_request_from_args(&args)?;
        let figure = super::artifact_access::read(MESH_NAME, || {
            crate::analysis::analysis_generate_study_run_mesh_figure(
                &request.study,
                &request.run_id,
                crate::analysis::AnalysisFigureGenerationOptions {
                    mesh_source: request.source,
                    show_solver_mesh_edges: request.show_edges,
                    apply_deformation_overlay: false,
                    include_comparison: false,
                    include_trends: false,
                    ..crate::analysis::AnalysisFigureGenerationOptions::default()
                },
            )
            .map_err(|err| builtin_error(MESH_NAME, &ERROR_OPERATION, err))
        })
        .await?;
        let handle = import_generated_figure(figure);
        Ok(Value::Num(f64::from(handle)))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = args;
        Err(builtin_error(
            MESH_NAME,
            &ERROR_OPERATION,
            "mesh for FEA objects requires the plot-core runtime feature",
        ))
    }
}

#[cfg(feature = "plot-core")]
fn generate_plot_figures(
    request: &FeaPlotRequest,
) -> BuiltinResult<Vec<crate::analysis::AnalysisGeneratedFigure>> {
    crate::analysis::analysis_generate_study_field_figures(
        &request.study,
        &request.run_id,
        &crate::analysis::AnalysisFieldSelection::new(&request.field_id, request.sample),
        crate::analysis::AnalysisFigureGenerationOptions {
            include_comparison: false,
            include_trends: false,
            max_mesh_result_figures: 8,
            mesh_source: request.options.mesh_source,
            show_solver_mesh_edges: request.options.show_solver_mesh_edges,
            apply_deformation_overlay: request.options.apply_deformation_overlay,
            ..crate::analysis::AnalysisFigureGenerationOptions::default()
        },
    )
    .map_err(|err| builtin_error(PLOT_NAME, &ERROR_OPERATION, err))
}

#[cfg(feature = "plot-core")]
fn select_generated_figure(
    figures: &mut Vec<crate::analysis::AnalysisGeneratedFigure>,
    field_id: &str,
) -> BuiltinResult<crate::analysis::AnalysisGeneratedFigure> {
    if figures.is_empty() {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_OPERATION,
            "plot could not generate a renderable FEA figure for this run",
        ));
    }
    if let Some(index) = figures.iter().position(|figure| {
        figure.field_ids.iter().any(|candidate| {
            crate::analysis::field_selection::field_id_matches(candidate, field_id)
        })
    }) {
        return Ok(figures.remove(index));
    }
    let available = figures
        .iter()
        .flat_map(|figure| figure.field_ids.iter())
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    Err(builtin_error(
        PLOT_NAME,
        &ERROR_INPUT,
        format!(
            "FEA field `{field_id}` did not produce a mesh figure; available figure fields: {available}"
        ),
    ))
}

#[cfg(feature = "plot-core")]
fn import_generated_figure(figure: crate::analysis::AnalysisGeneratedFigure) -> u32 {
    crate::builtins::plotting::import_runtime_figure(figure.figure)
}
