use crate::analysis::AnalysisStudySpec;
#[cfg(feature = "plot-core")]
use crate::analysis::{analysis_results_by_run_id_op, AnalysisResultsQuery};
use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_OPERATION};
use crate::builtins::fea::contracts::identities::{FEA_FIELD_CLASS, FEA_STUDY_CLASS, PLOT_NAME};
use crate::builtins::fea::errors::builtin_error;
#[cfg(feature = "plot-core")]
use crate::builtins::fea::errors::operation_error;
use crate::builtins::fea::geometry::scalar_string;
use crate::builtins::fea::results::field::field_id_matches;
#[cfg(feature = "plot-core")]
use crate::builtins::fea::results::field::find_field;
use crate::builtins::fea::results::query::{
    run_id_context_from_value, run_id_from_value, study_context_from_value,
};
use crate::builtins::fea::value_decode::bool_from_value;
#[cfg(feature = "plot-core")]
use crate::operations::OperationContext;
use crate::BuiltinResult;
#[cfg(feature = "plot-core")]
use runmat_analysis_core::AnalysisFieldValues;
use runmat_value::Value;

pub(in crate::builtins::fea) fn create_plot_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(feature = "plot-core")]
    {
        let request = plot_request_from_args(&args)?;
        reject_requested_device_field(&request)?;
        let mut figures = generate_plot_figures(&request.study, &request.run_id, &request.options)?;
        let figure = select_generated_figure(&mut figures, request.field_id.as_deref())?;
        let handle = import_generated_figure(figure)?;
        Ok(Value::Num(f64::from(handle)))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = args;
        Err(builtin_error(
            PLOT_NAME,
            &ERROR_OPERATION,
            "fea.plot requires the plot-core runtime feature",
        ))
    }
}

#[cfg(feature = "plot-core")]
pub(in crate::builtins::fea) fn reject_requested_device_field(
    request: &FeaPlotRequest,
) -> BuiltinResult<()> {
    let Some(field_id) = request.field_id.as_deref() else {
        return Ok(());
    };
    let results = analysis_results_by_run_id_op(
        &request.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .map(|envelope| envelope.data)
    .map_err(|err| operation_error(PLOT_NAME, &ERROR_OPERATION, err))?;
    if find_field(results.fields, field_id)
        .is_some_and(|field| matches!(field.values, AnalysisFieldValues::DeviceRef(_)))
    {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            format!(
                "FEA field `{field_id}` is device-backed and cannot be plotted without explicit host materialization"
            ),
        ));
    }
    Ok(())
}

pub(in crate::builtins::fea) struct FeaPlotRequest {
    pub(in crate::builtins::fea) study: AnalysisStudySpec,
    pub(in crate::builtins::fea) run_id: String,
    pub(in crate::builtins::fea) field_id: Option<String>,
    pub(in crate::builtins::fea) options: FeaPlotOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::builtins::fea) struct FeaPlotOptions {
    pub(in crate::builtins::fea) field_id: Option<String>,
    pub(in crate::builtins::fea) mesh_source: crate::analysis::AnalysisFigureMeshSource,
    pub(in crate::builtins::fea) show_solver_mesh_edges: bool,
    pub(in crate::builtins::fea) apply_deformation_overlay: bool,
}

impl Default for FeaPlotOptions {
    fn default() -> Self {
        Self {
            field_id: None,
            mesh_source: crate::analysis::AnalysisFigureMeshSource::Auto,
            show_solver_mesh_edges: false,
            apply_deformation_overlay: true,
        }
    }
}

pub(in crate::builtins::fea) fn plot_request_from_args(
    args: &[Value],
) -> BuiltinResult<FeaPlotRequest> {
    if args.is_empty() {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            "fea.plot requires a run, results, field, or study/run pair",
        ));
    }

    let (core, options) = split_plot_options(args)?;
    match core {
        [single] => plot_request_from_context_value(single, options),
        [first, second] if is_fea_study(first) => {
            let study = study_context_from_value(PLOT_NAME, first)?;
            let run_id = run_id_from_value(PLOT_NAME, second)?;
            Ok(FeaPlotRequest {
                study,
                run_id,
                field_id: options.field_id.clone(),
                options,
            })
        }
        [first, second] => {
            let mut request = plot_request_from_context_value(first, options)?;
            request.field_id = Some(scalar_string(second, PLOT_NAME, &ERROR_INPUT)?);
            if request.options.field_id.is_some() {
                request.field_id = request.options.field_id.clone();
            }
            Ok(request)
        }
        [first, second, third] if is_fea_study(first) => {
            let study = study_context_from_value(PLOT_NAME, first)?;
            let run_id = run_id_from_value(PLOT_NAME, second)?;
            let field_id = match options.field_id.clone() {
                Some(field_id) => Some(field_id),
                None => Some(scalar_string(third, PLOT_NAME, &ERROR_INPUT)?),
            };
            Ok(FeaPlotRequest {
                study,
                run_id,
                field_id,
                options,
            })
        }
        _ => Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            "fea.plot supports plot(run, field), plot(results, field), plot(field), or plot(study, runId, field)",
        )),
    }
}

pub(in crate::builtins::fea) fn split_plot_options(
    args: &[Value],
) -> BuiltinResult<(&[Value], FeaPlotOptions)> {
    let mut options = FeaPlotOptions::default();
    let mut end = args.len();
    while end >= 2 && is_plot_option_name(&args[end - 2]) {
        let key = scalar_string(&args[end - 2], PLOT_NAME, &ERROR_INPUT)?.to_ascii_lowercase();
        match key.as_str() {
            "field" | "fieldid" | "field_id" => {
                options.field_id = Some(scalar_string(&args[end - 1], PLOT_NAME, &ERROR_INPUT)?);
            }
            "mesh" => {
                options.show_solver_mesh_edges =
                    plot_mesh_option_shows_solver_edges(&args[end - 1])?;
            }
            "overlay" => {
                options.mesh_source = plot_overlay_option_mesh_source(&args[end - 1])?;
            }
            "deformed" => {
                options.apply_deformation_overlay = bool_from_value(PLOT_NAME, &args[end - 1])?;
            }
            _ => unreachable!("is_plot_option_name only accepts supported plot option names"),
        }
        end -= 2;
    }
    Ok((&args[..end], options))
}

pub(in crate::builtins::fea) fn is_plot_option_name(value: &Value) -> bool {
    scalar_string(value, PLOT_NAME, &ERROR_INPUT)
        .map(|name| {
            matches!(
                name.to_ascii_lowercase().as_str(),
                "field" | "fieldid" | "field_id" | "mesh" | "overlay" | "deformed"
            )
        })
        .unwrap_or(false)
}

pub(in crate::builtins::fea) fn plot_mesh_option_shows_solver_edges(
    value: &Value,
) -> BuiltinResult<bool> {
    let mesh = scalar_string(value, PLOT_NAME, &ERROR_INPUT)?;
    match mesh.to_ascii_lowercase().as_str() {
        "solver" | "solver_edges" | "solveredges" | "edges" => Ok(true),
        "cad" | "geometry" | "surface" | "none" => Ok(false),
        other => Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            format!(
                "unsupported fea.plot mesh option `{other}`; expected solver, solver_edges, cad, geometry, surface, or none"
            ),
        )),
    }
}

pub(in crate::builtins::fea) fn plot_overlay_option_mesh_source(
    value: &Value,
) -> BuiltinResult<crate::analysis::AnalysisFigureMeshSource> {
    let overlay = scalar_string(value, PLOT_NAME, &ERROR_INPUT)?;
    match overlay.to_ascii_lowercase().as_str() {
        "auto" => Ok(crate::analysis::AnalysisFigureMeshSource::Auto),
        "solver" | "mesh" | "boundary" | "solver_boundary" | "solverboundary" => {
            Ok(crate::analysis::AnalysisFigureMeshSource::Solver)
        }
        "cad" | "cad_reference" | "reference" | "geometry" | "surface" => {
            Ok(crate::analysis::AnalysisFigureMeshSource::CadReference)
        }
        other => Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            format!("unsupported fea.plot overlay option `{other}`; expected auto, solver, or cad"),
        )),
    }
}

pub(in crate::builtins::fea) fn is_fea_study(value: &Value) -> bool {
    matches!(value, Value::Object(object) if object.class_name == FEA_STUDY_CLASS)
}

pub(in crate::builtins::fea) fn plot_request_from_context_value(
    value: &Value,
    options: FeaPlotOptions,
) -> BuiltinResult<FeaPlotRequest> {
    let study = study_context_from_value(PLOT_NAME, value)?;
    let run_id = run_id_context_from_value(value)
        .or_else(|| run_id_from_value(PLOT_NAME, value).ok())
        .ok_or_else(|| {
            builtin_error(
                PLOT_NAME,
                &ERROR_INPUT,
                "fea.plot requires a run_id; use a fea.RunResult from fea.run or pass fea.plot(study, runId, field)",
            )
        })?;
    let field_id = options.field_id.clone().or_else(|| match value {
        Value::Object(object) if object.class_name == FEA_FIELD_CLASS => object
            .properties
            .get("field_id")
            .and_then(|value| match value {
                Value::String(field_id) => Some(field_id.clone()),
                _ => None,
            }),
        _ => None,
    });
    Ok(FeaPlotRequest {
        study,
        run_id,
        field_id,
        options,
    })
}

#[cfg(feature = "plot-core")]
pub(in crate::builtins::fea) fn generate_plot_figures(
    study: &AnalysisStudySpec,
    run_id: &str,
    options: &FeaPlotOptions,
) -> BuiltinResult<Vec<crate::analysis::AnalysisGeneratedFigure>> {
    crate::analysis::analysis_generate_study_run_figures(
        study,
        run_id,
        crate::analysis::AnalysisFigureGenerationOptions {
            include_comparison: false,
            include_trends: false,
            max_mesh_result_figures: 8,
            mesh_source: options.mesh_source,
            show_solver_mesh_edges: options.show_solver_mesh_edges,
            apply_deformation_overlay: options.apply_deformation_overlay,
            ..crate::analysis::AnalysisFigureGenerationOptions::default()
        },
    )
    .map_err(|err| builtin_error(PLOT_NAME, &ERROR_OPERATION, err))
}

#[cfg(feature = "plot-core")]
pub(in crate::builtins::fea) fn select_generated_figure(
    figures: &mut Vec<crate::analysis::AnalysisGeneratedFigure>,
    field_id: Option<&str>,
) -> BuiltinResult<crate::analysis::AnalysisGeneratedFigure> {
    if figures.is_empty() {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_OPERATION,
            "fea.plot could not generate a renderable FEA figure for this run",
        ));
    }
    let Some(field_id) = field_id else {
        if let Some(index) = default_generated_figure_index(figures) {
            return Ok(figures.remove(index));
        }
        return Ok(figures.remove(0));
    };
    if let Some(index) = figures.iter().position(|figure| {
        figure
            .field_ids
            .iter()
            .any(|candidate| field_id_matches(candidate, field_id))
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
        format!("FEA field `{field_id}` did not produce a mesh figure; available figure fields: {available}"),
    ))
}

#[cfg(feature = "plot-core")]
pub(in crate::builtins::fea) fn default_generated_figure_index(
    figures: &[crate::analysis::AnalysisGeneratedFigure],
) -> Option<usize> {
    let mut best: Option<(usize, u8)> = None;
    for (index, figure) in figures.iter().enumerate() {
        let score = default_generated_figure_score(figure);
        if best
            .map(|(_, best_score)| score > best_score)
            .unwrap_or(true)
        {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| index)
}

#[cfg(feature = "plot-core")]
pub(in crate::builtins::fea) fn default_generated_figure_score(
    figure: &crate::analysis::AnalysisGeneratedFigure,
) -> u8 {
    let kind_score = match figure.kind {
        crate::analysis::AnalysisGeneratedFigureKind::MeshResult => 40,
        crate::analysis::AnalysisGeneratedFigureKind::Modal
        | crate::analysis::AnalysisGeneratedFigureKind::Electromagnetic => 35,
        crate::analysis::AnalysisGeneratedFigureKind::Summary
        | crate::analysis::AnalysisGeneratedFigureKind::Convergence => 20,
        crate::analysis::AnalysisGeneratedFigureKind::Comparison
        | crate::analysis::AnalysisGeneratedFigureKind::Trend => 15,
    };
    figure
        .field_ids
        .iter()
        .map(|field_id| default_field_figure_score(field_id))
        .max()
        .unwrap_or(kind_score)
        .max(kind_score)
}

#[cfg(feature = "plot-core")]
pub(in crate::builtins::fea) fn default_field_figure_score(field_id: &str) -> u8 {
    let normalized = field_id.to_ascii_lowercase();
    if normalized.contains("residual")
        || normalized.contains("iteration")
        || normalized.contains("orthogonality")
        || normalized.contains("condition")
    {
        return 25;
    }
    if normalized.contains("von_mises") || normalized.contains("stress") {
        return 95;
    }
    if normalized.contains("temperature")
        || normalized.contains("heat_flux")
        || normalized.contains("velocity")
        || normalized.contains("pressure")
        || normalized.contains("magnetic_flux_density")
        || normalized.contains("electric_field")
        || normalized.contains("sound_pressure")
        || normalized.contains("coupling")
    {
        return 90;
    }
    if normalized.contains("mode_shape") || normalized.contains("displacement") {
        return 85;
    }
    if normalized.starts_with("structural.")
        || normalized.starts_with("modal.")
        || normalized.starts_with("thermal.")
        || normalized.starts_with("transient.")
        || normalized.starts_with("nonlinear.")
        || normalized.starts_with("em.")
        || normalized.starts_with("electro_thermal.")
        || normalized.starts_with("thermo_mechanical.")
        || normalized.starts_with("acoustic.")
        || normalized.starts_with("cfd.")
        || normalized.starts_with("fluid.")
        || normalized.starts_with("cht.")
        || normalized.starts_with("fsi.")
    {
        return 70;
    }
    40
}

#[cfg(feature = "plot-core")]
pub(in crate::builtins::fea) fn import_generated_figure(
    figure: crate::analysis::AnalysisGeneratedFigure,
) -> BuiltinResult<u32> {
    Ok(crate::builtins::plotting::import_runtime_figure(
        figure.figure,
    ))
}
