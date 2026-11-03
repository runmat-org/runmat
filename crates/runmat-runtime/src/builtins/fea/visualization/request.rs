//! FEA plot/mesh argument parsing; only meaningful with plot rendering.

use runmat_value::Value;

use super::super::{
    analysis_results_by_run_id_op, builtin_error, find_field, normalize_token, object_payload,
    operation_error, run_id_context_from_value, run_id_from_value, scalar_string,
    study_context_value, AnalysisFieldValues, AnalysisResultsQuery, AnalysisStudySpec,
    BuiltinResult, OperationContext, ERROR_INPUT, ERROR_OPERATION, FEA_STUDY_CLASS, MESH_NAME,
    PLOT_NAME,
};

pub(in crate::builtins::fea) struct FeaPlotRequest {
    pub(in crate::builtins::fea) study: AnalysisStudySpec,
    pub(in crate::builtins::fea) run_id: String,
    pub(in crate::builtins::fea) field_id: String,
    pub(in crate::builtins::fea) sample:
        Option<runmat_analysis_fea::results::field::RecoveredFieldSample>,
    pub(in crate::builtins::fea) options: FeaPlotOptions,
}

pub(in crate::builtins::fea) struct FeaMeshRequest {
    pub(in crate::builtins::fea) study: AnalysisStudySpec,
    pub(in crate::builtins::fea) run_id: String,
    pub(in crate::builtins::fea) source: crate::analysis::AnalysisFigureMeshSource,
    pub(in crate::builtins::fea) show_edges: bool,
}

pub(in crate::builtins::fea) fn mesh_request_from_args(
    args: &[Value],
) -> BuiltinResult<FeaMeshRequest> {
    if args.is_empty() {
        return Err(builtin_error(
            MESH_NAME,
            &ERROR_INPUT,
            "mesh requires an FEA run, results, or field",
        ));
    }
    let (core, source, show_edges) = split_mesh_options(args)?;
    let [context] = core else {
        return Err(builtin_error(
            MESH_NAME,
            &ERROR_INPUT,
            "mesh accepts one FEA run, results, or field followed by display Name, Value options",
        ));
    };
    let study = study_context_from_value(MESH_NAME, context)?;
    let run_id = run_id_context_from_value(context)
        .or_else(|| run_id_from_value(MESH_NAME, context).ok())
        .ok_or_else(|| {
            builtin_error(
                MESH_NAME,
                &ERROR_INPUT,
                "mesh requires an FEA object with run context",
            )
        })?;
    Ok(FeaMeshRequest {
        study,
        run_id,
        source,
        show_edges,
    })
}

fn split_mesh_options(
    args: &[Value],
) -> BuiltinResult<(&[Value], crate::analysis::AnalysisFigureMeshSource, bool)> {
    let mut source = crate::analysis::AnalysisFigureMeshSource::Solver;
    let mut show_edges = true;
    let mut end = args.len();
    while end >= 2 {
        let Ok(name) = scalar_string(&args[end - 2], MESH_NAME, &ERROR_INPUT) else {
            break;
        };
        match normalize_token(&name).as_str() {
            "source" => {
                source = mesh_source_from_value(&args[end - 1])?;
                end -= 2;
            }
            "edges" => {
                show_edges = bool_from_value(MESH_NAME, &args[end - 1])?;
                end -= 2;
            }
            _ => break,
        }
    }
    Ok((&args[..end], source, show_edges))
}

fn mesh_source_from_value(
    value: &Value,
) -> BuiltinResult<crate::analysis::AnalysisFigureMeshSource> {
    let source = scalar_string(value, MESH_NAME, &ERROR_INPUT)?;
    match normalize_token(&source).as_str() {
        "solver" => Ok(crate::analysis::AnalysisFigureMeshSource::Solver),
        "cad" => Ok(crate::analysis::AnalysisFigureMeshSource::Cad),
        "cadreference" => Ok(crate::analysis::AnalysisFigureMeshSource::CadReference),
        "auto" => Ok(crate::analysis::AnalysisFigureMeshSource::Auto),
        _ => Err(builtin_error(
            MESH_NAME,
            &ERROR_INPUT,
            format!(
                "unsupported FEA mesh source `{source}`; expected solver, cad, cad_reference, or auto"
            ),
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::builtins::fea) struct FeaPlotOptions {
    pub(in crate::builtins::fea) mesh_source: crate::analysis::AnalysisFigureMeshSource,
    pub(in crate::builtins::fea) show_solver_mesh_edges: bool,
    pub(in crate::builtins::fea) apply_deformation_overlay: bool,
}

impl Default for FeaPlotOptions {
    fn default() -> Self {
        Self {
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
            "plot requires an fea.Field",
        ));
    }

    let (core, options) = split_plot_options(args)?;
    let [field] = core else {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            "plot accepts one fea.Field followed by visualization Name, Value options",
        ));
    };
    let Value::Object(object) = field else {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            "plot requires an fea.Field",
        ));
    };
    if !object
        .class_name()
        .is(runmat_builtins::fea_contract::identities::FEA_FIELD_CLASS)
    {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            "plot requires an fea.Field",
        ));
    }
    let field_id = object
        .properties()
        .get("field_id")
        .and_then(|value| match value {
            Value::String(field_id) => field_id.text_ref().cloned(),
            _ => None,
        })
        .ok_or_else(|| {
            builtin_error(
                PLOT_NAME,
                &ERROR_INPUT,
                "fea.Field is missing field identity",
            )
        })?;
    Ok(FeaPlotRequest {
        study: study_context_from_value(PLOT_NAME, field)?,
        run_id: run_id_context_from_value(field).ok_or_else(|| {
            builtin_error(PLOT_NAME, &ERROR_INPUT, "fea.Field is missing run context")
        })?,
        field_id: field_id.to_string(),
        sample: super::super::field_access::selection_from_field_object(PLOT_NAME, object)?.sample,
        options,
    })
}

fn split_plot_options(args: &[Value]) -> BuiltinResult<(&[Value], FeaPlotOptions)> {
    let mut options = FeaPlotOptions::default();
    let mut end = args.len();
    while end >= 2 && is_plot_option_name(&args[end - 2]) {
        let key = scalar_string(&args[end - 2], PLOT_NAME, &ERROR_INPUT)?.to_ascii_lowercase();
        match key.as_str() {
            "edges" => {
                options.show_solver_mesh_edges = bool_from_value(PLOT_NAME, &args[end - 1])?;
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

fn is_plot_option_name(value: &Value) -> bool {
    scalar_string(value, PLOT_NAME, &ERROR_INPUT)
        .map(|name| {
            matches!(
                name.to_ascii_lowercase().as_str(),
                "edges" | "overlay" | "deformed"
            )
        })
        .unwrap_or(false)
}

fn plot_overlay_option_mesh_source(
    value: &Value,
) -> BuiltinResult<crate::analysis::AnalysisFigureMeshSource> {
    let overlay = scalar_string(value, PLOT_NAME, &ERROR_INPUT)?;
    match overlay.to_ascii_lowercase().as_str() {
        "auto" => Ok(crate::analysis::AnalysisFigureMeshSource::Auto),
        "solver" => Ok(crate::analysis::AnalysisFigureMeshSource::Solver),
        "cad" => Ok(crate::analysis::AnalysisFigureMeshSource::CadReference),
        other => Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            format!("unsupported FEA plot overlay option `{other}`; expected auto, solver, or cad"),
        )),
    }
}

fn study_context_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<AnalysisStudySpec> {
    let Some(study) = study_context_value(value) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin}: FEA visualization requires study geometry context; pass a fea.RunResult from fea.run(study), a derived fea.Results/fea.Field, or call {builtin}(study, run ID)"),
        ));
    };
    object_payload(builtin, &study, FEA_STUDY_CLASS)
}

fn bool_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<bool> {
    bool::try_from(value).map_err(|err| builtin_error(builtin, &ERROR_INPUT, err))
}

pub(super) fn reject_requested_device_field(request: &FeaPlotRequest) -> BuiltinResult<()> {
    let field_id = request.field_id.as_str();
    let query = crate::analysis::bounded_default_results_query(
        &request.run_id,
        AnalysisResultsQuery::default(),
    )
    .map_err(|err| builtin_error(PLOT_NAME, &ERROR_OPERATION, err))?;
    let results =
        analysis_results_by_run_id_op(&request.run_id, query, OperationContext::new(None, None))
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
