use crate::analysis::AnalysisAcousticRunOptions;
use crate::analysis::AnalysisCfdRunOptions;
use crate::analysis::AnalysisChtRunOptions;
use crate::analysis::AnalysisElectromagneticRunOptions;
use crate::analysis::AnalysisFsiRunOptions;
use crate::analysis::AnalysisModalRunOptions;
use crate::analysis::AnalysisNonlinearRunOptions;
use crate::analysis::AnalysisRunKind;
use crate::analysis::AnalysisRunOptions;
use crate::analysis::AnalysisThermalRunOptions;
use crate::analysis::AnalysisTransientRunOptions;
use crate::builtins::fea::contracts::descriptors::ERROR_INPUT;
use crate::builtins::fea::contracts::identities::{FEA_RUN_OPTIONS_CLASS, RUN_OPTIONS_NAME};
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::{parse_scalar_enum, scalar_string};
use crate::builtins::fea::model::assembly::run_options_to_object;
use crate::builtins::fea::model::codec::object_payload;
use crate::builtins::fea::options_json::{
    canonical_field_name, expect_name_value_tail, json_deserialize, typed_json_with_overrides,
    value_to_json,
};
use crate::builtins::fea::study::{ResolvedRunOptions, RunOptionsPayload};
use crate::builtins::fea::value_decode::usize_from_value;
use crate::BuiltinResult;
use runmat_value::Value;

pub(in crate::builtins::fea) fn create_run_options_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            RUN_OPTIONS_NAME,
            &ERROR_INPUT,
            "fea.runOptions requires a solver",
        ));
    }
    let kind_text = scalar_string(&args[0], RUN_OPTIONS_NAME, &ERROR_INPUT)?;
    let run_kind = parse_scalar_enum::<AnalysisRunKind>(&kind_text, "solver")?;
    let fields = run_options_fields_from_name_values(&args[1..])?;
    let data = run_options_json_for_kind(RUN_OPTIONS_NAME, run_kind, fields)?;
    run_options_to_object(RunOptionsPayload {
        run_kind,
        options: data,
    })
}

pub(in crate::builtins::fea) fn run_options_fields_from_name_values(
    args: &[Value],
) -> BuiltinResult<serde_json::Map<String, serde_json::Value>> {
    const EXACT_FIELDS: &[&str] = &[
        "mode_count",
        "step_count",
        "max_linear_iters",
        "max_step_retries",
        "increment_count",
        "max_newton_iters",
        "max_line_search_backtracks",
        "tangent_refresh_interval",
        "harmonic_max_iterations",
    ];
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(RUN_OPTIONS_NAME, args)? {
        let raw = scalar_string(pair.name, RUN_OPTIONS_NAME, &ERROR_INPUT)?;
        let key = canonical_field_name(&raw);
        if key == "prep_context" {
            return Err(builtin_error(
                RUN_OPTIONS_NAME,
                &ERROR_INPUT,
                "fea.runOptions does not expose the internal PrepContext; use PrepArtifactId or PrepCalibrationProfile",
            ));
        }
        let value = if EXACT_FIELDS.contains(&key.as_str()) {
            serde_json::Value::from(usize_from_value(RUN_OPTIONS_NAME, pair.value)? as u64)
        } else {
            value_to_json(RUN_OPTIONS_NAME, pair.value)?
        };
        if fields.insert(key.clone(), value).is_some() {
            return Err(builtin_error(
                RUN_OPTIONS_NAME,
                &ERROR_INPUT,
                format!("duplicate fea.runOptions option `{key}`"),
            ));
        }
    }
    Ok(fields)
}

pub(in crate::builtins::fea) fn run_options_payload_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<RunOptionsPayload> {
    object_payload(builtin, value, FEA_RUN_OPTIONS_CLASS)
}

pub(in crate::builtins::fea) fn resolved_run_options_from_payload(
    builtin: &'static str,
    payload: RunOptionsPayload,
    expected_kind: AnalysisRunKind,
) -> BuiltinResult<ResolvedRunOptions> {
    if payload.run_kind != expected_kind {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "run options kind {:?} does not match selected study solver {:?}",
                payload.run_kind, expected_kind
            ),
        ));
    }
    let mut resolved = ResolvedRunOptions::default();
    match payload.run_kind {
        AnalysisRunKind::LinearStatic => {
            resolved.linear_static = Some(json_deserialize(
                builtin,
                payload.options,
                "linear_static run options",
            )?);
        }
        AnalysisRunKind::Modal => {
            resolved.modal = Some(json_deserialize(
                builtin,
                payload.options,
                "modal run options",
            )?);
        }
        AnalysisRunKind::Acoustic => {
            resolved.acoustic = Some(json_deserialize(
                builtin,
                payload.options,
                "acoustic run options",
            )?);
        }
        AnalysisRunKind::Thermal => {
            resolved.thermal = Some(json_deserialize(
                builtin,
                payload.options,
                "thermal run options",
            )?);
        }
        AnalysisRunKind::Transient => {
            resolved.transient = Some(json_deserialize(
                builtin,
                payload.options,
                "transient run options",
            )?);
        }
        AnalysisRunKind::Cfd => {
            resolved.cfd = Some(json_deserialize(
                builtin,
                payload.options,
                "cfd run options",
            )?);
        }
        AnalysisRunKind::Cht => {
            resolved.cht = Some(json_deserialize(
                builtin,
                payload.options,
                "cht run options",
            )?);
        }
        AnalysisRunKind::Fsi => {
            resolved.fsi = Some(json_deserialize(
                builtin,
                payload.options,
                "fsi run options",
            )?);
        }
        AnalysisRunKind::Nonlinear => {
            resolved.nonlinear = Some(json_deserialize(
                builtin,
                payload.options,
                "nonlinear run options",
            )?);
        }
        AnalysisRunKind::Electromagnetic => {
            resolved.electromagnetic = Some(json_deserialize(
                builtin,
                payload.options,
                "electromagnetic run options",
            )?);
        }
    }
    Ok(resolved)
}

pub(in crate::builtins::fea) fn run_options_json_for_kind(
    builtin: &'static str,
    run_kind: AnalysisRunKind,
    fields: serde_json::Map<String, serde_json::Value>,
) -> BuiltinResult<serde_json::Value> {
    match run_kind {
        AnalysisRunKind::LinearStatic => typed_json_with_overrides::<AnalysisRunOptions>(
            builtin,
            AnalysisRunOptions::default(),
            fields,
            "linear_static run options",
        ),
        AnalysisRunKind::Modal => typed_json_with_overrides::<AnalysisModalRunOptions>(
            builtin,
            AnalysisModalRunOptions::default(),
            fields,
            "modal run options",
        ),
        AnalysisRunKind::Acoustic => typed_json_with_overrides::<AnalysisAcousticRunOptions>(
            builtin,
            AnalysisAcousticRunOptions::default(),
            fields,
            "acoustic run options",
        ),
        AnalysisRunKind::Thermal => typed_json_with_overrides::<AnalysisThermalRunOptions>(
            builtin,
            AnalysisThermalRunOptions::default(),
            fields,
            "thermal run options",
        ),
        AnalysisRunKind::Transient => typed_json_with_overrides::<AnalysisTransientRunOptions>(
            builtin,
            AnalysisTransientRunOptions::default(),
            fields,
            "transient run options",
        ),
        AnalysisRunKind::Cfd => typed_json_with_overrides::<AnalysisCfdRunOptions>(
            builtin,
            AnalysisCfdRunOptions::default(),
            fields,
            "cfd run options",
        ),
        AnalysisRunKind::Cht => typed_json_with_overrides::<AnalysisChtRunOptions>(
            builtin,
            AnalysisChtRunOptions::default(),
            fields,
            "cht run options",
        ),
        AnalysisRunKind::Fsi => typed_json_with_overrides::<AnalysisFsiRunOptions>(
            builtin,
            AnalysisFsiRunOptions::default(),
            fields,
            "fsi run options",
        ),
        AnalysisRunKind::Nonlinear => typed_json_with_overrides::<AnalysisNonlinearRunOptions>(
            builtin,
            AnalysisNonlinearRunOptions::default(),
            fields,
            "nonlinear run options",
        ),
        AnalysisRunKind::Electromagnetic => {
            typed_json_with_overrides::<AnalysisElectromagneticRunOptions>(
                builtin,
                AnalysisElectromagneticRunOptions::default(),
                fields,
                "electromagnetic run options",
            )
        }
    }
}
