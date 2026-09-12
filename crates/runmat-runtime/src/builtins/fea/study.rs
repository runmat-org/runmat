use crate::analysis::AnalysisAcousticRunOptions;
use crate::analysis::AnalysisCfdRunOptions;
use crate::analysis::AnalysisChtRunOptions;
use crate::analysis::AnalysisCreateModelIntentSpec;
use crate::analysis::AnalysisCreateModelProfile;
use crate::analysis::AnalysisElectromagneticRunOptions;
use crate::analysis::AnalysisFsiRunOptions;
use crate::analysis::AnalysisModalRunOptions;
use crate::analysis::AnalysisNonlinearRunOptions;
use crate::analysis::AnalysisRunKind;
use crate::analysis::AnalysisRunOptions;
use crate::analysis::AnalysisStudySpec;
use crate::analysis::AnalysisStudySweepSpec;
use crate::analysis::AnalysisThermalRunOptions;
use crate::analysis::AnalysisTransientRunOptions;
use crate::builtins::fea::contracts::descriptors::ERROR_INPUT;
use crate::builtins::fea::contracts::identities::{STUDY_NAME, SWEEP_NAME};
use crate::builtins::fea::errors::{builtin_error, sanitize_id};
use crate::builtins::fea::geometry::{
    geometry_asset_from_value, parse_scalar_enum, resolve_study_profile_and_run_kind, scalar_string,
};
use crate::builtins::fea::model::assembly::build_model_from_parts;
use crate::builtins::fea::model::codec::{
    boundary_condition_vec_from_value, domain_vec_from_value, interface_vec_from_value,
    load_case_vec_from_value, material_assignment_vec_from_value, material_vec_from_value,
    model_from_value, step_vec_from_value, study_vec_from_value,
};
use crate::builtins::fea::options_json::{expect_name_value_tail, option_key};
use crate::builtins::fea::output::{study_to_object, sweep_to_object};
use crate::builtins::fea::run_options::{
    resolved_run_options_from_payload, run_options_payload_from_value,
};
use crate::builtins::fea::value_decode::{logical_from_value, parse_model_defaults_mode};
use crate::BuiltinResult;
use runmat_analysis_core::AnalysisInterface;
use runmat_analysis_core::AnalysisModel;
use runmat_analysis_core::AnalysisStep;
use runmat_analysis_core::BoundaryCondition;
use runmat_analysis_core::LoadCase;
use runmat_analysis_core::MaterialAssignment;
use runmat_analysis_core::MaterialModel;
use runmat_analysis_core::ReferenceFrame;
use runmat_analysis_fea::ComputeBackend;
use runmat_value::Value;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::builtins::fea) struct RunOptionsPayload {
    pub(in crate::builtins::fea) run_kind: AnalysisRunKind,
    pub(in crate::builtins::fea) options: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::builtins::fea) struct DomainPayload {
    pub(in crate::builtins::fea) kind: String,
    pub(in crate::builtins::fea) data: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::fea) enum ModelDefaultsMode {
    ProfileScaffold,
    None,
}

impl Default for ModelDefaultsMode {
    fn default() -> Self {
        Self::ProfileScaffold
    }
}

#[derive(Debug, Default)]
pub(in crate::builtins::fea) struct StudyConstructorOptions {
    pub(in crate::builtins::fea) run_kind: Option<AnalysisRunKind>,
    pub(in crate::builtins::fea) profile: Option<AnalysisCreateModelProfile>,
    pub(in crate::builtins::fea) backend: Option<ComputeBackend>,
    pub(in crate::builtins::fea) model_id: Option<String>,
    pub(in crate::builtins::fea) model: Option<AnalysisModel>,
    pub(in crate::builtins::fea) frame: Option<ReferenceFrame>,
    pub(in crate::builtins::fea) model_defaults: ModelDefaultsMode,
    pub(in crate::builtins::fea) materials: Vec<MaterialModel>,
    pub(in crate::builtins::fea) material_assignments: Vec<MaterialAssignment>,
    pub(in crate::builtins::fea) boundary_conditions: Vec<BoundaryCondition>,
    pub(in crate::builtins::fea) loads: Vec<LoadCase>,
    pub(in crate::builtins::fea) steps: Vec<AnalysisStep>,
    pub(in crate::builtins::fea) domains: Vec<DomainPayload>,
    pub(in crate::builtins::fea) interfaces: Vec<AnalysisInterface>,
    pub(in crate::builtins::fea) run_options: Option<RunOptionsPayload>,
}

pub(in crate::builtins::fea) fn create_study_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study requires id and geometry arguments",
        ));
    }
    let study_id = scalar_string(&args[0], STUDY_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(STUDY_NAME, &args[1])?;
    let options = StudyConstructorOptions::parse(&args[2..])?;
    let (profile, run_kind) = resolve_study_profile_and_run_kind(&options)?;
    let model_id = options.model_id.clone().unwrap_or_else(|| {
        options
            .model
            .as_ref()
            .map(|model| model.model_id.0.clone())
            .unwrap_or_else(|| format!("{}_model", sanitize_id(&study_id)))
    });
    let model = match options.model {
        Some(model) => Some(model),
        None if options.has_model_components() => Some(build_model_from_parts(
            STUDY_NAME,
            &geometry,
            model_id.clone(),
            profile,
            options.model_defaults,
            options.frame,
            options.materials,
            options.material_assignments,
            options.boundary_conditions,
            options.loads,
            options.steps,
            options.domains,
            options.interfaces,
        )?),
        None => None,
    };
    let run_options = options
        .run_options
        .map(|payload| resolved_run_options_from_payload(STUDY_NAME, payload, run_kind))
        .transpose()?
        .unwrap_or_default();
    let spec = AnalysisStudySpec {
        study_id,
        geometry,
        create_model_intent: AnalysisCreateModelIntentSpec {
            model_id,
            profile,
            prep_context: None,
        },
        model,
        run_kind,
        backend: options.backend.unwrap_or(ComputeBackend::Cpu),
        mesh_options: None,
        outputs: Vec::new(),
        analysis_mesh_artifact_path: None,
        analysis_mesh_evidence_artifact_path: None,
        linear_static_run_options: run_options.linear_static,
        modal_run_options: run_options.modal,
        acoustic_run_options: run_options.acoustic,
        thermal_run_options: run_options.thermal,
        transient_run_options: run_options.transient,
        cfd_run_options: run_options.cfd,
        cht_run_options: run_options.cht,
        fsi_run_options: run_options.fsi,
        nonlinear_run_options: run_options.nonlinear,
        electromagnetic_run_options: run_options.electromagnetic,
    };
    study_to_object(spec)
}

impl StudyConstructorOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(builtin_error(
                STUDY_NAME,
                &ERROR_INPUT,
                "fea.study options must be Name, Value pairs",
            ));
        }
        let mut options = Self::default();
        let mut seen = HashSet::new();
        for pair in args.chunks(2) {
            let key = option_key(&pair[0], STUDY_NAME)?;
            let canonical = match key.as_str() {
                "runkind" | "kind" => "runkind",
                "materialassignments" | "assignments" => "materialassignments",
                "boundaryconditions" | "bcs" => "boundaryconditions",
                "loads" | "loadcases" => "loads",
                "runoptions" | "options" => "runoptions",
                other => other,
            };
            if !seen.insert(canonical.to_string()) {
                return Err(builtin_error(
                    STUDY_NAME,
                    &ERROR_INPUT,
                    format!("duplicate fea.study option `{canonical}`"),
                ));
            }
            match key.as_str() {
                "runkind" | "kind" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.run_kind = Some(parse_scalar_enum(&text, "RunKind")?);
                }
                "profile" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.profile = Some(parse_scalar_enum(&text, "Profile")?);
                }
                "backend" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.backend = Some(parse_scalar_enum(&text, "Backend")?);
                }
                "modelid" => {
                    options.model_id = Some(scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?);
                }
                "model" => {
                    options.model = Some(model_from_value(STUDY_NAME, &pair[1])?);
                }
                "frame" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.frame = Some(parse_scalar_enum(&text, "Frame")?);
                }
                "defaults" => {
                    options.model_defaults = parse_model_defaults_mode(&scalar_string(
                        &pair[1],
                        STUDY_NAME,
                        &ERROR_INPUT,
                    )?)?;
                }
                "materials" => options.materials = material_vec_from_value(STUDY_NAME, &pair[1])?,
                "materialassignments" | "assignments" => {
                    options.material_assignments =
                        material_assignment_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "boundaryconditions" | "bcs" => {
                    options.boundary_conditions =
                        boundary_condition_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "loads" | "loadcases" => {
                    options.loads = load_case_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "steps" => options.steps = step_vec_from_value(STUDY_NAME, &pair[1])?,
                "domains" => options.domains = domain_vec_from_value(STUDY_NAME, &pair[1])?,
                "interfaces" => {
                    options.interfaces = interface_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "runoptions" | "options" => {
                    options.run_options =
                        Some(run_options_payload_from_value(STUDY_NAME, &pair[1])?);
                }
                other => {
                    return Err(builtin_error(
                        STUDY_NAME,
                        &ERROR_INPUT,
                        format!("unsupported fea.study option `{other}`"),
                    ));
                }
            }
        }
        Ok(options)
    }

    fn has_model_components(&self) -> bool {
        self.frame.is_some()
            || !self.materials.is_empty()
            || !self.material_assignments.is_empty()
            || !self.boundary_conditions.is_empty()
            || !self.loads.is_empty()
            || !self.steps.is_empty()
            || !self.domains.is_empty()
            || !self.interfaces.is_empty()
    }
}

#[derive(Debug, Default)]
pub(in crate::builtins::fea) struct ModelConstructorOptions {
    pub(in crate::builtins::fea) profile: Option<AnalysisCreateModelProfile>,
    pub(in crate::builtins::fea) frame: Option<ReferenceFrame>,
    pub(in crate::builtins::fea) defaults: ModelDefaultsMode,
    pub(in crate::builtins::fea) materials: Vec<MaterialModel>,
    pub(in crate::builtins::fea) material_assignments: Vec<MaterialAssignment>,
    pub(in crate::builtins::fea) boundary_conditions: Vec<BoundaryCondition>,
    pub(in crate::builtins::fea) loads: Vec<LoadCase>,
    pub(in crate::builtins::fea) steps: Vec<AnalysisStep>,
    pub(in crate::builtins::fea) domains: Vec<DomainPayload>,
    pub(in crate::builtins::fea) interfaces: Vec<AnalysisInterface>,
}

#[derive(Debug, Default)]
pub(in crate::builtins::fea) struct ResolvedRunOptions {
    pub(in crate::builtins::fea) linear_static: Option<AnalysisRunOptions>,
    pub(in crate::builtins::fea) modal: Option<AnalysisModalRunOptions>,
    pub(in crate::builtins::fea) acoustic: Option<AnalysisAcousticRunOptions>,
    pub(in crate::builtins::fea) thermal: Option<AnalysisThermalRunOptions>,
    pub(in crate::builtins::fea) transient: Option<AnalysisTransientRunOptions>,
    pub(in crate::builtins::fea) cfd: Option<AnalysisCfdRunOptions>,
    pub(in crate::builtins::fea) cht: Option<AnalysisChtRunOptions>,
    pub(in crate::builtins::fea) fsi: Option<AnalysisFsiRunOptions>,
    pub(in crate::builtins::fea) nonlinear: Option<AnalysisNonlinearRunOptions>,
    pub(in crate::builtins::fea) electromagnetic: Option<AnalysisElectromagneticRunOptions>,
}

pub(in crate::builtins::fea) fn create_sweep_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            SWEEP_NAME,
            &ERROR_INPUT,
            "fea.sweep requires id and studies arguments",
        ));
    }
    let sweep_id = scalar_string(&args[0], SWEEP_NAME, &ERROR_INPUT)?;
    let studies = study_vec_from_value(SWEEP_NAME, &args[1])?;
    let mut fail_fast = true;
    let mut fail_fast_seen = false;
    for pair in expect_name_value_tail(SWEEP_NAME, &args[2..])? {
        match pair.key.as_str() {
            "failfast" => {
                if fail_fast_seen {
                    return Err(builtin_error(
                        SWEEP_NAME,
                        &ERROR_INPUT,
                        "duplicate fea.sweep option `failfast`",
                    ));
                }
                fail_fast_seen = true;
                fail_fast = logical_from_value(SWEEP_NAME, pair.value)?;
            }
            other => {
                return Err(builtin_error(
                    SWEEP_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.sweep option `{other}`"),
                ));
            }
        }
    }
    sweep_to_object(AnalysisStudySweepSpec {
        sweep_id,
        studies,
        fail_fast,
    })
}
