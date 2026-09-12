use crate::builtins::fea::contracts::descriptors::ERROR_INPUT;
use crate::builtins::fea::contracts::identities::{LOAD_CASE_NAME, STEP_NAME};
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::{parse_scalar_enum, scalar_string};
use crate::builtins::fea::model::assembly::{load_case_to_object, step_to_object};
use crate::builtins::fea::options_json::{
    json_fields_from_name_values, normalize_token, reject_unknown_fields, remove_optional_f64,
    remove_required_f64, remove_required_vector3,
};
use crate::BuiltinResult;
use runmat_analysis_core::AnalysisStep;
use runmat_analysis_core::AnalysisStepKind;
use runmat_analysis_core::LoadCase;
use runmat_analysis_core::LoadKind;
use runmat_value::Value;

pub(in crate::builtins::fea) fn create_load_case_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            LOAD_CASE_NAME,
            &ERROR_INPUT,
            "fea.loadCase requires id, region, and kind arguments",
        ));
    }
    let load_id = scalar_string(&args[0], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let region_id = scalar_string(&args[1], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[2], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let mut fields = json_fields_from_name_values(LOAD_CASE_NAME, &args[3..])?;
    let kind = match normalize_token(&kind_text).as_str() {
        "force" => {
            let [fx, fy, fz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::Force { fx, fy, fz }
        }
        "moment" | "torque" => {
            let [mx, my, mz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::Moment { mx, my, mz }
        }
        "pressure" => LoadKind::Pressure {
            magnitude_pa: remove_required_f64(&mut fields, LOAD_CASE_NAME, "magnitude_pa")?,
        },
        "bodyforce" => {
            let [gx, gy, gz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::BodyForce { gx, gy, gz }
        }
        "currentdensity" => {
            let [jx, jy, jz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::CurrentDensity {
                jx,
                jy,
                jz,
                phase_rad: remove_optional_f64(&mut fields, LOAD_CASE_NAME, "phase_rad")?
                    .unwrap_or_default(),
                amplitude_scale: remove_optional_f64(
                    &mut fields,
                    LOAD_CASE_NAME,
                    "amplitude_scale",
                )?
                .unwrap_or(1.0),
            }
        }
        "coilcurrent" => LoadKind::CoilCurrent {
            current_a: remove_required_f64(&mut fields, LOAD_CASE_NAME, "current_a")?,
            phase_rad: remove_optional_f64(&mut fields, LOAD_CASE_NAME, "phase_rad")?
                .unwrap_or_default(),
            amplitude_scale: remove_optional_f64(&mut fields, LOAD_CASE_NAME, "amplitude_scale")?
                .unwrap_or(1.0),
        },
        "heatsource" => LoadKind::HeatSource {
            volumetric_w_per_m3: remove_required_f64(
                &mut fields,
                LOAD_CASE_NAME,
                "volumetric_w_per_m3",
            )?,
        },
        other => {
            return Err(builtin_error(
                LOAD_CASE_NAME,
                &ERROR_INPUT,
                format!("unsupported load kind `{other}`"),
            ));
        }
    };
    reject_unknown_fields(LOAD_CASE_NAME, fields)?;
    load_case_to_object(LoadCase {
        load_id,
        region_id,
        kind,
    })
}

pub(in crate::builtins::fea) fn create_step_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(builtin_error(
            STEP_NAME,
            &ERROR_INPUT,
            "fea.step requires exactly id and kind arguments",
        ));
    }
    let step_id = scalar_string(&args[0], STEP_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[1], STEP_NAME, &ERROR_INPUT)?;
    let kind = parse_scalar_enum::<AnalysisStepKind>(&kind_text, "AnalysisStepKind")?;
    step_to_object(AnalysisStep { step_id, kind })
}
