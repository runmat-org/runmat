use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::fea::contracts::descriptors::ERROR_INPUT;
use crate::builtins::fea::contracts::identities::BOUNDARY_CONDITION_NAME;
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::{parse_scalar_enum_for_builtin, scalar_string};
use crate::builtins::fea::model::assembly::boundary_condition_to_object;
use crate::builtins::fea::options_json::{
    canonical_field_name, expect_name_value_tail, normalize_token,
};
use crate::BuiltinResult;
use runmat_analysis_core::BoundaryCondition;
use runmat_analysis_core::BoundaryConditionKind;
use runmat_value::IntValue;
use runmat_value::NumericScalar;
use runmat_value::Value;
use std::collections::HashMap;

pub(in crate::builtins::fea) fn create_boundary_condition_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            "fea.boundaryCondition requires id, region, and kind arguments",
        ));
    }
    let bc_id = scalar_string(&args[0], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let region_id = scalar_string(&args[1], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[2], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let mut fields = boundary_fields_from_name_values(&args[3..])?;
    let kind = match normalize_token(&kind_text).as_str() {
        "prescribedrotation" => BoundaryConditionKind::PrescribedRotation {
            rx: remove_required_boundary_f64(&mut fields, "rx")?,
            ry: remove_required_boundary_f64(&mut fields, "ry")?,
            rz: remove_required_boundary_f64(&mut fields, "rz")?,
        },
        "acousticimpedance" => BoundaryConditionKind::AcousticImpedance {
            specific_impedance_pa_s_per_m: remove_required_boundary_f64(
                &mut fields,
                "specific_impedance_pa_s_per_m",
            )?,
        },
        "thermalprescribedtemperature" => BoundaryConditionKind::ThermalPrescribedTemperature {
            temperature_k: remove_required_boundary_f64(&mut fields, "temperature_k")?,
        },
        "thermalheatflux" => BoundaryConditionKind::ThermalHeatFlux {
            heat_flux_w_per_m2: remove_required_boundary_f64(&mut fields, "heat_flux_w_per_m2")?,
        },
        "thermalconvection" => BoundaryConditionKind::ThermalConvection {
            ambient_temperature_k: remove_required_boundary_f64(
                &mut fields,
                "ambient_temperature_k",
            )?,
            coefficient_w_per_m2k: remove_required_boundary_f64(
                &mut fields,
                "coefficient_w_per_m2k",
            )?,
        },
        "cfdinletvelocity" => BoundaryConditionKind::CfdInletVelocity {
            velocity_m_per_s: remove_required_boundary_f64(&mut fields, "velocity_m_per_s")?,
        },
        "cfdoutletpressure" => BoundaryConditionKind::CfdOutletPressure {
            pressure_pa: remove_required_boundary_f64(&mut fields, "pressure_pa")?,
        },
        _ => parse_scalar_enum_for_builtin::<BoundaryConditionKind>(
            BOUNDARY_CONDITION_NAME,
            &kind_text,
            "BoundaryConditionKind",
        )?,
    };
    reject_unknown_boundary_fields(fields)?;
    boundary_condition_to_object(BoundaryCondition {
        bc_id,
        region_id,
        kind,
    })
}

pub(in crate::builtins::fea) fn boundary_fields_from_name_values(
    args: &[Value],
) -> BuiltinResult<HashMap<String, &Value>> {
    let mut fields = HashMap::new();
    for pair in expect_name_value_tail(BOUNDARY_CONDITION_NAME, args)? {
        let raw = scalar_string(pair.name, BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
        let key = canonical_field_name(&raw);
        if fields.insert(key.clone(), pair.value).is_some() {
            return Err(builtin_error(
                BOUNDARY_CONDITION_NAME,
                &ERROR_INPUT,
                format!("duplicate fea.boundaryCondition option `{key}`"),
            ));
        }
    }
    Ok(fields)
}

pub(in crate::builtins::fea) fn remove_required_boundary_f64(
    fields: &mut HashMap<String, &Value>,
    key: &str,
) -> BuiltinResult<f64> {
    let Some(value) = fields.remove(key) else {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            format!("missing required option `{key}`"),
        ));
    };
    boundary_numeric_scalar_f64(value, key)
}

pub(in crate::builtins::fea) fn boundary_numeric_scalar_f64(
    value: &Value,
    key: &str,
) -> BuiltinResult<f64> {
    let converted = match value {
        Value::Num(value) => *value,
        Value::Int(value) => boundary_integer_to_f64(value),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            boundary_numeric_storage_scalar_to_f64(
                tensor
                    .numeric_value_at(0)
                    .expect("validated scalar tensor storage"),
            )
        }
        _ => {
            return Err(builtin_error(
                BOUNDARY_CONDITION_NAME,
                &ERROR_INPUT,
                format!("numeric option `{key}` must be a real numeric scalar"),
            ))
        }
    };
    if !converted.is_finite() {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            format!("numeric option `{key}` must be finite"),
        ));
    }
    Ok(converted)
}

pub(in crate::builtins::fea) fn boundary_integer_to_f64(value: &IntValue) -> f64 {
    match value {
        IntValue::I8(value) => f64::from(*value),
        IntValue::I16(value) => f64::from(*value),
        IntValue::I32(value) => f64::from(*value),
        IntValue::I64(value) => *value as f64,
        IntValue::U8(value) => f64::from(*value),
        IntValue::U16(value) => f64::from(*value),
        IntValue::U32(value) => f64::from(*value),
        IntValue::U64(value) => *value as f64,
    }
}

pub(in crate::builtins::fea) fn boundary_numeric_storage_scalar_to_f64(
    value: NumericScalar,
) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        NumericScalar::I8(value) => f64::from(value),
        NumericScalar::I16(value) => f64::from(value),
        NumericScalar::I32(value) => f64::from(value),
        NumericScalar::I64(value) => value as f64,
        NumericScalar::U8(value) => f64::from(value),
        NumericScalar::U16(value) => f64::from(value),
        NumericScalar::U32(value) => f64::from(value),
        NumericScalar::U64(value) => value as f64,
    }
}

pub(in crate::builtins::fea) fn reject_unknown_boundary_fields(
    fields: HashMap<String, &Value>,
) -> BuiltinResult<()> {
    if let Some(key) = fields.keys().next() {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            format!("unsupported fea.boundaryCondition option `{key}`"),
        ));
    }
    Ok(())
}
