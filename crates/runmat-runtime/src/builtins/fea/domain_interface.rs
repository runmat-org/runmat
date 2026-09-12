use crate::builtins::fea::contracts::descriptors::ERROR_INPUT;
use crate::builtins::fea::contracts::identities::{DOMAIN_NAME, INTERFACE_NAME};
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::scalar_string;
use crate::builtins::fea::model::assembly::{domain_to_object, interface_to_object};
use crate::builtins::fea::options_json::{
    canonical_field_name, expect_name_value_tail, json_deserialize, json_fields_from_name_values,
    json_with_overrides, normalize_token, typed_domain_data, value_to_json,
};
use crate::builtins::fea::study::DomainPayload;
use crate::BuiltinResult;
use runmat_analysis_core::AnalysisInterface;
use runmat_analysis_core::AnalysisInterfaceKind;
use runmat_analysis_core::CfdDomain;
use runmat_analysis_core::ElectroThermalDomain;
use runmat_analysis_core::ElectromagneticDomain;
use runmat_analysis_core::ThermoMechanicalDomain;
use runmat_value::Value;

pub(in crate::builtins::fea) fn create_domain_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            DOMAIN_NAME,
            &ERROR_INPUT,
            "fea.domain requires a domain kind",
        ));
    }
    let kind_text = scalar_string(&args[0], DOMAIN_NAME, &ERROR_INPUT)?;
    let kind = normalize_token(&kind_text);
    let fields = json_fields_from_name_values(DOMAIN_NAME, &args[1..])?;
    let payload = match kind.as_str() {
        "thermomechanical" => DomainPayload {
            kind: "thermo_mechanical".to_string(),
            data: typed_domain_data::<ThermoMechanicalDomain>(
                DOMAIN_NAME,
                "thermo_mechanical domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_temperature_k": 293.15,
                        "applied_temperature_delta_k": 0.0,
                        "field_artifact_id": null,
                        "field_source": null,
                        "region_temperature_deltas": [],
                        "time_profile": []
                    }),
                    fields,
                    "thermo_mechanical domain",
                )?,
            )?,
        },
        "electrothermal" => DomainPayload {
            kind: "electro_thermal".to_string(),
            data: typed_domain_data::<ElectroThermalDomain>(
                DOMAIN_NAME,
                "electro_thermal domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_temperature_k": 293.15,
                        "applied_voltage_v": 0.0,
                        "region_conductivity_scales": [],
                        "time_profile": []
                    }),
                    fields,
                    "electro_thermal domain",
                )?,
            )?,
        },
        "electromagnetic" => DomainPayload {
            kind: "electromagnetic".to_string(),
            data: typed_domain_data::<ElectromagneticDomain>(
                DOMAIN_NAME,
                "electromagnetic domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_frequency_hz": 0.0,
                        "applied_current_a": 0.0
                    }),
                    fields,
                    "electromagnetic domain",
                )?,
            )?,
        },
        "cfd" => DomainPayload {
            kind: "cfd".to_string(),
            data: typed_domain_data::<CfdDomain>(
                DOMAIN_NAME,
                "cfd domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "solve_family": "steady_state",
                        "reference_density_kg_per_m3": 1.225,
                        "dynamic_viscosity_pa_s": 1.8e-5,
                        "inlet_velocity_m_per_s": 0.0,
                        "turbulence_intensity": 0.0,
                        "time_profile": []
                    }),
                    fields,
                    "cfd domain",
                )?,
            )?,
        },
        other => {
            return Err(builtin_error(
                DOMAIN_NAME,
                &ERROR_INPUT,
                format!("unsupported FEA domain kind `{other}`"),
            ));
        }
    };
    domain_to_object(payload)
}

pub(in crate::builtins::fea) fn create_interface_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            INTERFACE_NAME,
            &ERROR_INPUT,
            "fea.interface requires id, primary region, and secondary region arguments",
        ));
    }
    let interface_id = scalar_string(&args[0], INTERFACE_NAME, &ERROR_INPUT)?;
    let primary_region_id = scalar_string(&args[1], INTERFACE_NAME, &ERROR_INPUT)?;
    let secondary_region_id = scalar_string(&args[2], INTERFACE_NAME, &ERROR_INPUT)?;
    let mut kind = "contact".to_string();
    let mut kind_seen = false;
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(INTERFACE_NAME, &args[3..])? {
        if pair.key == "kind" {
            if kind_seen {
                return Err(builtin_error(
                    INTERFACE_NAME,
                    &ERROR_INPUT,
                    "duplicate fea.interface option `kind`",
                ));
            }
            kind_seen = true;
            kind = scalar_string(pair.value, INTERFACE_NAME, &ERROR_INPUT)?;
        } else {
            let key =
                canonical_field_name(&scalar_string(pair.name, INTERFACE_NAME, &ERROR_INPUT)?);
            if fields
                .insert(key.clone(), value_to_json(INTERFACE_NAME, pair.value)?)
                .is_some()
            {
                return Err(builtin_error(
                    INTERFACE_NAME,
                    &ERROR_INPUT,
                    format!("duplicate fea.interface option `{key}`"),
                ));
            }
        }
    }
    let kind = match normalize_token(&kind).as_str() {
        "contact" => AnalysisInterfaceKind::Contact(json_deserialize(
            INTERFACE_NAME,
            json_with_overrides(
                INTERFACE_NAME,
                serde_json::json!({
                    "penalty_stiffness_scale": 1.0,
                    "max_penetration_ratio": 0.0,
                    "friction_coefficient": 0.0
                }),
                fields,
                "contact interface",
            )?,
            "contact interface",
        )?),
        "fluid_structure" | "fluidstructure" | "fsi" => {
            AnalysisInterfaceKind::FluidStructure(json_deserialize(
                INTERFACE_NAME,
                json_with_overrides(
                    INTERFACE_NAME,
                    serde_json::json!({
                        "normal_stiffness_pa_per_m": 1.0e9,
                        "damping_ratio": 0.0,
                        "relaxation_factor": 0.5
                    }),
                    fields,
                    "fluid-structure interface",
                )?,
                "fluid-structure interface",
            )?)
        }
        "conjugate_heat_transfer" | "conjugateheattransfer" | "cht" => {
            AnalysisInterfaceKind::ConjugateHeatTransfer(json_deserialize(
                INTERFACE_NAME,
                json_with_overrides(
                    INTERFACE_NAME,
                    serde_json::json!({
                        "thermal_conductance_w_per_m2k": 500.0,
                        "contact_resistance_m2k_per_w": 0.0,
                        "relaxation_factor": 0.5
                    }),
                    fields,
                    "conjugate heat-transfer interface",
                )?,
                "conjugate heat-transfer interface",
            )?)
        }
        other => {
            return Err(builtin_error(
                INTERFACE_NAME,
                &ERROR_INPUT,
                format!("unsupported interface kind `{other}`"),
            ));
        }
    };
    interface_to_object(AnalysisInterface {
        interface_id,
        primary_region_id,
        secondary_region_id,
        kind,
    })
}
