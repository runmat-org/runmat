use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::AnalysisModel;

use crate::{
    FeaElectroThermalContext, FeaThermoMechanicalContext, FeaThermoRegionTemperatureDelta,
};

use super::{
    model::ComplexScalar, RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations,
};

pub(crate) fn realize_coupling_contexts(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
) -> Result<
    (
        Option<FeaThermoMechanicalContext>,
        Option<FeaElectroThermalContext>,
    ),
    String,
> {
    Ok((
        realize_thermo_mechanical_context(model, excitations)?,
        realize_electro_thermal_context(model, excitations)?,
    ))
}

fn realize_thermo_mechanical_context(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
) -> Result<Option<FeaThermoMechanicalContext>, String> {
    let Some(domain) = model.thermo_mechanical.as_ref() else {
        return Ok(None);
    };
    // A declared domain alone couples nothing: structural-only studies may share the model.
    // The prescribed-profile coupling belongs to thermomechanical steps that do not consume a
    // solved thermal state; solved-state preloads receive their temperatures explicitly.
    let prescribed_profile_step = model.sequence.steps.iter().any(|step| {
        step.family
            == runmat_analysis_core::PhysicsFamily::Coupled(
                runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
            )
            && !step
                .state_inputs
                .iter()
                .any(|input| input.required_kind == runmat_analysis_core::StateKind::Thermal)
    });
    if !prescribed_profile_step {
        return Ok(None);
    }
    let reference_temperature_k = domain.reference_temperature.canonical_value();
    if !reference_temperature_k.is_finite() || reference_temperature_k <= 0.0 {
        return Err("thermo-mechanical reference temperature must be finite and positive".into());
    }
    let thermal_materials = model
        .materials
        .iter()
        .map(|material| {
            material
                .require_thermal()
                .map_err(|error| format!("thermo-mechanical material is incomplete: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if thermal_materials.is_empty() {
        return Err("thermo-mechanical analysis requires at least one thermal material".into());
    }
    if thermal_materials.iter().any(|thermal| {
        !thermal
            .expansion_coefficient_per_k
            .canonical_value()
            .is_finite()
            || thermal.expansion_coefficient_per_k.canonical_value() < 0.0
    }) {
        return Err(
            "thermo-mechanical expansion coefficients must be finite and non-negative".into(),
        );
    }
    let thermal_expansion_coefficient = thermal_materials
        .iter()
        .map(|thermal| thermal.expansion_coefficient_per_k.canonical_value())
        .sum::<f64>()
        / thermal_materials.len() as f64;
    let mut whole_model_delta = None;
    let mut region_deltas = BTreeMap::new();
    let model_region_ids = model
        .material_assignments
        .iter()
        .filter_map(|assignment| assignment.selected_region_id())
        .chain(
            model
                .structural
                .iter()
                .flat_map(|structural| structural.elements.iter())
                .map(|element| element.region_id.as_str()),
        )
        .collect::<BTreeSet<_>>();
    for excitation in excitations.iter() {
        let RealizedExcitationKind::TemperatureField(value) = excitation.kind else {
            continue;
        };
        let temperature_k = real_scalar(value, "temperature-field")?;
        let delta = temperature_k - reference_temperature_k;
        match &excitation.target {
            RealizedExcitationTarget::WholeModel => set_unique_temperature(
                &mut whole_model_delta,
                delta,
                "whole-model temperature field",
            )?,
            RealizedExcitationTarget::Region(target) => {
                let region_id = &target.region_id;
                if !model_region_ids.is_empty() && !model_region_ids.contains(region_id.as_str()) {
                    return Err(format!(
                        "temperature field targets unknown model region {region_id}"
                    ));
                }
                if region_deltas
                    .insert(region_id.clone(), (target.clone(), delta))
                    .is_some()
                {
                    return Err(format!(
                        "multiple active temperature fields target region {region_id} at one sample"
                    ));
                }
            }
            RealizedExcitationTarget::ReferencePoint { .. }
            | RealizedExcitationTarget::Port(_)
            | RealizedExcitationTarget::Interface(_) => {
                return Err(
                    "temperature-field coupling supports whole-model and region targets".into(),
                );
            }
        }
    }
    Ok(Some(FeaThermoMechanicalContext {
        reference_temperature_k: runmat_types::physical::Temperature::new(reference_temperature_k),
        applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(
            whole_model_delta.unwrap_or(0.0),
        ),
        thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(
            thermal_expansion_coefficient,
        ),
        region_temperature_deltas: region_deltas
            .into_iter()
            .map(
                |(_, (target, temperature_delta_k))| FeaThermoRegionTemperatureDelta {
                    target,
                    temperature_delta_k,
                },
            )
            .collect(),
    }))
}

fn realize_electro_thermal_context(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
) -> Result<Option<FeaElectroThermalContext>, String> {
    let Some(domain) = model.electro_thermal.as_ref() else {
        return Ok(None);
    };
    if !domain.reference_temperature.canonical_value().is_finite()
        || domain.reference_temperature.canonical_value() <= 0.0
    {
        return Err("electro-thermal reference temperature must be finite and positive".into());
    }
    let electrical_materials = model
        .materials
        .iter()
        .filter_map(|material| material.electrical.as_ref())
        .collect::<Vec<_>>();
    if electrical_materials.iter().any(|material| {
        !material.conductivity_s_per_m.canonical_value().is_finite()
            || material.conductivity_s_per_m.canonical_value() <= 0.0
            || !material
                .resistive_heating_coefficient
                .canonical_value()
                .is_finite()
            || material.resistive_heating_coefficient.canonical_value() < 0.0
    }) {
        return Err(
            "electro-thermal material conductivity must be finite and positive and resistive heating must be finite and non-negative"
                .into(),
        );
    }
    let (base_electrical_conductivity_s_per_m, resistive_heating_coefficient) =
        if electrical_materials.is_empty() {
            (1.0, 0.0)
        } else {
            let count = electrical_materials.len() as f64;
            (
                electrical_materials
                    .iter()
                    .map(|material| material.conductivity_s_per_m.canonical_value().max(1.0e-12))
                    .sum::<f64>()
                    / count,
                electrical_materials
                    .iter()
                    .map(|material| {
                        material
                            .resistive_heating_coefficient
                            .canonical_value()
                            .max(0.0)
                    })
                    .sum::<f64>()
                    / count,
            )
        };
    let mut applied_voltage_v: Option<f64> = None;
    for excitation in excitations.iter() {
        let RealizedExcitationKind::TerminalVoltage(value) = excitation.kind else {
            continue;
        };
        let voltage = real_scalar(value, "terminal-voltage")?;
        if applied_voltage_v.is_none_or(|current| voltage.abs() > current.abs()) {
            applied_voltage_v = Some(voltage);
        }
    }
    Ok(Some(FeaElectroThermalContext {
        reference_temperature_k: runmat_types::physical::Temperature::new(
            domain.reference_temperature.canonical_value(),
        ),
        applied_voltage_v: applied_voltage_v.unwrap_or(0.0),
        base_electrical_conductivity_s_per_m,
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(
            resistive_heating_coefficient,
        ),
    }))
}

fn real_scalar(value: ComplexScalar, label: &str) -> Result<f64, String> {
    if !value.real.is_finite() || !value.imaginary.is_finite() {
        return Err(format!("{label} realization must be finite"));
    }
    if value.imaginary != 0.0 {
        return Err(format!(
            "{label} realization must be real for time-domain coupling"
        ));
    }
    Ok(value.real)
}

fn set_unique_temperature(slot: &mut Option<f64>, value: f64, label: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("multiple active {label}s exist at one sample"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisProcedure, CoupledPhysicsFamily, ProcedureStep, Temperature,
    };

    use crate::fixtures::{append_fixture_temperature_field, fixture_model, FixtureId};

    use super::*;
    use crate::excitation::{realize_step_excitations, ExcitationSamplePoint};

    #[test]
    fn coupling_context_uses_realized_sample_instead_of_authored_extrema() {
        let mut model = fixture_model(FixtureId::ThermoMechanicalKickoff);
        model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        let step = ProcedureStep::coupled(
            "thermomechanical_static",
            CoupledPhysicsFamily::Thermomechanical,
            AnalysisProcedure::Static,
        );
        model.sequence.steps = vec![step.clone()];
        let realized = realize_step_excitations(&model, &step, ExcitationSamplePoint::Static)
            .expect("fixture excitations should realize");
        let (thermo, electro) =
            realize_coupling_contexts(&model, &realized).expect("coupling context should realize");
        assert_eq!(
            thermo
                .unwrap()
                .applied_temperature_delta_k
                .canonical_value(),
            0.0
        );
        assert!(electro.is_none());
    }

    #[test]
    fn coupling_context_interpolates_the_executed_time_sample() {
        let mut model = fixture_model(FixtureId::ThermoMechanicalKickoff);
        let reference_temperature_k = 293.15;
        model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
            reference_temperature: Temperature::new(reference_temperature_k),
        });
        append_fixture_temperature_field(
            &mut model,
            "temperature_history",
            reference_temperature_k,
            runmat_analysis_core::ExcitationTarget::WholeModel,
        );
        let runmat_analysis_core::PhysicalSignal::Temperature(signal) =
            model.signals.last_mut().expect("temperature signal")
        else {
            panic!("fixture helper must append a temperature signal");
        };
        signal.representation = runmat_analysis_core::SignalRepresentation::Tabulated(
            runmat_analysis_core::TabulatedSignal {
                domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
                knots: [(0.0, 10.0), (1.0, 50.0)]
                    .into_iter()
                    .map(|(time_s, delta_k)| {
                        runmat_analysis_core::SignalKnot::continuous(
                            time_s,
                            runmat_analysis_core::SignalOrdinate::Real(
                                runmat_analysis_core::SignalValue::Scalar(Temperature::new(
                                    reference_temperature_k + delta_k,
                                )),
                            ),
                        )
                    })
                    .collect(),
                interpolation: runmat_analysis_core::SignalInterpolation::Linear,
                extrapolation: runmat_analysis_core::SignalExtrapolation::Reject,
            },
        );
        signal.evidence.sample_count = 2;
        let step = ProcedureStep::coupled(
            "thermomechanical_transient",
            CoupledPhysicsFamily::Thermomechanical,
            AnalysisProcedure::TransientResponse,
        );
        model.sequence.steps = vec![step.clone()];
        let realized = realize_step_excitations(&model, &step, ExcitationSamplePoint::time(0.25))
            .expect("temperature history should interpolate");

        let (thermo, _) =
            realize_coupling_contexts(&model, &realized).expect("coupling context should realize");

        assert!(
            (thermo
                .unwrap()
                .applied_temperature_delta_k
                .canonical_value()
                - 20.0)
                .abs()
                < 1.0e-12
        );
    }
}
