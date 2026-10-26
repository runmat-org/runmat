//! Retained source/physical-work consistency. Never reconstruct an electric field.
use super::*;
use crate::excitation::ProjectionSample;
use runmat_analysis_core::{PhysicalQuantityKind, StaticElectricalEquation};

pub(super) fn validate(r: &CanonicalElectricalResponse) -> AnalysisExecutionResult<()> {
    if let Some(source) = &r.charge_sources {
        let quantity = match r.equation {
            StaticElectricalEquation::Electrostatics => PhysicalQuantityKind::ElectricChargeDensity,
            StaticElectricalEquation::SteadyConduction => PhysicalQuantityKind::ElectricCurrent,
        };
        if source.nodal_source.len() != r.ordered_node_ids.len()
            || source.nodal_source.iter().any(|value| !value.is_finite())
            || !source.integrated_source.is_finite()
            || !source.absolute_source.is_finite()
            || source.absolute_source < 0.0
            || source.absolute_source < source.integrated_source.abs()
        {
            return Err(invalid("invalid retained electrical source"));
        }
        let sum = source.nodal_source.iter().sum::<f64>();
        let scale = source
            .nodal_source
            .iter()
            .map(|v| v.abs())
            .sum::<f64>()
            .max(source.absolute_source)
            .max(f64::MIN_POSITIVE);
        if (sum - source.integrated_source).abs()
            > 128.0 * f64::EPSILON * r.ordered_node_ids.len() as f64 * scale
        {
            return Err(invalid(
                "retained nodal charge disagrees with source integral",
            ));
        }
        for evidence in &source.projection_evidence {
            evidence.validate().map_err(invalid)?;
            if !evidence.accepted
                || evidence.quantity != quantity
                || evidence.sample != ProjectionSample::Static
            {
                return Err(invalid(
                    "invalid retained electrical source projection evidence",
                ));
            }
        }
    }
    let volume_work = r.charge_sources.as_ref().map_or(0.0, |source| {
        r.potential_increment_v
            .canonical_values()
            .iter()
            .zip(&source.nodal_source)
            .map(|(potential, charge)| potential * charge)
            .sum::<f64>()
    });
    let increments = r
        .ordered_node_ids
        .iter()
        .copied()
        .zip(r.potential_increment_v.canonical_values().iter().copied())
        .collect::<std::collections::BTreeMap<_, _>>();
    let outward = r
        .ordered_node_ids
        .iter()
        .copied()
        .zip(r.nodal_outward_flux.canonical_values().iter().copied())
        .collect::<std::collections::BTreeMap<_, _>>();
    let terminal_work = r
        .terminals
        .iter()
        .flat_map(|terminal| &terminal.ordered_node_ids)
        .map(|node| -increments[node] * outward[node])
        .sum::<f64>();
    let work_scale = volume_work
        .abs()
        .max(terminal_work.abs())
        .max(r.volume_source_work.canonical_value().abs())
        .max(r.terminal_work.canonical_value().abs())
        .max(f64::MIN_POSITIVE);
    let bound = 128.0 * f64::EPSILON * r.ordered_node_ids.len() as f64 * work_scale;
    if !volume_work.is_finite()
        || !terminal_work.is_finite()
        || !r.volume_source_work.canonical_value().is_finite()
        || !r.terminal_work.canonical_value().is_finite()
        || (volume_work - r.volume_source_work.canonical_value()).abs() > bound
        || (terminal_work - r.terminal_work.canonical_value()).abs() > bound
    {
        return Err(invalid(
            "retained source/terminal work disagrees with gauge-departure arrays",
        ));
    }
    let energy = match r.constitutive {
        ElectricalConstitutiveResponse::Electrostatics { total_energy_j, .. } => {
            2.0 * total_energy_j.canonical_value()
        }
        ElectricalConstitutiveResponse::SteadyConduction {
            total_joule_power_w,
            ..
        } => total_joule_power_w.canonical_value(),
    };
    let work = r.volume_source_work.canonical_value() + r.terminal_work.canonical_value();
    let error = (energy - work).abs() / energy.abs().max(work.abs()).max(f64::MIN_POSITIVE);
    if error != r.relative_energy_identity_error {
        return Err(invalid(
            "electrical energy closure disagrees with source/terminal work",
        ));
    }
    Ok(())
}
