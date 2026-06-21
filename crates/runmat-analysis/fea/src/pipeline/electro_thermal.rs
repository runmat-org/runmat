use runmat_analysis_core::AnalysisField;

use crate::{
    assembly::ElectroThermalAssemblySummary,
    contracts::{
        fea_electro_thermal_temperature_field_id, fea_electro_thermal_thermal_residual_field_id,
        FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY, FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
        FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL, FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT,
    },
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const MIN_CONDUCTIVITY: f64 = 1.0e-12;

#[derive(Default)]
pub(crate) struct ElectroThermalFields {
    pub(crate) static_fields: Vec<AnalysisField>,
    pub(crate) temperature_snapshots: Vec<AnalysisField>,
    pub(crate) thermal_residual_snapshots: Vec<AnalysisField>,
    pub(crate) diagnostics: Vec<FeaDiagnostic>,
}

#[derive(Debug)]
struct ElectroThermalPotentialSolve {
    potential: Vec<f64>,
    segment_conductance: Vec<f64>,
    segment_current: Vec<f64>,
    residual_norm: f64,
    equation_scale: f64,
    current_balance_residual: f64,
    potential_span_v: f64,
    integrated_joule_heat_w: f64,
    condition_estimate: f64,
}

pub(crate) fn recover_electro_thermal_fields(
    summary: Option<&ElectroThermalAssemblySummary>,
    progress_factors: &[f64],
    residual_norms: &[f64],
    dof_count: usize,
) -> ElectroThermalFields {
    let Some(summary) = summary.filter(|summary| summary.enabled) else {
        return ElectroThermalFields::default();
    };

    let node_count = dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1);
    let solve = solve_scalar_potential(summary, node_count);
    let heating_scale = if solve.integrated_joule_heat_w > 1.0e-12 {
        summary.joule_heating_scale.max(0.0) / solve.integrated_joule_heat_w
    } else {
        0.0
    };

    let mut electric_field = Vec::with_capacity(node_count * VECTOR_COMPONENT_COUNT);
    let mut current_density = Vec::with_capacity(node_count * VECTOR_COMPONENT_COUNT);
    let mut joule_heat = Vec::with_capacity(node_count);
    for index in 0..node_count {
        let left_segment = index
            .saturating_sub(1)
            .min(solve.segment_current.len().saturating_sub(1));
        let right_segment = index.min(solve.segment_current.len().saturating_sub(1));
        let current = match solve.segment_current.len() {
            0 => 0.0,
            _ if index == 0 => solve.segment_current[0],
            len if index >= len => solve.segment_current[len - 1],
            _ => 0.5 * (solve.segment_current[left_segment] + solve.segment_current[right_segment]),
        };
        let conductance = match solve.segment_conductance.len() {
            0 => MIN_CONDUCTIVITY,
            _ if index == 0 => solve.segment_conductance[0],
            len if index >= len => solve.segment_conductance[len - 1],
            _ => {
                0.5 * (solve.segment_conductance[left_segment]
                    + solve.segment_conductance[right_segment])
            }
        };
        let local_field = if conductance.abs() > MIN_CONDUCTIVITY {
            current / conductance
        } else {
            0.0
        };
        let local_joule_heat = current * current / conductance.max(MIN_CONDUCTIVITY);

        electric_field.extend_from_slice(&[local_field, 0.0, 0.0]);
        current_density.extend_from_slice(&[current, 0.0, 0.0]);
        joule_heat.push((local_joule_heat * heating_scale).max(0.0));
    }

    let diagnostics = vec![electro_thermal_potential_solve_diagnostic(
        node_count, &solve,
    )];

    let static_fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL,
            vec![node_count],
            solve.potential,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            electric_field,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            current_density,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT,
            vec![node_count],
            joule_heat,
        ),
    ];

    let mut temperature_snapshots = Vec::with_capacity(progress_factors.len());
    let mut thermal_residual_snapshots = Vec::with_capacity(progress_factors.len());
    for (index, progress_factor) in progress_factors.iter().copied().enumerate() {
        let current_factor = progress_factor.clamp(0.0, 1.0);
        let temperature_rise = summary.joule_heating_scale.max(0.0)
            * current_factor
            * (1.0 + summary.temporal_profile_variation);
        temperature_snapshots.push(AnalysisField::host_f64(
            fea_electro_thermal_temperature_field_id(index),
            vec![node_count],
            vec![summary.reference_temperature_k + temperature_rise; node_count],
        ));

        let residual = residual_norms
            .get(index)
            .copied()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0)
            * (1.0 + summary.conductivity_spread_ratio.ln().max(0.0));
        thermal_residual_snapshots.push(AnalysisField::host_f64(
            fea_electro_thermal_thermal_residual_field_id(index),
            vec![1],
            vec![residual],
        ));
    }

    ElectroThermalFields {
        static_fields,
        temperature_snapshots,
        thermal_residual_snapshots,
        diagnostics,
    }
}

fn solve_scalar_potential(
    summary: &ElectroThermalAssemblySummary,
    node_count: usize,
) -> ElectroThermalPotentialSolve {
    let node_count = node_count.max(1);
    let potential_span_v = summary.applied_voltage_v.abs();
    if node_count == 1 {
        return ElectroThermalPotentialSolve {
            potential: vec![summary.applied_voltage_v],
            segment_conductance: Vec::new(),
            segment_current: Vec::new(),
            residual_norm: 0.0,
            equation_scale: potential_span_v.max(1.0),
            current_balance_residual: 0.0,
            potential_span_v,
            integrated_joule_heat_w: 0.0,
            condition_estimate: 1.0,
        };
    }

    let segment_count = node_count - 1;
    let segment_conductance = conductivity_profile(summary, segment_count);
    let resistance_sum = segment_conductance
        .iter()
        .map(|conductance| 1.0 / conductance.max(MIN_CONDUCTIVITY))
        .sum::<f64>()
        .max(MIN_CONDUCTIVITY);
    let signed_current = summary.applied_voltage_v / resistance_sum;
    let mut potential = Vec::with_capacity(node_count);
    let mut cursor = summary.applied_voltage_v;
    potential.push(cursor);
    for conductance in &segment_conductance {
        cursor -= signed_current / conductance.max(MIN_CONDUCTIVITY);
        potential.push(cursor);
    }
    if let Some(last) = potential.last_mut() {
        *last = 0.0;
    }

    let segment_current = segment_conductance
        .iter()
        .enumerate()
        .map(|(index, conductance)| {
            let voltage_drop = potential[index] - potential[index + 1];
            conductance * voltage_drop
        })
        .collect::<Vec<_>>();

    let equation_scale = segment_current
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(potential_span_v)
        .max(1.0);
    let max_residual = segment_current
        .windows(2)
        .map(|pair| (pair[0] - pair[1]).abs())
        .fold(0.0_f64, f64::max);
    let residual_norm = max_residual / equation_scale;
    let current_balance_residual =
        if let (Some(first), Some(last)) = (segment_current.first(), segment_current.last()) {
            (first - last).abs() / first.abs().max(last.abs()).max(1.0e-12)
        } else {
            0.0
        };
    let integrated_joule_heat_w = segment_current
        .iter()
        .zip(segment_conductance.iter())
        .map(|(current, conductance)| current * current / conductance.max(MIN_CONDUCTIVITY))
        .sum::<f64>()
        .abs();
    let (min_conductance, max_conductance) = segment_conductance
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(min_value, max_value), value| {
            (min_value.min(*value), max_value.max(*value))
        });
    let condition_estimate = if min_conductance.is_finite() && min_conductance > 0.0 {
        (max_conductance / min_conductance).max(1.0)
    } else {
        1.0
    };

    ElectroThermalPotentialSolve {
        potential,
        segment_conductance,
        segment_current,
        residual_norm,
        equation_scale,
        current_balance_residual,
        potential_span_v,
        integrated_joule_heat_w,
        condition_estimate,
    }
}

fn conductivity_profile(summary: &ElectroThermalAssemblySummary, segment_count: usize) -> Vec<f64> {
    let base = summary
        .base_electrical_conductivity_s_per_m
        .max(MIN_CONDUCTIVITY);
    let spread = summary.conductivity_spread_ratio.max(1.0);
    let centered_spread = spread.sqrt();
    (0..segment_count)
        .map(|index| {
            let xi = if segment_count <= 1 {
                0.5
            } else {
                index as f64 / (segment_count - 1) as f64
            };
            let profile = 1.0 / centered_spread + (centered_spread - 1.0 / centered_spread) * xi;
            let temporal_adjustment = 1.0 + summary.temporal_profile_variation * (0.5 - xi) * 0.1;
            (base * profile * temporal_adjustment).max(MIN_CONDUCTIVITY)
        })
        .collect()
}

fn electro_thermal_potential_solve_diagnostic(
    node_count: usize,
    solve: &ElectroThermalPotentialSolve,
) -> FeaDiagnostic {
    let severity = if solve.residual_norm <= 1.0e-10 && solve.current_balance_residual <= 1.0e-10 {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    FeaDiagnostic {
        code: "FEA_ET_POTENTIAL_SOLVE".to_string(),
        severity,
        message: format!(
            "basis=resistor_network node_count={} segment_count={} potential_span_v={} residual_norm={} equation_scale={} current_balance_residual={} integrated_joule_heat_w={} condition_estimate={}",
            node_count,
            solve.segment_conductance.len(),
            solve.potential_span_v,
            solve.residual_norm,
            solve.equation_scale,
            solve.current_balance_residual,
            solve.integrated_joule_heat_w,
            solve.condition_estimate,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary() -> ElectroThermalAssemblySummary {
        ElectroThermalAssemblySummary {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.8e7,
            resistive_heating_coefficient: 3.5e-4,
            joule_heating_scale: 10.0,
            conductivity_spread_ratio: 1.08,
            temporal_profile_variation: 0.15,
            region_scale_count: 2,
            coupling_fingerprint: 42,
        }
    }

    #[test]
    fn scalar_potential_solve_balances_current() {
        let solve = solve_scalar_potential(&summary(), 16);

        assert!((solve.potential[0] - 36.0).abs() <= 1.0e-9);
        assert!(solve.potential.last().copied().unwrap_or(1.0).abs() <= 1.0e-12);
        assert!(solve.residual_norm <= 1.0e-10);
        assert!(solve.current_balance_residual <= 1.0e-10);
        assert!(solve.integrated_joule_heat_w > 0.0);
    }

    #[test]
    fn recovered_fields_use_potential_solve_diagnostic() {
        let fields = recover_electro_thermal_fields(Some(&summary()), &[1.0], &[0.01], 48);

        assert_eq!(fields.static_fields.len(), 4);
        assert_eq!(fields.diagnostics.len(), 1);
        assert_eq!(fields.diagnostics[0].code, "FEA_ET_POTENTIAL_SOLVE");
        assert!(fields.diagnostics[0]
            .message
            .contains("current_balance_residual="));
    }
}
