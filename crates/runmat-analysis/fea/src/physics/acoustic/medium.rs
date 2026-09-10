use runmat_analysis_core::{AnalysisModel, BoundaryOperator};

#[derive(Debug, Clone, Copy)]
pub(super) struct AcousticMaterialSummary {
    pub density_kg_per_m3: f64,
    pub speed_of_sound_m_per_s: f64,
    pub damping_ratio: f64,
    pub explicit_material_count: usize,
    pub coverage_ratio: f64,
}

impl AcousticMaterialSummary {
    pub(super) fn characteristic_impedance_pa_s_per_m(self) -> f64 {
        (self.density_kg_per_m3 * self.speed_of_sound_m_per_s).max(1.0)
    }
}

pub(super) fn material_summary(
    model: &AnalysisModel,
    mode_count: usize,
) -> AcousticMaterialSummary {
    let mut density_sum = 0.0;
    let mut speed_sum = 0.0;
    let mut damping_sum = 0.0;
    let mut explicit_material_count = 0usize;
    for material in &model.materials {
        let Some(acoustic) = &material.acoustic else {
            continue;
        };
        density_sum += acoustic.density_kg_per_m3.max(1.0e-9);
        speed_sum += acoustic.speed_of_sound_m_per_s.max(1.0);
        damping_sum += acoustic.damping_ratio.max(0.0);
        explicit_material_count += 1;
    }
    if explicit_material_count == 0 {
        let reference_temperature_k = reference_temperature_k(model);
        return AcousticMaterialSummary {
            density_kg_per_m3: 1.225 * (293.15 / reference_temperature_k.max(1.0)),
            speed_of_sound_m_per_s: 331.3 * (reference_temperature_k / 273.15).sqrt(),
            damping_ratio: 0.02 + 0.002 * mode_count.saturating_sub(1).min(12) as f64,
            explicit_material_count,
            coverage_ratio: 0.0,
        };
    }
    let inv_count = 1.0 / explicit_material_count as f64;
    AcousticMaterialSummary {
        density_kg_per_m3: density_sum * inv_count,
        speed_of_sound_m_per_s: speed_sum * inv_count,
        damping_ratio: damping_sum * inv_count,
        explicit_material_count,
        coverage_ratio: explicit_material_count as f64 / model.materials.len().max(1) as f64,
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AcousticBoundarySummary {
    pub acoustic_boundary_count: usize,
    pub rigid_wall_count: usize,
    pub radiation_boundary_count: usize,
    pub impedance_boundary_count: usize,
    pub coverage_ratio: f64,
    pub mean_specific_impedance_pa_s_per_m: f64,
    pub radiation_loss_factor: f64,
    pub impedance_loss_factor: f64,
}

impl AcousticBoundarySummary {
    pub(super) fn has_acoustic_boundary_data(self) -> bool {
        self.acoustic_boundary_count > 0
    }
}

pub(super) fn boundary_summary(
    model: &AnalysisModel,
    characteristic_impedance_pa_s_per_m: f64,
) -> AcousticBoundarySummary {
    let mut rigid_wall_count = 0usize;
    let mut radiation_boundary_count = 0usize;
    let mut impedance_boundary_count = 0usize;
    let mut impedance_sum = 0.0;
    for boundary in &model.boundaries {
        match &boundary.operator {
            BoundaryOperator::AcousticRigidWall => rigid_wall_count += 1,
            BoundaryOperator::AcousticRadiation => radiation_boundary_count += 1,
            BoundaryOperator::AcousticImpedance { impedance } => {
                impedance_boundary_count += 1;
                impedance_sum += impedance.pascal_seconds_per_meter.max(1.0);
            }
            _ => {}
        }
    }
    let acoustic_boundary_count =
        rigid_wall_count + radiation_boundary_count + impedance_boundary_count;
    let mean_specific_impedance_pa_s_per_m = if impedance_boundary_count == 0 {
        characteristic_impedance_pa_s_per_m
    } else {
        impedance_sum / impedance_boundary_count as f64
    };
    let impedance_ratio =
        characteristic_impedance_pa_s_per_m / mean_specific_impedance_pa_s_per_m.max(1.0);
    AcousticBoundarySummary {
        acoustic_boundary_count,
        rigid_wall_count,
        radiation_boundary_count,
        impedance_boundary_count,
        coverage_ratio: acoustic_boundary_count as f64 / model.boundaries.len().max(1) as f64,
        mean_specific_impedance_pa_s_per_m,
        radiation_loss_factor: 0.05 * radiation_boundary_count as f64,
        impedance_loss_factor: 0.025
            * impedance_boundary_count as f64
            * impedance_ratio.clamp(0.1, 10.0),
    }
}

fn reference_temperature_k(model: &AnalysisModel) -> f64 {
    if model.materials.is_empty() {
        293.15
    } else {
        model
            .materials
            .iter()
            .map(|material| material.thermal.reference_temperature_k.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    }
}
