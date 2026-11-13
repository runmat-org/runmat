use runmat_analysis_core::{AnalysisModel, AnalysisProcedure, PhysicsFamily};

use super::medium::AcousticBoundarySummary;

#[derive(Debug, Clone, Copy)]
pub(super) struct AcousticDomainTopology {
    pub node_count: usize,
    pub dims: [usize; 3],
    pub spacing: [f64; 3],
    pub edge_count: usize,
    pub boundary_node_count: usize,
    pub active_dimension_count: usize,
}

impl AcousticDomainTopology {
    pub(super) fn coords(self, index: usize) -> [usize; 3] {
        let x_dim = self.dims[0].max(1);
        let y_dim = self.dims[1].max(1);
        let plane = x_dim.saturating_mul(y_dim).max(1);
        let z = index / plane;
        let rem = index % plane;
        [rem % x_dim, rem / x_dim, z]
    }

    fn index(self, coords: [usize; 3]) -> Option<usize> {
        if coords
            .iter()
            .zip(self.dims.iter())
            .any(|(coord, dim)| *coord >= *dim)
        {
            return None;
        }
        let index = coords[0]
            + coords[1].saturating_mul(self.dims[0])
            + coords[2].saturating_mul(self.dims[0].saturating_mul(self.dims[1]));
        (index < self.node_count).then_some(index)
    }

    pub(super) fn is_boundary_node(self, index: usize) -> bool {
        let coords = self.coords(index);
        coords
            .iter()
            .zip(self.dims.iter())
            .any(|(coord, dim)| *dim > 1 && (*coord == 0 || *coord + 1 == *dim))
    }

    pub(super) fn average_node_degree(self) -> f64 {
        if self.node_count == 0 {
            0.0
        } else {
            2.0 * self.edge_count as f64 / self.node_count as f64
        }
    }

    pub(super) fn volume_m3(self) -> f64 {
        self.spacing
            .iter()
            .zip(self.dims.iter())
            .map(|(spacing, dim)| spacing * dim.saturating_sub(1).max(1) as f64)
            .product::<f64>()
            .max(1.0e-12)
    }

    pub(super) fn neighbor(self, index: usize, axis: usize, offset: isize) -> Option<usize> {
        let mut coords = self.coords(index);
        coords[axis] = coords[axis].checked_add_signed(offset)?;
        self.index(coords)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AcousticGraphEdge {
    pub left: usize,
    pub right: usize,
    pub stiffness: f64,
}

#[derive(Debug, Clone)]
pub(super) struct AcousticDomainSystem {
    pub diag_real: Vec<f64>,
    pub diag_imag: Vec<f64>,
    pub edges: Vec<AcousticGraphEdge>,
}

pub(super) fn node_count(model: &AnalysisModel) -> usize {
    model.excitations.len().saturating_mul(3).clamp(3, 512)
}

pub(super) fn drive_frequency_hz(mode_count: usize, node_count: usize) -> f64 {
    (125.0 * mode_count.max(1) as f64 * (node_count as f64).sqrt()).clamp(50.0, 20_000.0)
}

pub(super) fn sweep_frequencies_hz(drive_frequency_hz: f64) -> Vec<f64> {
    let mut frequencies = Vec::new();
    for scale in [0.75, 1.0, 1.25] {
        let frequency = (drive_frequency_hz * scale).clamp(50.0, 20_000.0);
        if !frequencies
            .iter()
            .any(|existing| f64::abs(*existing - frequency) <= 1.0e-9)
        {
            frequencies.push(frequency);
        }
    }
    frequencies
}

pub(super) fn topology(node_count: usize) -> AcousticDomainTopology {
    let n = node_count.max(1);
    let z_dim = if n >= 24 { 2 } else { 1 };
    let y_dim = if n >= 3 {
        ((n as f64 / z_dim as f64).sqrt().ceil() as usize).max(2)
    } else {
        1
    };
    let x_dim = n.div_ceil(y_dim * z_dim).max(1);
    let dims = [x_dim, y_dim, z_dim];
    let spacing = dims.map(|dim| {
        if dim <= 1 {
            1.0
        } else {
            1.0 / dim.saturating_sub(1) as f64
        }
    });
    let base = AcousticDomainTopology {
        node_count: n,
        dims,
        spacing,
        edge_count: 0,
        boundary_node_count: 0,
        active_dimension_count: dims.iter().filter(|dim| **dim > 1).count(),
    };
    let mut edge_count = 0;
    let mut boundary_node_count = 0;
    for node in 0..n {
        boundary_node_count += usize::from(base.is_boundary_node(node));
        for axis in 0..3 {
            edge_count += usize::from(base.neighbor(node, axis, 1).is_some());
        }
    }
    AcousticDomainTopology {
        edge_count,
        boundary_node_count,
        ..base
    }
}

pub(super) fn source_vectors(
    model: &AnalysisModel,
    node_count: usize,
    frequency_hz: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut source_real = vec![0.0; node_count.max(1)];
    let mut source_imag = vec![0.0; node_count.max(1)];
    for (index, (target_id, real, imaginary)) in pressure_sources(model, frequency_hz)?
        .into_iter()
        .enumerate()
    {
        let node = (index * 3 + target_id.len()) % source_real.len();
        source_real[node] += real;
        source_imag[node] += imaginary;
    }
    Ok((source_real, source_imag))
}

pub(super) fn helmholtz_operator(
    topology: AcousticDomainTopology,
    drive_frequency_hz: f64,
    speed_of_sound_m_per_s: f64,
    damping_ratio: f64,
    boundary: &AcousticBoundarySummary,
) -> AcousticDomainSystem {
    let n = topology.node_count.max(1);
    let omega = 2.0 * std::f64::consts::PI * drive_frequency_hz.max(1.0);
    let wave_number = omega / speed_of_sound_m_per_s.max(1.0);
    let mut diag_real = vec![0.0; n];
    let mut diag_imag = vec![0.0; n];
    let mut edges = Vec::with_capacity(topology.edge_count);
    for node in 0..n {
        for axis in 0..3 {
            let Some(next_index) = topology.neighbor(node, axis, 1) else {
                continue;
            };
            let stiffness = 1.0 / topology.spacing[axis].max(1.0e-9).powi(2);
            diag_real[node] += stiffness;
            diag_real[next_index] += stiffness;
            edges.push(AcousticGraphEdge {
                left: node,
                right: next_index,
                stiffness,
            });
        }
    }
    let mass_term =
        (wave_number * topology.volume_m3().cbrt()).powi(2) / topology.node_count.max(1) as f64;
    let damping = (2.0 * damping_ratio.max(0.0) * mass_term.max(1.0e-9)).max(1.0e-9);
    let boundary_loss = (boundary.radiation_loss_factor + boundary.impedance_loss_factor)
        * wave_number.abs().max(1.0e-9)
        / topology.boundary_node_count.max(1) as f64;
    for node in 0..n {
        diag_real[node] -= mass_term;
        diag_imag[node] += damping;
        if topology.is_boundary_node(node) {
            diag_imag[node] += boundary_loss;
            diag_real[node] += 0.02 * boundary.rigid_wall_count as f64;
        }
    }
    AcousticDomainSystem {
        diag_real,
        diag_imag,
        edges,
    }
}

pub(super) fn residual_norm(
    system: &AcousticDomainSystem,
    pressure_real: &[f64],
    pressure_imag: &[f64],
    source_real: &[f64],
    source_imag: &[f64],
) -> f64 {
    let mut residual_sq = 0.0;
    for i in 0..system.diag_real.len() {
        let mut applied_real =
            system.diag_real[i] * pressure_real[i] - system.diag_imag[i] * pressure_imag[i];
        let mut applied_imag =
            system.diag_real[i] * pressure_imag[i] + system.diag_imag[i] * pressure_real[i];
        for edge in system
            .edges
            .iter()
            .filter(|edge| edge.left == i || edge.right == i)
        {
            let neighbor = if edge.left == i {
                edge.right
            } else {
                edge.left
            };
            applied_real -= edge.stiffness * pressure_real[neighbor];
            applied_imag -= edge.stiffness * pressure_imag[neighbor];
        }
        let real = applied_real - source_real.get(i).copied().unwrap_or(0.0);
        let imag = applied_imag - source_imag.get(i).copied().unwrap_or(0.0);
        residual_sq += real * real + imag * imag;
    }
    residual_sq.sqrt() / source_norm(source_real, source_imag).max(1.0)
}

pub(super) fn source_norm(source_real: &[f64], source_imag: &[f64]) -> f64 {
    source_real
        .iter()
        .zip(source_imag.iter().chain(std::iter::repeat(&0.0)))
        .map(|(real, imag)| real * real + imag * imag)
        .sum::<f64>()
        .sqrt()
}

pub(super) fn pressure_sources(
    model: &AnalysisModel,
    frequency_hz: f64,
) -> Result<Vec<(String, f64, f64)>, String> {
    let matching_steps = model
        .sequence
        .steps
        .iter()
        .filter(|step| {
            step.family == PhysicsFamily::Acoustic
                && step.procedure == AnalysisProcedure::HarmonicResponse
        })
        .collect::<Vec<_>>();
    let [step] = matching_steps.as_slice() else {
        return Err(format!(
            "acoustic excitation realization requires exactly one harmonic acoustic step, found {}",
            matching_steps.len()
        ));
    };
    let realized = crate::excitation::realize_step_excitations(
        model,
        step,
        crate::excitation::ExcitationSamplePoint::frequency(frequency_hz),
    )
    .map_err(|error| format!("acoustic pressure excitation realization failed: {error}"))?;
    if realized.iter().any(|load| {
        matches!(
            load.kind,
            crate::excitation::RealizedExcitationKind::AcousticIncidentPressure(_)
                | crate::excitation::RealizedExcitationKind::VolumeVelocity(_)
        )
    }) {
        return Err(
            "incident pressure and total volume velocity require canonical exterior-face Helmholtz execution".into(),
        );
    }
    Ok(realized
        .iter()
        .filter_map(|excitation| {
            let value = match excitation.kind {
                crate::excitation::RealizedExcitationKind::Pressure(value)
                | crate::excitation::RealizedExcitationKind::AcousticPressure(value) => value,
                _ => return None,
            };
            let target = match &excitation.target {
                crate::excitation::RealizedExcitationTarget::Region(target) => {
                    target.region_id.clone()
                }
                crate::excitation::RealizedExcitationTarget::Port(id)
                | crate::excitation::RealizedExcitationTarget::Interface(id) => id.clone(),
                crate::excitation::RealizedExcitationTarget::ReferencePoint {
                    reference_point_id,
                    ..
                } => reference_point_id.clone(),
                crate::excitation::RealizedExcitationTarget::WholeModel => {
                    "whole_model".to_string()
                }
            };
            Some((target, value.real, value.imaginary))
        })
        .collect())
}
