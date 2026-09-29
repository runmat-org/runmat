use super::{assembly::AcousticAssembly, AcousticHarmonicError, AcousticHarmonicPolicy};
use crate::{
    algebra::spmv,
    excitation::{ProjectionScalar, ProjectionVector3, SpatialProjectionEvidence},
    solve::krylov::{restarted_gmres, GmresPolicy},
    topology::AnalysisProjection,
};
use std::convert::Infallible;

#[derive(Debug, Clone)]
pub struct CanonicalAcousticResponse {
    pub frequency_hz: f64,
    pub projection: AnalysisProjection,
    /// Peak pressure phasors in Pa, in canonical projection node order.
    pub pressure_pa: Vec<ProjectionScalar>,
    /// Volume-averaged particle velocity phasors in m/s, in projection element order.
    pub element_velocity_m_per_s: Vec<ProjectionVector3>,
    /// Volume average of 0.5 Re(p conjugate(v)), from peak phasors, in element order.
    pub element_mean_intensity_w_per_m2: Vec<[f64; 3]>,
    /// Volume average of (|p|²/(rho c²) + rho |v|²)/4, in element order.
    pub element_mean_energy_density_j_per_m3: Vec<f64>,
    pub iterations: u32,
    pub relative_residual: f64,
    /// Positive into the fluid; includes prescribed-pressure reaction flux.
    pub mean_input_power_w: f64,
    pub mean_impedance_loss_w: f64,
    pub power_relative_error: f64,
    pub source_evidence: Vec<SpatialProjectionEvidence>,
}

pub(super) fn solve(
    assembly: AcousticAssembly,
    frequency_hz: f64,
    policy: AcousticHarmonicPolicy,
) -> Result<CanonicalAcousticResponse, AcousticHarmonicError> {
    let size = assembly.projection.nodes.len();
    let free = assembly
        .prescribed
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.is_none().then_some(i))
        .collect::<Vec<_>>();
    let mut real = assembly
        .prescribed
        .iter()
        .map(|p| p.map_or(0.0, |p| p.real))
        .collect::<Vec<_>>();
    let mut imaginary = assembly
        .prescribed
        .iter()
        .map(|p| p.map_or(0.0, |p| p.imaginary))
        .collect::<Vec<_>>();
    let (lifting_real, lifting_imaginary) = assembly.apply(&real, &imaginary);
    let mut rhs = free
        .iter()
        .map(|&i| assembly.source[i].real - lifting_real[i])
        .collect::<Vec<_>>();
    rhs.extend(
        free.iter()
            .map(|&i| assembly.source[i].imaginary - lifting_imaginary[i]),
    );
    let mut iterations = 0;
    let mut relative_residual = 0.0;
    if !free.is_empty() {
        let count = free.len();
        let result = restarted_gmres(
            &rhs,
            &vec![0.0; rhs.len()],
            GmresPolicy {
                relative_tolerance: policy.relative_tolerance,
                restart: policy.restart,
                maximum_iterations: policy.maximum_iterations,
            },
            |value| {
                let mut r = vec![0.0; size];
                let mut im = vec![0.0; size];
                for (j, &i) in free.iter().enumerate() {
                    r[i] = value[j];
                    im[i] = value[j + count];
                }
                let (ar, ai) = assembly.apply(&r, &im);
                let mut value = free.iter().map(|&i| ar[i]).collect::<Vec<_>>();
                value.extend(free.iter().map(|&i| ai[i]));
                Ok::<_, Infallible>(value)
            },
            |a, b| Ok::<_, Infallible>(a.iter().zip(b).map(|(a, b)| a * b).sum()),
            crate::is_fea_cancelled,
        )
        .expect("validated acoustic block application")
        .ok_or(AcousticHarmonicError::Breakdown)?;
        if result.cancelled {
            return Err(AcousticHarmonicError::Cancelled);
        }
        if result.breakdown {
            return Err(AcousticHarmonicError::Breakdown);
        }
        if !result.converged {
            return Err(AcousticHarmonicError::NotConverged);
        }
        let norm = rhs.iter().map(|x| x * x).sum::<f64>().sqrt();
        relative_residual = if norm == 0.0 {
            result.residual_norm
        } else {
            result.residual_norm / norm
        };
        iterations = result.iterations;
        for (j, &i) in free.iter().enumerate() {
            real[i] = result.solution[j];
            imaginary[i] = result.solution[j + count];
        }
    }
    let (ar, ai) = assembly.apply(&real, &imaginary);
    let br = spmv(&assembly.impedance, &real).expect("validated pressure shape");
    let bi = spmv(&assembly.impedance, &imaginary).expect("validated pressure shape");
    let mean_impedance_loss_w = 0.5
        * (0..size)
            .map(|i| real[i] * br[i] + imaginary[i] * bi[i])
            .sum::<f64>();
    let mean_input_power_w = (0..size)
        .map(|i| {
            let (source_real, source_imaginary) = if assembly.prescribed[i].is_some() {
                // Sum specified natural flux with pressure-boundary reaction flux.
                (ar[i], ai[i])
            } else {
                (assembly.source[i].real, assembly.source[i].imaginary)
            };
            (real[i] * source_imaginary - imaginary[i] * source_real) / (2.0 * assembly.omega)
        })
        .sum::<f64>();
    // Include reactive apparent power so a lossless cavity is not normalized by zero
    // real power. No dimensional floor may hide a small-amplitude imbalance.
    let power_scale = (0..size)
        .map(|i| {
            let source = if assembly.prescribed[i].is_some() {
                (ar[i], ai[i])
            } else {
                (assembly.source[i].real, assembly.source[i].imaginary)
            };
            real[i].hypot(imaginary[i]) * source.0.hypot(source.1) / (2.0 * assembly.omega)
        })
        .sum::<f64>()
        .max(mean_impedance_loss_w.abs());
    let power_relative_error = if power_scale == 0.0 {
        0.0
    } else {
        (mean_input_power_w - mean_impedance_loss_w).abs() / power_scale
    };
    let recovered = super::recovery::recover(&assembly, &real, &imaginary)?;
    if real
        .iter()
        .chain(&imaginary)
        .any(|value| !value.is_finite())
        || recovered.velocity.iter().any(|v| {
            v.real
                .iter()
                .chain(&v.imaginary)
                .any(|value| !value.is_finite())
        })
        || [
            mean_input_power_w,
            mean_impedance_loss_w,
            power_relative_error,
            relative_residual,
        ]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(AcousticHarmonicError::Operator(
            "nonfinite recovered field or power".into(),
        ));
    }
    let pressure_pa = real
        .into_iter()
        .zip(imaginary)
        .map(|(real, imaginary)| ProjectionScalar { real, imaginary })
        .collect();
    Ok(CanonicalAcousticResponse {
        frequency_hz,
        projection: assembly.projection,
        pressure_pa,
        element_velocity_m_per_s: recovered.velocity,
        element_mean_intensity_w_per_m2: recovered.intensity,
        element_mean_energy_density_j_per_m3: recovered.energy,
        iterations,
        relative_residual,
        mean_input_power_w,
        mean_impedance_loss_w,
        power_relative_error,
        source_evidence: assembly.source_evidence,
    })
}
