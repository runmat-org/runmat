//! Physical power authority shared by solve acceptance and retained response reporting.
use super::{AcousticAssembly, AcousticHarmonicError};
use crate::algebra::spmv;

pub(super) struct PowerClosure {
    pub mean_input_power_w: f64,
    pub mean_impedance_loss_w: f64,
    pub power_relative_error: f64,
}

pub(super) fn evaluate(
    assembly: &AcousticAssembly,
    real: &[f64],
    imaginary: &[f64],
) -> Result<PowerClosure, AcousticHarmonicError> {
    let size = assembly.projection.nodes.len();
    let (ar, ai) = assembly.apply(real, imaginary);
    let br = spmv(&assembly.impedance, real).expect("validated pressure shape");
    let bi = spmv(&assembly.impedance, imaginary).expect("validated pressure shape");
    let passive_power_w = 0.5
        * (0..size)
            .map(|i| real[i] * br[i] + imaginary[i] * bi[i])
            .sum::<f64>();
    let source_work_w = (0..size)
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
        .max(passive_power_w.abs());
    // At an incident boundary v_n=Y(p-2 p_inc). Its passive Y|p|²/2 is part
    // of Robin source-work bookkeeping, NOT an outgoing physical loss. Remove
    // that same term from both channels; the residual and original apparent
    // power normalization remain unchanged. Accepted input is then net incoming
    // boundary power (incoming minus reflected), rather than twice incident power.
    let mut incident_passive_power_w = 0.0;
    for (face, admittance) in &assembly.incident_faces {
        assembly
            .projection
            .visit_boundary_quadrature(face, |point| {
                let mut pressure = [0.0; 2];
                for (&node, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                    pressure[0] += shape * real[node];
                    pressure[1] += shape * imaginary[node];
                }
                incident_passive_power_w += 0.5
                    * admittance
                    * (pressure[0].powi(2) + pressure[1].powi(2))
                    * point.area_weight_m2;
            })
            .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
    }
    let mean_input_power_w = source_work_w - incident_passive_power_w;
    let mean_impedance_loss_w = passive_power_w - incident_passive_power_w;
    let power_relative_error = if power_scale == 0.0 {
        0.0
    } else {
        (mean_input_power_w - mean_impedance_loss_w).abs() / power_scale
    };
    if [
        power_scale,
        mean_input_power_w,
        mean_impedance_loss_w,
        power_relative_error,
    ]
    .iter()
    .any(|v| !v.is_finite())
    {
        return Err(AcousticHarmonicError::Operator(
            "nonfinite acoustic power closure".into(),
        ));
    }
    Ok(PowerClosure {
        mean_input_power_w,
        mean_impedance_loss_w,
        power_relative_error,
    })
}
