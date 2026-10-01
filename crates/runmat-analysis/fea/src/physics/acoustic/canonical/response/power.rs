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
    if [
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
