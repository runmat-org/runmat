//! Recover products at the same physical quadrature points as the pressure operator.
//! The product of element-averaged pressure and velocity is NOT average intensity.
use super::{assembly::AcousticAssembly, AcousticHarmonicError};
use crate::excitation::ProjectionVector3;

pub(super) struct RecoveredAcoustics {
    pub velocity: Vec<ProjectionVector3>,
    pub intensity: Vec<[f64; 3]>,
    pub energy: Vec<f64>,
}

pub(super) fn recover(
    assembly: &AcousticAssembly,
    real: &[f64],
    imaginary: &[f64],
) -> Result<RecoveredAcoustics, AcousticHarmonicError> {
    let count = assembly.projection.elements.len();
    let mut result = RecoveredAcoustics {
        velocity: Vec::with_capacity(count),
        intensity: Vec::with_capacity(count),
        energy: Vec::with_capacity(count),
    };
    // Independent per-element recovery on the admitted pool, collected in element order.
    let elements = &assembly.projection.elements;
    let recovered = crate::algebra::element_assembly::ordered_map_range(count, |index| {
        let element = &elements[index];
        let rho = assembly.density.canonical_values()[index];
        let mut velocity = ProjectionVector3 {
            real: [0.0; 3],
            imaginary: [0.0; 3],
        };
        let mut intensity = [0.0; 3];
        let mut energy = 0.0;
        let mut volume = 0.0;
        assembly
            .projection
            .visit_element_quadrature(element, |point| {
                let mut pressure = [0.0; 2];
                let mut v_real = [0.0; 3];
                let mut v_imaginary = [0.0; 3];
                for (local, &node) in element.node_indices.iter().enumerate() {
                    pressure[0] += point.shape_values()[local] * real[node];
                    pressure[1] += point.shape_values()[local] * imaginary[node];
                    for axis in 0..3 {
                        let factor = point.physical_shape_gradients()[local][axis]
                            / assembly.derivative_rate.canonical_value()
                            / rho;
                        v_real[axis] -= imaginary[node] * factor;
                        v_imaginary[axis] += real[node] * factor;
                    }
                }
                let weight = point.volume_weight_m3;
                volume += weight;
                let mut speed_squared = 0.0;
                for axis in 0..3 {
                    velocity.real[axis] += v_real[axis] * weight;
                    velocity.imaginary[axis] += v_imaginary[axis] * weight;
                    intensity[axis] += 0.5
                        * (pressure[0] * v_real[axis] + pressure[1] * v_imaginary[axis])
                        * weight;
                    speed_squared += v_real[axis].powi(2) + v_imaginary[axis].powi(2);
                }
                energy += 0.25
                    * (assembly.compressibility.canonical_values()[index]
                        * (pressure[0].powi(2) + pressure[1].powi(2))
                        + rho * speed_squared)
                    * weight;
            })
            .map_err(|e| AcousticHarmonicError::Topology(e.to_string()))?;
        velocity.real = velocity.real.map(|v| v / volume);
        velocity.imaginary = velocity.imaginary.map(|v| v / volume);
        intensity = intensity.map(|v| v / volume);
        energy /= volume;
        if velocity
            .real
            .iter()
            .chain(&velocity.imaginary)
            .chain(&intensity)
            .chain(std::iter::once(&energy))
            .any(|v| !v.is_finite())
            || energy < 0.0
        {
            return Err(AcousticHarmonicError::Operator(
                "nonfinite or negative acoustic recovery".into(),
            ));
        }
        Ok((velocity, intensity, energy))
    });
    for item in recovered {
        let (velocity, intensity, energy) = item?;
        result.velocity.push(velocity);
        result.intensity.push(intensity);
        result.energy.push(energy);
    }
    Ok(result)
}
