use crate::assembly::StructuralMaterialSummary;

const TENSOR_COMPONENT_COUNT: usize = 6;

pub(super) fn recover_stress(strain: &[f64], material: StructuralMaterialSummary) -> Vec<f64> {
    let lambda = material.lame_lambda_pa.max(0.0);
    let mu = material.shear_modulus_pa.canonical_value().max(0.0);
    let mut stress = vec![0.0; strain.len()];
    for (element_index, strain_tensor) in strain.chunks_exact(TENSOR_COMPONENT_COUNT).enumerate() {
        let trace = strain_tensor[0] + strain_tensor[1] + strain_tensor[2];
        let base = element_index * TENSOR_COMPONENT_COUNT;
        stress[base] = lambda * trace + 2.0 * mu * strain_tensor[0];
        stress[base + 1] = lambda * trace + 2.0 * mu * strain_tensor[1];
        stress[base + 2] = lambda * trace + 2.0 * mu * strain_tensor[2];
        stress[base + 3] = mu * strain_tensor[3];
        stress[base + 4] = mu * strain_tensor[4];
        stress[base + 5] = mu * strain_tensor[5];
    }
    stress
}

pub(super) fn recover_von_mises(stress: &[f64]) -> Vec<f64> {
    stress
        .chunks_exact(TENSOR_COMPONENT_COUNT)
        .map(|tensor| {
            let sxx = tensor[0];
            let syy = tensor[1];
            let szz = tensor[2];
            let txy = tensor[3];
            let tyz = tensor[4];
            let txz = tensor[5];
            (0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2))
                + 3.0 * (txy.powi(2) + tyz.powi(2) + txz.powi(2)))
            .sqrt()
        })
        .collect()
}

pub(super) fn recover_strain_energy_density(strain: &[f64], stress: &[f64]) -> Vec<f64> {
    strain
        .chunks_exact(TENSOR_COMPONENT_COUNT)
        .zip(stress.chunks_exact(TENSOR_COMPONENT_COUNT))
        .map(|(strain_tensor, stress_tensor)| {
            0.5 * strain_tensor
                .iter()
                .zip(stress_tensor.iter())
                .map(|(strain, stress)| strain * stress)
                .sum::<f64>()
        })
        .collect()
}
