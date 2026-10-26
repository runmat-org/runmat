use super::fixture::{
    channel, complete_physical_basis, constant_spectrum, converged_policy, full_frequency_bounds,
};
use crate::solve::dynamics::solve_modal_random_response;

#[test]
fn spectral_density_scaling_is_linear_in_variance_and_square_root_in_rms() {
    let (summary, basis) = complete_physical_basis();
    let bounds = full_frequency_bounds(&basis);
    let channels = vec![channel("scale", [0.73, -0.41, 0.29])];
    let base = constant_spectrum(
        bounds,
        channels.clone(),
        runmat_analysis_core::CrossPowerSpectralDensity::diagonal(
            &runmat_types::physical::Quantity::new([2.0e-4]),
        ),
    );
    let scaled = constant_spectrum(
        bounds,
        channels,
        runmat_analysis_core::CrossPowerSpectralDensity::diagonal(
            &runmat_types::physical::Quantity::new([1.8e-3]),
        ),
    );
    let policy = converged_policy(0.999_999_999);
    let base_result = solve_modal_random_response(&summary, &basis, &base, policy).unwrap();
    let scaled_result = solve_modal_random_response(&summary, &basis, &scaled, policy).unwrap();
    for (base, scaled) in base_result
        .dof_variance()
        .iter()
        .zip(scaled_result.dof_variance())
        .filter(|(base, _)| **base > 1.0e-24)
    {
        assert!(((scaled / base) - 9.0).abs() < 2.0e-8);
    }
    for (base, scaled) in base_result
        .dof_rms()
        .iter()
        .zip(scaled_result.dof_rms())
        .filter(|(base, _)| **base > 1.0e-12)
    {
        assert!(((scaled / base) - 3.0).abs() < 2.0e-8);
    }
    assert_eq!(base_result.fields().len(), scaled_result.fields().len());
    for (base_field, scaled_field) in base_result.fields().iter().zip(scaled_result.fields()) {
        assert_eq!(base_field.quantity(), scaled_field.quantity());
        assert_eq!(base_field.shape(), scaled_field.shape());
        for (base, scaled) in base_field
            .variance()
            .iter()
            .zip(scaled_field.variance())
            .filter(|(base, _)| **base > 1.0e-24)
        {
            assert!(((scaled / base) - 9.0).abs() < 2.0e-8);
        }
        for (base, scaled) in base_field
            .rms()
            .iter()
            .zip(scaled_field.rms())
            .filter(|(base, _)| **base > 1.0e-12)
        {
            assert!(((scaled / base) - 3.0).abs() < 2.0e-8);
        }
    }
}
