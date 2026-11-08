//! Generalized-Newtonian channel flow against the closed-form power-law profile.
use super::*;
use runmat_analysis_core::*;

/// Fully developed power-law flow between plates y = 0 and y = 1 driven by a body force G:
/// u = n/(n+1) (G/K)^(1/n) (h^((n+1)/n) − |y − h|^((n+1)/n)), h = ½.
fn power_law_velocity(y: f64, consistency: f64, index: f64, force: f64) -> f64 {
    let h: f64 = 0.5;
    let e = (index + 1.0) / index;
    index / (index + 1.0)
        * (force / consistency).powf(1.0 / index)
        * (h.powf(e) - (y - h).abs().powf(e))
}

// witness: FEA-FLUID-MATERIALS-001/generalized_newtonian @ generalized_newtonian_flow/tet4/numerical_kernel;
//   coverage: partial; check: power-law-channel; boundary: shear-thinning power-law (n = 0.5) channel converges to the closed-form
//   profile; Carreau, Cross and yield-stress laws, temperature dependence and the
//   shear-rate/apparent-viscosity fields are not asserted here.
#[test]
fn shear_thinning_power_law_channel_converges_to_the_closed_form_profile() {
    let (consistency, index) = (1.0, 0.5);
    let law = GeneralizedNewtonianLaw::PowerLaw {
        consistency_pa_s_n: consistency,
        power_index: index,
        minimum_shear_rate_per_s: 1e-4,
    };
    let mut errors = Vec::new();
    for cells in [2, 4, 8] {
        let mut f = case(cells, ReferenceCase::Duct, false);
        f.material.rheology = Some(law);
        f.policy.maximum_operator_work = None;
        f.material.dynamic_viscosity = DynamicViscosity::new(law.low_shear_viscosity());
        // Every authored boundary carries the plane power-law profile (u depends on y only).
        for constraint in &mut f.constraints {
            let FluidConstraintKind::PrescribedVelocity { target, value } = &mut constraint.kind
            else {
                continue;
            };
            let face = f
                .source
                .boundary_faces
                .iter()
                .find(|face| {
                    face.provenance
                        .iter()
                        .any(|e| target.matches_persistent_entity(e))
                })
                .unwrap();
            *value = PrescribedValue::Constant(
                [
                    power_law_velocity(face.centroid_m[1], consistency, index, 1.0),
                    0.0,
                    0.0,
                ]
                .map(Velocity::new),
            );
        }
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let r = p.solve().unwrap();
        let peak = power_law_velocity(0.5, consistency, index, 1.0);
        let error = (p
            .topology
            .cells
            .iter()
            .zip(r.velocity_m_per_s.canonical_values())
            .map(|(cell, u)| {
                (u[0] - power_law_velocity(cell.centroid_m[1], consistency, index, 1.0)).powi(2)
                    + u[1].powi(2)
                    + u[2].powi(2)
            })
            .sum::<f64>()
            / p.topology.cells.len() as f64)
            .sqrt()
            / peak;
        errors.push(error);
    }
    assert!(
        errors.windows(2).all(|pair| pair[1] < pair[0]),
        "{errors:?}"
    );
    assert!(errors[2] < 0.05, "{errors:?}");
}

#[test]
fn newtonian_limit_of_the_law_reproduces_the_newtonian_solution() {
    // n = 1 is Newtonian with μ = K: the same field as the constant-viscosity solve.
    let reference = case(2, ReferenceCase::Duct, false);
    let expected = CanonicalFluidProblem::prepare(reference.input())
        .unwrap()
        .solve()
        .unwrap();
    let mut f = case(2, ReferenceCase::Duct, false);
    f.material.rheology = Some(GeneralizedNewtonianLaw::PowerLaw {
        consistency_pa_s_n: 1.0,
        power_index: 1.0,
        minimum_shear_rate_per_s: 1e-6,
    });
    let actual = CanonicalFluidProblem::prepare(f.input())
        .unwrap()
        .solve()
        .unwrap();
    for (u, v) in actual
        .velocity_m_per_s
        .canonical_values()
        .iter()
        .zip(expected.velocity_m_per_s.canonical_values())
    {
        assert!(norm(sub(*u, *v)) < 1e-9);
    }
}
