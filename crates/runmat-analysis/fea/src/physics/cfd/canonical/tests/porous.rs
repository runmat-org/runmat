//! Darcy–Forchheimer porous zones against the closed-form uniform-flow pressure gradient.
use super::super::qualification::mass_flow::uniform;
use super::*;
use runmat_analysis_core::*;

// witness: FEA-FLUID-MATERIALS-001/porous_zones @ porous_media_flow/tet4/numerical_kernel;
//   coverage: partial; check: darcy-forchheimer; boundary: uniform flow through a porous slab has the pressure gradient
//   μU/K + ½ρC₂U² (Darcy and Forchheimer terms); Brinkman layers, anisotropy, local thermal
//   equilibrium and the Ergun packed bed are not covered.
#[test]
fn uniform_flow_through_a_porous_zone_has_the_darcy_forchheimer_gradient() {
    let (speed, density, permeability, inertial) = (0.01, 2.0, 1e-3, 50.0);
    let mut f = uniform(3, speed, density, false);
    let mut regions = f
        .topology
        .cells
        .iter()
        .map(|cell| cell.region.clone())
        .collect::<Vec<_>>();
    regions.sort();
    regions.dedup();
    f.material.porous_zones = vec![PorousZone {
        regions,
        law: PorousMediumLaw {
            permeability_m2: permeability,
            inertial_coefficient_per_m: inertial,
            porosity: 0.8,
        },
    }];
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let r = p.solve().unwrap();
    let mu = f.material.dynamic_viscosity.canonical_value();
    let gradient = mu * speed / permeability + 0.5 * density * inertial * speed * speed;
    for u in r.velocity_m_per_s.canonical_values() {
        assert!(norm(sub(*u, [speed, 0.0, 0.0])) < 1e-6 * speed, "{u:?}");
    }
    // p(x) − p(x') = G (x' − x) between any two cells.
    let (first, last) = p
        .topology
        .cells
        .iter()
        .zip(r.pressure_fluctuation_pa.canonical_values())
        .fold(None::<((f64, f64), (f64, f64))>, |acc, (cell, value)| {
            let point = (cell.centroid_m[0], *value);
            Some(match acc {
                None => (point, point),
                Some((lo, hi)) => (
                    if point.0 < lo.0 { point } else { lo },
                    if point.0 > hi.0 { point } else { hi },
                ),
            })
        })
        .unwrap();
    let measured = (first.1 - last.1) / (last.0 - first.0);
    assert!(
        (measured - gradient).abs() < 1e-6 * gradient,
        "measured {measured} expected {gradient}"
    );
}
