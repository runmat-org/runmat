//! Analytic PEC cube mode, independent of RunMat's element matrix or solver.
use super::super::*;
use crate::results::field::{SampledVolumeFieldTopology, VolumeFieldSamplingRule};
use runmat_meshing_core::{fixtures, ElementOrder, SolverMeshArtifact};

pub(super) fn problem(
    order: ElementOrder,
    cells: usize,
    phase: ComplexPhasor,
) -> (SolverMeshArtifact, MaxwellInput) {
    let mesh = match order {
        ElementOrder::Tet4 => fixtures::slab_mesh(cells),
        ElementOrder::Tet10 => fixtures::quadratic_slab_mesh(cells),
    };
    let sampling =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronKeastFifteen)
            .unwrap();
    let volumes = sampling
        .element_identities()
        .iter()
        .map(|&element_identity| MaxwellVolumeData {
            element_identity,
            permeability_h_per_m: runmat_types::physical::MagneticPermeability::new(1.0),
            permittivity_f_per_m: runmat_types::physical::ElectricPermittivity::new(1.0),
            conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(0.3),
        })
        .collect();
    let source_current = sampling
        .integration_samples_for(&mesh)
        .unwrap()
        .iter()
        .map(|sample| {
            let e = electric(sample.position_m, phase);
            // exp(+iwt), omega=1: J=i*(2*pi^2-1+i*sigma)*E.
            let reactive = 2.0 * std::f64::consts::PI.powi(2) - 1.0;
            MaxwellCurrentSample {
                sample_identity: sample.identity,
                current_density_a_per_m2: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::ElectricCurrentDensity,
                    _,
                >::new(e.map(|value| ComplexPhasor {
                    real: -0.3 * value.real - reactive * value.imaginary,
                    imaginary: reactive * value.real - 0.3 * value.imaginary,
                })),
            }
        })
        .collect();
    let mut grounded_faces: Vec<_> = mesh
        .topology
        .boundary_faces
        .iter()
        .map(|face| face.stable_identity)
        .collect();
    grounded_faces.sort();
    (
        mesh,
        MaxwellInput {
            angular_frequency_rad_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::AngularFrequency,
                _,
            >::new(1.0),
            sampling,
            volumes,
            source_current,
            grounded_faces,
            drives: Vec::new(),
        },
    )
}

pub(super) fn electric(position: [f64; 3], phase: ComplexPhasor) -> [ComplexPhasor; 3] {
    let pi = std::f64::consts::PI;
    let shape = (pi * position[0]).sin() * (pi * position[2]).sin();
    [
        ComplexPhasor {
            real: 0.0,
            imaginary: 0.0,
        },
        ComplexPhasor {
            real: shape * phase.real,
            imaginary: shape * phase.imaginary,
        },
        ComplexPhasor {
            real: 0.0,
            imaginary: 0.0,
        },
    ]
}

pub(super) fn flux(position: [f64; 3], phase: ComplexPhasor) -> [ComplexPhasor; 3] {
    let pi = std::f64::consts::PI;
    // B=i*curl(E)/omega; omega=1. This checks Faraday's sign and phase.
    let curl = [
        -pi * (pi * position[0]).sin() * (pi * position[2]).cos(),
        0.0,
        pi * (pi * position[0]).cos() * (pi * position[2]).sin(),
    ];
    curl.map(|value| ComplexPhasor {
        real: -value * phase.imaginary,
        imaginary: value * phase.real,
    })
}

pub(super) fn longitudinal(position: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = position;
    // E=grad[x(1-x)y(1-y)z(1-z)] has tangential PEC trace and nonzero charge.
    [
        (1.0 - 2.0 * x) * y * (1.0 - y) * z * (1.0 - z),
        x * (1.0 - x) * (1.0 - 2.0 * y) * z * (1.0 - z),
        x * (1.0 - x) * y * (1.0 - y) * (1.0 - 2.0 * z),
    ]
}
