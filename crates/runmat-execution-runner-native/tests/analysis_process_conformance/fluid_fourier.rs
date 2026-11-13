//! Original non-affine Fourier duct through scheduling, cold replay and publication.
use runmat_analysis_fea::physics::cfd::canonical::qualification::{
    cell_centroid_velocity_error,
    fixture::{case, ReferenceCase},
    reference, wall_shear_error,
};

pub(super) async fn run() {
    assert!(reference::duct_tail(255) < 1e-5);
    let mut errors = Vec::new();
    let mut flows = Vec::new();
    let mut shears = Vec::new();
    for n in [2, 3, 4] {
        let fixture = case(n, ReferenceCase::Duct, false);
        let cold = super::fluid_study_fixture::cold_fixture(&fixture, "original-fourier").await;

        let original = cold.result.result.original();
        errors.push(cell_centroid_velocity_error(&fixture, original));
        shears.push(wall_shear_error(&fixture, &original.residual));
        let flow: f64 = fixture
            .topology
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.neighbor.is_none() && reference::outlet(f.centroid_m, false))
            .map(|(i, _)| original.face_volume_flux_m3_per_s.canonical_values()[i])
            .sum();
        flows.push((flow - reference::duct_flow(255)).abs() / reference::duct_flow(255));
        assert!(
            original.residual.global_boundary_flux_m3_per_s.abs()
                < runmat_types::physical::Quantity::new(1e-8)
        );
        let mut net = [1.0, 0.0, 0.0];
        for (i, face) in fixture.topology.faces.iter().enumerate() {
            if face.neighbor.is_some() {
                continue;
            }
            for (axis, value) in net.iter_mut().enumerate() {
                *value += original
                    .residual
                    .face_forces
                    .viscous_traction_n
                    .canonical_values()[i][axis]
                    + original
                        .residual
                        .face_forces
                        .pressure_traction_n
                        .canonical_values()[i][axis]
                    - original
                        .residual
                        .face_forces
                        .advective_momentum_n
                        .canonical_values()[i][axis];
            }
        }
        assert!(
            net.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-6,
            "independent original volume-force/stress closure {net:?}"
        );

        cold.result
            .numerical_receipt
            .validate_against(cold.product.workload(), &cold.result.result, &fixture.mesh)
            .unwrap();
        eprintln!(
            "scheduled original Fourier n={n}: centroid velocity={}, flow={}, wall shear={}",
            errors.last().unwrap(),
            flows.last().unwrap(),
            shears.last().unwrap()
        );
    }
    for (values, limit) in [(&errors, 0.35), (&flows, 0.25), (&shears, 0.35)] {
        assert!(
            values[1] < values[0] && values[2] < values[1] && values[2] < limit,
            "original Fourier convergence {values:?}, limit={limit}"
        );
    }
}
