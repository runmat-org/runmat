//! Independently check a solved connected Tet4 patch against an affine elasticity solution.

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, MaterialElasticityModel,
    PrescribedStructuralDofs, PrescribedValue, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::fixtures::connected_tetrahedron_solver_mesh;

use super::{
    experimental_linear_static_options, fixture_model, host_field, ComputeBackend, FixtureId,
    LinearStaticSolveOptions, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
    FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_STRAIN,
    FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
};

#[test]
fn connected_tet4_static_dirichlet_patch_recovers_affine_interior_and_boundary_tractions() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let MaterialElasticityModel::IsotropicLinear(elasticity) = model.materials[0]
        .elasticity
        .as_mut()
        .expect("fixture elasticity")
    else {
        panic!("expected isotropic linear elasticity");
    };
    elasticity.youngs_modulus_pa = 100.0;
    elasticity.poisson_ratio = 0.25;
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints = (1..=4)
        .map(|node_id| {
            let coordinates = match node_id {
                1 => [0.0, 0.0, 0.0],
                2 => [1.0, 0.0, 0.0],
                3 => [0.0, 1.0, 0.0],
                4 => [0.0, 0.0, 1.0],
                _ => unreachable!(),
            };
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("affine_boundary_node_{node_id}"),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{node_id}")),
                    translations: [
                        StructuralTranslationDof::Ux,
                        StructuralTranslationDof::Uy,
                        StructuralTranslationDof::Uz,
                    ]
                    .into_iter()
                    .zip([0.1, 0.2, 0.3])
                    .enumerate()
                    .map(|(axis, (dof, gradient))| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(
                            gradient * coordinates[axis],
                        )),
                    })
                    .collect(),
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(connected_tetrahedron_solver_mesh()),
            solver_mesh_artifact_path: Some("connected-static-patch.rmmesh".into()),
            ..experimental_linear_static_options()
        },
    )
    .expect("connected affine Dirichlet patch should solve");

    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert_eq!(displacement.len(), 15);
    for (actual, expected) in displacement[12..15].iter().zip([0.025, 0.05, 0.075]) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "{actual} != {expected}"
        );
    }

    // E=100 Pa, nu=0.25 gives lambda=mu=40 Pa. The affine strain is
    // diag(0.1,0.2,0.3), hence sigma=diag(32,40,48) Pa and w=12.8 J/m^3.
    let strain = host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN);
    let stress = host_field(&result, FEA_FIELD_STRUCTURAL_STRESS);
    let energy_density = host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY);
    assert_eq!(strain.len(), 24);
    assert_eq!(stress.len(), 24);
    assert_eq!(energy_density.len(), 4);
    for element in 0..4 {
        for (actual, expected) in strain[element * 6..][..6]
            .iter()
            .zip([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        {
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "strain {actual} != {expected}"
            );
        }
        for (actual, expected) in stress[element * 6..][..6]
            .iter()
            .zip([32.0, 40.0, 48.0, 0.0, 0.0, 0.0])
        {
            assert!(
                (actual - expected).abs() < 1.0e-8,
                "stress {actual} != {expected}"
            );
        }
        assert!((energy_density[element] - 12.8).abs() < 1.0e-9);
    }
    assert!(
        (host_field(&result, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] - 12.8 / 6.0).abs()
            < 1.0e-9
    );

    // The analytical exterior nodal traction is integral(N_i sigma*n) dA.
    // The published reaction field uses the same sign as K*u, and the
    // unconstrained centroid has none.
    let reaction = host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE);
    for (actual, traction) in reaction.chunks_exact(3).zip([
        [-16.0 / 3.0, -20.0 / 3.0, -8.0],
        [16.0 / 3.0, 0.0, 0.0],
        [0.0, 20.0 / 3.0, 0.0],
        [0.0, 0.0, 8.0],
        [0.0, 0.0, 0.0],
    ]) {
        for (&actual, traction) in actual.iter().zip(traction) {
            assert!(
                (actual - traction).abs() < 1.0e-8,
                "reaction {actual} vs traction {traction}"
            );
        }
    }
}
