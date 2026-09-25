use std::f64::consts::TAU;

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, FormulationRequirement,
    MaterialElasticityModel, PrescribedStructuralDofs, PrescribedValue, ProcedureStep,
    RegionTarget, StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof,
    TranslationDofValue,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, SolverMeshArtifact, StableDigest,
};

use crate::{
    fixtures::{fixture_model, replace_fixture_excitations, FixtureId},
    pipeline::modal_governance::ModalTruncationPolicy,
    ComputeBackend, ModalSolveOptions,
};

// Independent ten-point rule for a converged reference of the rational curved-element
// stiffness integral. Production uses eight points; its mass rule uses five.
const GAUSS_TEN_UNIT: [(f64, f64); 10] = [
    (0.013_046_735_741_414, 0.033_335_672_154_344),
    (0.067_468_316_655_507_5, 0.074_725_674_575_29),
    (0.160_295_215_850_488, 0.109_543_181_257_991),
    (0.283_302_302_935_376_5, 0.134_633_359_654_998),
    (0.425_562_830_509_184_5, 0.147_762_112_357_376_5),
    (0.574_437_169_490_815_5, 0.147_762_112_357_376_5),
    (0.716_697_697_064_623_5, 0.134_633_359_654_998),
    (0.839_704_784_149_512, 0.109_543_181_257_991),
    (0.932_531_683_344_492_5, 0.074_725_674_575_29),
    (0.986_953_264_258_586, 0.033_335_672_154_344),
];

fn curved_corner_x_stiffness_reference(warp: f64, youngs: f64, poisson: f64) -> f64 {
    let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let shear = youngs / (2.0 * (1.0 + poisson));
    let mut stiffness = 0.0;
    for (r, r_weight) in GAUSS_TEN_UNIT {
        for (s, s_weight) in GAUSS_TEN_UNIT {
            for (t, t_weight) in GAUSS_TEN_UNIT {
                let l0 = (1.0 - r) * (1.0 - s) * (1.0 - t);
                let determinant = 1.0 + 4.0 * warp * (l0 - r);
                let transverse = -4.0 * warp * r;
                let slope = 4.0 * l0 - 1.0;
                let gx = -slope / determinant;
                let gy = slope * (transverse / determinant - 1.0);
                let gz = gy;
                let reference_weight =
                    r_weight * s_weight * t_weight * (1.0 - r).powi(2) * (1.0 - s);
                stiffness += reference_weight
                    * determinant
                    * ((lambda + 2.0 * shear) * gx * gx + shear * (gy * gy + gz * gz));
            }
        }
    }
    stiffness
}

fn constrain_everything_except_corner_x(mesh: &SolverMeshArtifact) -> Vec<AnalysisConstraint> {
    mesh.topology
        .nodes
        .iter()
        .map(|node| {
            let translations = [
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .filter(|dof| node.node_id != 1 || *dof != StructuralTranslationDof::Ux)
            .map(|dof| TranslationDofValue {
                dof,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            })
            .collect();
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("modal_support_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations,
                    rotations: Vec::new(),
                }),
            })
        })
        .collect()
}

#[test]
fn curved_tet10_modal_frequency_matches_independent_one_dof_reference() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let warp = 0.1;
    mesh.topology.nodes[4].coordinates_m[0] += warp;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_modal("curved_tet10_modal")];
    replace_fixture_excitations(&mut model, Vec::new());
    model.constraints = constrain_everything_except_corner_x(&mesh);
    let MaterialElasticityModel::IsotropicLinear(elasticity) = model.materials[0]
        .elasticity
        .as_mut()
        .expect("fixture elasticity")
    else {
        panic!("expected isotropic linear elasticity");
    };
    elasticity.youngs_modulus_pa = 100.0;
    elasticity.poisson_ratio = 0.25;
    model.materials[0].mass_density_kg_per_m3 = Some(1.0);
    model.materials[0].damping = None;

    let result = super::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: FormulationRequirement::Experimental,
            truncation: ModalTruncationPolicy::requested(1),
            solver_mesh: Some(mesh),
            ..ModalSolveOptions::default()
        },
    )
    .expect("physical curved Tet10 one-DOF modal case should solve");
    let [frequency_hz] = result.frequencies_hz().try_into().unwrap();
    let stiffness = curved_corner_x_stiffness_reference(warp, 100.0, 0.25);
    let exact_mass = 1.0 / 420.0 + warp / 315.0;
    let reference_frequency_hz = (stiffness / exact_mass).sqrt() / TAU;
    assert!(
        (frequency_hz - reference_frequency_hz).abs() < 1.0e-7 * reference_frequency_hz,
        "modal frequency {frequency_hz} Hz vs independent {reference_frequency_hz} Hz"
    );
}
