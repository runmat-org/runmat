use std::f64::consts::TAU;

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, MaterialElasticityModel,
    PrescribedStructuralDofs, PrescribedValue, ProcedureStep, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, SolverMeshArtifact, StableDigest,
};

use crate::{
    fixtures::{fixture_model, replace_fixture_excitations, FixtureId},
    pipeline::modal_governance::ModalTruncationPolicy,
    ComputeBackend, ModalSolveOptions,
};

mod harmonic;

// Independent 24-point rule for the rational curved-element stiffness reference.
// Production admits only a converged eight/twelve or twelve/sixteen pair.
const GAUSS_TWENTY_FOUR_UNIT: [(f64, f64); 24] = [
    (0.002_406_390_001_489_345, 0.006_170_614_899_993_392),
    (0.012_635_722_014_345_26, 0.014_265_694_314_466_84),
    (0.030_862_723_998_633_6, 0.022_138_719_408_709_71),
    (0.056_792_236_497_799_46, 0.029_649_292_457_718_27),
    (0.089_999_007_013_048_53, 0.036_673_240_705_540_14),
    (0.129_937_904_210_722_8, 0.043_095_080_765_976_71),
    (0.175_953_174_031_512_2, 0.048_809_326_052_057_05),
    (0.227_289_264_305_580_2, 0.053_722_135_057_982_87),
    (0.283_103_246_186_977_5, 0.057_752_834_026_862_86),
    (0.342_478_660_151_918_3, 0.060_835_236_463_901_76),
    (0.404_440_566_263_191_9, 0.062_918_728_173_414_25),
    (0.467_971_553_568_697_2, 0.063_969_097_673_376_15),
    (0.532_028_446_431_302_8, 0.063_969_097_673_376_15),
    (0.595_559_433_736_808_2, 0.062_918_728_173_414_25),
    (0.657_521_339_848_081_7, 0.060_835_236_463_901_76),
    (0.716_896_753_813_022_5, 0.057_752_834_026_862_86),
    (0.772_710_735_694_419_8, 0.053_722_135_057_982_87),
    (0.824_046_825_968_487_8, 0.048_809_326_052_057_05),
    (0.870_062_095_789_277_2, 0.043_095_080_765_976_71),
    (0.910_000_992_986_951_5, 0.036_673_240_705_540_14),
    (0.943_207_763_502_200_5, 0.029_649_292_457_718_27),
    (0.969_137_276_001_366_3, 0.022_138_719_408_709_71),
    (0.987_364_277_985_654_7, 0.014_265_694_314_466_84),
    (0.997_593_609_998_510_7, 0.006_170_614_899_993_392),
];

fn curved_corner_x_stiffness_reference(warp: f64, youngs: f64, poisson: f64) -> f64 {
    let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let shear = youngs / (2.0 * (1.0 + poisson));
    let mut stiffness = 0.0;
    for (r, r_weight) in GAUSS_TWENTY_FOUR_UNIT {
        for (s, s_weight) in GAUSS_TWENTY_FOUR_UNIT {
            for (t, t_weight) in GAUSS_TWENTY_FOUR_UNIT {
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
    for warp in [0.1, 0.2] {
        let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        {
            let quantity_change = warp;
            let quantity_index = 0;
            let quantity_slot = &mut mesh.topology.nodes[4].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] += quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
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
        elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(100.0);
        elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.25);
        model.materials[0].mass_density_kg_per_m3 =
            (Some(1.0)).map(runmat_types::physical::MassDensity::new);
        model.materials[0].damping = None;

        let result = super::run_modal_with_options(
            &model,
            ComputeBackend::Cpu,
            ModalSolveOptions {
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
}
