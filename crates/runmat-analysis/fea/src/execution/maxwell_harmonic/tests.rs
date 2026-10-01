use super::*;
use crate::physics::maxwell::ComplexPhasor;
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures, ElementOrder};
const FREQUENCY: f64 = 1e8;

fn workload() -> MaxwellHarmonicWorkload {
    let mut model = crate::physics::magnetic::test_support::authored_model(1.0);
    model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(FREQUENCY),
    });
    let mut step = ProcedureStep::electromagnetic_harmonic("potential");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![FREQUENCY]);
    model.sequence = AnalysisSequence::from_steps("maxwell", vec![step]);
    model.constraints.clear();
    for boundary in ["slab_left", "slab_right", "slab_insulated"] {
        model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: boundary.into(),
                kind: ElectricalConstraintKind::ElectricPerfectConductor {
                    target: RegionTarget::boundary(boundary),
                },
            }));
    }
    let PhysicalSignal::ElectricCurrentDensity(signal) = &mut model.signals[0] else {
        panic!()
    };
    let vector = |v: [f64; 3]| SignalValue::Vector3(v.map(ElectricCurrentDensity::new));
    signal.precision = SignalPrecision::Complex64;
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: vec![SignalKnot::continuous(
            FREQUENCY,
            SignalOrdinate::Complex {
                real: vector([0.0, 0.6, 0.0]),
                imaginary: vector([0.0, 0.8, 0.0]),
            },
        )],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::HoldEndpoint,
    });
    MaxwellHarmonicWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        formulation_requirement: FormulationRequirement::Experimental,
        relative_residual_target: 1e-10,
    }
}
fn mesh(order: ElementOrder) -> SolverMeshArtifact {
    match order {
        ElementOrder::Tet4 => fixtures::slab_mesh(2),
        ElementOrder::Tet10 => fixtures::quadratic_slab_mesh(2),
    }
}
#[test]
fn authored_complex_current_materials_and_cold_retained_fields_preserve_original_authority() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order);
        let workload = workload();
        let cold =
            MaxwellHarmonicWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(workload.identity().unwrap(), cold.identity().unwrap());
        let result = cold.execute(&mesh).unwrap();
        assert_eq!(
            result.response().angular_frequency_rad_per_s,
            std::f64::consts::TAU * FREQUENCY
        );
        for source in &result.input().source_current {
            assert_eq!(
                source.current_density_a_per_m2,
                [
                    ComplexPhasor::ZERO,
                    ComplexPhasor::new(0.6, 0.8),
                    ComplexPhasor::ZERO
                ]
            );
        }
        for law in &result.input().volumes {
            assert_eq!(
                law.permeability_h_per_m,
                crate::physics::magnetic::VACUUM_PERMEABILITY_H_PER_M
            );
            assert_eq!(
                law.permittivity_f_per_m,
                4.0 * crate::physics::electrical::VACUUM_PERMITTIVITY_F_PER_M
            );
            assert_eq!(law.conductivity_s_per_m, 3.0);
        }
        assert!(result.response().ohmic_power_w > 0.0);
        assert!(result.response().relative_physical_residual < 8e-10);
        assert!(result.response().relative_complex_power_error < 1.6e-9);
        let bytes = result.canonical_bytes().unwrap();
        let expected = Digest::sha256(&bytes);
        let recovered =
            MaxwellHarmonicSolverResult::from_canonical_bytes_for(&bytes, expected, &cold, &mesh)
                .unwrap();
        assert_eq!(result, recovered);
        assert_eq!(result.identity().unwrap(), recovered.identity().unwrap());
        assert!(MaxwellHarmonicSolverResult::from_canonical_bytes_for(
            &bytes,
            Digest::sha256(b"foreign"),
            &cold,
            &mesh
        )
        .is_err());
        let mut foreign = cold.clone();
        foreign.model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .conductivity_s_per_m = 4.0;
        assert!(recovered.validate_for(&foreign, &mesh).is_err());
        let mut substituted = serde_json::to_value(&recovered).unwrap();
        substituted["response"]["samples"][0]["electric_field_v_per_m"][0]["real"] =
            serde_json::json!(1.0);
        assert!(serde_json::from_value::<MaxwellHarmonicSolverResult>(substituted).is_err());
        let foreign_mesh = fixtures::slab_mesh(1);
        assert!(recovered.validate_for(&cold, &foreign_mesh).is_err());
    }
}
#[test]
fn authored_frequency_pec_material_source_and_lifecycle_negatives_fail_closed() {
    let mesh = mesh(ElementOrder::Tet4);
    let reject = |changed: MaxwellHarmonicWorkload| assert!(changed.execute(&mesh).is_err());
    let original = workload();
    let mut changed = original.clone();
    changed.model.constraints.pop();
    reject(changed);
    let mut changed = original.clone();
    changed.model.constraints[0] = AnalysisConstraint::Electrical(ElectricalConstraint {
        constraint_id: "potential".into(),
        kind: ElectricalConstraintKind::PrescribedPotential {
            target: RegionTarget::boundary("slab_left"),
            value: PrescribedValue::Constant(ElectricPotential::new(0.0)),
        },
    });
    reject(changed);
    let mut changed = original.clone();
    changed.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![FREQUENCY * 2.0]);
    reject(changed);
    let mut changed = original.clone();
    changed.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![FREQUENCY, FREQUENCY * 2.0]);
    reject(changed);
    let mut changed = original.clone();
    changed.formulation_requirement = FormulationRequirement::Production;
    reject(changed);
    let mut changed = original.clone();
    changed.model.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_s_per_m = 0.0;
    reject(changed);
    let mut changed = original.clone();
    changed.model.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .resistive_heating_coefficient = 0.1;
    reject(changed);
    let mut changed = original.clone();
    changed.model.constraints[0] = AnalysisConstraint::Magnetic(MagneticConstraint {
        constraint_id: "magnetic".into(),
        kind: MagneticConstraintKind::VectorPotentialGround {
            target: RegionTarget::boundary("slab_left"),
        },
    });
    reject(changed);
}
