use super::*;
use crate::physics::maxwell::ComplexPhasor;
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures, ElementOrder};
const FREQUENCY: f64 = 1e8;

#[test]
fn receipt_evidence_never_qualifies_missing_maxwell_discretization() {
    use crate::execution::{AccuracyEvaluationRequest, AccuracySourceKind, AccuracySourceReceipt};
    use crate::results::accuracy::{
        evaluate_accuracy, publication_gate, AccuracyStatus, PublicationDecision,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let workload = workload();
        let mesh = mesh(order);
        let result = workload.execute(&mesh).unwrap();
        let receipt = MaxwellNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
        let source = AccuracySourceReceipt::from_canonical_bytes(
            AccuracySourceKind::MaxwellHarmonic,
            &receipt.canonical_bytes().unwrap(),
        )
        .unwrap();
        let mut plan = AccuracyPlan::baseline("maxwell_accuracy");
        plan.criteria = [
            (
                "residual",
                ErrorContributionKind::Algebraic,
                RequiredAccuracyEvidence::ResidualBound,
            ),
            (
                "power",
                ErrorContributionKind::Algebraic,
                RequiredAccuracyEvidence::ConservationBalance,
            ),
            (
                "spatial",
                ErrorContributionKind::SpatialDiscretization,
                RequiredAccuracyEvidence::ConvergenceStudy,
            ),
        ]
        .into_iter()
        .map(|(id, contribution, evidence)| AccuracyCriterion {
            criterion_id: id.into(),
            target: AccuracyTarget::GlobalEquilibrium,
            contribution,
            evidence,
            tolerance: AccuracyTolerance::Relative(1e-7),
            mandatory: true,
        })
        .collect();
        let request = AccuracyEvaluationRequest::new(
            receipt.receipt().producer,
            AccuracySourceKind::MaxwellHarmonic,
            receipt.source_result(),
            None,
            plan,
        )
        .unwrap();
        let evidence = source.evidence(&request).unwrap();
        assert_eq!(evidence.len(), 2);
        let evaluated = evaluate_accuracy(&request.plan, &evidence).unwrap();
        assert_eq!(evaluated.report().status, AccuracyStatus::Indeterminate);
        for contribution in &evaluated.report().contributions {
            assert_eq!(
                contribution.status,
                if contribution.criterion_id == "spatial" {
                    AccuracyStatus::Indeterminate
                } else {
                    AccuracyStatus::Satisfied
                }
            );
        }
        assert_eq!(
            publication_gate(&request.plan, &evaluated)
                .unwrap()
                .decision,
            PublicationDecision::Block
        );
        let mut algebraic = request.plan.clone();
        algebraic.criteria.truncate(2);
        assert_eq!(
            evaluate_accuracy(&algebraic, &evidence)
                .unwrap()
                .report()
                .status,
            AccuracyStatus::Satisfied
        );
        let mut foreign = workload.clone();
        foreign.model.sequence.sequence_id = "foreign".into();
        let foreign_request = AccuracyEvaluationRequest::new(
            foreign.producer_identity().unwrap(),
            request.source_kind,
            request.source_result,
            None,
            request.plan.clone(),
        )
        .unwrap();
        assert!(source.evidence(&foreign_request).is_err());
    }
}

pub(super) fn workload() -> MaxwellHarmonicWorkload {
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
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(boundary)),
                    ),
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
        relative_residual_target: 1e-10,
        frequency: runmat_analysis_core::Frequency::new(FREQUENCY),
    }
}
pub(super) fn mesh(order: ElementOrder) -> SolverMeshArtifact {
    match order {
        ElementOrder::Tet4 => fixtures::slab_mesh(2),
        ElementOrder::Tet10 => fixtures::quadratic_slab_mesh(2),
    }
}
#[test]
fn original_maxwell_receipt_roundtrip_rejects_balance_phase_and_source_substitution() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order);
        let workload = workload();
        let result = workload.execute(&mesh).unwrap();
        let receipt = MaxwellNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
        let bytes = receipt.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= MAXWELL_RECEIPT_MAX_ENCODED_BYTES);
        let cold = MaxwellNumericalReceiptArtifact::from_canonical_bytes(&bytes).unwrap();
        cold.validate_against(&workload, &result, &mesh).unwrap();
        assert_eq!(receipt, cold);
        for (field, value) in [
            ("relative_physical_residual", serde_json::json!(1.0)),
            (
                "relative_charge_continuity_residual",
                serde_json::json!(1.0),
            ),
            ("reactive_power_var", serde_json::json!(1.0)),
            ("integration_samples", serde_json::json!(1)),
            ("angular_frequency_rad_per_s", serde_json::json!(0.0)),
        ] {
            let mut wire = serde_json::to_value(&cold).unwrap();
            wire["receipt"][field] = value;
            assert!(serde_json::from_value::<MaxwellNumericalReceiptArtifact>(wire).is_err());
        }
        let mut foreign = workload.clone();
        foreign.relative_residual_target *= 2.0;
        assert!(cold.validate_against(&foreign, &result, &mesh).is_err());
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
            result
                .response()
                .angular_frequency_rad_per_s
                .canonical_value(),
            std::f64::consts::TAU * FREQUENCY
        );
        for source in &result.input().source_current {
            assert_eq!(
                source.current_density_a_per_m2.canonical_value(),
                [
                    ComplexPhasor::ZERO,
                    ComplexPhasor::new(0.6, 0.8),
                    ComplexPhasor::ZERO
                ]
            );
        }
        for law in &result.input().volumes {
            assert_eq!(
                law.permeability_h_per_m.canonical_value(),
                crate::physics::magnetic::VACUUM_PERMEABILITY_H_PER_M.canonical_value()
            );
            assert_eq!(
                law.permittivity_f_per_m.canonical_value(),
                4.0 * crate::physics::electrical::VACUUM_PERMITTIVITY_F_PER_M.canonical_value()
            );
            assert_eq!(law.conductivity_s_per_m.canonical_value(), 3.0);
        }
        assert!(result.response().ohmic_power_w.canonical_value() > 0.0);
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
            .conductivity_s_per_m = runmat_types::physical::ElectricalConductivity::new(4.0);
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
    // A face without a ground is a symmetry plane of the current (n x H = 0), not an error.
    changed.execute(&mesh).unwrap();
    let mut changed = original.clone();
    changed.model.constraints[0] = AnalysisConstraint::Electrical(ElectricalConstraint {
        constraint_id: "potential".into(),
        kind: ElectricalConstraintKind::PrescribedPotential {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_left",
            )),
            value: PrescribedValue::Constant(ElectricPotential::new(0.0)),
        },
    });
    reject(changed);
    let mut changed = original.clone();
    changed.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![FREQUENCY * 2.0]);
    reject(changed);
    // Admitted: another authored frequency (its own case), zero conductivity, a
    // resistive-heating coefficient, and a magnetic ground in place of the PEC wall.
    let accept = |changed: MaxwellHarmonicWorkload| changed.execute(&mesh).unwrap();
    let mut changed = original.clone();
    changed.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![FREQUENCY, FREQUENCY * 2.0]);
    accept(changed);
    let mut changed = original.clone();
    changed.model.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .resistive_heating_coefficient = runmat_types::physical::InverseTemperature::new(0.1);
    accept(changed);
    let mut changed = original.clone();
    changed.model.constraints[0] = AnalysisConstraint::Magnetic(MagneticConstraint {
        constraint_id: "magnetic".into(),
        kind: MagneticConstraintKind::VectorPotentialGround {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_left",
            )),
        },
    });
    accept(changed);
}
