use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, PhysicalSignal, PrescribedStructuralDofs,
    PrescribedValue, RegionTarget, Rotation, RotationDofValue, Signal, SignalApproximationEvidence,
    SignalExtrapolation, SignalId, SignalInterpolation, SignalKnot, SignalOrdinate,
    SignalPointContinuity, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape,
    SignalValue, StructuralConstraint, StructuralConstraintKind, StructuralRotationDof,
    StructuralTranslationDof, TabulatedSignal, TabulatedSignalDomain, TranslationDofValue,
};

use crate::fixtures::{fixture_model, FixtureId};

use super::{validate, TransientConstraintAdmissionError};

fn model() -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.signals.clear();
    model
}

fn prescribed(
    id: &str,
    translations: Vec<TranslationDofValue>,
    rotations: Vec<RotationDofValue>,
) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: id.into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::authored(runmat_meshing_core::fixtures::fixture_face("root")),
            translations,
            rotations,
        }),
    })
}

fn signal_evidence(sample_count: usize) -> SignalApproximationEvidence {
    SignalApproximationEvidence {
        max_absolute_error: Some(0.0),
        max_relative_error: Some(0.0),
        derivative_max_absolute_error: None,
        sample_count,
    }
}

fn displacement_ordinate(value: f64) -> SignalOrdinate<Displacement> {
    SignalOrdinate::Real(SignalValue::Scalar(Displacement::new(value)))
}

fn rotation_ordinate(value: f64) -> SignalOrdinate<Rotation> {
    SignalOrdinate::Real(SignalValue::Scalar(Rotation::new(value)))
}

#[test]
fn inline_and_signal_constants_are_admitted() {
    let mut model = model();
    let displacement_id = SignalId("constant-displacement".into());
    let rotation_id = SignalId("constant-rotation".into());
    model.signals.push(PhysicalSignal::Displacement(Signal {
        signal_id: displacement_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(Displacement::new(
            0.25,
        ))),
        precision: SignalPrecision::Real64,
        evidence: signal_evidence(1),
        provenance: SignalProvenance::Authored,
    }));
    model.signals.push(PhysicalSignal::Rotation(Signal {
        signal_id: rotation_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(Rotation::new(0.5))),
        precision: SignalPrecision::Real64,
        evidence: signal_evidence(1),
        provenance: SignalProvenance::Authored,
    }));
    model.constraints.push(prescribed(
        "constant-kinematics",
        vec![
            TranslationDofValue {
                dof: StructuralTranslationDof::Ux,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            },
            TranslationDofValue {
                dof: StructuralTranslationDof::Uy,
                value: PrescribedValue::Signal(displacement_id),
            },
        ],
        vec![RotationDofValue {
            dof: StructuralRotationDof::Rz,
            value: PrescribedValue::Signal(rotation_id),
        }],
    ));

    validate(&model).expect("study-constant structural kinematics should be admitted");
}

#[test]
fn time_varying_translation_is_rejected() {
    let mut model = model();
    let signal_id = SignalId("translation-history".into());
    model
        .signals
        .push(PhysicalSignal::Displacement(tabulated_displacement(
            signal_id.clone(),
            vec![
                SignalKnot::continuous(0.0, displacement_ordinate(0.0)),
                SignalKnot::continuous(1.0, displacement_ordinate(1.0)),
            ],
        )));
    model.constraints.push(prescribed(
        "moving-support",
        vec![TranslationDofValue {
            dof: StructuralTranslationDof::Ux,
            value: PrescribedValue::Signal(signal_id),
        }],
        Vec::new(),
    ));

    assert_eq!(
        validate(&model),
        Err(
            TransientConstraintAdmissionError::TimeVaryingPrescribedKinematics {
                constraint_id: "moving-support".into(),
                quantity: "displacement",
            }
        )
    );
}

#[test]
fn discontinuous_normal_displacement_is_rejected() {
    let mut model = model();
    let signal_id = SignalId("normal-jump".into());
    model
        .signals
        .push(PhysicalSignal::Displacement(tabulated_displacement(
            signal_id.clone(),
            vec![
                SignalKnot::continuous(0.0, displacement_ordinate(0.0)),
                SignalKnot::jump(
                    0.5,
                    displacement_ordinate(0.0),
                    displacement_ordinate(1.0),
                    SignalPointContinuity::Right,
                ),
                SignalKnot::continuous(1.0, displacement_ordinate(1.0)),
            ],
        )));
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "jumping-normal-support".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::NormalDisplacement {
                target: RegionTarget::authored(runmat_meshing_core::fixtures::fixture_face("root")),
                value: PrescribedValue::Signal(signal_id),
            },
        }));

    assert_eq!(
        validate(&model),
        Err(
            TransientConstraintAdmissionError::TimeVaryingPrescribedKinematics {
                constraint_id: "jumping-normal-support".into(),
                quantity: "displacement",
            }
        )
    );
}

#[test]
fn time_varying_rotation_is_rejected() {
    let mut model = model();
    let signal_id = SignalId("rotation-history".into());
    model.signals.push(PhysicalSignal::Rotation(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(0.0, rotation_ordinate(0.0)),
                SignalKnot::continuous(1.0, rotation_ordinate(0.1)),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: signal_evidence(2),
        provenance: SignalProvenance::Authored,
    }));
    model.constraints.push(prescribed(
        "rotating-support",
        Vec::new(),
        vec![RotationDofValue {
            dof: StructuralRotationDof::Rx,
            value: PrescribedValue::Signal(signal_id),
        }],
    ));

    assert_eq!(
        validate(&model),
        Err(
            TransientConstraintAdmissionError::TimeVaryingPrescribedKinematics {
                constraint_id: "rotating-support".into(),
                quantity: "rotation",
            }
        )
    );
}

fn tabulated_displacement(
    signal_id: SignalId,
    knots: Vec<SignalKnot<Displacement>>,
) -> Signal<Displacement> {
    let sample_count = knots.len();
    Signal {
        signal_id,
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots,
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: signal_evidence(sample_count),
        provenance: SignalProvenance::Authored,
    }
}
