use super::super::{diagnostic, properties::*, region_source};
use super::values::*;
use crate::{
    BoundFeaProgram, CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration,
    CheckedFeaStatement, FeaDiagnostic, PhysicalDimension,
};
use runmat_analysis_core::{
    AuthoredConstraintKind, AuthoredConstraintTarget, AuthoredPairedConstraint,
    AuthoredPairingPolicy, AuthoredPeriodicConstraint, AuthoredPeriodicTransform,
    AuthoredPhaseShiftedPeriodicConstraint, AuthoredReferencePointCoupling,
    AuthoredRotationDofValue, AuthoredStructuralConstraintEquation,
    AuthoredStructuralConstraintEquationTerm, AuthoredStructuralDofs, AuthoredTargetCardinality,
    AuthoredThermalJump, AuthoredTranslationDofValue, ComplexPhase, Displacement, Length,
    PhysicalQuantityKind, PhysicalSignal, Position, ReferencePoint, Rotation,
    StructuralRotationDof, StructuralTranslationDof, TemperatureDifference, Translation,
};

const KINDS: &[&str] = &[
    "structural.prescribed",
    "structural.prescribed_motion",
    "structural.normal_displacement",
    "structural.rigid_coupling",
    "structural.distributing_coupling",
    "structural.tie",
    "structural.periodic",
    "structural.equation",
    "thermal.periodic",
    "thermal.tie",
    "thermal.affine_jump",
    "electrical.periodic",
    "magnetic.periodic",
    "magnetic.phase_shifted_periodic",
    "acoustic.periodic",
    "acoustic.phase_shifted_periodic",
    "fluid.periodic",
];

pub(super) fn recognizes(kind: &str) -> bool {
    KINDS.contains(&kind)
}

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    kind: &str,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraintKind> {
    match kind {
        "structural.prescribed" => prescribed(declaration, bound, signals, body, diagnostics),
        "structural.prescribed_motion" => {
            super::motion::lower(declaration, bound, signals, body, diagnostics)
        }
        "structural.normal_displacement" => {
            validate_assignments(declaration, body, &["target", "value"], diagnostics);
            Some(AuthoredConstraintKind::StructuralNormalDisplacement {
                target: target(declaration, bound, body, "target", diagnostics)?,
                value: prescribed_value(
                    declaration,
                    body,
                    "value",
                    (
                        PhysicalDimension::LENGTH,
                        PhysicalQuantityKind::Displacement,
                    ),
                    signals,
                    Displacement::new,
                    diagnostics,
                )?,
            })
        }
        "structural.rigid_coupling" | "structural.distributing_coupling" => {
            coupling(declaration, bound, kind, body, diagnostics)
        }
        "structural.tie" | "thermal.tie" => {
            validate_assignments(
                declaration,
                body,
                &["primary", "secondary", "tolerance"],
                diagnostics,
            );
            let pair = pair(declaration, bound, body, diagnostics)?;
            Some(if kind == "structural.tie" {
                AuthoredConstraintKind::StructuralTie(pair)
            } else {
                AuthoredConstraintKind::ThermalTie(pair)
            })
        }
        "structural.equation" => equation(declaration, bound, body, diagnostics),
        _ => periodic(declaration, bound, signals, kind, body, diagnostics),
    }
}

fn prescribed(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraintKind> {
    validate_assignments(
        declaration,
        body,
        &["target", "frame", "ux", "uy", "uz", "rx", "ry", "rz"],
        diagnostics,
    );
    let mut translations = Vec::new();
    for (name, dof) in [
        ("ux", StructuralTranslationDof::Ux),
        ("uy", StructuralTranslationDof::Uy),
        ("uz", StructuralTranslationDof::Uz),
    ] {
        if optional_assignment(body, name).is_some() {
            translations.push(AuthoredTranslationDofValue {
                dof,
                value: prescribed_value(
                    declaration,
                    body,
                    name,
                    (
                        PhysicalDimension::LENGTH,
                        PhysicalQuantityKind::Displacement,
                    ),
                    signals,
                    Displacement::new,
                    diagnostics,
                )?,
            });
        }
    }
    let mut rotations = Vec::new();
    for (name, dof) in [
        ("rx", StructuralRotationDof::Rx),
        ("ry", StructuralRotationDof::Ry),
        ("rz", StructuralRotationDof::Rz),
    ] {
        if optional_assignment(body, name).is_some() {
            rotations.push(AuthoredRotationDofValue {
                dof,
                value: prescribed_value(
                    declaration,
                    body,
                    name,
                    (
                        PhysicalDimension::DIMENSIONLESS,
                        PhysicalQuantityKind::Rotation,
                    ),
                    signals,
                    Rotation::new,
                    diagnostics,
                )?,
            });
        }
    }
    Some(AuthoredConstraintKind::StructuralPrescribed(
        AuthoredStructuralDofs {
            target: target(declaration, bound, body, "target", diagnostics)?,
            frame: frame(
                declaration,
                required_assignment(declaration, body, "frame", diagnostics)?,
                diagnostics,
            )?,
            translations,
            rotations,
        },
    ))
}

fn coupling(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    kind: &str,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraintKind> {
    validate_assignments(
        declaration,
        body,
        &["target", "reference_point_id", "position"],
        diagnostics,
    );
    let coupling = AuthoredReferencePointCoupling {
        target: target(declaration, bound, body, "target", diagnostics)?,
        reference_point: ReferencePoint {
            reference_point_id: required_string(
                declaration,
                body,
                "reference_point_id",
                diagnostics,
            )?,
            position: Position::new(required_vector(
                declaration,
                body,
                "position",
                PhysicalDimension::LENGTH,
                diagnostics,
            )?),
        },
    };
    Some(if kind == "structural.rigid_coupling" {
        AuthoredConstraintKind::StructuralRigidCoupling(coupling)
    } else {
        AuthoredConstraintKind::StructuralDistributingCoupling(coupling)
    })
}

fn periodic(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    kind: &str,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraintKind> {
    let mut supported = vec![
        "primary",
        "secondary",
        "tolerance",
        "translation",
        "rotation_origin",
        "rotation_axis",
        "rotation_angle",
    ];
    if kind == "thermal.affine_jump" {
        supported.push("jump");
    }
    if matches!(
        kind,
        "magnetic.phase_shifted_periodic" | "acoustic.phase_shifted_periodic"
    ) {
        supported.push("phase");
    }
    validate_assignments(declaration, body, &supported, diagnostics);
    let periodic = AuthoredPeriodicConstraint {
        pair: pair(declaration, bound, body, diagnostics)?,
        transform: transform(declaration, body, diagnostics)?,
    };
    Some(match kind {
        "structural.periodic" => AuthoredConstraintKind::StructuralPeriodic(periodic),
        "thermal.periodic" => AuthoredConstraintKind::ThermalPeriodic(periodic),
        "thermal.affine_jump" => AuthoredConstraintKind::ThermalAffineJump(AuthoredThermalJump {
            periodic,
            temperature_jump: prescribed_value(
                declaration,
                body,
                "jump",
                (
                    PhysicalDimension::TEMPERATURE,
                    PhysicalQuantityKind::TemperatureDifference,
                ),
                signals,
                TemperatureDifference::new,
                diagnostics,
            )?,
        }),
        "electrical.periodic" => AuthoredConstraintKind::ElectricalPeriodic(periodic),
        "magnetic.periodic" => AuthoredConstraintKind::MagneticPeriodic(periodic),
        "acoustic.periodic" => AuthoredConstraintKind::AcousticPeriodic(periodic),
        "fluid.periodic" => AuthoredConstraintKind::FluidPeriodic(periodic),
        "magnetic.phase_shifted_periodic" => AuthoredConstraintKind::MagneticPhaseShiftedPeriodic(
            phase_shifted(declaration, body, periodic, diagnostics)?,
        ),
        "acoustic.phase_shifted_periodic" => AuthoredConstraintKind::AcousticPhaseShiftedPeriodic(
            phase_shifted(declaration, body, periodic, diagnostics)?,
        ),
        _ => return None,
    })
}

fn pair(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredPairedConstraint> {
    Some(AuthoredPairedConstraint {
        primary: target(declaration, bound, body, "primary", diagnostics)?,
        secondary: target(declaration, bound, body, "secondary", diagnostics)?,
        pairing: match optional_quantity(
            declaration,
            body,
            "tolerance",
            PhysicalDimension::LENGTH,
            diagnostics,
        )? {
            Some(value) => AuthoredPairingPolicy::GeometricTolerance(Length::new(value)),
            None => AuthoredPairingPolicy::Matching,
        },
    })
}

fn transform(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredPeriodicTransform> {
    let translation = optional_vector(
        declaration,
        body,
        "translation",
        PhysicalDimension::LENGTH,
        diagnostics,
    )?;
    let origin = optional_vector(
        declaration,
        body,
        "rotation_origin",
        PhysicalDimension::LENGTH,
        diagnostics,
    )?;
    let axis = optional_vector(
        declaration,
        body,
        "rotation_axis",
        PhysicalDimension::DIMENSIONLESS,
        diagnostics,
    )?;
    let angle = optional_quantity(
        declaration,
        body,
        "rotation_angle",
        runmat_types::physical::DynamicDimension::quantity_type(PhysicalQuantityKind::Rotation),
        diagnostics,
    )?;
    match (translation, origin, axis, angle) {
        (None, None, None, None) => Some(AuthoredPeriodicTransform::Identity),
        (Some(value), None, None, None) => Some(AuthoredPeriodicTransform::Translation(
            Translation::new(value),
        )),
        (None, Some(origin), Some(axis), Some(angle)) => {
            Some(AuthoredPeriodicTransform::Rotation {
                origin: Position::new(origin),
                axis,
                angle: Rotation::new(angle),
            })
        }
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                "periodic transform requires no transform, one translation, or a complete rotation",
                diagnostics,
            );
            None
        }
    }
}

fn phase_shifted(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    periodic: AuthoredPeriodicConstraint,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredPhaseShiftedPeriodicConstraint> {
    Some(AuthoredPhaseShiftedPeriodicConstraint {
        periodic,
        phase: ComplexPhase::new(required_quantity(
            declaration,
            body,
            "phase",
            runmat_types::physical::DynamicDimension::quantity_type(
                PhysicalQuantityKind::ComplexPhase,
            ),
            diagnostics,
        )?),
    })
}

fn equation(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraintKind> {
    validate_assignments(
        declaration,
        body,
        &["terms", "right_hand_side"],
        diagnostics,
    );
    let expression = required_assignment(declaration, body, "terms", diagnostics)?;
    let CheckedExpressionKind::List(entries) = &expression.kind else {
        invalid(
            declaration,
            expression,
            "terms",
            "list of equation terms",
            diagnostics,
        );
        return None;
    };
    let mut terms = Vec::new();
    for entry in entries {
        let CheckedExpressionKind::Record(members) = &entry.kind else {
            invalid(
                declaration,
                entry,
                "terms",
                "equation-term record",
                diagnostics,
            );
            return None;
        };
        if members.len() != 3
            || !members.contains_key("target")
            || !members.contains_key("dof")
            || !members.contains_key("coefficient")
        {
            invalid(
                declaration,
                entry,
                "terms",
                "target/dof/coefficient record",
                diagnostics,
            );
            return None;
        }
        terms.push(AuthoredStructuralConstraintEquationTerm {
            target: target_expression(declaration, bound, members.get("target")?, diagnostics)?,
            dof: translation_dof(declaration, members.get("dof")?, diagnostics)?,
            coefficient: dimensionless_scalar(
                declaration,
                members.get("coefficient")?,
                "coefficient",
                diagnostics,
            )?,
        });
    }
    Some(AuthoredConstraintKind::StructuralEquation(
        AuthoredStructuralConstraintEquation {
            terms,
            right_hand_side: Displacement::new(required_quantity(
                declaration,
                body,
                "right_hand_side",
                PhysicalDimension::LENGTH,
                diagnostics,
            )?),
        },
    ))
}

fn target(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    body: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraintTarget> {
    target_expression(
        declaration,
        bound,
        required_assignment(declaration, body, name, diagnostics)?,
        diagnostics,
    )
}

fn target_expression(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraintTarget> {
    Some(AuthoredConstraintTarget {
        region: region_source(bound, declaration, expression, diagnostics)?,
        cardinality: AuthoredTargetCardinality::OneOrMore,
    })
}
