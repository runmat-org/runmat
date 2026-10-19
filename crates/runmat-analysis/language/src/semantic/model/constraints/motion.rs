use super::super::{properties::*, region_source};
use super::values::{invalid, quantity, translation_dof};
use crate::{
    BoundFeaProgram, CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration,
    CheckedFeaStatement, FeaDiagnostic, PhysicalDimension,
};
use runmat_analysis_core::{
    authored_motion_acceleration_is_compatible, AccelerationSignalId, AuthoredConstraintKind,
    AuthoredConstraintTarget, AuthoredPrescribedStructuralMotion, AuthoredTargetCardinality,
    Displacement, PhysicalSignal, SignalId, StructuralKinematicHistory,
    StructuralMotionReconstruction, TranslationDofMotion, Velocity,
};

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraintKind> {
    validate_assignments(declaration, body, &["target", "translations"], diagnostics);
    let target_expression = required_assignment(declaration, body, "target", diagnostics)?;
    let translations_expression =
        required_assignment(declaration, body, "translations", diagnostics)?;
    let CheckedExpressionKind::List(entries) = &translations_expression.kind else {
        invalid(
            declaration,
            translations_expression,
            "translations",
            "a list of motion records",
            diagnostics,
        );
        return None;
    };
    let translations = entries
        .iter()
        .map(|entry| translation(declaration, signals, entry, diagnostics))
        .collect::<Option<Vec<_>>>()?;
    Some(AuthoredConstraintKind::StructuralPrescribedMotion(
        AuthoredPrescribedStructuralMotion {
            target: AuthoredConstraintTarget {
                region: region_source(bound, declaration, target_expression, diagnostics)?,
                cardinality: AuthoredTargetCardinality::OneOrMore,
            },
            translations,
        },
    ))
}

fn translation(
    declaration: &CheckedFeaDeclaration,
    signals: &[PhysicalSignal],
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<TranslationDofMotion> {
    let CheckedExpressionKind::Record(members) = &expression.kind else {
        invalid(
            declaration,
            expression,
            "translations",
            "motion records",
            diagnostics,
        );
        return None;
    };
    let required = [
        "dof",
        "acceleration",
        "initial_displacement",
        "initial_velocity",
        "reconstruction",
    ];
    if members.len() != required.len() || required.iter().any(|name| !members.contains_key(*name)) {
        invalid(
            declaration,
            expression,
            "translations",
            "records containing exactly `dof`, `acceleration`, `initial_displacement`, `initial_velocity`, and `reconstruction`",
            diagnostics,
        );
        return None;
    }
    let acceleration_expression = members.get("acceleration")?;
    let acceleration = signal_reference(declaration, acceleration_expression, diagnostics)?;
    if !signals
        .iter()
        .any(|signal| valid_acceleration_signal(signal, &acceleration))
    {
        invalid(
            declaration,
            acceleration_expression,
            "acceleration",
            "a declared real global vector acceleration signal with piecewise-linear time history and reject extrapolation",
            diagnostics,
        );
        return None;
    }
    let reconstruction_expression = members.get("reconstruction")?;
    if !matches!(
        &reconstruction_expression.kind,
        CheckedExpressionKind::Enum(name) if name == "piecewise_linear_acceleration_exact"
    ) {
        invalid(
            declaration,
            reconstruction_expression,
            "reconstruction",
            "`piecewise_linear_acceleration_exact`",
            diagnostics,
        );
        return None;
    }
    Some(TranslationDofMotion {
        dof: translation_dof(declaration, members.get("dof")?, diagnostics)?,
        history: StructuralKinematicHistory::AccelerationDriven {
            acceleration: AccelerationSignalId(acceleration),
            initial_displacement: Displacement::new(quantity(
                declaration,
                members.get("initial_displacement")?,
                "initial_displacement",
                PhysicalDimension::LENGTH,
                diagnostics,
            )?),
            initial_velocity: Velocity::new(quantity(
                declaration,
                members.get("initial_velocity")?,
                "initial_velocity",
                PhysicalDimension::VELOCITY,
                diagnostics,
            )?),
            reconstruction: StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
        },
    })
}

fn valid_acceleration_signal(signal: &PhysicalSignal, id: &SignalId) -> bool {
    signal.signal_id() == id && authored_motion_acceleration_is_compatible(signal)
}

fn signal_reference(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalId> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        invalid(
            declaration,
            expression,
            "acceleration",
            "`signal.<name>`",
            diagnostics,
        );
        return None;
    };
    let [namespace, name] = path.as_slice() else {
        invalid(
            declaration,
            expression,
            "acceleration",
            "`signal.<name>`",
            diagnostics,
        );
        return None;
    };
    if namespace != "signal" {
        invalid(
            declaration,
            expression,
            "acceleration",
            "`signal.<name>`",
            diagnostics,
        );
        return None;
    }
    Some(SignalId(name.clone()))
}
