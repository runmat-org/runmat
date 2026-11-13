use super::{optional_frame, policy, signal_value, SignalWrapper};
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    FeaDiagnostic, FeaType, PhysicalDimension,
};
use runmat_analysis_core::{
    PhysicalSignal, Signal, SignalApproximationEvidence, SignalId, SignalKnot, SignalKnotValue,
    SignalOrdinate, SignalPointContinuity, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalShape, SignalValue, TabulatedSignal, TabulatedSignalDomain,
};

use super::super::{diagnostic, properties::*};

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    domain_name: &str,
    body: &[CheckedFeaStatement],
    quantity_kind: runmat_types::physical::QuantityKind,
    wrap: SignalWrapper,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PhysicalSignal> {
    validate_assignments(
        declaration,
        body,
        &[
            "axis",
            "samples",
            "jumps",
            "interpolation",
            "extrapolation",
            "frame",
        ],
        diagnostics,
    );
    let (domain, abscissa_dimension) = domain(declaration, domain_name, body, diagnostics)?;
    let mut knots = Vec::new();
    if let Some(samples) = assignment(body, "samples") {
        knots.extend(continuous_knots(
            declaration,
            samples,
            abscissa_dimension,
            quantity_kind,
            diagnostics,
        )?);
    }
    if let Some(jumps) = assignment(body, "jumps") {
        knots.extend(jump_knots(
            declaration,
            jumps,
            abscissa_dimension,
            quantity_kind,
            diagnostics,
        )?);
    }
    if knots.is_empty() {
        diagnostic(
            declaration,
            declaration.span,
            "tabulated signal requires at least one sample",
            diagnostics,
        );
        return None;
    }
    knots.sort_by(|left, right| left.abscissa.total_cmp(&right.abscissa));
    if knots
        .windows(2)
        .any(|pair| pair[0].abscissa.to_bits() == pair[1].abscissa.to_bits())
    {
        diagnostic(
            declaration,
            declaration.span,
            "tabulated signal coordinates must be unique",
            diagnostics,
        );
        return None;
    }
    let first = knot_ordinate(&knots[0]);
    let shape_source = ordinate_real(first);
    let vector = matches!(shape_source, SignalValue::Vector3(_));
    let frame = optional_frame(declaration, body, vector, diagnostics)?;
    let complex = knots
        .iter()
        .any(|knot| ordinate_is_complex(knot_ordinate(knot)));
    let interpolation = policy::interpolation(declaration, body, diagnostics)?;
    let extrapolation = policy::extrapolation(declaration, body, diagnostics)?;
    let sample_count = knots.len();
    Some(wrap(Signal {
        signal_id: SignalId(id.to_string()),
        shape: if vector {
            SignalShape::Vector3
        } else {
            SignalShape::Scalar
        },
        frame,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain,
            knots,
            interpolation,
            extrapolation,
        }),
        precision: if complex {
            SignalPrecision::Complex128
        } else {
            SignalPrecision::Real64
        },
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count,
        },
        provenance: SignalProvenance::Authored,
    }))
}

fn domain(
    declaration: &CheckedFeaDeclaration,
    name: &str,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(
    TabulatedSignalDomain,
    runmat_types::physical::DynamicDimension,
)> {
    match name {
        "time" => Some((
            TabulatedSignalDomain::TimeSeconds,
            PhysicalDimension::TIME.into(),
        )),
        "frequency" => Some((
            TabulatedSignalDomain::FrequencyHertz,
            PhysicalDimension::FREQUENCY.into(),
        )),
        "parameter" => {
            let axis = required_assignment(declaration, body, "axis", diagnostics)?;
            let CheckedExpressionKind::Reference(path) = &axis.kind else {
                invalid(
                    declaration,
                    axis,
                    "axis must reference `input.<name>`",
                    diagnostics,
                );
                return None;
            };
            let [namespace, parameter_id] = path.as_slice() else {
                invalid(
                    declaration,
                    axis,
                    "axis must reference `input.<name>`",
                    diagnostics,
                );
                return None;
            };
            if namespace != "input" {
                invalid(
                    declaration,
                    axis,
                    "axis must reference `input.<name>`",
                    diagnostics,
                );
                return None;
            }
            let dimension = match axis.value_type {
                FeaType::Integer | FeaType::Real => PhysicalDimension::DIMENSIONLESS.into(),
                FeaType::Quantity(dimension) => dimension,
                _ => {
                    invalid(
                        declaration,
                        axis,
                        "study-parameter axis must be numeric or a physical quantity",
                        diagnostics,
                    );
                    return None;
                }
            };
            Some((
                TabulatedSignalDomain::StudyParameter {
                    parameter_id: parameter_id.clone(),
                    quantity: Some(dimension),
                },
                dimension,
            ))
        }
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported tabulated signal domain `{name}`"),
                diagnostics,
            );
            None
        }
    }
}

fn continuous_knots(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    abscissa_dimension: runmat_types::physical::DynamicDimension,
    quantity_kind: runmat_types::physical::QuantityKind,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Vec<SignalKnot<f64>>> {
    let entries = list(declaration, expression, "samples", diagnostics)?;
    entries
        .iter()
        .map(|entry| {
            let members = record(declaration, entry, &["at", "value"], diagnostics)?;
            Some(SignalKnot::continuous(
                coordinate(
                    declaration,
                    members.get("at")?,
                    abscissa_dimension,
                    diagnostics,
                )?,
                ordinate(
                    declaration,
                    members.get("value")?,
                    quantity_kind,
                    diagnostics,
                )?,
            ))
        })
        .collect()
}

fn jump_knots(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    abscissa_dimension: runmat_types::physical::DynamicDimension,
    quantity_kind: runmat_types::physical::QuantityKind,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Vec<SignalKnot<f64>>> {
    let entries = list(declaration, expression, "jumps", diagnostics)?;
    entries
        .iter()
        .map(|entry| {
            let members = record(
                declaration,
                entry,
                &["at", "left", "right", "continuity"],
                diagnostics,
            )?;
            let continuity = match policy::enum_name(members.get("continuity")?) {
                Some("left") => SignalPointContinuity::Left,
                Some("right") => SignalPointContinuity::Right,
                _ => {
                    invalid(
                        declaration,
                        members.get("continuity")?,
                        "jump continuity must be `left` or `right`",
                        diagnostics,
                    );
                    return None;
                }
            };
            Some(SignalKnot::jump(
                coordinate(
                    declaration,
                    members.get("at")?,
                    abscissa_dimension,
                    diagnostics,
                )?,
                ordinate(
                    declaration,
                    members.get("left")?,
                    quantity_kind,
                    diagnostics,
                )?,
                ordinate(
                    declaration,
                    members.get("right")?,
                    quantity_kind,
                    diagnostics,
                )?,
                continuity,
            ))
        })
        .collect()
}

fn ordinate(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    dimension: runmat_types::physical::QuantityKind,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalOrdinate<f64>> {
    if let CheckedExpressionKind::Record(members) = &expression.kind {
        if members.len() == 2 && members.contains_key("real") && members.contains_key("imaginary") {
            return Some(SignalOrdinate::Complex {
                real: signal_value(declaration, members.get("real")?, dimension, diagnostics)?,
                imaginary: signal_value(
                    declaration,
                    members.get("imaginary")?,
                    dimension,
                    diagnostics,
                )?,
            });
        }
    }
    signal_value(declaration, expression, dimension, diagnostics).map(SignalOrdinate::Real)
}

fn coordinate(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    dimension: runmat_types::physical::DynamicDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let valid_type = if dimension == PhysicalDimension::DIMENSIONLESS.into() {
        (matches!(expression.value_type, FeaType::Integer | FeaType::Real)
            || expression
                .value_type
                .is_quantity(PhysicalDimension::DIMENSIONLESS.into()))
    } else {
        (expression.value_type).is_quantity(dimension)
    };
    if !valid_type {
        invalid(
            declaration,
            expression,
            "sample coordinate has the wrong physical dimension",
            diagnostics,
        );
        return None;
    }
    scalar(expression).or_else(|| {
        invalid(
            declaration,
            expression,
            "sample coordinate must be a finite scalar",
            diagnostics,
        );
        None
    })
}

fn list<'a>(
    declaration: &CheckedFeaDeclaration,
    expression: &'a CheckedExpression,
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a [CheckedExpression]> {
    let CheckedExpressionKind::List(entries) = &expression.kind else {
        invalid(
            declaration,
            expression,
            &format!("{name} must be a list of records"),
            diagnostics,
        );
        return None;
    };
    Some(entries)
}

fn record<'a>(
    declaration: &CheckedFeaDeclaration,
    expression: &'a CheckedExpression,
    required: &[&str],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a std::collections::BTreeMap<String, CheckedExpression>> {
    let CheckedExpressionKind::Record(members) = &expression.kind else {
        invalid(
            declaration,
            expression,
            "sample must be a record",
            diagnostics,
        );
        return None;
    };
    if members.len() != required.len() || required.iter().any(|name| !members.contains_key(*name)) {
        invalid(
            declaration,
            expression,
            &format!("sample record requires exactly `{}`", required.join("`, `")),
            diagnostics,
        );
        return None;
    }
    Some(members)
}

fn assignment<'a>(body: &'a [CheckedFeaStatement], name: &str) -> Option<&'a CheckedExpression> {
    body.iter().find_map(|statement| match &statement.kind {
        crate::CheckedFeaStatementKind::Assignment { target, value, .. }
            if target.as_slice() == [name] =>
        {
            Some(value)
        }
        _ => None,
    })
}

fn knot_ordinate(knot: &SignalKnot<f64>) -> &SignalOrdinate<f64> {
    match &knot.value {
        SignalKnotValue::Continuous(value) => value,
        SignalKnotValue::Jump { left_limit, .. } => left_limit,
    }
}

fn ordinate_real(ordinate: &SignalOrdinate<f64>) -> &SignalValue<f64> {
    match ordinate {
        SignalOrdinate::Real(value) | SignalOrdinate::Complex { real: value, .. } => value,
    }
}

fn ordinate_is_complex(ordinate: &SignalOrdinate<f64>) -> bool {
    matches!(ordinate, SignalOrdinate::Complex { .. })
}

fn invalid(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    message: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostic(
        declaration,
        expression.span,
        message.to_string(),
        diagnostics,
    );
}
