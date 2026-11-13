//! Explicit physical sample axes and numerical representation of a generated signal.
use super::*;
use crate::PhysicalDimension;

pub(super) fn sampling(
    declaration: &CheckedFeaDeclaration,
    domain: &str,
    quantity_kind: runmat_types::physical::QuantityKind,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(SignalSamplingPolicy, Option<f64>)> {
    if let Some(axis) = assignment(statements, "sample_points") {
        if statements.iter().any(|statement| {
            matches!(
                statement.kind,
                CheckedFeaStatementKind::UniformSample { .. }
            )
        }) {
            error(
                declaration,
                axis.span,
                "generated signal must choose sample_points or uniform sampling, not both",
                diagnostics,
            );
            return None;
        }
        return Some((
            points(declaration, domain, axis, diagnostics)?,
            tolerance(declaration, quantity_kind, statements, diagnostics),
        ));
    }
    let samples = statements
        .iter()
        .filter_map(|statement| match &statement.kind {
            CheckedFeaStatementKind::UniformSample { start, end, rate } => Some((start, end, rate)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if samples.len() != 1 {
        error(
            declaration,
            declaration.span,
            "generated signal requires exactly one uniform sampling declaration",
            diagnostics,
        );
        return None;
    }
    let (start, end, rate) = samples[0];
    if domain != "time"
        || !(start.value_type).is_quantity((PhysicalDimension::TIME).into())
        || !(end.value_type).is_quantity((PhysicalDimension::TIME).into())
        || !(rate.value_type).is_quantity((PhysicalDimension::FREQUENCY).into())
    {
        error(
            declaration,
            start.span,
            "uniform generated sampling currently requires a time domain, time bounds, and frequency rate",
            diagnostics,
        );
        return None;
    }
    let start = scalar(start)?;
    let end = scalar(end)?;
    let rate = scalar(rate)?;
    let intervals = (end - start) * rate;
    let rounded = intervals.round();
    if !start.is_finite()
        || !end.is_finite()
        || !rate.is_finite()
        || start >= end
        || rate <= 0.0
        || (intervals - rounded).abs() > 1e-9 * intervals.abs().max(1.0)
        || rounded > (usize::MAX - 1) as f64
    {
        error(
            declaration,
            declaration.span,
            "sampling bounds and rate must define at least two finite, exactly spaced samples",
            diagnostics,
        );
        return None;
    }
    Some((
        SignalSamplingPolicy::Uniform {
            start,
            end,
            sample_count: rounded as usize + 1,
        },
        tolerance(declaration, quantity_kind, statements, diagnostics),
    ))
}

fn tolerance(
    declaration: &CheckedFeaDeclaration,
    quantity_kind: runmat_types::physical::QuantityKind,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let tolerance =
        assignment(statements, "tolerance").and_then(|value| {
            let tolerance_kind =
                if quantity_kind == runmat_types::physical::QuantityKind::Temperature {
                    runmat_types::physical::QuantityKind::TemperatureDifference
                } else {
                    quantity_kind
                };
            if !value.value_type.is_quantity(
                runmat_types::physical::DynamicDimension::quantity_type(tolerance_kind),
            ) || value.validate_quantity_kind(tolerance_kind).is_err()
            {
                error(
                    declaration,
                    value.span,
                    "signal tolerance must have the declared signal quantity",
                    diagnostics,
                );
                None
            } else {
                scalar(value)
            }
        });
    if tolerance.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        error(
            declaration,
            declaration.span,
            "signal tolerance must be finite and positive",
            diagnostics,
        );
        return None;
    }
    tolerance
}

pub(super) fn points(
    declaration: &CheckedFeaDeclaration,
    domain: &str,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalSamplingPolicy> {
    let dimension = match domain {
        "time" => PhysicalDimension::TIME,
        "frequency" => PhysicalDimension::FREQUENCY,
        _ => return None,
    };
    let CheckedExpressionKind::List(entries) = &value.kind else {
        error(
            declaration,
            value.span,
            "sample_points requires a physical-axis list",
            diagnostics,
        );
        return None;
    };
    let points = entries
        .iter()
        .map(|entry| {
            ((entry.value_type).is_quantity((dimension).into()))
                .then(|| scalar(entry))
                .flatten()
        })
        .collect::<Option<Vec<_>>>();
    let Some(points) = points.filter(|points| {
        points.len() >= 2
            && points.iter().all(|point| point.is_finite())
            && points.windows(2).all(|pair| pair[0] < pair[1])
            && (domain != "frequency" || points[0] >= 0.0)
    }) else {
        error(declaration, value.span, "sample_points requires at least two finite, strictly increasing constants in the signal's physical axis; frequencies must be nonnegative", diagnostics);
        return None;
    };
    Some(SignalSamplingPolicy::CanonicalPoints(points))
}

pub(super) fn precision(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalPrecision> {
    let Some(value) = assignment(statements, "precision") else {
        return Some(SignalPrecision::Real64);
    };
    match super::super::policy::enum_name(value) {
        Some("real64") => Some(SignalPrecision::Real64),
        Some("complex128") => Some(SignalPrecision::Complex128),
        _ => {
            error(
                declaration,
                value.span,
                "generated signal precision must be real64 or complex128",
                diagnostics,
            );
            None
        }
    }
}
