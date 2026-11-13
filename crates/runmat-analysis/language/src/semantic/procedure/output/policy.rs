use std::collections::BTreeSet;

use runmat_analysis_core::{
    AuthoredOutputPolicy, CompressionPolicy, ObservationPurpose, PartialAccessPolicy,
    RetentionPolicy, SignalPrecision,
};

use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic,
};

use super::super::{lower, values};

pub(super) fn lower_output_definition<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(&'a CheckedExpression, AuthoredOutputPolicy)> {
    validate_members(declaration, statements, diagnostics)?;
    let value = values::assignment(statements, "value").or_else(|| {
        lower::error(
            declaration,
            declaration.span,
            "output block requires `value = <expression>`",
            diagnostics,
        );
        None
    })?;
    let mut policy = AuthoredOutputPolicy::default();
    if let Some(value) = values::assignment(statements, "precision") {
        policy.precision = Some(parse_precision(value).or_else(|| {
            invalid(
                declaration,
                value,
                "precision must be real64 or complex128",
                diagnostics,
            )
        })?);
    }
    if let Some(value) = values::assignment(statements, "compression") {
        policy.compression = parse_compression(value, statements).or_else(|| {
            invalid(
                declaration,
                value,
                "compression must be lossless, none, or error_bounded with maximum_relative_error",
                diagnostics,
            )
        })?;
    } else if values::assignment(statements, "maximum_relative_error").is_some() {
        return invalid(
            declaration,
            values::assignment(statements, "maximum_relative_error")?,
            "maximum_relative_error requires error_bounded compression",
            diagnostics,
        );
    }
    if let Some(value) = values::assignment(statements, "retention") {
        policy.retention = parse_retention(value, statements).or_else(|| {
            invalid(
                declaration,
                value,
                "retention must be canonical or until_dependents_complete without a bound, or latest_samples or explicit_samples with a positive maximum_samples",
                diagnostics,
            )
        })?;
    } else if values::assignment(statements, "maximum_samples").is_some() {
        invalid(
            declaration,
            values::assignment(statements, "maximum_samples")?,
            "maximum_samples requires latest_samples or explicit_samples retention",
            diagnostics,
        )?;
    }
    if let Some(value) = values::assignment(statements, "partial_access") {
        policy.partial_access = match enum_name(value).unwrap_or("") {
            "manifest_and_chunks" => PartialAccessPolicy::ManifestAndChunks,
            "whole_artifact" => PartialAccessPolicy::WholeArtifact,
            _ => {
                return invalid(
                    declaration,
                    value,
                    "partial_access must be manifest_and_chunks or whole_artifact",
                    diagnostics,
                )
            }
        };
    }
    if let Some(value) = values::assignment(statements, "purpose") {
        policy.purpose = match enum_name(value).unwrap_or("") {
            "publication" => ObservationPurpose::Publication,
            "solver_control" => ObservationPurpose::SolverControl,
            "visualization" => ObservationPurpose::Visualization,
            "downstream_state" => ObservationPurpose::DownstreamState,
            _ => return invalid(
                declaration,
                value,
                "purpose must be publication, solver_control, visualization, or downstream_state",
                diagnostics,
            ),
        };
    }
    Some((value, policy))
}

fn validate_members(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    const MEMBERS: &[&str] = &[
        "value",
        "precision",
        "compression",
        "maximum_relative_error",
        "retention",
        "maximum_samples",
        "partial_access",
        "purpose",
    ];
    let mut seen = BTreeSet::new();
    for statement in statements {
        let CheckedFeaStatementKind::Assignment { target, .. } = &statement.kind else {
            return invalid_statement(
                declaration,
                statement,
                "output blocks contain only policy assignments",
                diagnostics,
            );
        };
        let [name] = target.as_slice() else {
            return invalid_statement(
                declaration,
                statement,
                "output policy names are not nested paths",
                diagnostics,
            );
        };
        if !MEMBERS.contains(&name.as_str()) {
            return invalid_statement(
                declaration,
                statement,
                &format!("unknown output policy member `{name}`"),
                diagnostics,
            );
        }
        if !seen.insert(name.as_str()) {
            return invalid_statement(
                declaration,
                statement,
                &format!("output policy member `{name}` is duplicated"),
                diagnostics,
            );
        }
    }
    Some(())
}

fn parse_precision(value: &CheckedExpression) -> Option<SignalPrecision> {
    Some(match enum_name(value)? {
        "real64" => SignalPrecision::Real64,
        "complex128" => SignalPrecision::Complex128,
        _ => return None,
    })
}

fn parse_compression(
    value: &CheckedExpression,
    statements: &[CheckedFeaStatement],
) -> Option<CompressionPolicy> {
    let bound = values::assignment(statements, "maximum_relative_error");
    Some(match enum_name(value)? {
        "lossless" if bound.is_none() => CompressionPolicy::Lossless,
        "none" if bound.is_none() => CompressionPolicy::None,
        "error_bounded" => {
            let maximum_relative_error = bound.and_then(values::canonical_scalar)?;
            if !maximum_relative_error.is_finite()
                || maximum_relative_error <= 0.0
                || maximum_relative_error >= 1.0
            {
                return None;
            }
            CompressionPolicy::ErrorBounded {
                maximum_relative_error,
            }
        }
        _ => return None,
    })
}

fn parse_retention(
    value: &CheckedExpression,
    statements: &[CheckedFeaStatement],
) -> Option<RetentionPolicy> {
    let maximum =
        || values::assignment(statements, "maximum_samples").and_then(values::positive_usize);
    Some(match enum_name(value)? {
        "canonical" if maximum().is_none() => RetentionPolicy::Canonical,
        "until_dependents_complete" if maximum().is_none() => {
            RetentionPolicy::UntilDependentsComplete
        }
        "latest_samples" => RetentionPolicy::LatestSamples { count: maximum()? },
        "explicit_samples" => RetentionPolicy::ExplicitSamples {
            maximum_count: maximum()?,
        },
        _ => return None,
    })
}

fn enum_name(value: &CheckedExpression) -> Option<&str> {
    match &value.kind {
        CheckedExpressionKind::Enum(name) => Some(name),
        _ => None,
    }
}

fn invalid<T>(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    message: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    lower::error(declaration, value.span, message, diagnostics);
    None
}

fn invalid_statement<T>(
    declaration: &CheckedFeaDeclaration,
    statement: &CheckedFeaStatement,
    message: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    lower::error(declaration, statement.span, message, diagnostics);
    None
}
