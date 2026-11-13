use runmat_analysis_core::{AuthoredSeriesDomain, Position, SamplingPolicy, SeriesWeighting};

use crate::{CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, FeaDiagnostic};

use super::super::{lower, values};

pub(super) fn interpolation_positions(value: &CheckedExpression) -> Option<Vec<Position>> {
    if let Some(meters) = values::canonical_vector3(value) {
        return Some(vec![Position::new(meters)]);
    }
    let CheckedExpressionKind::List(values) = &value.kind else {
        return None;
    };
    (!values.is_empty())
        .then(|| {
            values
                .iter()
                .map(|value| {
                    Some(Position::new(super::super::values::canonical_vector3(
                        value,
                    )?))
                })
                .collect::<Option<Vec<_>>>()
        })
        .flatten()
}

pub(super) fn series_sampling(
    over: &CheckedExpression,
    at: &CheckedExpression,
) -> Option<(AuthoredSeriesDomain, SamplingPolicy)> {
    let values = values::canonical_list(at)?;
    match &over.kind {
        CheckedExpressionKind::Enum(name) if name == "time" => Some((
            AuthoredSeriesDomain::Time,
            SamplingPolicy::TimeSeconds(values),
        )),
        CheckedExpressionKind::Enum(name) if name == "frequency" => Some((
            AuthoredSeriesDomain::Frequency,
            SamplingPolicy::FrequencyHertz(values),
        )),
        CheckedExpressionKind::Enum(name) if name == "load_factor" => Some((
            AuthoredSeriesDomain::LoadFactor,
            SamplingPolicy::LoadFactors(values),
        )),
        CheckedExpressionKind::Enum(name) if name == "mode" => Some((
            AuthoredSeriesDomain::Mode,
            SamplingPolicy::ModeIndices(
                values
                    .into_iter()
                    .map(|value| {
                        (value.is_finite() && value >= 1.0 && value.fract() == 0.0)
                            .then_some(value as usize)
                    })
                    .collect::<Option<Vec<_>>>()?,
            ),
        )),
        CheckedExpressionKind::Reference(path) if path.len() == 2 && path[0] == "input" => Some((
            AuthoredSeriesDomain::StudyParameter {
                input_id: path[1].clone(),
            },
            SamplingPolicy::StudyParameterValues(values),
        )),
        _ => None,
    }
}

pub(super) fn checked_series_weighting(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SeriesWeighting> {
    series_weighting(value).or_else(|| {
        lower::error(
            declaration,
            value.span,
            "weighting must be uniform_samples or trapezoidal_independent_domain",
            diagnostics,
        );
        None
    })
}

pub(super) fn series_domain(value: &CheckedExpression) -> Option<AuthoredSeriesDomain> {
    match &value.kind {
        CheckedExpressionKind::Enum(name) => Some(match name.as_str() {
            "time" => AuthoredSeriesDomain::Time,
            "frequency" => AuthoredSeriesDomain::Frequency,
            "load_factor" => AuthoredSeriesDomain::LoadFactor,
            "mode" => AuthoredSeriesDomain::Mode,
            _ => return None,
        }),
        CheckedExpressionKind::Reference(path) if path.len() == 2 && path[0] == "input" => {
            Some(AuthoredSeriesDomain::StudyParameter {
                input_id: path[1].clone(),
            })
        }
        _ => None,
    }
}

fn series_weighting(value: &CheckedExpression) -> Option<SeriesWeighting> {
    let CheckedExpressionKind::Enum(name) = &value.kind else {
        return None;
    };
    match name.as_str() {
        "uniform_samples" => Some(SeriesWeighting::UniformSamples),
        "trapezoidal_independent_domain" => Some(SeriesWeighting::TrapezoidalIndependentDomain),
        _ => None,
    }
}
