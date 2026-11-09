use std::collections::BTreeSet;

use runmat_analysis_core::{
    AuthoredExtremaTiePolicy, AuthoredFieldProjection, AuthoredOutput, AuthoredOutputExpression,
    AuthoredPrincipalSelection, AuthoredRegionSource, AuthoredSeriesReduction,
    AuthoredSpatialReduction, AuthoredVectorComponent, FieldLocation, Position,
};

use crate::{
    operations::OutputOperation, BoundFeaProgram, CheckedCallArgument, CheckedExpression,
    CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatementKind, FeaDiagnostic,
};

use super::lower;

mod fields;
mod policy;
mod series;

pub(crate) use fields::{field_location, field_reference_type};
use series::{checked_series_weighting, interpolation_positions, series_domain, series_sampling};

pub(super) fn lower_outputs(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredOutput> {
    let mut available = BTreeSet::new();
    declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let (name, value, policy) = match &statement.kind {
                CheckedFeaStatementKind::Output { name, value } => (
                    name,
                    value,
                    runmat_analysis_core::AuthoredOutputPolicy::default(),
                ),
                CheckedFeaStatementKind::OutputDefinition { name, statements } => {
                    let (value, policy) =
                        policy::lower_output_definition(declaration, statements, diagnostics)?;
                    (name, value, policy)
                }
                _ => return None,
            };
            let expression = lower_expression(bound, declaration, value, &available, diagnostics)?;
            available.insert(name.as_str());
            Some(AuthoredOutput {
                output_id: name.clone(),
                expression,
                policy,
            })
        })
        .collect()
}

fn lower_expression(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    available: &BTreeSet<&str>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredOutputExpression> {
    match &expression.kind {
        CheckedExpressionKind::Reference(path) => match path.as_slice() {
            [analysis, step_id, field, location_name, quantity_name]
                if analysis == "analysis" && field == "field" =>
            {
                let (quantity, _) = field_reference_type(quantity_name)?;
                let location = field_location(location_name)?;
                if location == runmat_analysis_core::FieldLocation::Mode {
                    // No procedure publishes mode-located fields; every run would reject it.
                    super::lower::error(
                        declaration,
                        expression.span,
                        crate::semantic::pending::with_pending(
                            format!(
                                "no procedure publishes mode-located fields (`field.mode.{quantity_name}`) yet"
                            ),
                            Some(crate::semantic::pending::OUTPUT_MODE_LOCATION),
                        ),
                        diagnostics,
                    );
                    return None;
                }
                Some(AuthoredOutputExpression::AnalysisField {
                    step_id: step_id.clone(),
                    quantity,
                    location,
                })
            }
            [output, output_id] if output == "output" && available.contains(output_id.as_str()) => {
                Some(AuthoredOutputExpression::OutputReference {
                    output_id: output_id.clone(),
                })
            }
            _ => invalid(declaration, expression, diagnostics),
        },
        CheckedExpressionKind::OperationCall {
            lowering: lowering_id,
            arguments,
            ..
        } => match OutputOperation::from_lowering(lowering_id)? {
            operation @ (OutputOperation::Real
            | OutputOperation::Imaginary
            | OutputOperation::Phase
            | OutputOperation::Level
            | OutputOperation::Magnitude
            | OutputOperation::VonMises) => {
                let source = positional_or_named(arguments, 0, "field")?;
                Some(AuthoredOutputExpression::Projection {
                    projection: match operation {
                        OutputOperation::Real => AuthoredFieldProjection::Real,
                        OutputOperation::Imaginary => AuthoredFieldProjection::Imaginary,
                        OutputOperation::Phase => AuthoredFieldProjection::Phase,
                        OutputOperation::Level => AuthoredFieldProjection::Level,
                        OutputOperation::Magnitude => AuthoredFieldProjection::Magnitude,
                        OutputOperation::VonMises => AuthoredFieldProjection::VonMises,
                        _ => unreachable!(),
                    },
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                })
            }
            OutputOperation::Component => {
                let source = positional_or_named(arguments, 0, "field")?;
                let component = positional_or_named(arguments, 1, "component")?;
                let CheckedExpressionKind::Enum(component_name) = &component.kind else {
                    lower::error(
                        declaration,
                        component.span,
                        "component must be one of x, y, or z",
                        diagnostics,
                    );
                    return None;
                };
                let component = match component_name.as_str() {
                    "x" => AuthoredVectorComponent::X,
                    "y" => AuthoredVectorComponent::Y,
                    "z" => AuthoredVectorComponent::Z,
                    _ => {
                        lower::error(
                            declaration,
                            component.span,
                            "component must be one of x, y, or z",
                            diagnostics,
                        );
                        return None;
                    }
                };
                Some(AuthoredOutputExpression::Projection {
                    projection: AuthoredFieldProjection::Component(component),
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                })
            }
            OutputOperation::Principal => {
                let source = positional_or_named(arguments, 0, "field")?;
                let selector = positional_or_named(arguments, 1, "which")?;
                let CheckedExpressionKind::Enum(which) = &selector.kind else {
                    lower::error(
                        declaration,
                        selector.span,
                        "principal selector must be maximum or minimum",
                        diagnostics,
                    );
                    return None;
                };
                let which = match which.as_str() {
                    "maximum" => AuthoredPrincipalSelection::Maximum,
                    "minimum" => AuthoredPrincipalSelection::Minimum,
                    _ => {
                        lower::error(
                            declaration,
                            selector.span,
                            "principal selector must be maximum or minimum",
                            diagnostics,
                        );
                        return None;
                    }
                };
                Some(AuthoredOutputExpression::Projection {
                    projection: AuthoredFieldProjection::Principal(which),
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                })
            }
            OutputOperation::Recovery => {
                let source = positional_or_named(arguments, 0, "field")?;
                let to = positional_or_named(arguments, 1, "to")?;
                let using = positional_or_named(arguments, 2, "using")?;
                let (CheckedExpressionKind::Enum(to), CheckedExpressionKind::Enum(using)) =
                    (&to.kind, &using.kind)
                else {
                    lower::error(
                        declaration,
                        expression.span,
                        "recover requires `to = node` and `using = volume_weighted_average`",
                        diagnostics,
                    );
                    return None;
                };
                if to != "node" || using != "volume_weighted_average" {
                    lower::error(
                        declaration,
                        expression.span,
                        "recover currently supports scalar element von Mises stress with `to = node, using = volume_weighted_average`",
                        diagnostics,
                    );
                    return None;
                }
                Some(AuthoredOutputExpression::Recovery {
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                    output_location: FieldLocation::Node,
                    recovery: runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage,
                })
            }
            operation @ (OutputOperation::Integral
            | OutputOperation::Mean
            | OutputOperation::RootMeanSquare
            | OutputOperation::Norm
            | OutputOperation::Minimum
            | OutputOperation::Maximum
            | OutputOperation::Resultant
            | OutputOperation::Moment) => {
                let source = positional_or_named(arguments, 0, "field")?;
                let target = positional_or_named(arguments, 1, "over")?;
                let target =
                    super::super::model::region_source(bound, declaration, target, diagnostics)?;
                if matches!(target, AuthoredRegionSource::ModelInput(_)) {
                    lower::error(
                        declaration,
                        expression.span,
                        "study output regions must name an exact geometry region",
                        diagnostics,
                    );
                    return None;
                }
                Some(AuthoredOutputExpression::SpatialReduction {
                    reduction: match operation {
                        OutputOperation::Integral => AuthoredSpatialReduction::Integral,
                        OutputOperation::Mean => AuthoredSpatialReduction::Mean,
                        OutputOperation::RootMeanSquare => AuthoredSpatialReduction::RootMeanSquare,
                        OutputOperation::Norm => AuthoredSpatialReduction::EuclideanNorm,
                        OutputOperation::Minimum => AuthoredSpatialReduction::Minimum,
                        OutputOperation::Maximum => AuthoredSpatialReduction::Maximum,
                        OutputOperation::Resultant => AuthoredSpatialReduction::Resultant,
                        OutputOperation::Moment => {
                            let about = positional_or_named(arguments, 2, "about")?;
                            let meters = super::values::canonical_vector3(about).or_else(|| {
                                lower::error(
                                    declaration,
                                    about.span,
                                    "moment origin must be a constant three-component length",
                                    diagnostics,
                                );
                                None
                            })?;
                            AuthoredSpatialReduction::Moment {
                                about: Position::new(meters),
                            }
                        }
                        _ => unreachable!(),
                    },
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                    target,
                    ties: AuthoredExtremaTiePolicy::FirstCanonical,
                })
            }
            // The active intensity of a harmonic analysis through a face region: the flux of
            // its native element-average intensity field.
            OutputOperation::SoundPower => {
                let analysis = positional_or_named(arguments, 0, "analysis")?;
                let CheckedExpressionKind::Reference(path) = &analysis.kind else {
                    return invalid(declaration, analysis, diagnostics);
                };
                let [namespace, step_id] = path.as_slice() else {
                    return invalid(declaration, analysis, diagnostics);
                };
                if namespace != "analysis" {
                    return invalid(declaration, analysis, diagnostics);
                }
                let target = positional_or_named(arguments, 1, "through")?;
                let target =
                    super::super::model::region_source(bound, declaration, target, diagnostics)?;
                if matches!(target, AuthoredRegionSource::ModelInput(_)) {
                    lower::error(
                        declaration,
                        expression.span,
                        "study output regions must name an exact geometry region",
                        diagnostics,
                    );
                    return None;
                }
                Some(AuthoredOutputExpression::SpatialReduction {
                    reduction: AuthoredSpatialReduction::NormalFlux,
                    source: Box::new(AuthoredOutputExpression::AnalysisField {
                        step_id: step_id.clone(),
                        quantity: runmat_analysis_core::ObservationQuantity::AcousticIntensity,
                        location: FieldLocation::Element,
                    }),
                    target,
                    ties: AuthoredExtremaTiePolicy::FirstCanonical,
                })
            }
            // Two ports of one analysis. The pressure drop reads the finite-volume face pressure
            // of a fluid or conjugate analysis; the transmission loss reads the acoustic port
            // intensity (incident wave on the incident face, outgoing wave on the anechoic face).
            operation @ (OutputOperation::PressureDrop | OutputOperation::TransmissionLoss) => {
                let analysis = positional_or_named(arguments, 0, "analysis")?;
                let CheckedExpressionKind::Reference(path) = &analysis.kind else {
                    return invalid(declaration, analysis, diagnostics);
                };
                let [namespace, step_id] = path.as_slice() else {
                    return invalid(declaration, analysis, diagnostics);
                };
                if namespace != "analysis" {
                    return invalid(declaration, analysis, diagnostics);
                }
                let (names, reduction, quantity, location) = match operation {
                    OutputOperation::PressureDrop => (
                        ["from", "to"],
                        runmat_analysis_core::AuthoredPortReduction::MeanDifference,
                        runmat_analysis_core::ObservationQuantity::FluidPressure,
                        FieldLocation::Face,
                    ),
                    _ => (
                        ["incident", "transmitted"],
                        runmat_analysis_core::AuthoredPortReduction::TransmissionLoss,
                        runmat_analysis_core::ObservationQuantity::AcousticPortIntensity,
                        FieldLocation::Element,
                    ),
                };
                let mut regions = Vec::with_capacity(2);
                for (position, name) in names.into_iter().enumerate() {
                    let target = positional_or_named(arguments, position + 1, name)?;
                    let target = super::super::model::region_source(
                        bound,
                        declaration,
                        target,
                        diagnostics,
                    )?;
                    if matches!(target, AuthoredRegionSource::ModelInput(_)) {
                        lower::error(
                            declaration,
                            expression.span,
                            "study output regions must name an exact geometry region",
                            diagnostics,
                        );
                        return None;
                    }
                    regions.push(target);
                }
                let second = regions.pop()?;
                let first = regions.pop()?;
                Some(AuthoredOutputExpression::PortReduction {
                    reduction,
                    source: Box::new(AuthoredOutputExpression::AnalysisField {
                        step_id: step_id.clone(),
                        quantity,
                        location,
                    }),
                    first,
                    second,
                })
            }
            OutputOperation::NormalFlux => {
                let source = positional_or_named(arguments, 0, "field")?;
                let target = positional_or_named(arguments, 1, "through")?;
                let target =
                    super::super::model::region_source(bound, declaration, target, diagnostics)?;
                if matches!(target, AuthoredRegionSource::ModelInput(_)) {
                    lower::error(
                        declaration,
                        expression.span,
                        "study output regions must name an exact geometry region",
                        diagnostics,
                    );
                    return None;
                }
                Some(AuthoredOutputExpression::SpatialReduction {
                    reduction: AuthoredSpatialReduction::NormalFlux,
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                    target,
                    ties: AuthoredExtremaTiePolicy::FirstCanonical,
                })
            }
            OutputOperation::SpatialSample => {
                let source = positional_or_named(arguments, 0, "field")?;
                let target = positional_or_named(arguments, 1, "over")?;
                let target =
                    super::super::model::region_source(bound, declaration, target, diagnostics)?;
                if matches!(target, AuthoredRegionSource::ModelInput(_)) {
                    lower::error(
                        declaration,
                        expression.span,
                        "study output regions must name an exact geometry region",
                        diagnostics,
                    );
                    return None;
                }
                Some(AuthoredOutputExpression::SpatialSelection {
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                    target,
                })
            }
            OutputOperation::InterpolatePoint | OutputOperation::InterpolatePath => {
                let source = positional_or_named(arguments, 0, "field")?;
                let at = positional_or_named(arguments, 1, "at")?;
                let positions = interpolation_positions(at).or_else(|| {
                    lower::error(
                        declaration,
                        at.span,
                        "interpolation requires one point or a nonempty path of length vectors",
                        diagnostics,
                    );
                    None
                })?;
                Some(AuthoredOutputExpression::Interpolation {
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                    positions,
                })
            }
            OutputOperation::SampleSeries => {
                let source = positional_or_named(arguments, 0, "source")?;
                let over = positional_or_named(arguments, 1, "over")?;
                let at = positional_or_named(arguments, 2, "at")?;
                let (domain, sampling) = series_sampling(over, at).or_else(|| {
                    lower::error(
                        declaration,
                        expression.span,
                        "sample coordinates must match time, frequency, load_factor, mode, or input.<name>",
                        diagnostics,
                    );
                    None
                })?;
                Some(AuthoredOutputExpression::Sampling {
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                    sampling,
                    domain,
                })
            }
            operation @ (OutputOperation::MinimumSeries
            | OutputOperation::MaximumSeries
            | OutputOperation::AbsoluteMaximumSeries
            | OutputOperation::MeanSeries
            | OutputOperation::RootMeanSquareSeries
            | OutputOperation::VarianceSeries
            | OutputOperation::EnvelopeSeries) => {
                let source = positional_or_named(arguments, 0, "history")?;
                let over = positional_or_named(arguments, 1, "over")?;
                let domain = series_domain(over).or_else(|| {
                    lower::error(
                        declaration,
                        over.span,
                        "series reduction domain must be time, frequency, load_factor, mode, or input.<name>",
                        diagnostics,
                    );
                    None
                })?;
                let reduction = match operation {
                    OutputOperation::MinimumSeries => AuthoredSeriesReduction::Minimum,
                    OutputOperation::MaximumSeries => AuthoredSeriesReduction::Maximum,
                    OutputOperation::AbsoluteMaximumSeries => {
                        AuthoredSeriesReduction::AbsoluteMaximum
                    }
                    OutputOperation::MeanSeries => AuthoredSeriesReduction::Mean {
                        weighting: checked_series_weighting(
                            declaration,
                            positional_or_named(arguments, 2, "weighting")?,
                            diagnostics,
                        )?,
                    },
                    OutputOperation::RootMeanSquareSeries => {
                        AuthoredSeriesReduction::RootMeanSquare {
                            weighting: checked_series_weighting(
                                declaration,
                                positional_or_named(arguments, 2, "weighting")?,
                                diagnostics,
                            )?,
                        }
                    }
                    OutputOperation::VarianceSeries => AuthoredSeriesReduction::Variance {
                        weighting: checked_series_weighting(
                            declaration,
                            positional_or_named(arguments, 2, "weighting")?,
                            diagnostics,
                        )?,
                    },
                    OutputOperation::EnvelopeSeries => AuthoredSeriesReduction::Envelope,
                    _ => unreachable!(),
                };
                Some(AuthoredOutputExpression::SeriesReduction {
                    reduction,
                    source: Box::new(lower_expression(
                        bound,
                        declaration,
                        source,
                        available,
                        diagnostics,
                    )?),
                    domain,
                    ties: AuthoredExtremaTiePolicy::FirstCanonical,
                })
            }
            OutputOperation::CollectCases
            | OutputOperation::RetainEachCase
            | OutputOperation::MinimumCase
            | OutputOperation::MaximumCase => invalid(declaration, expression, diagnostics),
        },
        _ => invalid(declaration, expression, diagnostics),
    }
}

fn positional_or_named<'a>(
    arguments: &'a [CheckedCallArgument],
    position: usize,
    name: &str,
) -> Option<&'a CheckedExpression> {
    arguments
        .iter()
        .find_map(|argument| match argument {
            CheckedCallArgument::Named {
                name: argument_name,
                value,
                ..
            } if argument_name == name => Some(value),
            _ => None,
        })
        .or_else(|| match arguments.get(position) {
            Some(CheckedCallArgument::Positional(value)) => Some(value),
            _ => None,
        })
}

fn invalid(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredOutputExpression> {
    lower::error(
        declaration,
        expression.span,
        "output expression is not part of the closed typed observation language",
        diagnostics,
    );
    None
}
