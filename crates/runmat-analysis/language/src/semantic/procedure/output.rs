use std::collections::BTreeSet;

use runmat_analysis_core::{
    AuthoredExtremaTiePolicy, AuthoredFieldProjection, AuthoredOutput, AuthoredOutputExpression,
    AuthoredPrincipalSelection, AuthoredRegionSource, AuthoredSeriesReduction,
    AuthoredSpatialReduction, AuthoredVectorComponent, FieldLocation, ObservationQuantity, Position,
};

use crate::{
    operations::OutputOperation, BoundFeaProgram, CheckedCallArgument, CheckedExpression,
    CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatementKind, FeaDiagnostic, FeaType,
    PhysicalDimension,
};

use super::lower;

mod series;

use series::{
    checked_series_weighting, interpolation_positions, series_domain, series_sampling,
};

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
            let CheckedFeaStatementKind::Output { name, value } = &statement.kind else {
                return None;
            };
            let expression = lower_expression(bound, declaration, value, &available, diagnostics)?;
            available.insert(name.as_str());
            Some(AuthoredOutput {
                output_id: name.clone(),
                expression,
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
            | OutputOperation::Magnitude
            | OutputOperation::VonMises) => {
                let source = positional_or_named(arguments, 0, "field")?;
                Some(AuthoredOutputExpression::Projection {
                    projection: match operation {
                        OutputOperation::Real => AuthoredFieldProjection::Real,
                        OutputOperation::Imaginary => AuthoredFieldProjection::Imaginary,
                        OutputOperation::Phase => AuthoredFieldProjection::Phase,
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
                                about: Position { meters },
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

pub(crate) fn field_location(name: &str) -> Option<FieldLocation> {
    Some(match name {
        "node" => FieldLocation::Node,
        "cell" => FieldLocation::Cell,
        "element" => FieldLocation::Element,
        "integration_point" => FieldLocation::IntegrationPoint,
        "face" => FieldLocation::Face,
        "edge" => FieldLocation::Edge,
        "interface" => FieldLocation::Interface,
        "connector" => FieldLocation::Connector,
        "constraint" => FieldLocation::Constraint,
        "reference_point" => FieldLocation::ReferencePoint,
        "mode" => FieldLocation::Mode,
        "global" => FieldLocation::Global,
        _ => return None,
    })
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

pub(crate) fn field_reference_type(name: &str) -> Option<(ObservationQuantity, FeaType)> {
    use ObservationQuantity as Q;
    let quantity = ObservationQuantity::from_source_name(name)?;
    let (dimension, components) = match quantity {
        Q::Displacement => (PhysicalDimension::LENGTH, 3),
        Q::ContactGap => (PhysicalDimension::LENGTH, 1),
        Q::Rotation => (PhysicalDimension::DIMENSIONLESS, 3),
        Q::Velocity | Q::AcousticParticleVelocity | Q::FluidVelocity => {
            (PhysicalDimension::VELOCITY, 3)
        }
        Q::Acceleration => (PhysicalDimension::ACCELERATION, 3),
        Q::Strain => (PhysicalDimension::DIMENSIONLESS, 6),
        Q::Stress => (PhysicalDimension::PRESSURE, 6),
        Q::VonMisesStress
        | Q::BeamTorsionStress
        | Q::ShellVonMises
        | Q::ContactPressure
        | Q::FluidPressure
        | Q::AcousticPressure => (PhysicalDimension::PRESSURE, 1),
        Q::BeamBendingStress => (PhysicalDimension::PRESSURE, 2),
        Q::WallShearStress => (PhysicalDimension::PRESSURE, 3),
        Q::ReactionForce => (PhysicalDimension::FORCE, 3),
        Q::BeamAxialForce => (PhysicalDimension::FORCE, 1),
        Q::BeamShearForce => (PhysicalDimension::FORCE, 2),
        Q::ReactionMoment => (PhysicalDimension::MOMENT, 3),
        Q::BeamTorsionMoment => (PhysicalDimension::MOMENT, 1),
        Q::BeamBendingMoment => (PhysicalDimension::MOMENT, 2),
        Q::StrainEnergy => (PhysicalDimension::ENERGY, 1),
        Q::ShellMembraneForce => (
            PhysicalDimension::FORCE.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::ShellBendingMoment => (
            PhysicalDimension::MOMENT.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::ShellTransverseShear => (
            PhysicalDimension::FORCE.divide(PhysicalDimension::LENGTH)?,
            2,
        ),
        Q::Temperature => (PhysicalDimension::TEMPERATURE, 1),
        Q::TemperatureGradient => (
            PhysicalDimension::TEMPERATURE.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::HeatFlux => (PhysicalDimension::HEAT_FLUX, 3),
        Q::HeatRate => (PhysicalDimension::POWER, 1),
        Q::ElectricPotential => (PhysicalDimension::ELECTRIC_POTENTIAL, 1),
        Q::ElectricField => (
            PhysicalDimension::ELECTRIC_POTENTIAL.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::ElectricCurrent => (PhysicalDimension::ELECTRIC_CURRENT, 1),
        Q::ElectricCurrentDensity => (
            PhysicalDimension::ELECTRIC_CURRENT.divide(PhysicalDimension::AREA)?,
            3,
        ),
        Q::ElectricCharge => (
            PhysicalDimension::ELECTRIC_CURRENT.multiply(PhysicalDimension::TIME)?,
            1,
        ),
        Q::MagneticVectorPotential => (
            PhysicalDimension::MAGNETIC_FLUX_DENSITY.multiply(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::MagneticFluxDensity => (PhysicalDimension::MAGNETIC_FLUX_DENSITY, 3),
        Q::MagneticFieldIntensity => (
            PhysicalDimension::ELECTRIC_CURRENT.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::SoundPressureLevel
        | Q::ModalParticipation
        | Q::Eigenvalue
        | Q::Residual
        | Q::Balance
        | Q::ConstraintViolation => (PhysicalDimension::DIMENSIONLESS, 1),
        Q::Vorticity => (PhysicalDimension::FREQUENCY, 3),
        Q::ModalFrequency => (PhysicalDimension::FREQUENCY, 1),
        Q::MassFlowRate => (PhysicalDimension::MASS.divide(PhysicalDimension::TIME)?, 1),
    };
    let scalar = FeaType::Quantity(dimension);
    Some((
        quantity,
        if components == 1 {
            scalar
        } else {
            FeaType::Vector {
                element: Box::new(scalar),
                length: Some(components),
            }
        },
    ))
}
