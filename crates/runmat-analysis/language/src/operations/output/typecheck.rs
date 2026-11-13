use crate::{CheckedCallArgument, CheckedExpressionKind, FeaSeriesDomain, FeaType};

use super::OutputOperation;

impl OutputOperation {
    pub(crate) fn result_type(
        self,
        arguments: &[CheckedCallArgument],
    ) -> Result<FeaType, &'static str> {
        let source = arguments.first().map(|argument| match argument {
            CheckedCallArgument::Positional(value) | CheckedCallArgument::Named { value, .. } => {
                &value.value_type
            }
        });
        match (self, source) {
            (
                operation @ (Self::Magnitude
                | Self::Component
                | Self::Real
                | Self::Imaginary
                | Self::Phase
                | Self::Level
                | Self::VonMises
                | Self::Principal),
                Some(value),
            ) => project_type(operation, value),
            (Self::SoundPower, _) => {
                let measure = region_measure(argument_type(arguments, 1, "through"))
                    .ok_or("sound_power requires a face region")?;
                if measure != crate::PhysicalDimension::AREA {
                    return Err("sound_power requires a face region");
                }
                Ok(FeaType::Quantity(crate::PhysicalDimension::POWER.into()))
            }
            (Self::PressureDrop | Self::TransmissionLoss, _) => {
                let names = if self == Self::PressureDrop {
                    ["from", "to"]
                } else {
                    ["incident", "transmitted"]
                };
                for (position, name) in names.into_iter().enumerate() {
                    if region_measure(argument_type(arguments, position + 1, name))
                        != Some(crate::PhysicalDimension::AREA)
                    {
                        return Err(if self == Self::PressureDrop {
                            "pressure_drop reads two face regions (`from`, `to`)"
                        } else {
                            "transmission_loss reads two face regions (`incident`, `transmitted`)"
                        });
                    }
                }
                Ok(FeaType::Quantity(if self == Self::PressureDrop {
                    crate::PhysicalDimension::PRESSURE.into()
                } else {
                    crate::PhysicalDimension::DIMENSIONLESS.into()
                }))
            }
            (Self::NormalFlux, Some(FeaType::Field(field))) => {
                let measure = region_measure(argument_type(arguments, 1, "through"))
                    .ok_or("flux requires a face region")?;
                if measure != crate::PhysicalDimension::AREA {
                    return Err("flux requires a face region");
                }
                let FeaType::Vector {
                    element,
                    length: Some(3),
                } = field.value.as_ref()
                else {
                    return Err("flux requires a three-component vector field");
                };
                let FeaType::Quantity(dimension) = element.as_ref() else {
                    return Err("flux requires a physical vector field");
                };
                dimension
                    .multiply(measure.into())
                    .map(FeaType::Quantity)
                    .ok_or("flux dimension overflows")
            }
            (Self::Integral, Some(FeaType::Field(field))) => {
                let measure = region_measure(argument_type(arguments, 1, "over"))
                    .ok_or("integral requires an edge, face, solid, or cell region")?;
                scale_numeric_type(&field.value, measure)
                    .ok_or("integral requires a numeric field with a representable dimension")
            }
            (Self::Mean | Self::RootMeanSquare, Some(FeaType::Field(field)))
                if numeric_shape(&field.value) =>
            {
                Ok(field.value.as_ref().clone())
            }
            (Self::Norm, Some(FeaType::Field(field))) => scalar_dimension(&field.value)
                .map(FeaType::Quantity)
                .ok_or("norm requires a numeric field"),
            (Self::Minimum | Self::Maximum, Some(FeaType::Field(field)))
                if field.value.is_numeric() =>
            {
                Ok(FeaType::AttributedExtremum(field.value.clone()))
            }
            (Self::Minimum | Self::Maximum, Some(FeaType::Field(_))) => {
                Err("spatial extrema require an explicit scalar field projection")
            }
            (Self::Resultant | Self::Moment, Some(FeaType::Field(field))) => {
                resultant_type(self, field, argument_type(arguments, 1, "over"))
            }
            (Self::SpatialSample, Some(FeaType::Field(field))) => Ok(FeaType::Field(field.clone())),
            (Self::Recovery, Some(FeaType::Field(field))) => {
                let to = enum_argument(arguments, 1, "to")
                    .ok_or("recover requires the closed `to = node` location")?;
                let using = enum_argument(arguments, 2, "using").ok_or(
                    "recover requires the closed `using = volume_weighted_average` method",
                )?;
                if field.location != Some(runmat_analysis_core::FieldLocation::Element)
                    || field.quantity
                        != Some(runmat_analysis_core::ObservationQuantity::VonMisesStress)
                    || !field.value.is_numeric()
                    || to != "node"
                    || using != "volume_weighted_average"
                {
                    return Err(
                        "recover currently supports scalar element von Mises stress to nodes using volume_weighted_average",
                    );
                }
                let mut field = field.clone();
                field.location = Some(runmat_analysis_core::FieldLocation::Node);
                Ok(FeaType::Field(field))
            }
            (Self::InterpolatePoint, Some(FeaType::Field(field))) => {
                Ok(field.value.as_ref().clone())
            }
            (Self::InterpolatePath, Some(FeaType::Field(field))) => {
                let mut field = field.clone();
                field.location = Some(runmat_analysis_core::FieldLocation::Point);
                Ok(FeaType::Field(field))
            }
            (Self::SampleSeries, Some(value)) => {
                let domain = series_domain(argument_expression(arguments, 1, "over"))
                    .ok_or("sample requires time, frequency, load_factor, mode, or input.<name>")?;
                validate_sample_axis(&domain, argument_type(arguments, 2, "at"))?;
                Ok(FeaType::History(crate::FeaHistoryType {
                    sample: Box::new(value.clone()),
                    domain: Some(domain),
                }))
            }
            (
                operation @ (Self::MinimumSeries
                | Self::MaximumSeries
                | Self::AbsoluteMaximumSeries
                | Self::MeanSeries
                | Self::RootMeanSquareSeries
                | Self::VarianceSeries
                | Self::EnvelopeSeries),
                Some(FeaType::History(history)),
            ) => {
                let over = series_domain(argument_expression(arguments, 1, "over"))
                    .ok_or("series reduction requires its explicit independent domain")?;
                if history
                    .domain
                    .as_ref()
                    .is_some_and(|domain| domain != &over)
                {
                    return Err("series reduction domain does not match its history");
                }
                series_result_type(operation, &history.sample)
            }
            (Self::CollectCases, Some(value)) if value.is_numeric() => {
                Ok(FeaType::CaseTable(Box::new(value.clone())))
            }
            (Self::RetainEachCase, Some(FeaType::Field(field))) => {
                Ok(FeaType::CaseTable(Box::new(FeaType::Field(field.clone()))))
            }
            (Self::MinimumCase | Self::MaximumCase, Some(value)) if value.is_numeric() => {
                Ok(FeaType::AttributedExtremum(Box::new(value.clone())))
            }
            _ => Err("output operation requires an analysis field"),
        }
    }
}

fn project_type(operation: OutputOperation, value: &FeaType) -> Result<FeaType, &'static str> {
    if let FeaType::History(history) = value {
        return Ok(FeaType::History(crate::FeaHistoryType {
            sample: Box::new(project_type(operation, &history.sample)?),
            domain: history.domain.clone(),
        }));
    }
    let FeaType::Field(field) = value else {
        return Err("field projection requires an analysis field or field history");
    };
    match operation {
        OutputOperation::Magnitude => match field.value.as_ref() {
            FeaType::Vector {
                element,
                length: Some(3),
            } if element.is_numeric() => {
                Ok(projected_field(field, element.clone(), field.quantity))
            }
            // The modulus of a scalar phasor. The study's procedure decides whether the
            // field is complex; authored-output validation rejects a real scalar source.
            scalar if scalar.is_numeric() => Ok(FeaType::Field(field.clone())),
            _ => Err("magnitude requires a three-component or complex scalar numeric field"),
        },
        OutputOperation::Component => match field.value.as_ref() {
            FeaType::Vector {
                element,
                length: Some(3),
            } if element.is_numeric() => {
                Ok(projected_field(field, element.clone(), field.quantity))
            }
            _ => Err("component requires a three-component numeric field"),
        },
        OutputOperation::Real | OutputOperation::Imaginary => Ok(FeaType::Field(field.clone())),
        // Sound pressure level of a complex scalar pressure, dB re 20 uPa (RMS).
        OutputOperation::Level => match field.value.as_ref() {
            value if value.is_numeric() => Ok(projected_field(
                field,
                Box::new(FeaType::Quantity(
                    crate::PhysicalDimension::DIMENSIONLESS.into(),
                )),
                Some(runmat_analysis_core::ObservationQuantity::SoundPressureLevel),
            )),
            _ => Err("level requires a complex scalar acoustic pressure field"),
        },
        OutputOperation::Phase => Ok(projected_field(
            field,
            Box::new(phase_shape(&field.value)),
            None,
        )),
        OutputOperation::VonMises => match field.value.as_ref() {
            FeaType::Vector {
                element,
                length: Some(6),
            } if (element.as_ref()).is_quantity((crate::PhysicalDimension::PRESSURE).into()) => {
                Ok(projected_field(
                    field,
                    element.clone(),
                    Some(runmat_analysis_core::ObservationQuantity::VonMisesStress),
                ))
            }
            _ => Err("von_mises requires a symmetric stress field"),
        },
        OutputOperation::Principal => match field.value.as_ref() {
            FeaType::Vector {
                element,
                length: Some(6),
            } if element.is_numeric()
                && matches!(
                    field.quantity,
                    Some(
                        runmat_analysis_core::ObservationQuantity::Stress
                            | runmat_analysis_core::ObservationQuantity::Strain
                    )
                ) =>
            {
                Ok(projected_field(field, element.clone(), field.quantity))
            }
            _ => Err("principal requires a real symmetric tensor field"),
        },
        _ => Err("operation is not a field projection"),
    }
}

fn series_result_type(
    operation: OutputOperation,
    sample: &FeaType,
) -> Result<FeaType, &'static str> {
    let value = match sample {
        FeaType::Field(field) => field.value.as_ref(),
        // A spatial extremum carries a scalar value and its governing entity.
        // Reducing its history compares those values; the resulting extremum
        // carries the governing sample as well as the spatial attribution.
        FeaType::AttributedExtremum(value) => value.as_ref(),
        value => value,
    };
    if !numeric_shape(value) {
        return Err("series reduction requires numeric history samples");
    }
    match operation {
        OutputOperation::MinimumSeries
        | OutputOperation::MaximumSeries
        | OutputOperation::AbsoluteMaximumSeries
            if value.is_numeric() =>
        {
            Ok(FeaType::AttributedExtremum(Box::new(value.clone())))
        }
        OutputOperation::MinimumSeries
        | OutputOperation::MaximumSeries
        | OutputOperation::AbsoluteMaximumSeries => {
            Err("series extrema require an explicit scalar projection")
        }
        OutputOperation::MeanSeries | OutputOperation::RootMeanSquareSeries => Ok(value.clone()),
        OutputOperation::VarianceSeries => {
            square_numeric_type(value).ok_or("variance dimension overflows")
        }
        OutputOperation::EnvelopeSeries => Ok(FeaType::Envelope(Box::new(value.clone()))),
        _ => Err("operation is not a series reduction"),
    }
}

fn square_numeric_type(value: &FeaType) -> Option<FeaType> {
    match value {
        FeaType::Quantity(dimension) => Some(FeaType::Quantity(dimension.power(2)?)),
        FeaType::Integer | FeaType::Real | FeaType::Complex => Some(value.clone()),
        FeaType::Vector { element, length } => Some(FeaType::Vector {
            element: Box::new(square_numeric_type(element)?),
            length: *length,
        }),
        _ => None,
    }
}

fn series_domain(value: Option<&crate::CheckedExpression>) -> Option<FeaSeriesDomain> {
    match &value?.kind {
        CheckedExpressionKind::Enum(name) => Some(match name.as_str() {
            "time" => FeaSeriesDomain::Time,
            "frequency" => FeaSeriesDomain::Frequency,
            "load_factor" => FeaSeriesDomain::LoadFactor,
            "mode" => FeaSeriesDomain::Mode,
            _ => return None,
        }),
        CheckedExpressionKind::Reference(path) if path.len() == 2 && path[0] == "input" => {
            Some(FeaSeriesDomain::StudyParameter(path[1].clone()))
        }
        _ => None,
    }
}

fn validate_sample_axis(
    domain: &FeaSeriesDomain,
    values: Option<&FeaType>,
) -> Result<(), &'static str> {
    let Some(FeaType::Vector { element, .. }) = values else {
        return Err("sample coordinates must be an explicit numeric list");
    };
    let valid = match domain {
        FeaSeriesDomain::Time => {
            (element.as_ref()).is_quantity((crate::PhysicalDimension::TIME).into())
        }
        FeaSeriesDomain::Frequency => {
            (element.as_ref()).is_quantity((crate::PhysicalDimension::FREQUENCY).into())
        }
        FeaSeriesDomain::Mode => matches!(element.as_ref(), FeaType::Integer),
        FeaSeriesDomain::LoadFactor => {
            (matches!(element.as_ref(), FeaType::Integer | FeaType::Real)
                || element.is_quantity(crate::PhysicalDimension::DIMENSIONLESS.into()))
        }
        FeaSeriesDomain::StudyParameter(_) => element.is_numeric(),
    };
    valid
        .then_some(())
        .ok_or("sample coordinate dimension does not match its independent domain")
}

fn argument_expression<'a>(
    arguments: &'a [CheckedCallArgument],
    position: usize,
    name: &str,
) -> Option<&'a crate::CheckedExpression> {
    arguments
        .iter()
        .find_map(|argument| match argument {
            CheckedCallArgument::Named {
                name: candidate,
                value,
                ..
            } if candidate == name => Some(value),
            _ => None,
        })
        .or_else(|| {
            arguments.get(position).map(|argument| match argument {
                CheckedCallArgument::Positional(value)
                | CheckedCallArgument::Named { value, .. } => value,
            })
        })
}

fn enum_argument<'a>(
    arguments: &'a [CheckedCallArgument],
    position: usize,
    name: &str,
) -> Option<&'a str> {
    match &argument_expression(arguments, position, name)?.kind {
        CheckedExpressionKind::Enum(value) => Some(value),
        _ => None,
    }
}

fn argument_type<'a>(
    arguments: &'a [CheckedCallArgument],
    position: usize,
    name: &str,
) -> Option<&'a FeaType> {
    arguments
        .iter()
        .find_map(|argument| match argument {
            CheckedCallArgument::Named {
                name: candidate,
                value,
                ..
            } if candidate == name => Some(&value.value_type),
            _ => None,
        })
        .or_else(|| {
            arguments.get(position).map(|argument| match argument {
                CheckedCallArgument::Positional(value)
                | CheckedCallArgument::Named { value, .. } => &value.value_type,
            })
        })
}

fn numeric_shape(value: &FeaType) -> bool {
    value.is_numeric() || matches!(value, FeaType::Vector { element, .. } if element.is_numeric())
}

fn scalar_dimension(value: &FeaType) -> Option<runmat_types::physical::DynamicDimension> {
    match value {
        FeaType::Quantity(dimension) => Some(*dimension),
        FeaType::Integer | FeaType::Real | FeaType::Complex => {
            Some(crate::PhysicalDimension::DIMENSIONLESS.into())
        }
        FeaType::Vector { element, .. } => scalar_dimension(element),
        _ => None,
    }
}

fn scale_numeric_type(value: &FeaType, measure: crate::PhysicalDimension) -> Option<FeaType> {
    match value {
        FeaType::Quantity(dimension) => {
            Some(FeaType::Quantity(dimension.multiply(measure.into())?))
        }
        FeaType::Integer | FeaType::Real | FeaType::Complex => {
            Some(FeaType::Quantity((measure).into()))
        }
        FeaType::Vector { element, length } => Some(FeaType::Vector {
            element: Box::new(scale_numeric_type(element, measure)?),
            length: *length,
        }),
        _ => None,
    }
}

fn region_measure(value: Option<&FeaType>) -> Option<crate::PhysicalDimension> {
    Some(match value? {
        FeaType::Region(crate::FeaTopologyRole::Vertex) => return None,
        FeaType::Region(crate::FeaTopologyRole::Edge) => crate::PhysicalDimension::LENGTH,
        FeaType::Region(crate::FeaTopologyRole::Face) => crate::PhysicalDimension::AREA,
        FeaType::Region(crate::FeaTopologyRole::Solid | crate::FeaTopologyRole::Cell) => {
            crate::PhysicalDimension::VOLUME
        }
        _ => return None,
    })
}

fn resultant_type(
    operation: OutputOperation,
    field: &crate::FeaFieldType,
    target: Option<&FeaType>,
) -> Result<FeaType, &'static str> {
    use runmat_analysis_core::ObservationQuantity as Q;
    let quantity = field
        .quantity
        .ok_or("resultant requires a known physical field quantity")?;
    if operation == OutputOperation::Moment && quantity == Q::ReactionMoment {
        return Err("moment cannot take a second moment of reaction moments");
    }
    let compatible = matches!(
        field.value.as_ref(),
        FeaType::Vector {
            length: Some(3),
            ..
        }
    ) || quantity == Q::Stress
        || matches!(
            quantity,
            Q::AcousticPressure | Q::FluidPressure | Q::ContactPressure
        );
    if !compatible {
        return Err("resultant requires a vector field, boundary stress, or boundary pressure");
    }
    let mut dimension = quantity.physical_dimension();
    if !matches!(quantity, Q::ReactionForce | Q::ReactionMoment) {
        dimension = dimension
            .multiply(region_measure(target).ok_or("resultant requires a measurable region")?)
            .ok_or("resultant dimension overflows")?;
    }
    if operation == OutputOperation::Moment {
        dimension = dimension
            .multiply(crate::PhysicalDimension::LENGTH)
            .ok_or("moment dimension overflows")?;
    }
    Ok(FeaType::Vector {
        element: Box::new(FeaType::Quantity((dimension).into())),
        length: Some(3),
    })
}

fn projected_field(
    field: &crate::FeaFieldType,
    value: Box<FeaType>,
    quantity: Option<runmat_analysis_core::ObservationQuantity>,
) -> FeaType {
    FeaType::Field(crate::FeaFieldType {
        value,
        location: field.location,
        quantity,
    })
}

fn phase_shape(value: &FeaType) -> FeaType {
    match value {
        FeaType::Vector { length, .. } => FeaType::Vector {
            element: Box::new(FeaType::Quantity(
                runmat_types::physical::DynamicDimension::quantity_type(
                    runmat_types::physical::QuantityKind::ComplexPhase,
                ),
            )),
            length: *length,
        },
        _ => FeaType::Quantity(runmat_types::physical::DynamicDimension::quantity_type(
            runmat_types::physical::QuantityKind::ComplexPhase,
        )),
    }
}
