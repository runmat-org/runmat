use runmat_types::{
    AnalysisValueType, OperationCatalogExtension, OperationCatalogExtensionIdentity,
    OperationDistribution, OperationEffectFacts, OperationFacts, OperationFrontend,
    OperationIdentity, OperationLoweringId, OperationOverloadFacts, OperationOverloadIdentity,
    OperationParameter, OperationParameterArity, OperationQuantityRule, OperationSignature,
    OperationTypePattern, TopologyApplicability,
};

use crate::{CheckedCallArgument, FeaType};

pub fn output_operation_extension() -> OperationCatalogExtension {
    OperationCatalogExtension {
        identity: id::<OperationCatalogExtensionIdentity>("runmat.fea.outputs"),
        operations: vec![
            projection("real"),
            projection("imag"),
            projection("phase"),
            magnitude(),
            component(),
            principal(),
            projection("von_mises"),
            spatial_reduction("integral"),
            spatial_reduction("mean"),
            spatial_reduction("rms"),
            spatial_reduction("norm"),
            extremum("min"),
            extremum("max"),
            spatial_reduction("resultant"),
            moment(),
            case_operation("collect", "fea.output.case.collect"),
            case_operation("retain_each", "fea.output.case.retain_each"),
        ],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputOperation {
    Real,
    Imaginary,
    Phase,
    Magnitude,
    Component,
    Principal,
    VonMises,
    Integral,
    Mean,
    RootMeanSquare,
    Norm,
    Minimum,
    Maximum,
    Resultant,
    Moment,
    CollectCases,
    RetainEachCase,
    MinimumCase,
    MaximumCase,
}

impl OutputOperation {
    pub(crate) fn from_lowering(lowering: &OperationLoweringId) -> Option<Self> {
        match lowering.as_str() {
            "fea.output.real" => Some(Self::Real),
            "fea.output.imag" => Some(Self::Imaginary),
            "fea.output.phase" => Some(Self::Phase),
            "fea.output.magnitude" => Some(Self::Magnitude),
            "fea.output.component" => Some(Self::Component),
            "fea.output.principal" => Some(Self::Principal),
            "fea.output.von_mises" => Some(Self::VonMises),
            "fea.output.integral" => Some(Self::Integral),
            "fea.output.mean" => Some(Self::Mean),
            "fea.output.rms" => Some(Self::RootMeanSquare),
            "fea.output.norm" => Some(Self::Norm),
            "fea.output.min" => Some(Self::Minimum),
            "fea.output.max" => Some(Self::Maximum),
            "fea.output.resultant" => Some(Self::Resultant),
            "fea.output.moment" => Some(Self::Moment),
            "fea.output.case.collect" => Some(Self::CollectCases),
            "fea.output.case.retain_each" => Some(Self::RetainEachCase),
            "fea.output.case.min" => Some(Self::MinimumCase),
            "fea.output.case.max" => Some(Self::MaximumCase),
            _ => None,
        }
    }

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
            (Self::Magnitude, Some(FeaType::Field(field))) => match field.value.as_ref() {
                FeaType::Vector {
                    element,
                    length: Some(3),
                } if element.is_numeric() => {
                    Ok(projected_field(field, element.clone(), field.quantity))
                }
                _ => Err("magnitude requires a three-component numeric field"),
            },
            (Self::Component, Some(FeaType::Field(field))) => match field.value.as_ref() {
                FeaType::Vector {
                    element,
                    length: Some(3),
                } if element.is_numeric() => {
                    Ok(projected_field(field, element.clone(), field.quantity))
                }
                _ => Err("component requires a three-component numeric field"),
            },
            (Self::Real | Self::Imaginary, Some(FeaType::Field(field))) => {
                Ok(FeaType::Field(field.clone()))
            }
            (Self::Phase, Some(FeaType::Field(field))) => Ok(projected_field(
                field,
                Box::new(dimensionless_shape(&field.value)),
                None,
            )),
            (Self::VonMises, Some(FeaType::Field(field))) => match field.value.as_ref() {
                FeaType::Vector {
                    element,
                    length: Some(6),
                } if element.as_ref() == &FeaType::Quantity(crate::PhysicalDimension::PRESSURE) => {
                    Ok(projected_field(
                        field,
                        element.clone(),
                        Some(runmat_analysis_core::ObservationQuantity::VonMisesStress),
                    ))
                }
                _ => Err("von_mises requires a symmetric stress field"),
            },
            (Self::Principal, Some(FeaType::Field(field))) => match field.value.as_ref() {
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

fn scalar_dimension(value: &FeaType) -> Option<crate::PhysicalDimension> {
    match value {
        FeaType::Quantity(dimension) => Some(*dimension),
        FeaType::Integer | FeaType::Real | FeaType::Complex => {
            Some(crate::PhysicalDimension::DIMENSIONLESS)
        }
        FeaType::Vector { element, .. } => scalar_dimension(element),
        _ => None,
    }
}

fn scale_numeric_type(value: &FeaType, measure: crate::PhysicalDimension) -> Option<FeaType> {
    match value {
        FeaType::Quantity(dimension) => Some(FeaType::Quantity(dimension.multiply(measure)?)),
        FeaType::Integer | FeaType::Real | FeaType::Complex => Some(FeaType::Quantity(measure)),
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
        element: Box::new(FeaType::Quantity(dimension)),
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

fn dimensionless_shape(value: &FeaType) -> FeaType {
    match value {
        FeaType::Vector { length, .. } => FeaType::Vector {
            element: Box::new(FeaType::Quantity(crate::PhysicalDimension::DIMENSIONLESS)),
            length: *length,
        },
        _ => FeaType::Quantity(crate::PhysicalDimension::DIMENSIONLESS),
    }
}

fn magnitude() -> OperationFacts {
    operation(
        "magnitude",
        "Compute the scalar magnitude of a vector analysis field.",
        vec![parameter(
            "field",
            field(),
            OperationParameterArity::Required,
        )],
        field(),
        OperationDistribution::PartitionLocal,
        TopologyApplicability::AnyGeometry,
        "fea.output.magnitude",
    )
}

fn projection(name: &str) -> OperationFacts {
    operation(
        name,
        &format!("Apply the typed `{name}` representation to an analysis field."),
        vec![parameter(
            "field",
            field(),
            OperationParameterArity::Required,
        )],
        field(),
        OperationDistribution::PartitionLocal,
        TopologyApplicability::AnyGeometry,
        &format!("fea.output.{name}"),
    )
}

fn component() -> OperationFacts {
    operation(
        "component",
        "Select one Cartesian component of a vector analysis field.",
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter(
                "component",
                OperationTypePattern::StringScalar,
                OperationParameterArity::Required,
            ),
        ],
        field(),
        OperationDistribution::PartitionLocal,
        TopologyApplicability::AnyGeometry,
        "fea.output.component",
    )
}

fn principal() -> OperationFacts {
    operation(
        "principal",
        "Select the maximum or minimum principal value of a symmetric tensor field.",
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter(
                "which",
                OperationTypePattern::StringScalar,
                OperationParameterArity::Required,
            ),
        ],
        field(),
        OperationDistribution::PartitionLocal,
        TopologyApplicability::AnyGeometry,
        "fea.output.principal",
    )
}

fn spatial_reduction(name: &str) -> OperationFacts {
    operation(
        name,
        &format!("Apply the typed spatial `{name}` reduction to an analysis field."),
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter("over", region(), OperationParameterArity::Required),
        ],
        OperationTypePattern::Any,
        OperationDistribution::DeterministicReduction,
        TopologyApplicability::Region,
        &format!("fea.output.{name}"),
    )
}

fn moment() -> OperationFacts {
    operation(
        "moment",
        "Integrate a force resultant moment about an explicit point.",
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter("over", region(), OperationParameterArity::Required),
            parameter(
                "about",
                OperationTypePattern::VectorQuantity {
                    dimension: runmat_types::QuantityDimension::Length,
                    length: Some(3),
                },
                OperationParameterArity::Required,
            ),
        ],
        OperationTypePattern::Any,
        OperationDistribution::DeterministicReduction,
        TopologyApplicability::Region,
        "fea.output.moment",
    )
}

fn extremum(name: &str) -> OperationFacts {
    let mut facts = operation(
        name,
        &format!("Find the attributed spatial {name}imum of a scalar analysis field."),
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter("over", region(), OperationParameterArity::Required),
        ],
        OperationTypePattern::Analysis(AnalysisValueType::AttributedExtremum),
        OperationDistribution::DeterministicReduction,
        TopologyApplicability::Region,
        &format!("fea.output.{name}"),
    );
    facts.overloads.push(OperationOverloadFacts {
        identity: id::<OperationOverloadIdentity>(format!("{name}:fea-cases")),
        frontends: vec![OperationFrontend::Fea],
        signature: OperationSignature {
            inputs: vec![
                parameter(
                    "value",
                    OperationTypePattern::NumericScalar,
                    OperationParameterArity::Required,
                ),
                parameter(
                    "over",
                    OperationTypePattern::Any,
                    OperationParameterArity::Required,
                ),
            ],
            outputs: vec![parameter(
                "result",
                OperationTypePattern::Analysis(AnalysisValueType::AttributedExtremum),
                OperationParameterArity::Required,
            )],
        },
        quantity: OperationQuantityRule::CheckedByLowering,
        topology: TopologyApplicability::AnyGeometry,
        effects: OperationEffectFacts::default(),
        distribution: OperationDistribution::DeterministicReduction,
        lowering: id::<OperationLoweringId>(format!("fea.output.case.{name}")),
    });
    facts
}

fn case_operation(name: &str, lowering: &str) -> OperationFacts {
    operation(
        name,
        "Collect one typed result from every independently schedulable sweep case.",
        vec![parameter(
            "value",
            OperationTypePattern::Any,
            OperationParameterArity::Required,
        )],
        OperationTypePattern::Analysis(AnalysisValueType::CaseTable),
        OperationDistribution::CheckedByLowering,
        TopologyApplicability::AnyGeometry,
        lowering,
    )
}

fn operation(
    name: &str,
    summary: &str,
    inputs: Vec<OperationParameter>,
    output: OperationTypePattern,
    distribution: OperationDistribution,
    topology: TopologyApplicability,
    lowering: &str,
) -> OperationFacts {
    OperationFacts {
        identity: id::<OperationIdentity>(name),
        category: "analysis output".into(),
        summary: summary.into(),
        overloads: vec![OperationOverloadFacts {
            identity: id::<OperationOverloadIdentity>(format!("{name}:fea-output")),
            frontends: vec![OperationFrontend::Fea],
            signature: OperationSignature {
                inputs,
                outputs: vec![parameter(
                    "result",
                    output,
                    OperationParameterArity::Required,
                )],
            },
            quantity: OperationQuantityRule::CheckedByLowering,
            topology,
            effects: OperationEffectFacts::default(),
            distribution,
            lowering: id::<OperationLoweringId>(lowering),
        }],
    }
}

fn parameter(
    name: &str,
    value: OperationTypePattern,
    arity: OperationParameterArity,
) -> OperationParameter {
    OperationParameter {
        name: name.into(),
        value,
        arity,
        default: None,
    }
}

fn field() -> OperationTypePattern {
    OperationTypePattern::Analysis(AnalysisValueType::Field)
}

fn region() -> OperationTypePattern {
    OperationTypePattern::Analysis(AnalysisValueType::Region)
}

trait OperationId: Sized {
    fn parse(value: String) -> Self;
}

fn id<T: OperationId>(value: impl Into<String>) -> T {
    T::parse(value.into())
}

macro_rules! operation_id {
    ($kind:ty) => {
        impl OperationId for $kind {
            fn parse(value: String) -> Self {
                <$kind>::new(value).expect("static output-operation identities are valid")
            }
        }
    };
}

operation_id!(OperationCatalogExtensionIdentity);
operation_id!(OperationIdentity);
operation_id!(OperationOverloadIdentity);
operation_id!(OperationLoweringId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_operations_are_one_shared_public_identity_with_fea_overloads() {
        let extension = output_operation_extension();
        assert_eq!(extension.operations.len(), 17);
        assert_eq!(extension.operations[11].identity.as_str(), "min");
        assert_eq!(extension.operations[12].identity.as_str(), "max");
        assert!(extension.operations.iter().all(|operation| operation
            .overloads
            .iter()
            .all(|overload| overload.frontends == [OperationFrontend::Fea])));
    }
}
