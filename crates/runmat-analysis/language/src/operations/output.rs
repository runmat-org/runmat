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
            projection("von_mises"),
            extremum("min"),
            extremum("max"),
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
    VonMises,
    Minimum,
    Maximum,
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
            "fea.output.von_mises" => Some(Self::VonMises),
            "fea.output.min" => Some(Self::Minimum),
            "fea.output.max" => Some(Self::Maximum),
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
            (Self::Magnitude, Some(FeaType::Field(value))) => match value.as_ref() {
                FeaType::Vector {
                    element,
                    length: Some(3),
                } if element.is_numeric() => Ok(FeaType::Field(element.clone())),
                _ => Err("magnitude requires a three-component numeric field"),
            },
            (Self::Component, Some(FeaType::Field(value))) => match value.as_ref() {
                FeaType::Vector {
                    element,
                    length: Some(3),
                } if element.is_numeric() => Ok(FeaType::Field(element.clone())),
                _ => Err("component requires a three-component numeric field"),
            },
            (Self::Real | Self::Imaginary, Some(FeaType::Field(value))) => {
                Ok(FeaType::Field(value.clone()))
            }
            (Self::Phase, Some(FeaType::Field(value))) => {
                Ok(FeaType::Field(Box::new(dimensionless_shape(value))))
            }
            (Self::VonMises, Some(FeaType::Field(value))) => match value.as_ref() {
                FeaType::Vector {
                    element,
                    length: Some(6),
                } if element.as_ref() == &FeaType::Quantity(crate::PhysicalDimension::PRESSURE) => {
                    Ok(FeaType::Field(element.clone()))
                }
                _ => Err("von_mises requires a symmetric stress field"),
            },
            (Self::Minimum | Self::Maximum, Some(FeaType::Field(value))) if value.is_numeric() => {
                Ok(FeaType::AttributedExtremum(value.clone()))
            }
            (Self::Minimum | Self::Maximum, Some(FeaType::Field(_))) => {
                Err("spatial extrema require an explicit scalar field projection")
            }
            (Self::CollectCases, Some(value)) if value.is_numeric() => {
                Ok(FeaType::CaseTable(Box::new(value.clone())))
            }
            (Self::RetainEachCase, Some(FeaType::Field(value))) => {
                Ok(FeaType::CaseTable(Box::new(FeaType::Field(value.clone()))))
            }
            (Self::MinimumCase | Self::MaximumCase, Some(value)) if value.is_numeric() => {
                Ok(FeaType::AttributedExtremum(Box::new(value.clone())))
            }
            _ => Err("output operation requires an analysis field"),
        }
    }
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
        assert_eq!(extension.operations.len(), 10);
        assert_eq!(extension.operations[6].identity.as_str(), "min");
        assert_eq!(extension.operations[7].identity.as_str(), "max");
        assert!(extension.operations.iter().all(|operation| operation
            .overloads
            .iter()
            .all(|overload| overload.frontends == [OperationFrontend::Fea])));
    }
}
