use runmat_types::{
    AnalysisValueType, OperationCatalogExtension, OperationCatalogExtensionIdentity,
    OperationDistribution, OperationEffectFacts, OperationFacts, OperationFrontend,
    OperationIdentity, OperationLoweringId, OperationOverloadFacts, OperationOverloadIdentity,
    OperationParameter, OperationParameterArity, OperationQuantityRule, OperationSignature,
    OperationTypePattern, TopologyApplicability,
};

mod typecheck;

pub fn output_operation_extension() -> OperationCatalogExtension {
    let mut mean = spatial_reduction("mean");
    mean.overloads.push(series_reduction_overload("mean", true));
    let mut rms = spatial_reduction("rms");
    rms.overloads.push(series_reduction_overload("rms", true));
    OperationCatalogExtension {
        identity: id::<OperationCatalogExtensionIdentity>("runmat.fea.outputs"),
        operations: vec![
            projection("real"),
            projection("imag"),
            projection("phase"),
            projection("level"),
            magnitude(),
            component(),
            principal(),
            projection("von_mises"),
            recovery(),
            spatial_reduction("integral"),
            mean,
            rms,
            spatial_reduction("norm"),
            extremum("min"),
            extremum("max"),
            spatial_reduction("resultant"),
            flux(),
            sound_power(),
            port_operation(
                "pressure_drop",
                "Area-weighted mean static pressure over one face region minus that over another, Pa.",
                ["from", "to"],
                "fea.output.pressure_drop",
            ),
            port_operation(
                "transmission_loss",
                "Transmission loss of an acoustic harmonic analysis per frequency, dB: incident wave power through one face region over the outgoing wave power through an anechoic face region.",
                ["incident", "transmitted"],
                "fea.output.transmission_loss",
            ),
            moment(),
            sample(),
            interpolate(),
            series_operation("absmax", false),
            series_operation("variance", true),
            series_operation("envelope", false),
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
    Level,
    Magnitude,
    Component,
    Principal,
    VonMises,
    Recovery,
    Integral,
    Mean,
    RootMeanSquare,
    Norm,
    Minimum,
    Maximum,
    Resultant,
    Moment,
    NormalFlux,
    SoundPower,
    PressureDrop,
    TransmissionLoss,
    SpatialSample,
    InterpolatePoint,
    InterpolatePath,
    SampleSeries,
    MinimumSeries,
    MaximumSeries,
    AbsoluteMaximumSeries,
    MeanSeries,
    RootMeanSquareSeries,
    VarianceSeries,
    EnvelopeSeries,
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
            "fea.output.level" => Some(Self::Level),
            "fea.output.magnitude" => Some(Self::Magnitude),
            "fea.output.component" => Some(Self::Component),
            "fea.output.principal" => Some(Self::Principal),
            "fea.output.von_mises" => Some(Self::VonMises),
            "fea.output.recover" => Some(Self::Recovery),
            "fea.output.integral" => Some(Self::Integral),
            "fea.output.mean" => Some(Self::Mean),
            "fea.output.rms" => Some(Self::RootMeanSquare),
            "fea.output.norm" => Some(Self::Norm),
            "fea.output.min" => Some(Self::Minimum),
            "fea.output.max" => Some(Self::Maximum),
            "fea.output.resultant" => Some(Self::Resultant),
            "fea.output.moment" => Some(Self::Moment),
            "fea.output.flux" => Some(Self::NormalFlux),
            "fea.output.sound_power" => Some(Self::SoundPower),
            "fea.output.pressure_drop" => Some(Self::PressureDrop),
            "fea.output.transmission_loss" => Some(Self::TransmissionLoss),
            "fea.output.spatial.sample" => Some(Self::SpatialSample),
            "fea.output.interpolate.point" => Some(Self::InterpolatePoint),
            "fea.output.interpolate.path" => Some(Self::InterpolatePath),
            "fea.output.series.sample" => Some(Self::SampleSeries),
            "fea.output.series.min" => Some(Self::MinimumSeries),
            "fea.output.series.max" => Some(Self::MaximumSeries),
            "fea.output.series.absmax" => Some(Self::AbsoluteMaximumSeries),
            "fea.output.series.mean" => Some(Self::MeanSeries),
            "fea.output.series.rms" => Some(Self::RootMeanSquareSeries),
            "fea.output.series.variance" => Some(Self::VarianceSeries),
            "fea.output.series.envelope" => Some(Self::EnvelopeSeries),
            "fea.output.case.collect" => Some(Self::CollectCases),
            "fea.output.case.retain_each" => Some(Self::RetainEachCase),
            "fea.output.case.min" => Some(Self::MinimumCase),
            "fea.output.case.max" => Some(Self::MaximumCase),
            _ => None,
        }
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

fn recovery() -> OperationFacts {
    operation(
        "recover",
        "Recover scalar element von Mises stress at nodes with an explicit field-transfer method.",
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter(
                "to",
                OperationTypePattern::StringScalar,
                OperationParameterArity::Required,
            ),
            parameter(
                "using",
                OperationTypePattern::StringScalar,
                OperationParameterArity::Required,
            ),
        ],
        field(),
        OperationDistribution::PartitionLocal,
        TopologyApplicability::MeshElement,
        "fea.output.recover",
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

fn flux() -> OperationFacts {
    operation(
        "flux",
        "Integrate the outward normal component of a vector field through a face region.",
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter("through", region(), OperationParameterArity::Required),
        ],
        OperationTypePattern::Any,
        OperationDistribution::DeterministicReduction,
        TopologyApplicability::Region,
        "fea.output.flux",
    )
}

fn sound_power() -> OperationFacts {
    operation(
        "sound_power",
        "Mean acoustic power of a harmonic analysis through a face region (outward positive).",
        vec![
            parameter(
                "analysis",
                OperationTypePattern::Any,
                OperationParameterArity::Required,
            ),
            parameter("through", region(), OperationParameterArity::Required),
        ],
        OperationTypePattern::Any,
        OperationDistribution::DeterministicReduction,
        TopologyApplicability::Region,
        "fea.output.sound_power",
    )
}

/// An analysis reduced over two named face regions (ports) into one value per sample.
fn port_operation(name: &str, summary: &str, ports: [&str; 2], lowering: &str) -> OperationFacts {
    operation(
        name,
        summary,
        vec![
            parameter(
                "analysis",
                OperationTypePattern::Any,
                OperationParameterArity::Required,
            ),
            parameter(ports[0], region(), OperationParameterArity::Required),
            parameter(ports[1], region(), OperationParameterArity::Required),
        ],
        OperationTypePattern::Any,
        OperationDistribution::DeterministicReduction,
        TopologyApplicability::Region,
        lowering,
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
                    dimension: runmat_types::physical::DynamicDimension::from(
                        runmat_types::PhysicalDimension::LENGTH,
                    ),
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
    facts.overloads.push(series_reduction_overload(name, false));
    facts
}

fn sample() -> OperationFacts {
    let mut facts = operation(
        "sample",
        "Select a bounded spatial region or explicit independent-domain samples.",
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter("over", region(), OperationParameterArity::Required),
        ],
        field(),
        OperationDistribution::PartitionLocal,
        TopologyApplicability::Region,
        "fea.output.spatial.sample",
    );
    facts.overloads.push(OperationOverloadFacts {
        identity: id::<OperationOverloadIdentity>("sample:fea-series"),
        frontends: vec![OperationFrontend::Fea],
        signature: OperationSignature {
            inputs: vec![
                parameter(
                    "source",
                    OperationTypePattern::Any,
                    OperationParameterArity::Required,
                ),
                parameter(
                    "over",
                    OperationTypePattern::Any,
                    OperationParameterArity::Required,
                ),
                parameter(
                    "at",
                    OperationTypePattern::NumericArray,
                    OperationParameterArity::Required,
                ),
            ],
            outputs: vec![parameter(
                "result",
                history(),
                OperationParameterArity::Required,
            )],
        },
        quantity: OperationQuantityRule::CheckedByLowering,
        topology: TopologyApplicability::AnyGeometry,
        effects: OperationEffectFacts::default(),
        distribution: OperationDistribution::PartitionLocal,
        lowering: id::<OperationLoweringId>("fea.output.series.sample"),
    });
    facts
}

fn interpolate() -> OperationFacts {
    let point = OperationTypePattern::VectorQuantity {
        dimension: runmat_types::physical::DynamicDimension::from(
            runmat_types::PhysicalDimension::LENGTH,
        ),
        length: Some(3),
    };
    let mut facts = operation(
        "interpolate",
        "Interpolate a field at an explicit physical point.",
        vec![
            parameter("field", field(), OperationParameterArity::Required),
            parameter("at", point.clone(), OperationParameterArity::Required),
        ],
        field(),
        OperationDistribution::PartitionLocal,
        TopologyApplicability::AnyGeometry,
        "fea.output.interpolate.point",
    );
    facts.overloads.push(OperationOverloadFacts {
        identity: id::<OperationOverloadIdentity>("interpolate:fea-path"),
        frontends: vec![OperationFrontend::Fea],
        signature: OperationSignature {
            inputs: vec![
                parameter("field", field(), OperationParameterArity::Required),
                parameter(
                    "at",
                    OperationTypePattern::List(Box::new(point)),
                    OperationParameterArity::Required,
                ),
            ],
            outputs: vec![parameter(
                "result",
                field(),
                OperationParameterArity::Required,
            )],
        },
        quantity: OperationQuantityRule::CheckedByLowering,
        topology: TopologyApplicability::AnyGeometry,
        effects: OperationEffectFacts::default(),
        distribution: OperationDistribution::PartitionLocal,
        lowering: id::<OperationLoweringId>("fea.output.interpolate.path"),
    });
    facts
}

fn series_operation(name: &str, weighting: bool) -> OperationFacts {
    OperationFacts {
        identity: id::<OperationIdentity>(name),
        category: "analysis output".into(),
        summary: format!("Apply the typed `{name}` reduction over an observation history."),
        overloads: vec![series_reduction_overload(name, weighting)],
    }
}

fn series_reduction_overload(name: &str, weighting: bool) -> OperationOverloadFacts {
    let mut inputs = vec![
        parameter("history", history(), OperationParameterArity::Required),
        parameter(
            "over",
            OperationTypePattern::Any,
            OperationParameterArity::Required,
        ),
    ];
    if weighting {
        inputs.push(parameter(
            "weighting",
            OperationTypePattern::StringScalar,
            OperationParameterArity::Required,
        ));
    }
    OperationOverloadFacts {
        identity: id::<OperationOverloadIdentity>(format!("{name}:fea-series")),
        frontends: vec![OperationFrontend::Fea],
        signature: OperationSignature {
            inputs,
            outputs: vec![parameter(
                "result",
                OperationTypePattern::Any,
                OperationParameterArity::Required,
            )],
        },
        quantity: OperationQuantityRule::CheckedByLowering,
        topology: TopologyApplicability::AnyGeometry,
        effects: OperationEffectFacts::default(),
        distribution: OperationDistribution::DeterministicReduction,
        lowering: id::<OperationLoweringId>(format!("fea.output.series.{name}")),
    }
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

fn history() -> OperationTypePattern {
    OperationTypePattern::Analysis(AnalysisValueType::History)
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

    // witness: FEA-AUTHORING-OUTPUT-002/shared_query_catalog @
    //   shared_infrastructure/mesh_independent/source_check; coverage: partial;
    //   boundary: Output operations are one shared public identity with FEA overloads.
    #[test]
    fn output_operations_are_one_shared_public_identity_with_fea_overloads() {
        let extension = output_operation_extension();
        assert_eq!(extension.operations.len(), 28);
        for expected in ["recover", "min", "max"] {
            assert!(extension
                .operations
                .iter()
                .any(|operation| operation.identity.as_str() == expected));
        }
        assert!(extension.operations.iter().all(|operation| operation
            .overloads
            .iter()
            .all(|overload| overload.frontends == [OperationFrontend::Fea])));
    }
}
