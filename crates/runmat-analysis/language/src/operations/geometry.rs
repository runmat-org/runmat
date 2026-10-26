use runmat_types::{
    AnalysisValueType, OperationCatalogExtension, OperationCatalogExtensionIdentity,
    OperationDistribution, OperationEffectFacts, OperationFacts, OperationFrontend,
    OperationIdentity, OperationLoweringId, OperationOverloadFacts, OperationOverloadIdentity,
    OperationParameter, OperationParameterArity, OperationQuantityRule, OperationSignature,
    OperationTypePattern, TopologyApplicability,
};

pub fn geometry_operation_extension() -> OperationCatalogExtension {
    OperationCatalogExtension {
        identity: identity::<OperationCatalogExtensionIdentity>("runmat.fea.geometry"),
        operations: vec![
            operation(
                "box",
                &[
                    ("width", length()),
                    ("depth", length()),
                    ("height", length()),
                ],
            ),
            operation(
                "cylinder",
                &[
                    ("radius", length()),
                    ("height", length()),
                    ("axis", numeric_array()),
                    ("origin", length_vector()),
                ],
            ),
            operation(
                "sphere",
                &[("radius", length()), ("center", length_vector())],
            ),
            operation(
                "transform",
                &[("source", feature()), ("matrix", numeric_array())],
            ),
            operation(
                "instance",
                &[
                    ("source", feature()),
                    (
                        "transforms",
                        OperationTypePattern::List(Box::new(numeric_array())),
                    ),
                ],
            ),
            operation(
                "extrude",
                &[
                    ("profile", feature()),
                    ("distance", length()),
                    ("direction", numeric_array()),
                ],
            ),
            operation(
                "revolve",
                &[
                    ("profile", feature()),
                    ("axis", numeric_array()),
                    ("origin", length_vector()),
                    ("angle", angle()),
                ],
            ),
            variadic_operation("union", "operands", 2),
            variadic_operation("subtract", "operands", 2),
            variadic_operation("intersect", "operands", 2),
            operation(
                "hole",
                &[
                    ("target", feature()),
                    ("radius", length()),
                    ("depth", length()),
                    ("axis", numeric_array()),
                    ("origin", length_vector()),
                ],
            ),
            operation(
                "pocket",
                &[
                    ("target", feature()),
                    ("profile", feature()),
                    ("depth", length()),
                    ("direction", numeric_array()),
                ],
            ),
            operation(
                "fillet",
                &[
                    ("target", feature()),
                    ("edges", region()),
                    ("radius", length()),
                ],
            ),
            operation(
                "chamfer",
                &[
                    ("target", feature()),
                    ("edges", region()),
                    ("distance", length()),
                ],
            ),
            operation("regenerate", &[("source", feature())]),
            global_operation(),
        ],
    }
}

fn operation(name: &str, parameters: &[(&str, OperationTypePattern)]) -> OperationFacts {
    facts(
        name,
        parameters
            .iter()
            .map(|(parameter, value)| OperationParameter {
                name: (*parameter).to_string(),
                value: value.clone(),
                arity: OperationParameterArity::Required,
                default: None,
            })
            .collect(),
        OperationTypePattern::Analysis(AnalysisValueType::GeometryFeature),
    )
}

fn length() -> OperationTypePattern {
    OperationTypePattern::Quantity(runmat_types::physical::DynamicDimension::from(
        runmat_types::PhysicalDimension::LENGTH,
    ))
}

fn angle() -> OperationTypePattern {
    OperationTypePattern::Quantity(runmat_types::physical::DynamicDimension::quantity_type(
        runmat_types::physical::QuantityKind::Angle,
    ))
}

fn length_vector() -> OperationTypePattern {
    OperationTypePattern::VectorQuantity {
        dimension: runmat_types::physical::DynamicDimension::from(
            runmat_types::PhysicalDimension::LENGTH,
        ),
        length: Some(3),
    }
}

fn feature() -> OperationTypePattern {
    OperationTypePattern::Analysis(AnalysisValueType::GeometryFeature)
}

fn region() -> OperationTypePattern {
    OperationTypePattern::Analysis(AnalysisValueType::Region)
}

fn numeric_array() -> OperationTypePattern {
    OperationTypePattern::NumericArray
}

fn variadic_operation(name: &str, parameter: &str, minimum_operands: usize) -> OperationFacts {
    let mut inputs = Vec::with_capacity(minimum_operands);
    for index in 0..minimum_operands {
        inputs.push(OperationParameter {
            name: if index == 0 {
                parameter.to_string()
            } else {
                format!("{parameter}_{index}")
            },
            value: OperationTypePattern::Analysis(AnalysisValueType::GeometryFeature),
            arity: if index + 1 == minimum_operands {
                OperationParameterArity::Variadic
            } else {
                OperationParameterArity::Required
            },
            default: None,
        });
    }
    facts(
        name,
        inputs,
        OperationTypePattern::Analysis(AnalysisValueType::GeometryFeature),
    )
}

fn global_operation() -> OperationFacts {
    facts(
        "global",
        vec![OperationParameter {
            name: "components".to_string(),
            value: OperationTypePattern::NumericArray,
            arity: OperationParameterArity::Required,
            default: None,
        }],
        OperationTypePattern::NumericArray,
    )
}

fn facts(
    name: &str,
    inputs: Vec<OperationParameter>,
    output: OperationTypePattern,
) -> OperationFacts {
    OperationFacts {
        identity: identity::<OperationIdentity>(name),
        category: "geometry".to_string(),
        summary: format!("Typed FEA geometry `{name}` operation."),
        overloads: vec![OperationOverloadFacts {
            identity: identity::<OperationOverloadIdentity>(format!("{name}:fea-geometry")),
            frontends: vec![OperationFrontend::Fea],
            signature: OperationSignature {
                inputs,
                outputs: vec![OperationParameter {
                    name: "result".to_string(),
                    value: output,
                    arity: OperationParameterArity::Required,
                    default: None,
                }],
            },
            quantity: OperationQuantityRule::CheckedByLowering,
            topology: TopologyApplicability::AnyGeometry,
            effects: OperationEffectFacts::default(),
            distribution: OperationDistribution::HostOnly,
            lowering: identity::<OperationLoweringId>(format!("fea.geometry.{name}")),
        }],
    }
}

trait Identity: Sized {
    fn create(value: String) -> Self;
}

macro_rules! identity_impl {
    ($type:ty) => {
        impl Identity for $type {
            fn create(value: String) -> Self {
                <$type>::new(value).expect("static geometry operation identities are valid")
            }
        }
    };
}

identity_impl!(OperationCatalogExtensionIdentity);
identity_impl!(OperationIdentity);
identity_impl!(OperationOverloadIdentity);
identity_impl!(OperationLoweringId);

fn identity<T: Identity>(value: impl Into<String>) -> T {
    T::create(value.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn geometry_operations_have_unique_closed_lowerings() {
        let extension = geometry_operation_extension();
        let names = extension
            .operations
            .iter()
            .map(|operation| operation.identity.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), extension.operations.len());
        assert!(extension.operations.iter().all(|operation| {
            operation.overloads.len() == 1
                && operation.overloads[0]
                    .lowering
                    .as_str()
                    .starts_with("fea.geometry.")
        }));
    }
}
