use super::{
    builtin_catalog_entries, BuiltinCatalogEntry, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinResidencyPolicy, BuiltinSignatureDescriptor,
};
use runmat_types::{
    OperationCatalogExtension, OperationDefaultValue, OperationDistribution, OperationEffectFacts,
    OperationFacts, OperationFrontend, OperationIdentity, OperationLoweringId,
    OperationOverloadFacts, OperationOverloadIdentity, OperationParameter, OperationParameterArity,
    OperationQuantityRule, OperationSignature, OperationTypePattern, TopologyApplicability,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const COMPOSED_OPERATION_CATALOG_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedOperationCatalog {
    operations: Vec<OperationFacts>,
    fingerprint: [u8; 32],
}

impl ComposedOperationCatalog {
    pub fn operations(&self) -> &[OperationFacts] {
        &self.operations
    }

    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    pub fn operation(&self, identity: &OperationIdentity) -> Option<&OperationFacts> {
        self.operations
            .binary_search_by(|candidate| candidate.identity.cmp(identity))
            .ok()
            .map(|index| &self.operations[index])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OperationCatalogCompositionError {
    #[error("duplicate operation catalog extension `{0}`")]
    DuplicateExtension(String),
    #[error("operation `{operation}` has conflicting {field}")]
    ConflictingOperationFacts {
        operation: String,
        field: &'static str,
    },
    #[error("operation `{0}` must have a category")]
    MissingCategory(String),
    #[error("operation `{0}` must have documentation")]
    MissingDocumentation(String),
    #[error("operation `{0}` has no overloads")]
    MissingOverloads(String),
    #[error("operation `{operation}` repeats overload `{overload}`")]
    DuplicateOverload { operation: String, overload: String },
    #[error("operation `{operation}` overload `{overload}` has no frontend")]
    MissingFrontend { operation: String, overload: String },
    #[error("operation `{operation}` overload `{overload}` repeats a frontend")]
    DuplicateFrontend { operation: String, overload: String },
    #[error("operation `{operation}` has ambiguous overloads `{left}` and `{right}`")]
    AmbiguousOverload {
        operation: String,
        left: String,
        right: String,
    },
    #[error("invalid identity in the standard builtin catalog: {0}")]
    InvalidStandardIdentity(String),
    #[error("cannot fingerprint the operation catalog: {0}")]
    Fingerprint(String),
}

pub fn compose_operation_catalog(
    extensions: &[OperationCatalogExtension],
) -> Result<ComposedOperationCatalog, OperationCatalogCompositionError> {
    let mut operations = BTreeMap::<OperationIdentity, OperationFacts>::new();
    for entry in builtin_catalog_entries() {
        insert_operation(&mut operations, standard_operation(entry)?)?;
    }
    let mut standard = operations
        .keys()
        .cloned()
        .collect::<BTreeSet<OperationIdentity>>();

    let mut extension_ids = BTreeSet::new();
    let mut ordered_extensions = extensions.iter().collect::<Vec<_>>();
    ordered_extensions.sort_unstable_by(|left, right| left.identity.cmp(&right.identity));
    for extension in ordered_extensions {
        if !extension_ids.insert(extension.identity.as_str()) {
            return Err(OperationCatalogCompositionError::DuplicateExtension(
                extension.identity.to_string(),
            ));
        }
        let mut extension_operations = extension.operations.to_vec();
        extension_operations.sort_unstable_by(|left, right| left.identity.cmp(&right.identity));
        for operation in extension_operations {
            // Analysis-language operations shadow a same-named standard builtin
            // (e.g. FEA `union` is geometry, not set union); extensions still
            // compose with each other under the ordinary conflict rules.
            if standard.remove(&operation.identity) {
                operations.remove(&operation.identity);
            }
            insert_operation(&mut operations, operation)?;
        }
    }

    let mut operations = operations.into_values().collect::<Vec<_>>();
    for operation in &mut operations {
        for overload in &mut operation.overloads {
            overload.frontends.sort_unstable();
        }
        operation
            .overloads
            .sort_unstable_by(|left, right| left.identity.cmp(&right.identity));
        // Standard builtins resolve overloads at run time (MATLAB semantics);
        // analysis-language operations must be statically unambiguous.
        validate_operation(operation, !standard.contains(&operation.identity))?;
    }
    let fingerprint = catalog_fingerprint(&operations)?;
    Ok(ComposedOperationCatalog {
        operations,
        fingerprint,
    })
}

fn insert_operation(
    operations: &mut BTreeMap<OperationIdentity, OperationFacts>,
    mut incoming: OperationFacts,
) -> Result<(), OperationCatalogCompositionError> {
    match operations.get_mut(&incoming.identity) {
        Some(existing) => {
            if existing.category != incoming.category {
                return Err(
                    OperationCatalogCompositionError::ConflictingOperationFacts {
                        operation: existing.identity.to_string(),
                        field: "category",
                    },
                );
            }
            if existing.summary != incoming.summary {
                return Err(
                    OperationCatalogCompositionError::ConflictingOperationFacts {
                        operation: existing.identity.to_string(),
                        field: "documentation summary",
                    },
                );
            }
            existing.overloads.append(&mut incoming.overloads);
        }
        None => {
            operations.insert(incoming.identity.clone(), incoming);
        }
    }
    Ok(())
}

fn validate_operation(
    operation: &OperationFacts,
    statically_resolved: bool,
) -> Result<(), OperationCatalogCompositionError> {
    let name = operation.identity.to_string();
    if operation.category.trim().is_empty() {
        return Err(OperationCatalogCompositionError::MissingCategory(name));
    }
    if operation.summary.trim().is_empty() {
        return Err(OperationCatalogCompositionError::MissingDocumentation(name));
    }
    if operation.overloads.is_empty() {
        return Err(OperationCatalogCompositionError::MissingOverloads(name));
    }
    let mut identities = BTreeSet::new();
    for overload in &operation.overloads {
        let overload_name = overload.identity.to_string();
        if !identities.insert(overload.identity.clone()) {
            return Err(OperationCatalogCompositionError::DuplicateOverload {
                operation: name,
                overload: overload_name,
            });
        }
        if overload.frontends.is_empty() {
            return Err(OperationCatalogCompositionError::MissingFrontend {
                operation: name,
                overload: overload_name,
            });
        }
        let unique = overload.frontends.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != overload.frontends.len() {
            return Err(OperationCatalogCompositionError::DuplicateFrontend {
                operation: name,
                overload: overload_name,
            });
        }
    }
    if !statically_resolved {
        return Ok(());
    }
    for (index, left) in operation.overloads.iter().enumerate() {
        for right in &operation.overloads[index + 1..] {
            let shares_frontend = left
                .frontends
                .iter()
                .any(|frontend| right.frontends.contains(frontend));
            if shares_frontend && left.signature == right.signature {
                return Err(OperationCatalogCompositionError::AmbiguousOverload {
                    operation: name,
                    left: left.identity.to_string(),
                    right: right.identity.to_string(),
                });
            }
        }
    }
    Ok(())
}

fn standard_operation(
    entry: &BuiltinCatalogEntry,
) -> Result<OperationFacts, OperationCatalogCompositionError> {
    let identity = OperationIdentity::new(entry.identity.name).map_err(|error| {
        OperationCatalogCompositionError::InvalidStandardIdentity(error.to_string())
    })?;
    let overloads = entry
        .descriptor
        .signatures
        .iter()
        .enumerate()
        .map(|(ordinal, signature)| standard_overload(entry, ordinal, signature))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OperationFacts {
        identity,
        category: entry.category.to_string(),
        summary: entry.documentation.summary.to_string(),
        overloads,
    })
}

fn standard_overload(
    entry: &BuiltinCatalogEntry,
    ordinal: usize,
    signature: &BuiltinSignatureDescriptor,
) -> Result<OperationOverloadFacts, OperationCatalogCompositionError> {
    // Signature labels are documentation and may repeat; the ordinal keeps the
    // overload identity unique and stable for the declared signature order.
    let overload_identity = format!("{}#{ordinal}:{}", entry.identity.name, signature.label);
    let identity = OperationOverloadIdentity::new(overload_identity).map_err(|error| {
        OperationCatalogCompositionError::InvalidStandardIdentity(error.to_string())
    })?;
    let lowering = OperationLoweringId::new(entry.identity.name).map_err(|error| {
        OperationCatalogCompositionError::InvalidStandardIdentity(error.to_string())
    })?;
    let distribution = match entry.placement.residency {
        BuiltinResidencyPolicy::Host | BuiltinResidencyPolicy::GatherToHost => {
            OperationDistribution::HostOnly
        }
        BuiltinResidencyPolicy::PreserveInputs
        | BuiltinResidencyPolicy::ProduceResident
        | BuiltinResidencyPolicy::Dynamic => OperationDistribution::CheckedByLowering,
    };
    Ok(OperationOverloadFacts {
        identity,
        frontends: vec![OperationFrontend::Matlab],
        signature: OperationSignature {
            inputs: signature.inputs.iter().map(parameter).collect(),
            outputs: signature.outputs.iter().map(parameter).collect(),
        },
        quantity: OperationQuantityRule::NotApplicable,
        topology: TopologyApplicability::NotApplicable,
        effects: OperationEffectFacts {
            effects: entry.contract.effect_set(),
            capabilities: entry.contract.capability_set(),
        },
        distribution,
        lowering,
    })
}

fn parameter(parameter: &BuiltinParamDescriptor) -> OperationParameter {
    OperationParameter {
        name: parameter.name.to_string(),
        value: match parameter.ty {
            BuiltinParamType::Any => OperationTypePattern::Any,
            BuiltinParamType::NumericScalar => OperationTypePattern::NumericScalar,
            BuiltinParamType::IntegerScalar => OperationTypePattern::IntegerScalar,
            BuiltinParamType::StringScalar => OperationTypePattern::StringScalar,
            BuiltinParamType::NumericArray => OperationTypePattern::NumericArray,
            BuiltinParamType::LogicalArray => OperationTypePattern::LogicalArray,
            BuiltinParamType::SizeArg => OperationTypePattern::Size,
            BuiltinParamType::LikePrototype => OperationTypePattern::Prototype,
            BuiltinParamType::AxesHandle => OperationTypePattern::Axes,
            BuiltinParamType::StyleSpec => OperationTypePattern::Style,
            BuiltinParamType::PropertyName => OperationTypePattern::PropertyName,
            BuiltinParamType::PropertyValue => OperationTypePattern::PropertyValue,
            BuiltinParamType::Callable => OperationTypePattern::Callable,
        },
        arity: match parameter.arity {
            BuiltinParamArity::Required => OperationParameterArity::Required,
            BuiltinParamArity::Optional => OperationParameterArity::Optional,
            BuiltinParamArity::Variadic => OperationParameterArity::Variadic,
        },
        default: parameter.default.map(|value| {
            value
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .map_or_else(
                    || OperationDefaultValue::EnumCase(value.to_string()),
                    |value| OperationDefaultValue::String(value.to_string()),
                )
        }),
    }
}

fn catalog_fingerprint(
    operations: &[OperationFacts],
) -> Result<[u8; 32], OperationCatalogCompositionError> {
    #[derive(Serialize)]
    struct FingerprintInput<'a> {
        schema: u32,
        operations: &'a [OperationFacts],
    }

    let encoded = serde_json::to_vec(&FingerprintInput {
        schema: COMPOSED_OPERATION_CATALOG_SCHEMA,
        operations,
    })
    .map_err(|error| OperationCatalogCompositionError::Fingerprint(error.to_string()))?;
    let mut hash = Sha256::new();
    hash.update(b"runmat-composed-operation-catalog");
    hash.update(encoded);
    Ok(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::OperationCatalogExtensionIdentity;

    fn extension(identity: &str, operation: &str, overload: &str) -> OperationCatalogExtension {
        OperationCatalogExtension {
            identity: OperationCatalogExtensionIdentity::new(identity).unwrap(),
            operations: vec![OperationFacts {
                identity: OperationIdentity::new(operation).unwrap(),
                category: "analysis".to_string(),
                summary: "Analysis operation.".to_string(),
                overloads: vec![OperationOverloadFacts {
                    identity: OperationOverloadIdentity::new(overload).unwrap(),
                    frontends: vec![OperationFrontend::Fea],
                    signature: OperationSignature {
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                    },
                    quantity: OperationQuantityRule::CheckedByLowering,
                    topology: TopologyApplicability::Region,
                    effects: OperationEffectFacts::default(),
                    distribution: OperationDistribution::DeterministicReduction,
                    lowering: OperationLoweringId::new(overload).unwrap(),
                }],
            }],
        }
    }

    #[test]
    fn composition_and_fingerprint_are_order_independent() {
        let left = extension("analysis-output", "maximum", "maximum:fea-field");
        let right = extension("geometry", "box", "box:fea-geometry");
        let first = compose_operation_catalog(&[left.clone(), right.clone()]).unwrap();
        let second = compose_operation_catalog(&[right, left]).unwrap();
        assert_eq!(first.operations(), second.operations());
        assert_eq!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn duplicate_overloads_are_rejected() {
        let first = extension("one", "maximum", "maximum:fea-field");
        let second = extension("two", "maximum", "maximum:fea-field");
        let error = compose_operation_catalog(&[first, second]).unwrap_err();
        assert!(matches!(
            error,
            OperationCatalogCompositionError::DuplicateOverload { .. }
        ));
    }

    #[test]
    fn analysis_operations_shadow_same_named_standard_builtins() {
        // `envelope` is both a standard signal builtin and an FEA output operation.
        let standard = compose_operation_catalog(&[]).unwrap();
        assert!(standard
            .operations()
            .iter()
            .any(|operation| operation.identity.as_str() == "envelope"));
        let shadowed =
            compose_operation_catalog(&[extension("analysis-output", "envelope", "envelope:fea")])
                .unwrap();
        let envelope = shadowed
            .operations()
            .iter()
            .find(|operation| operation.identity.as_str() == "envelope")
            .unwrap();
        assert_eq!(envelope.overloads.len(), 1);
        assert_eq!(envelope.overloads[0].identity.as_str(), "envelope:fea");
    }

    #[test]
    fn standard_catalog_has_a_stable_read_only_projection() {
        let first = compose_operation_catalog(&[]).unwrap();
        let second = compose_operation_catalog(&[]).unwrap();
        assert!(!first.operations().is_empty());
        assert_eq!(first.fingerprint(), second.fingerprint());
    }
}
