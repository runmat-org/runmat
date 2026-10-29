//! Shared contract facts for typed FEA field selectors
//! (`fea.StructuralField.*`, `fea.ResponseSpectrumField.*`,
//! `fea.RandomResponseField.*`, and `fea.OutputField.named`). Each selector is
//! a pure host constructor of an immutable value object.

use super::constructor_inference::{host_text, object_fact, possible_object};
use super::descriptors::ERRORS;
use super::identities::{
    FEA_OUTPUT_FIELD_CLASS, FEA_RANDOM_RESPONSE_FIELD_CLASS, FEA_RESPONSE_SPECTRUM_FIELD_CLASS,
    FEA_STRUCTURAL_FIELD_CLASS,
};
use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerAuditDescriptor,
    BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
pub const SELECTOR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "field",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Immutable typed field selector for fea.field.",
}];
pub const LINEAR_QUANTITY_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "quantity",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A linear fea.StructuralField quantity.",
}];
pub const OUTPUT_NAME_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Authored output identifier declared by the FEA study.",
}];

pub const fn selector_descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    }
}

pub const STRUCTURAL_SELECTOR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Structural field selectors have no integer inputs.",
    };
pub const STATISTICAL_SELECTOR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Statistical field selectors have no integer inputs.",
    };

fn infer_nullary(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: StaticClassIdentity,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            "field selector constructors take no arguments",
            request.arguments.len(),
        ));
    }
    finish_fixed(entry, request, object_fact(class), diagnostics)
}

fn infer_statistical(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: StaticClassIdentity,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            "statistical field selectors take one fea.StructuralField quantity",
            request.arguments.len(),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|quantity| !possible_object(quantity, &[FEA_STRUCTURAL_FIELD_CLASS]))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-FIELD-SELECTOR",
            "statistical field selectors require a fea.StructuralField quantity such as fea.StructuralField.stress()",
            0,
        ));
    }
    finish_fixed(entry, request, object_fact(class), diagnostics)
}

/// `fea.StructuralField.*()`
pub fn infer_structural(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_nullary(request, entry, FEA_STRUCTURAL_FIELD_CLASS)
}

/// `fea.ResponseSpectrumField.<nodal quantity>()`
pub fn infer_response_spectrum_nodal(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_nullary(request, entry, FEA_RESPONSE_SPECTRUM_FIELD_CLASS)
}

/// `fea.ResponseSpectrumField.envelope(quantity)`
pub fn infer_response_spectrum_envelope(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_statistical(request, entry, FEA_RESPONSE_SPECTRUM_FIELD_CLASS)
}

/// `fea.RandomResponseField.rms(quantity)` and `.variance(quantity)`
pub fn infer_random_response(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_statistical(request, entry, FEA_RANDOM_RESPONSE_FIELD_CLASS)
}

/// The runtime's authored-output identifier rule.
pub fn valid_output_name(name: &str) -> bool {
    let mut characters = name.chars();
    matches!(characters.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
        && name.len() <= 512
}

/// `fea.OutputField.named(name)`
pub fn infer_output_named(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            "fea.OutputField.named requires one output name",
            request.arguments.len(),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|name| !host_text(name))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-TEXT",
            "fea.OutputField.named requires a text output name",
            0,
        ));
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|name| !valid_output_name(&name))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-OUTPUT-NAME",
            "authored output name must be a valid FEA identifier",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        object_fact(FEA_OUTPUT_FIELD_CLASS),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
    use crate::fea_contract::constructor_inference::object_fact;
    use crate::{BuiltinCatalogProvenance, BuiltinDistributedPolicy, BuiltinDocumentation};
    use runmat_types::{
        DynamicReason, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
        ValueFact, ValueKindFact,
    };

    const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.stress()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
    const DESCRIPTOR: BuiltinDescriptor = selector_descriptor(&SIGNATURES);

    fn entry() -> BuiltinCatalogEntry {
        catalog_entry::entry(ConstructorEntryDefinition {
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            name: "fea.StructuralField.stress",
            documentation: BuiltinDocumentation::EMPTY,
            descriptor: &DESCRIPTOR,
            inference: infer_structural,
            integer_capabilities: &[],
            integer_audit: Some(&STRUCTURAL_SELECTOR_INTEGER_AUDIT),
            filesystem_read: false,
            distributed: BuiltinDistributedPolicy::Unsupported,
        })
    }

    fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    fn codes(inference: &CallInference) -> Vec<&str> {
        inference
            .diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect()
    }

    fn class(inference: &CallInference) -> Option<String> {
        match &inference.outputs[0].kind {
            ValueKindFact::Object(object) => object.runtime_class.as_ref().map(ToString::to_string),
            _ => None,
        }
    }

    #[test]
    fn selectors_are_pure_and_return_their_exact_class() {
        let entry = entry();
        assert_eq!(
            entry.contract.effects,
            &[runmat_types::EffectKind::MayThrow]
        );
        let structural = infer_structural(&request(vec![], vec![]), &entry);
        assert!(structural.diagnostics.is_empty());
        assert_eq!(class(&structural).as_deref(), Some("fea.StructuralField"));
        let nodal = infer_response_spectrum_nodal(&request(vec![], vec![]), &entry);
        assert_eq!(class(&nodal).as_deref(), Some("fea.ResponseSpectrumField"));
        let extra = infer_structural(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                vec![LiteralValue::Unknown],
            ),
            &entry,
        );
        assert!(codes(&extra).contains(&"RM-CATALOG-FEA-ARITY"));
    }

    #[test]
    fn statistical_selectors_require_a_structural_quantity() {
        let entry = entry();
        let valid = infer_random_response(
            &request(
                vec![object_fact(FEA_STRUCTURAL_FIELD_CLASS)],
                vec![LiteralValue::Unknown],
            ),
            &entry,
        );
        assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
        assert_eq!(class(&valid).as_deref(), Some("fea.RandomResponseField"));
        let text = infer_response_spectrum_envelope(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                vec![LiteralValue::String("stress".into())],
            ),
            &entry,
        );
        assert!(codes(&text).contains(&"RM-CATALOG-FEA-FIELD-SELECTOR"));
        assert_eq!(class(&text).as_deref(), Some("fea.ResponseSpectrumField"));
        let unknown = infer_random_response(
            &request(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                vec![LiteralValue::Unknown],
            ),
            &entry,
        );
        assert!(unknown.diagnostics.is_empty());
        let missing = infer_random_response(&request(vec![], vec![]), &entry);
        assert!(codes(&missing).contains(&"RM-CATALOG-FEA-ARITY"));
    }

    #[test]
    fn output_names_follow_the_runtime_identifier_rule() {
        let entry = entry();
        assert!(valid_output_name("peak_stress"));
        assert!(valid_output_name("_x1"));
        assert!(!valid_output_name("1x"));
        assert!(!valid_output_name("not valid"));
        assert!(!valid_output_name(&"a".repeat(513)));
        let invalid = infer_output_named(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                vec![LiteralValue::String("not valid".into())],
            ),
            &entry,
        );
        assert!(codes(&invalid).contains(&"RM-CATALOG-FEA-OUTPUT-NAME"));
        assert_eq!(class(&invalid).as_deref(), Some("fea.OutputField"));
        let numeric = infer_output_named(
            &request(
                vec![ValueFact::scalar(ValueKindFact::Logical)],
                vec![LiteralValue::Unknown],
            ),
            &entry,
        );
        assert!(codes(&numeric).contains(&"RM-CATALOG-FEA-TEXT"));
    }
}
