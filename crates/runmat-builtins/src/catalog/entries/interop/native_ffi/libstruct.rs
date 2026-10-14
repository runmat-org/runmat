use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::descriptor::{scalar_text, ERRORS};
use super::documentation_support::{
    NATIVE_FIXTURE, NATIVE_HOST, POINTER_IMPLEMENTATION as IMPLEMENTATION,
};
use super::inference_support::{arity, infer_with, owned_pointer, text_at};
use crate::*;
use runmat_types::*;

const HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Session-owned native structure handle.",
};
const TYPE: BuiltinParamDescriptor = scalar_text(
    "type",
    "Scalar text, number, or logical qualified C structure type.",
);
const INITIALIZER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "structure",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar structure initializer checked against the prepared layout.",
};
pub const LIBSTRUCT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "value = libstruct(type)",
            inputs: &[TYPE],
            outputs: &[HANDLE],
        },
        BuiltinSignatureDescriptor {
            label: "value = libstruct(type, structure)",
            inputs: &[TYPE, INITIALIZER],
            outputs: &[HANDLE],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "value = libstruct(type, structure_with_integer_fields)", inputs: &[], computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Only integer fields declared by the prepared structure layout are admitted. Field storage and checked conversion follow their ABI widths; the public result is a foreign handle. No generic integer field class follows from the type text alone." }];

pub const LIBSTRUCT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "libstruct" },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &LIBSTRUCT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::EnvironmentWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[
            CapabilityRequirement::NativeCode,
            CapabilityRequirement::ForeignRuntime,
        ],
    },
    placement: HOST_PLACEMENT,
    link: FOREIGN_LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, Some(2), &mut diagnostics);
    text_at(request, 0, &mut diagnostics);
    if request
        .arguments
        .get(1)
        .is_some_and(|fact| !matches!(fact.kind, ValueKindFact::Unknown | ValueKindFact::Struct(_)))
    {
        diagnostics.push(catalog::inference::argument_error(
            "RM-NATIVE-FFI-STRUCT",
            "libstruct initializer must be a scalar structure",
            1,
        ));
    }
    infer_with(request, entry, vec![owned_pointer()], diagnostics)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("libstruct"), slug: Some("libstruct"),
    summary: "Create owned native storage for a C structure declared by a loaded interface.",
    description: "`libstruct(type)` allocates storage using the structure layout from a loaded prepared header. `libstruct(type, structure)` initializes that storage from a scalar structure value. Qualify the type with its library alias when more than one loaded interface declares the same structure name.",
    keywords: &["libstruct", "native", "C", "structure", "layout"],
    related: &["loadlibrary", "libpointer", "calllib", "setdatatype"],
    sections: &[BuiltinDocumentationSection { heading: "Layout and ownership", paragraphs: &["The prepared ABI manifest, not field-name guessing, determines native layout. The resulting foreign handle owns its storage and remains tied to the originating native session/thread. A mismatched initializer raises a native call error."] }],
    examples: &[BuiltinExample {
        id: "declared-structure", title: "Allocate a declared C structure",
        program: "loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\nrecord = libstruct('fixture.fixture_record');\nassert(record.left == 0 && record.right == 0);\nunloadlibrary('fixture');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(record.left == 0 && record.right == 0);" },
    }],
    example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Manifest-backed structure allocation", location: "native_ffi_session::library_and_pointer_state_are_owned_by_one_runtime_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
