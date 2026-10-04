use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::documentation_support::{IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST};
use super::inference_support::{arity, infer_with, owned_pointer, text_at};
use super::{LIBSTRUCT_DESCRIPTOR, NATIVE_FFI_INTEGER_AUDIT};
use crate::*;
use runmat_types::*;

pub const LIBSTRUCT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
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
    integer_capabilities: &[],
    integer_audit: Some(&NATIVE_FFI_INTEGER_AUDIT),
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
        program: "loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\nrecord = libstruct('fixture.fixture_record');\nassert(~isempty(record));\nunloadlibrary('fixture');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(record));" },
    }],
    example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Manifest-backed structure allocation", location: "tests::native_ffi_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
