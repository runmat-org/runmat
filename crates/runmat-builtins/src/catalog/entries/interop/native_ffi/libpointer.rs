use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::documentation_support::{IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST};
use super::inference_support::{arity, infer_with, owned_pointer, text_at};
use super::{LIBPOINTER_DESCRIPTOR, NATIVE_FFI_INTEGER_AUDIT};
use crate::*;
use runmat_types::*;

pub const LIBPOINTER_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "libpointer" },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &LIBPOINTER_DESCRIPTOR,
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
    arity(request, 0, Some(2), &mut diagnostics);
    text_at(request, 0, &mut diagnostics);
    infer_with(request, entry, vec![owned_pointer()], diagnostics)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("libpointer"), slug: Some("libpointer"),
    summary: "Create a typed, session-owned native pointer value.",
    description: "`libpointer()` creates an opaque null pointer. `libpointer(type)` and `libpointer(type, value)` allocate storage for a normalized C pointer type and optionally initialize it from the value. The native adapter owns the allocation and tracks the pointer's origin-thread, session lifetime, and type metadata.",
    keywords: &["libpointer", "native", "pointer", "C", "ownership"],
    related: &["calllib", "libstruct", "setdatatype", "loadlibrary"],
    sections: &[BuiltinDocumentationSection { heading: "Type and storage", paragraphs: &["Accepted scalar pointer spellings include signed and unsigned fixed-width integers, logical, single, and double types. The declared C ABI type controls range checks; integer payloads are not silently converted through double storage. A returned borrowed pointer from `calllib` remains borrowed rather than becoming caller-owned through `libpointer`."] }],
    examples: &[BuiltinExample {
        id: "int32-pointer", title: "Create exact integer pointer storage",
        program: "p = libpointer('int32Ptr', int32(7));\nassert(p.Value == int32(7));",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(p.Value == int32(7));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does libpointer copy a C-returned pointer?", answer: "No. Pointers returned from C keep their borrowed ownership and must be interpreted under the native adapter's handle contract." }],
    links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Caller-owned pointer storage", location: "tests::native_ffi_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
