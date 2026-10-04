use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::documentation_support::{IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST};
use super::inference_support::{arity, infer_with, text_at};
use super::{NATIVE_FFI_INTEGER_AUDIT, SETDATATYPE_DESCRIPTOR};
use crate::*;
use runmat_types::*;

pub const SETDATATYPE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "setdatatype",
    },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &SETDATATYPE_DESCRIPTOR,
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
    suppress_auto_output: true,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 3, None, &mut diagnostics);
    if request.arguments.first().is_some_and(|fact| {
        !matches!(
            fact.kind,
            ValueKindFact::Unknown | ValueKindFact::Foreign(_)
        )
    }) {
        diagnostics.push(catalog::inference::argument_error(
            "RM-NATIVE-FFI-POINTER",
            "setdatatype requires a native pointer handle",
            0,
        ));
    }
    text_at(request, 1, &mut diagnostics);
    for index in 2..request.arguments.len() {
        if !matches!(
            request.arguments[index].kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) {
            diagnostics.push(catalog::inference::argument_error(
                "RM-NATIVE-FFI-DIMENSION",
                "pointer dimensions must be positive integer scalars",
                index,
            ));
        }
    }
    infer_with(request, entry, vec![], diagnostics)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("setdatatype"), slug: Some("setdatatype"),
    summary: "Declare the pointee type and extent of an opaque native pointer.",
    description: "`setdatatype(pointer, type, dimensions...)` records a typed view of an opaque pointer returned by a native library. It does not allocate or copy the pointee. It applies only to an opaque borrowed pointer, not caller-owned `libpointer` storage, and requires at least one checked dimension.",
    keywords: &["setdatatype", "native", "pointer", "C", "extent"],
    related: &["libpointer", "calllib", "loadlibrary"],
    sections: &[BuiltinDocumentationSection { heading: "Borrowed memory", paragraphs: &["The foreign handle retains its original borrowed ownership, origin affinity, and session lifetime. Dimensions are validated with checked multiplication before the view is published; assigning a type does not authorize an unsafe copy from an unknown address."] }],
    examples: &[BuiltinExample {
        id: "borrowed-int32", title: "Type an opaque pointer returned from C",
        program: "loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\np = calllib('fixture', 'fixture_borrowed_value', int32(1));\nsetdatatype(p, 'int32Ptr', 1);\nassert(~isempty(p));\nunloadlibrary('fixture');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(p));" },
    }],
    example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Borrowed pointer type and extent", location: "tests::native_ffi_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
