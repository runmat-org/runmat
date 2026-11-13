use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::descriptor::{scalar_text, ERRORS};
use super::documentation_support::{
    NATIVE_FIXTURE, NATIVE_HOST, POINTER_IMPLEMENTATION as IMPLEMENTATION,
};
use super::inference_support::{arity, infer_with, owned_pointer, text_at};
use crate::*;
use runmat_types::*;

const POINTER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pointer",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Session-owned native pointer handle.",
};
const TYPE: BuiltinParamDescriptor = scalar_text(
    "type",
    "Scalar text, number, or logical normalized C pointer type.",
);
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Initializer validated against the declared C ABI element type and storage extent.",
};
pub const LIBPOINTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "pointer = libpointer()",
            inputs: &[],
            outputs: &[POINTER],
        },
        BuiltinSignatureDescriptor {
            label: "pointer = libpointer(type)",
            inputs: &[TYPE],
            outputs: &[POINTER],
        },
        BuiltinSignatureDescriptor {
            label: "pointer = libpointer(type, value)",
            inputs: &[TYPE, VALUE],
            outputs: &[POINTER],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability { name: "value", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "An integer initializer is accepted only when the declared pointer pointee type and checked conversion admit it." }];
const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "pointer = libpointer(integer_type, integer_value)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "The normalized pointer type selects exact native storage. Initializer conversion checks the declared width and range; the public result is a foreign handle, not an integer array." }];

pub const LIBPOINTER_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
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
    integer_capabilities: INTEGER_CAPABILITIES,
    integer_audit: None,
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
    description: "`libpointer()` creates a null `voidPtr` handle. `libpointer(type)` creates a null handle of the normalized C pointer type without allocating pointee storage. `libpointer(type, value)` allocates adapter-owned storage initialized from the value. The native adapter tracks origin-thread, session lifetime, and type metadata in every form.",
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
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Caller-owned pointer storage", location: "native_ffi_session::library_and_pointer_state_are_owned_by_one_runtime_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
