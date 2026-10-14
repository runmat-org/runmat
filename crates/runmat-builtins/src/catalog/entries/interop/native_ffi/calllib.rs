use super::contract::{FOREIGN_LINK, HOST_PLACEMENT};
use super::descriptor::{scalar_text, ERRORS};
use super::documentation_support::{
    CALL_IMPLEMENTATION as IMPLEMENTATION, NATIVE_FIXTURE, NATIVE_HOST,
};
use super::inference_support::{arity, infer_dynamic, text_at};
use crate::*;
use runmat_types::*;

const ARGUMENTS: BuiltinParamDescriptor = BuiltinParamDescriptor { name: "arguments", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic, default: None, description: "Native ABI values interpreted by the prepared function signature, including exact integer arrays and foreign handles." };
const OUTPUTS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "outputs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Return value and output parameters determined by the prepared function signature.",
};
pub const CALLLIB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "[outputs] = calllib(alias, function, arguments...)",
        inputs: &[
            scalar_text(
                "alias",
                "Scalar text, number, or logical loaded library alias.",
            ),
            scalar_text(
                "function",
                "Scalar text, number, or logical declared C symbol name.",
            ),
            ARGUMENTS,
        ],
        outputs: &[OUTPUTS],
    }],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability { name: "arguments", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Exact integer scalar and tensor storage is marshalled according to the prepared C parameter type; scalar double conversion is checked at the ABI boundary." }];
const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "[outputs] = calllib(alias, function, integer_arguments...)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "The prepared function signature determines admissible integer classes, widths, ranges, return storage, and output parameters. A generic calllib invocation cannot promise a particular result class." }];

pub const CALLLIB_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "calllib" },
    category: "interop/native",
    documentation: DOCUMENTATION,
    descriptor: &CALLLIB_DESCRIPTOR,
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
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
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
    arity(request, 2, None, &mut diagnostics);
    text_at(request, 0, &mut diagnostics);
    text_at(request, 1, &mut diagnostics);
    // The prepared manifest fixes ABI outputs at runtime. No public C symbol
    // name alone proves its return class or output-parameter count here.
    infer_dynamic(request, entry, diagnostics)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("calllib"), slug: Some("calllib"),
    summary: "Call a declared function in a loaded native C library.",
    description: "`calllib(alias, function, arguments...)` resolves the public C function name through the exact prepared interface manifest, converts arguments according to its declared ABI types, and returns the native result followed by declared output parameters. Requested output count is checked against the declaration.",
    keywords: &["calllib", "native", "C", "shared library", "ABI", "pointer"],
    related: &["loadlibrary", "libfunctions", "libpointer", "libstruct", "setdatatype"],
    sections: &[
        BuiltinDocumentationSection { heading: "Values and ownership", paragraphs: &["Exact fixed-width integer storage is preserved at the ABI boundary. Pointer and structure values retain adapter-owned handle metadata; a borrowed pointer returned by C is not copied into caller-owned memory. Native callbacks and output parameters follow the prepared function signature."] },
        BuiltinDocumentationSection { heading: "Isolation", paragraphs: &["In-process calls execute on the native session lane; an explicitly isolated adapter runs the same manifest contract in its process. A browser does not execute this native call and reports an unsupported-host error."] },
    ],
    examples: &[BuiltinExample {
        id: "add-int32", title: "Call a declared integer function",
        program: "loadlibrary('fixture', 'fixture.h', 'Alias', 'fixture');\nanswer = calllib('fixture', 'fixture_add', int32(2), int32(3));\nassert(answer == int32(5));\nunloadlibrary('fixture');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeForeignRuntime,
        fixture: NATIVE_FIXTURE, requirements: NATIVE_HOST,
        verification: BuiltinExampleVerification::Assertions { source: "assert(answer == int32(5));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can I call an exported symbol absent from the header?", answer: "No. The function must be declared in the prepared interface manifest, even if the library exports its name." }],
    links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Native ABI calls and callbacks", location: "native_ffi_session::library_and_pointer_state_are_owned_by_one_runtime_session" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
