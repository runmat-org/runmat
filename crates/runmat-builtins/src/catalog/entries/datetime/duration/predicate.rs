use super::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use runmat_types::{CallInference, CallRequest, ShapeFact};

const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value whose exact runtime class is inspected.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical scalar indicating a duration value.",
}];
const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "tf = isduration(X)",
        inputs: &INPUT,
        outputs: &OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Duration predicate", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/duration/builtins.rs") };
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("isduration"), slug: Some("isduration"), summary: "Test whether a value has the duration class.",
    description: "`isduration(X)` returns a host logical scalar. It checks exact runtime class identity without reading a numeric payload, so numeric and unrelated object values return false.",
    keywords: &["isduration", "duration", "predicate"], related: &["duration", "days"],
    sections: &[BuiltinDocumentationSection { heading: "Execution", paragraphs: &["The predicate does not download resident numeric data and is available in native and WebAssembly runtimes. It has no workspace, file, network, or provider effect."] }],
    examples: &[BuiltinExample { id: "class-check", title: "Distinguish duration from numeric values", program: "d = seconds(2); a = isduration(d); b = isduration(2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(a); assert(~b);" } }],
    example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "MATLAB isduration reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/isduration.html") }, SOURCE], media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Duration class predicate", location: "crates/runmat-runtime/src/builtins/duration/tests.rs" }], notes: &[] }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.len() == 1 {
        vec![]
    } else {
        vec![argument_error(
            "RM-ISDURATION-ARITY",
            "isduration expects one input",
            0,
        )]
    };
    finish_fixed(
        entry,
        request,
        facts::logical(ShapeFact::Scalar),
        diagnostics,
    )
}
pub const ISDURATION_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "isduration",
    &DESCRIPTOR,
    DOCUMENTATION,
    infer,
    false,
    &[],
    &[],
);
