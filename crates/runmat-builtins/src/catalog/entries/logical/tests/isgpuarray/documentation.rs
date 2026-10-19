use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};
const SECTIONS: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection { heading: "Explicit device values", paragraphs: &["`isgpuarray(A)` returns true for an explicit gpuArray value and false for host values. The query reads handle provenance only and never downloads device data.", "RunMat may place intermediate values on an accelerator automatically. Those internal placement decisions do not change the source-level class into `gpuArray`; `isgpuarray` is true only when the program explicitly establishes gpuArray identity."] }];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "gpuarray", title: "Check an explicit gpuArray", program: "G = gpuArray(reshape(1:12, [3 4]));\ntf = isgpuarray(G)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "host", title: "Check a host array", program: "A = [1 2 3];\ntf = isgpuarray(A)", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "gather", title: "Observe an explicit gather boundary", program: "G = gpuArray(single([1 2 3]));\nA = gather(G);\ntf_gpu = isgpuarray(G);\ntf_host = isgpuarray(A)", display_output: Some("tf_gpu = true and tf_host = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(tf_gpu);\nassert(~tf_host);" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `isgpuarray` gather data?", answer: "No. It reads explicit handle identity without transferring payload data." },
    BuiltinDocumentationFaq { question: "What happens without an acceleration provider?", answer: "Host values return false. Existing explicit handles retain their identity, while constructing a new gpuArray requires a provider." },
    BuiltinDocumentationFaq { question: "Do automatically accelerated values return true?", answer: "No. Automatic placement is an execution detail; only explicit gpuArray identity returns true." },
];
const RELATED: &[&str] = &[
    "gpuArray",
    "gather",
    "islogical",
    "isnumeric",
    "isreal",
    "issparse",
    "isa",
];
const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") }, BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") }, BuiltinDocumentationLink { label: "isa", target: BuiltinDocumentationLinkTarget::Builtin("isa") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isgpuarray.rs") }];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "gpuArray identity predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isgpuarray.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host, explicit, and automatic-residency behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/isgpuarray/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU explicit identity and automatic-residency distinction", location: "crates/runmat-runtime/src/builtins/logical/tests/isgpuarray/tests.rs::wgpu_explicit_identity_is_distinct_from_automatic_residency" }], notes: &[] };
pub(super) const ISGPUARRAY_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("isgpuarray"), slug: Some("isgpuarray"), summary: "Determine whether a value has explicit gpuArray identity.", description: "`isgpuarray` returns one logical scalar without reading or transferring device payload data.", keywords: &["isgpuarray", "gpuArray", "accelerator", "residency", "gather"], related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable) };
