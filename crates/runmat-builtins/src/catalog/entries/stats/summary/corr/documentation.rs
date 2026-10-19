use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
authority: BuiltinDocumentationAuthority::Catalog, title: Some("corr"), slug: Some("corr"),
summary:"Compute pairwise correlation coefficients and p-values.", description:"`corr(X)` returns pairwise correlations between columns of `X`. `corr(X,Y)` returns correlations between columns of `X` and columns of `Y`. `[R,P] = corr(...)` also returns approximate p-values for testing no correlation.",
keywords:&["corr","correlation","pearson","spearman","kendall","statistics","tail"], related:&["corrcoef","cov"],
sections:&[
BuiltinDocumentationSection {heading:"Behavior", paragraphs:&["Rows are observations and columns are variables.",
"`Type` supports `\"Pearson\"`, `\"Spearman\"`, and `\"Kendall\"`.",
"`Rows` supports `\"all\"`, `\"complete\"`, and `\"pairwise\"`.",
"`Tail` supports `\"both\"`, `\"right\"`, and `\"left\"` for p-value direction.",
"Constant or insufficient data pairs produce `NaN`."]},
BuiltinDocumentationSection {heading:"Classes and execution", paragraphs:&[
"Inputs must be real observation matrices. Non-singleton dimensions beyond the second are rejected.",
"Outputs are host double values, including when observations use single precision. Resident observations are gathered through their owning provider; correlation is a materializing fusion boundary, not a direct GPU kernel.",
"Integer observations and integer weights are separately gated RunMat extensions. Integer observations retain exact ordering and centered integer differences before the floating statistical calculation.",
"Weights must be finite, nonnegative, and match the observation count. Weighted p-values are NaN.",
"Requests beyond the coefficient and p-value outputs currently receive scalar zero padding."
]},
],
examples:&[BuiltinExample {
id: "example-1", title: "Column correlations",
program: "R = corr([1 3; 2 2; 3 1])", display_output: None,
compatibility: BuiltinExampleCompatibility::Matlab,
harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
requirements: BuiltinExampleRequirements::NONE,
verification: BuiltinExampleVerification::Assertions {source:"assert(max(abs(R(:)-[1;-1;-1;1]))<1e-12);"},
},
BuiltinExample {
id: "example-2", title: "Pairwise NaN omission",
program: "R = corr([1; 2; NaN; 4], [1; 2; 3; 4], \"Rows\", \"pairwise\")", display_output: None,
compatibility: BuiltinExampleCompatibility::Matlab,
harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
requirements: BuiltinExampleRequirements::NONE,
verification: BuiltinExampleVerification::Assertions {source:"assert(abs(R-1)<1e-12);"},
},
BuiltinExample {
id: "example-3", title: "Rank correlation and p-values",
program: "x=[1;2;3;4]; y=[1;2;3;4];\n[R,P] = corr(x, y, \"Type\", \"Kendall\", \"Tail\", \"right\")", display_output: None,
compatibility: BuiltinExampleCompatibility::Matlab,
harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
requirements: BuiltinExampleRequirements::NONE,
verification: BuiltinExampleVerification::Assertions {source:"assert(abs(R-1)<1e-12); assert(P>=0 && P<=1);"},
}], example_exemption:None,
faqs:&[BuiltinDocumentationFaq {question:"Does corr support rank correlations?",answer:"Spearman and Kendall rank correlations are supported through the `Type` option. P-values use standard large-sample approximations."}],
links:&[
BuiltinDocumentationLink {label:"corrcoef",target:BuiltinDocumentationLinkTarget::Builtin("corrcoef")},
BuiltinDocumentationLink {label:"cov",target:BuiltinDocumentationLinkTarget::Builtin("cov")},
BuiltinDocumentationLink {label:"Source",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/corr/mod.rs")},
], media:&[],
evidence:BuiltinDocumentationEvidence {implementation:&[],verification:&[BuiltinEvidenceReference {
kind:BuiltinEvidenceKind::UnitTest,label:"Correlation semantics and integer boundaries",location:"builtins::stats::summary::corr::tests"
}], notes:&["Legacy descriptions, behaviors, FAQ, links and three examples retained. The rank example now defines its previously missing x/y fixtures. Null feature/integration fields omitted. Host gather and non-fusion behavior preserved; source follows modular implementation."]},
introduced:None,status:Some(BuiltinDocumentationStatus::Stable)
};
