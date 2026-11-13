use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"Text that is a valid MATLAB identifier creates a symbolic variable with that name.",
"Numeric, integer, logical, and scalar numeric tensor inputs create scalar symbolic constants with class `sym`. Integer scalars are represented exactly, including `int64` and `uint64` values outside the exactly representable range of double precision.",
"Text containing a finite numeric scalar creates that numeric symbolic constant.",
"Text containing a simple integer rational such as `1/3` creates an exact scalar symbolic rational.",
"Symbolic values are host-only and do not participate in GPU fusion or residency.",
"Bare integer text is parsed directly into arbitrary-precision integer storage, including values above 2^53. Integer numerators and denominators are exact; decimal and exponent text use finite binary floating-point constants rather than inferred integer intent.",
"Numeric scalar nonfinite constants retain existing symbolic numeric behavior. Numeric arrays with more than one element, complex inputs, logical-array containers, GPU values, and options are outside this scalar constructor. Text must be a string scalar or a character row with no non-singleton higher dimensions.",
] },
BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
"This portable host operation is pure apart from errors and never suspends. It performs no callbacks, workspace changes, filesystem, randomness, time services, provider hooks, or GPU fusion.",
"Interpreter and compiled calls share the default runtime binding. Zero requested outputs evaluates and discards the return value; more than one output is rejected before processing. Distributed input is unsupported. Device kernels and provider benchmarks do not apply to host-only text and symbolic values.",
] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Create a symbolic variable",
        program: "x = sym('x')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(x,'sym')); assert(strcmp(char(x),'x'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Create a symbolic numeric constant",
        program: "a = sym(2)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(a,'sym')); assert(strcmp(char(a),'2'));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Create an exact symbolic rational",
        program: "r = sym('1/3')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(r,'sym')); assert(strcmp(char(r),'1/3'));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Preserve a wide authored integer literal",
        program: "wide = sym('9007199254740993')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(char(wide), '9007199254740993'));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog, title: Some("sym"), slug: Some("sym"),
 summary: "Create a scalar symbolic variable or constant.", description: "`sym` creates scalar symbolic variables from identifier text and scalar symbolic constants from numeric input. Nonconstant symbolic values remain symbolic through supported arithmetic, trigonometric, and limit operations.",
 keywords: &["sym","symbolic","variable","algebra"], related: &["syms","limit","int","vpa","digits"],
 sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
 faqs: &[
 BuiltinDocumentationFaq { question: "Does `sym` create symbolic arrays?", answer: "This constructor accepts scalar symbolic values and scalar numeric or text inputs only. Symbolic array containers used by other symbolic operations are not constructed by sym." },
 ],
 links: &[BuiltinDocumentationLink { label: "syms", target: BuiltinDocumentationLinkTarget::Builtin("syms") },
BuiltinDocumentationLink { label: "limit", target: BuiltinDocumentationLinkTarget::Builtin("limit") },
BuiltinDocumentationLink { label: "int", target: BuiltinDocumentationLinkTarget::Builtin("int") },
BuiltinDocumentationLink { label: "vpa", target: BuiltinDocumentationLinkTarget::Builtin("vpa") },
BuiltinDocumentationLink { label: "digits", target: BuiltinDocumentationLinkTarget::Builtin("digits") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/sym.rs") }], media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/sym.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "unit", location: "builtins::math::symbolic::sym::tests" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "integration", location: "runmat_vm::bytecode::compile::tests::compile_interprets_symbolic_limit_workflow" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Identity inference tests", location: "crates/runmat-builtins/src/catalog/entries/math/symbolic/sym/inference.rs" }],
 notes: &["Legacy behaviors, syntax, examples, FAQs and links are reconciled here. Typed placement replaces GPU/fusion metadata. Example assertions verify semantic values rather than presentation. Integrated native, browser and documentation execution remains required; catalog metadata alone is not runtime evidence."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
