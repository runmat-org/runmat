use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"Command syntax is supported for identifiers, for example `syms x h`; function syntax supports literal text names, for example `syms('x')`.",
"Declared identifiers are assigned scalar symbolic variables of the same name in the active workspace.",
"Symbolic function declarations such as `syms Y(X)` assign `X` as a symbolic variable and `Y` as a symbolic function expression. Calls such as `Y(0)` then produce symbolic applications rather than numeric indexing.",
"Common assumption tokens after the first declaration, such as `real`, `positive`, and `integer`, are accepted and ignored by the current scalar symbolic core.",
"`syms` is a sink builtin: it does not print or return a useful value when called as a statement.",
"All declaration syntax is validated before workspace writes begin. Existing bindings are overwritten. A workspace assignment failure can leave earlier assignments in place; callers must not replay this operation as a pure computation.",
"A writable caller workspace is required. Statements suppress automatic output; requesting one value yields an empty 0-by-0 double placeholder. Zero outputs still declares symbols. More than one output fails before mutation.",
"Array declarations are not implemented. Assumptions are accepted but not enforced. The clear token after a declaration is ignored rather than clearing workspace bindings.",
] },
BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
"This portable host operation writes caller workspace bindings and may throw. It is not reorderable across workspace access. No callbacks, filesystem, randomness, time services, provider hooks, or fusion kernels are used.",
"Interpreter and compiled calls share the default runtime binding. Zero requested outputs evaluates and discards the return value; more than one output is rejected before processing. Distributed input is unsupported. Device kernels and provider benchmarks do not apply to host-only text and symbolic values.",
] },
];
const EXAMPLES: &[BuiltinExample] = &[
BuiltinExample {
 id: "example-1", title: "Declare two symbolic variables", program: "syms x h",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isa(x,'sym')); assert(strcmp(char(x),'x')); assert(strcmp(char(h),'h'));" },
},
BuiltinExample {
 id: "example-2", title: "Use declared variables in a symbolic expression", program: "syms x\nf = sin(x)/x",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isa(f,'sym')); assert(isa(x,'sym')); assert(strcmp(char(x),'x'));" },
},
BuiltinExample {
 id: "example-3", title: "Declare and apply a symbolic function", program: "syms Y(X)\ncond = Y(0) == 0\ndydx = diff(Y, X)",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Y,'sym')); assert(strcmp(char(Y),'Y(X)')); assert(isa(cond,'sym')); assert(isa(dydx,'sym'));" },
},
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog, title: Some("syms"), slug: Some("syms"),
 summary: "Declare scalar symbolic variables in the active workspace.", description: "`syms` supports MATLAB command syntax such as `syms x h`, symbolic function declarations such as `syms Y(X)`, and string function syntax such as `syms('x')` by declaring each named symbol in the caller workspace before later expressions are resolved.",
 keywords: &["syms","symbolic","variable","workspace"], related: &["sym","limit","int"],
 sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
 faqs: &[
 BuiltinDocumentationFaq { question: "Are symbolic assumptions enforced?", answer: "Assumption tokens are parsed for MATLAB command compatibility, but the current scalar symbolic evaluator treats expressions as real-valued and does not maintain an assumption database." },
 ],
 links: &[BuiltinDocumentationLink { label: "sym", target: BuiltinDocumentationLinkTarget::Builtin("sym") },
BuiltinDocumentationLink { label: "limit", target: BuiltinDocumentationLinkTarget::Builtin("limit") },
BuiltinDocumentationLink { label: "int", target: BuiltinDocumentationLinkTarget::Builtin("int") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/syms.rs") }], media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/syms.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "unit", location: "builtins::math::symbolic::syms::tests" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "integration", location: "runmat_vm::bytecode::compile::tests::compile_interprets_symbolic_limit_workflow" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "symbolic_function", location: "runmat_vm::bytecode::compile::tests::compile_interprets_symbolic_function_declaration_workflow" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Identity inference tests", location: "crates/runmat-builtins/src/catalog/entries/math/symbolic/syms/inference.rs" }],
 notes: &["Legacy behaviors, syntax, examples, FAQs and links are reconciled here. Typed placement replaces GPU/fusion metadata. Example assertions verify semantic values rather than presentation. Integrated native, browser and documentation execution remains required; catalog metadata alone is not runtime evidence."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
