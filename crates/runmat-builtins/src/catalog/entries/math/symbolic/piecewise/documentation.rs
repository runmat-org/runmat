use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("piecewise"),
    slug: Some("piecewise"),
    summary: "Construct a scalar symbolic piecewise expression.",
    description: "`piecewise` currently materializes scalar symbolic condition/value arguments as a symbolic function expression. It is intended for workflows that start with `syms` and need to preserve symbolic expressions for later symbolic operations.",
    keywords: &["piecewise", "symbolic", "condition", "syms"],
    related: &["syms", "sym"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["Accepts symbolic arguments and scalar numeric, integer, or logical constants.", "Returns a scalar symbolic expression such as `piecewise(lt(abs(t), 2), 1, gt(abs(t), 2), 0)`.", "Does not yet evaluate symbolic conditions or simplify branches.", "Rejects non-scalar arrays and unsupported argument types with a `RunMat:piecewise:InvalidInput` diagnostic."],
        },
        BuiltinDocumentationSection {
            heading: "Supported scalar forms and limits",
            paragraphs: &["Requires at least two arguments. Further condition/value arguments and an optional final default are stored in order. The constructor does not test whether condition expressions are boolean and does not select, evaluate or simplify a branch.", "Real numeric, scalar logical and symbolic scalar carriers are accepted, including one-element N-D numeric tensors. Native integers from every class become exact BigInt-backed symbolic rationals without binary64 narrowing, including full int64/uint64 extrema.", "Logical-array and symbolic-array containers (even one-element), complex values, sparse, empty/nonscalar, text and resident/distributed arguments reject. Use sym to construct a symbolic value from text before calling piecewise.", "The result is always one host symbolic scalar. Zero requested outputs constructs and discards it; requesting more than one output fails before construction."],
        },
        BuiltinDocumentationSection {
            heading: "Placement and effects",
            paragraphs: &["`piecewise` constructs host-only symbolic expressions and does not participate in GPU fusion.", "GPU and distributed operands, including secondary controls, are not supported. These functions do not upload or gather values and have no device kernel or GPU fusion recipe.", "Constructing and transforming symbolic expression trees currently requires unrestricted host allocation. Finite Managed calls reject before unowned symbolic allocation; complete finite expression ownership is not yet implemented.", "This portable host operation is pure apart from errors. It does not use callbacks, workspace mutation, filesystem, clock or random services."],
        },
    ],
    examples: &[BuiltinExample {
        id: "example-1",
        title: "Construct a symbolic piecewise expression",
        program: "syms t\nf = piecewise(abs(t)<2, 1, abs(t)>2, 0)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(f,'sym')); assert(strcmp(char(f),'piecewise(lt(abs(t), 2), 1, gt(abs(t), 2), 0)'));",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does `piecewise` implement full Symbolic Math Toolbox simplification?",
        answer: "No. It preserves scalar symbolic piecewise expressions so scripts can continue past declaration and expression-construction steps.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "syms",
            target: BuiltinDocumentationLinkTarget::Builtin("syms"),
        },
        BuiltinDocumentationLink {
            label: "sym",
            target: BuiltinDocumentationLinkTarget::Builtin("sym"),
        },
        BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/math/symbolic/piecewise.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/piecewise.rs"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/symbolic/sym.piecewise.html"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/piecewise.rs"),
            },
            BuiltinDocumentationLink {
                label: "Scalar text and numeric conversion",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/support.rs"),
            },
            BuiltinDocumentationLink {
                label: "Invocation allocation boundary",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/invocation.rs"),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::symbolic::piecewise::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "runmat_vm::tests::symbolic::symbolic_syms_piecewise_repro_binds_workspace_variables",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Scalar form inference",
                location: "crates/runmat-builtins/src/catalog/entries/math/symbolic/inference/tests.rs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
