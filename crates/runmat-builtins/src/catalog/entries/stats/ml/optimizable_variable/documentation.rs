use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("optimizableVariable"),
    slug: Some("optimizableVariable"),
    summary: "Create Bayesian optimization variable metadata.",
    description: "`optimizableVariable(Name,Range)` constructs a host metadata object for a variable used by `bayesopt`.",
    keywords: &["optimizableVariable", "bayesopt", "optimization", "variable"],
    related: &["bayesopt"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Range and type",
            paragraphs: &[
                "Name must be a nonempty text scalar. Numeric Range contains exactly two finite, increasing bounds and defaults to Type='real'. Type accepts real, integer and categorical; integer bounds must be whole numbers.",
                "Categorical choices may be a string array, a scalar string or character row with Type='categorical', or a cell array of text scalars. Choices must be nonempty and unique.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Transform, Optimize and residency",
            paragraphs: &[
                "Transform is 'none' by default. 'log' requires a positive real lower bound or a nonnegative integer lower bound and is not available for categorical variables. Optimize defaults to true and accepts a scalar logical value or numeric 0 or 1.",
                "The result has Name, Range, Type, Transform and Optimize properties. It is host metadata: host-visible names and effective option values are checked before transfer. Resident numeric Range and Optimize values are materialized by their owning provider for deferred validation; resident numeric Type and Transform values are rejected before download because these options require text. Typed-integer Range storage is a RunMat-only extension and must convert to double exactly; MATLAB-compatible mode rejects it before provider access. There is no device kernel for metadata construction.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Real variable",
            program: "v = optimizableVariable(\"depth\", [1 10]); disp(v);",
            display_output: Some("v is an optimizableVariable object with a real Range of [1 10]."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(v.Name, 'depth')); assert(strcmp(v.Type, 'real')); assert(all(v.Range == [1 10]));" },
        },
        BuiltinExample {
            id: "example-2", title: "Categorical variable",
            program: "v = optimizableVariable(\"kernel\", [\"linear\" \"rbf\"], \"Type\", \"categorical\"); disp(v);",
            display_output: Some("v has the two categorical choices linear and rbf."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(v.Name, 'kernel')); assert(strcmp(v.Type, 'categorical')); assert(numel(v.Range) == 2); assert(all(v.Range == [\"linear\" \"rbf\"]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does the object contain?", answer: "Name, Range, Type, Transform and Optimize describe the variable consumed by bayesopt." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "bayesopt", target: BuiltinDocumentationLinkTarget::Builtin("bayesopt") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/optimizable_variable.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Variable constructor and range admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/optimizable_variable.rs") },
            BuiltinDocumentationLink { label: "Range validation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/optimizable_variable/range.rs") },
            BuiltinDocumentationLink { label: "Typed options", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/optimizable_variable/options.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Variable construction and range tests", location: "builtins::stats::ml::optimizable_variable::tests" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
