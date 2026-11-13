use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("builtin"),
    slug: Some("builtin"),
    summary: "Invoke a registered builtin implementation by name.",
    description: "`[varargout] = builtin(name, varargin)` forwards the supplied arguments and requested output count to the selected registered builtin. The target, not this dispatcher, owns input admission, numeric/storage behavior, output facts, and errors.",
    keywords: &["builtin", "dispatch", "subsref", "subsasgn"],
    related: &["feval", "subsref", "subsasgn", "find"],
    sections: &[
        BuiltinDocumentationSection { heading: "Dispatch and output count", paragraphs: &[
            "`name` is a scalar builtin name. Positional arguments after it are forwarded unchanged. The caller's requested output count is forwarded, including multiple outputs. The selected target determines valid arity, options, output class and shape, and failure identifiers; a source-known name alone does not prove those runtime facts for arbitrary arguments.",
            "For `subsref` and `subsasgn`, the registered implementation executes the standard subscript descriptor, including chained member, parentheses, and brace steps. It bypasses the original receiver's indexing override while preserving member visibility, dependent-property accessors, and ordinary dispatch on nested receivers. Assignments use the shared indexing implementation rather than the public `setfield` extension.",
            "Method and target calls may execute user code, mutate objects, suspend, throw, or use ambient services. An unknown or dynamically resolved target remains effectful. This wrapper is not a replay boundary for post-effect failures.",
        ] },
        BuiltinDocumentationSection { heading: "Integer and device behavior", paragraphs: &[
            "All eight integer classes pass unchanged to an admitted target. Scalar-double mixing, overflow, output class, and computation domain are function-specific; builtin does no arithmetic or numeric conversion itself.",
            "The selected target owns GPU/provider admission, transfers, output residency and fusion. Dispatch does not imply host-only execution or gather resident payloads merely because the target name is text. Target-specific provider capabilities and error ordering remain in force.",
            "The wrapper is portable runtime dispatch with no separate native symbol or target-specific scheduler. Calls may reach process-stack methods and inherit their reentry, callback, ownership and affinity requirements. Distributed or foreign behavior is likewise target-dependent, not a blanket portability guarantee.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "exact-integer", title: "Preserve an integer target's output class",
            program: "value = builtin('plus', uint64(7), uint64(3));",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isa(value,'uint64')); assert(value == uint64(10));" },
        },
        BuiltinExample {
            id: "requested-outputs", title: "Forward two requested outputs",
            program: "[row, col] = builtin('find', [0 4; 5 0]);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(row,[2;1])); assert(isequal(col,[1;2]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does builtin convert integer arguments to double?", answer: "No. It forwards them unchanged; the selected builtin's integer contract applies." },
        BuiltinDocumentationFaq { question: "Can a delegated call use a GPU provider?", answer: "Yes, if the selected builtin admits that form. builtin itself neither launches a kernel nor forces a gather." },
        BuiltinDocumentationFaq { question: "Who decides how many outputs are valid?", answer: "The selected builtin receives the caller's requested output count and owns its output contract." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "feval", target: BuiltinDocumentationLinkTarget::Builtin("feval") },
        BuiltinDocumentationLink { label: "subsref", target: BuiltinDocumentationLinkTarget::Builtin("subsref") },
        BuiltinDocumentationLink { label: "subsasgn", target: BuiltinDocumentationLinkTarget::Builtin("subsasgn") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime builtin dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/builtin.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact integer, requested outputs, and resident target delegation", location: "crates/runmat-runtime/src/builtins/introspection/builtin.rs::tests" }],
        notes: &["No editable reference JSON sidecar existed for this identity at the signed base. Composed example execution and provider qualification remain required."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
