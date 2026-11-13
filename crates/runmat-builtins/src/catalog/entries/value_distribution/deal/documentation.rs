use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("deal"),
    slug: Some("deal"),
    summary: "Distribute input values to requested outputs.",
    description: "`[B1,...,Bn] = deal(A1,...,An)` copies matching inputs positionally. `[B1,...,Bn] = deal(A)` copies one input to every requested output.",
    keywords: &["deal", "multiple outputs", "comma-separated list", "assignment"],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "With multiple inputs, the number of requested outputs must equal the number of inputs.",
                "With one input, any requested output count receives the same value.",
                "Values pass through structurally without numeric conversion, preserving integer class, shape, exact values, and handle identity.",
                "Resident pass-through is a RunMat-only extension because the current public page documents GPU code generation but not interactive GPU Arrays. The gate runs without gather or provider access.",
                "RunMat does not retain the former one-output cell-packing behavior; use a cell-array literal when a packed cell row is intended.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Syntax",
            paragraphs: &[
                "[B1,...,Bn] = deal(A1,...,An)\n[B1,...,Bn] = deal(A)",
                "Outputs preserve their corresponding input values.",
                "Multiple inputs require the same number of outputs.",
                "One input is replicated to every requested output.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Residency",
            paragraphs: &[
                "`deal` performs no numeric computation. RunMat can pass resident values through without gather behind an explicit compatibility gate.",
                "`deal` is output-list construction and is not fused.",
                "When its RunMat-only extension is enabled, `deal` preserves a resident handle and its owner without gather, upload, allocation, or provider dispatch.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution",
            paragraphs: &[
                "Focused tests cover matching distribution, single-input replication, count errors, all eight exact integer storage classes, zero-output behavior, and pre-access resident gating.",
                "No numerical kernel, provider access, filesystem access, callback, or suspension occurs. Nested values and handles retain their ownership and alias identity. The internal OutputList carrier remains at the current builtin ABI; requested and discarded outputs are validated before copying.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Distribute matching values",
            program: "[a,b] = deal(int64(7), uint64(9))",
            display_output: Some("a is int64(7) and b is uint64(9)."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(a,'int64')); assert(isa(b,'uint64')); assert(a == int64(7)); assert(b == uint64(9));",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Replicate one value",
            program: "u = uint64(9007199254740992) + uint64(1); [a,b,c] = deal(u)",
            display_output: Some("All three outputs preserve the exact uint64 value."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(a,'uint64')); assert(a == u); assert(b == u); assert(c == u); assert(a > uint64(9007199254740992));",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Compile exact integer distribution",
            program: "wide = uint64(9007199254740992) + uint64(1);\n[left, right] = deal(wide);\n[~, selected] = deal(uint8(3), wide);\ndeal(wide);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements {
                host: BuiltinExampleHostRequirement::NativeOnly,
                engine: BuiltinExampleEngine::Aot,
                ..BuiltinExampleRequirements::NONE
            },
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(left,'uint64') && isa(right,'uint64') && isa(selected,'uint64')); assert(left == wide && right == wide && selected == wide); assert(selected > uint64(9007199254740992));",
            },
        },
        BuiltinExample {
            id: "example-4",
            title: "Catch a compiled output-count error once",
            program: "global dealCalls; dealCalls = 0; caught = false;\ntry\n operation = @deal;\n [a,b,c] = operation(nextDealInput(), uint16(8));\ncatch failure\n caught = strcmp(failure.identifier, 'RunMat:deal:InputOutputCountMismatch');\nend\nassert(caught && dealCalls == 1);\nfunction value = nextDealInput()\nglobal dealCalls; dealCalls = dealCalls + 1;\nvalue = uint64(9007199254740992) + uint64(1);\nend",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements {
                host: BuiltinExampleHostRequirement::NativeOnly,
                engine: BuiltinExampleEngine::Aot,
                ..BuiltinExampleRequirements::NONE
            },
            verification: BuiltinExampleVerification::Succeeds,
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does `deal` convert integer inputs to double?",
        answer: "No. It copies values without arithmetic or class conversion.",
    }],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/common/deal.rs`",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/common/deal.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Distribution and ownership",
            location: "builtins::common::deal::tests",
        }],
        notes: &[
            "Retained integration reference: crates/runmat-vm/tests/integer_builtin_compatibility_datetime_to_dcgain.rs",
            "The runtime binding is shared by VM/native/JIT/AOT routes; exact-revision execution remains an integrated verification requirement.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
