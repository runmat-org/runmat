use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Independent GPU random state",
        paragraphs: &[
            "`gpurng` queries or changes the session GPU random stream used by GPU output forms of `rand`, `randi`, `randn`, and `randperm`. `rng` controls the separate ordinary CPU stream; seeding one does not reseed the other. Automatic placement of an ordinary CPU random call must not change its stream or draw order.",
            "`gpurng(\"default\")` selects Threefry4x64_20 with seed zero. A numeric seed restarts the current GPU generator; `gpurng(\"shuffle\")` chooses a fresh seed. The returned scalar structure records the previous Type, Seed, and opaque State, and a prior returned snapshot can be restored.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Generator availability and execution",
        paragraphs: &[
            "The documented GPU algorithms are Threefry, Philox, and CombRecursive. This RunMat implementation currently executes Threefry; requests for the other algorithms fail before changing the stream. They are not aliases of Threefry. Snapshot bytes are RunMat-owned state, not a claim of byte-for-byte compatibility with MATLAB's opaque state encoding.",
            "A request-bound GPU child range determines each output independently of execution placement. Supported provider forms can generate on the device; other admitted forms may compute from the GPU child stream on the host and restore the requested GPU residency through the exact owner. An operational provider failure after effects is terminal, not a signal to draw again on another route.",
            "Threefry words and normal Box-Muller inputs follow the same RunMat child mapping across supported providers. Floating transcendental last bits may differ by backend. The default seed and generator identity do not assert MATLAB's undocumented seed-to-counter or distribution-lane mapping.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "snapshot-restore",
        title: "Save and restore GPU random state",
        program: "gpurng('default'); s = gpurng; a = rand(2,2,'gpuArray'); gpurng(s); b = rand(2,2,'gpuArray'); same = isequal(gather(a),gather(b))",
        display_output: Some("same is true after the GPU state is restored"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(same); assert(strcmpi(s.Type,'threefry')); assert(s.Seed == 0);",
        },
    },
    BuiltinExample {
        id: "separate-streams",
        title: "Seed CPU and GPU streams separately",
        program: "rng('default'); gpurng('default'); cpu = rand(1,3); gpu = gather(rand(1,3,'gpuArray'))",
        display_output: Some("cpu and gpu are each reproducible from their own defaults"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "rng('default'); assert(isequal(cpu,rand(1,3))); gpurng('default'); assert(isequal(gpu,gather(rand(1,3,'gpuArray'))));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does rng seed the GPU stream?", answer: "No. Use gpurng for GPU random output. rng and gpurng control independent session streams." },
    BuiltinDocumentationFaq { question: "Can I restore a GPU stream?", answer: "Yes. Capture s = gpurng, then call gpurng(s) to restore that RunMat snapshot." },
    BuiltinDocumentationFaq { question: "Are Philox and CombRecursive supported?", answer: "They are documented GPU generators, but remain unimplemented in RunMat and are rejected before stream mutation; they are not renamed Threefry streams." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "MathWorks gpurng",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/parallel-computing/gpurng.html",
        ),
    },
    BuiltinDocumentationLink {
        label: "rand",
        target: BuiltinDocumentationLinkTarget::Builtin("rand"),
    },
    BuiltinDocumentationLink {
        label: "randi",
        target: BuiltinDocumentationLinkTarget::Builtin("randi"),
    },
    BuiltinDocumentationLink {
        label: "randn",
        target: BuiltinDocumentationLinkTarget::Builtin("randn"),
    },
    BuiltinDocumentationLink {
        label: "randperm",
        target: BuiltinDocumentationLinkTarget::Builtin("randperm"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "GPU stream state", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/gpurng.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "GPU stream query, seed, and restore", location: "crates/runmat-runtime/src/builtins/stats/random/gpurng.rs" }],
    notes: &["Alternate documented generators and full device generation for every random constructor remain open."],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gpurng"),
    slug: Some("gpurng"),
    summary: "Control the independent GPU random stream.",
    description: "`gpurng` queries, seeds, and restores RunMat's session GPU random stream independently of the CPU `rng` stream.",
    keywords: &["gpurng", "GPU", "Threefry", "seed", "random", "state"],
    related: &["rng", "rand", "randi", "randn", "randperm"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("R2011b"),
    status: Some(BuiltinDocumentationStatus::Partial),
};
