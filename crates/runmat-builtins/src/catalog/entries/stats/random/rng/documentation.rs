//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`rng()` returns a structure describing the current generator (fields `Type`, `Seed`, and `State`).",
            "`rng(seed)` reads a non-negative integer seed below `2^32` from scalar double or any of the eight native integer classes. Seeds at or above `2^32` are an independently gated RunMat extension.",
            "`rng('default')` restores the host MT19937 engine with user seed 0. It is reproducible within RunMat; MATLAB sequence and state-wire equivalence are separate conformance obligations.",
            "`rng('shuffle')` requires an invocation context. A valid unsigned RUNMAT_RNG_SHUFFLE_SEED from its session environment overrides seeding without a Clock read; otherwise the admitted Clock supplies the wall-time duration. The same duration yields the same seed, without ambient clock or address entropy. Missing context or unavailable Clock fails before changing RNG state.",
            "`rng(seed, 'twister')` selects the host MT19937 generator. RunMat does not claim MATLAB normal-transform or byte-for-byte snapshot interchange compatibility.",
            "`rng(S)` restores a structure previously returned by RunMat's `rng`, including the supported legacy LCG snapshot format. Independently constructed native-integer Seed or legacy LCG State fields require the separately gated typed-state extension. The explicit device RNG is independent: rng does not forward host MT19937 state to, or reseed, a device provider.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["rng is host-stream state control, not an accelerator numeric kernel. The explicit device RNG is independent and is managed by gpurng; no host state token is forwarded to a provider.", "rng changes only the host generator. It does not synchronize or reseed the explicit device RNG. Automatic tensor placement does not turn a host snapshot into device RNG state, and CPU/GPU random-sequence identity is not claimed."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation limits",
        paragraphs: &["RunMat snapshots are not a MATLAB state-wire interchange format, and exact MATLAB random-sequence/normal-transform parity is not established. Independent host/device stream APIs remain distinct."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Resetting the generator for reproducible simulation runs",
        program: "rng(0);\nu = rand(1, 4);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(u), [1 4])); assert(all(u >= 0 & u < 1)); rng(0); assert(isequal(u, rand(1,4)));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Saving and restoring RNG state around a computation",
        program: "rng(0);\ns = rng;\nrng(1337);\nA = randn(2, 2);\nrng(s);            % restore original stream\nB = randn(2, 2);   % continues from the saved state",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "rng(s); assert(isequal(B, randn(2,2))); assert(isequal(size(A), [2 2])); assert(strcmp(s.Type, 'twister')); assert(isequal(size(s.State), [625 1])); assert(isa(s.State, 'uint32'));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Scrambling the generator with the system clock",
        program: "rng('shuffle');\nu = rand(1, 3)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(u), [1 3])); assert(all(u >= 0 & u < 1));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Replay host-generated data after transferring it to the GPU",
        program: "rng(2024);\nG = gpuArray(rand(1, 3));\nC = gather(G);\nrng(2024);\nH = rand(1, 3);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isgpuarray(G)); assert(isequal(C, H)); assert(isequal(size(C), [1 3]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Creating deterministic permutations with `rng` and `randperm`",
        program: "rng(42);\np = randperm(6);\nrng(42);\nq = randperm(6)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(p,q)); assert(isequal(sort(p), 1:6));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Preserve a wide native seed exactly",
        program: "seed = bitshift(uint64(1),53) + uint64(1); rng(seed); s = rng; u = rand(1,3); rng(s); again = rand(1,3);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(s.Seed,'uint64')); assert(s.Seed==seed); assert(isa(s.State,'uint32')); assert(isequal(size(s.State),[625 1])); assert(isequal(u,again));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rng"),
    slug: Some("rng"),
    summary: "Configure, query, and restore RunMat's global pseudorandom number generator through MATLAB-shaped forms.",
    description: "`rng` queries or reconfigures the host pseudorandom generator used by `rand`, `randn`, `randi`, `randperm`, and host-backed distribution samplers. RunMat uses MT19937 for the public twister engine. Its normal transform, exact sequences, and state-wire format are not a claim of MATLAB stream interchange compatibility.",
    keywords: &["rng", "seed", "twister", "shuffle", "random state"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What generators does RunMat support?",
            answer: "The public 'twister' spelling selects RunMat's host MT19937 generator. Legacy LCG snapshots remain restorable for backward compatibility. Additional generator APIs and MATLAB sequence/state-wire parity are separate obligations.",
        },
        BuiltinDocumentationFaq {
            question: "What numeric range is valid for seeds?",
            answer: "MATLAB-compatible mode accepts non-negative integer seeds below `2^32`. RunMat mode extends the domain through `u64`; typed values remain exact, and snapshot Seed fields use native `uint64` when binary64 would round.",
        },
        BuiltinDocumentationFaq {
            question: "How can I force a specific seed for `rng('shuffle')` in tests?",
            answer: "Populate RUNMAT_RNG_SHUFFLE_SEED with an unsigned integer in the invocation's frozen session environment. rng('shuffle') uses that override without reading Clock. Without a valid override, an admitted Clock is required; the builtin never consults the process environment, ambient wall clock, or address entropy.",
        },
        BuiltinDocumentationFaq {
            question: "Does `rng` affect GPU random numbers?",
            answer: "No. The explicit device RNG has independent state managed by gpurng. rng configures the host stream and does not pass MT19937 state to device providers or claim CPU/GPU sequence identity.",
        },
        BuiltinDocumentationFaq {
            question: "How do I restore a saved state structure?",
            answer: "Pass a structure returned by the same RunMat build back to rng: s = rng; ...; rng(s); resumes the exact host stream. A RunMat snapshot is not a MATLAB Twister interchange format; legacy LCG snapshots are also accepted.",
        },
        BuiltinDocumentationFaq {
            question: "What is stored in the `State` field?",
            answer: "For MT19937, State is a 625-by-1 uint32 tensor: 624 state words followed by RunMat's next-word index. Legacy LCG snapshots use the older two-word representation. Seed is stored as double when exact, or native uint64 when binary64 would round it. Restoring the returned structure preserves the exact supported engine state.",
        },
        BuiltinDocumentationFaq {
            question: "Can I create independent streams?",
            answer: "Not yet. RunMat exposes MATLAB-compatible `RandStream` APIs in a separate roadmap item; for now `rng` operates on the single global stream.",
        },
        BuiltinDocumentationFaq {
            question: "Does `rng` change automatically when I call other random functions?",
            answer: "Sampling consumes and advances the current stream but does not reseed it. Explicit rng calls configure or restore the host stream; provider-specific state is separate. A discarded result can still consume randomness, according to the called sampler's output-count semantics.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "rand",
            target: BuiltinDocumentationLinkTarget::Builtin("rand"),
        },
        BuiltinDocumentationLink {
            label: "randn",
            target: BuiltinDocumentationLinkTarget::Builtin("randn"),
        },
        BuiltinDocumentationLink {
            label: "randi",
            target: BuiltinDocumentationLinkTarget::Builtin("randi"),
        },
        BuiltinDocumentationLink {
            label: "randperm",
            target: BuiltinDocumentationLinkTarget::Builtin("randperm"),
        },
        BuiltinDocumentationLink {
            label: "gpuArray",
            target: BuiltinDocumentationLinkTarget::Builtin("gpuarray"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/rng.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/rng.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::stats::random::rng::tests",
        }],
        notes: &["Integer compatibility and placement are declared in the canonical contract. Examples assert shape, class, bounds, or state replay rather than fixed random transcripts."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
