use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink {
            label: "Collective and point-to-point execution",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/interpreter/dispatch/parallel/collective.rs"),
        },
        BuiltinDocumentationLink {
            label: "Executor-neutral collective protocol",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-execution/src/collective/mod.rs"),
        },
    ],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Typed point-to-point execution",
            location: "crates/runmat-core/src/tests.rs::modern_spmd_point_to_point_surface_executes_through_typed_bytecode",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Modern and compatible aggregate execution",
            location: "crates/runmat-core/src/tests.rs::modern_and_legacy_spmd_aggregate_surfaces_use_runtime_language_semantics",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Multi-rank ordering and results",
            location: "crates/runmat-core/src/tests.rs::local_multi_rank_spmd_preserves_rank_order_and_collective_results",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "SPMD execution",
        paragraphs: &[
            "Collective and point-to-point functions run inside an active `spmd` region. Ranks are one-based within the admitted worker gang. Every rank follows the compiler-owned operation order, which keeps matching calls deterministic across local, browser, isolated-process, and remote execution.",
            "The `spmd*` names are the current surface. The corresponding `lab*`, `gplus`, `gcat`, and `gop` names remain compatible spellings with the same execution and value semantics.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Values and messages",
        paragraphs: &[
            "Messages use RunMat's canonical execution codec. Supported values retain their class, shape, container structure, and exact integer payload; a transfer does not route typed data through `double`.",
            "Source and destination ranks must identify members of the active gang. Tags are exact nonnegative integers and default to zero when omitted. A receive may select a particular source and tag; `spmdProbe` and `labProbe` test the same selection without consuming a message.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Aggregate results",
        paragraphs: &[
            "An aggregate without a destination returns the result to every rank. With a destination, the result is delivered to that rank and the other ranks receive the operation's documented empty result. Concatenation uses the requested one-based dimension, and functional reductions require an associative binary callable.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Failures and cancellation",
        paragraphs: &[
            "A collective used outside compiler-owned SPMD execution returns a structured lowering error. Invalid ranks, tags, dimensions, reducers, mismatched operation order, worker failure, cancellation, and an unsatisfied communication dependency terminate the gang with a structured parallel error instead of waiting indefinitely.",
        ],
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Where can these functions run?",
        answer: "Inside an active `spmd` region. They are lowered with the region's worker-gang identity and operation sequence.",
    },
    BuiltinDocumentationFaq {
        question: "Are ranks zero-based?",
        answer: "No. Rank values are one-based within the active SPMD gang.",
    },
    BuiltinDocumentationFaq {
        question: "Do messages preserve integer classes?",
        answer: "Yes. The execution codec preserves supported typed values, including exact fixed-width integer payloads.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when destination is omitted from an aggregate?",
        answer: "The aggregate result is returned to every rank.",
    },
    BuiltinDocumentationFaq {
        question: "Can an unmatched communication wait forever?",
        answer: "No. RunMat detects an unsatisfied gang communication dependency and reports a structured parallel error.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Parallel execution",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/execution/parallel"),
}];

const LAB_BARRIER_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "synchronize-two-labs",
    title: "Synchronize two labs",
    program: "pool = parpool(2);\nspmd\n    before = labindex;\n    labBarrier();\n    after = labindex;\nend",
    display_output: Some("both labs continue after reaching the barrier"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(before{1} == 1 && after{1} == 1);\nassert(before{2} == 2 && after{2} == 2);\ndelete(pool);" },
}];
const SPMD_BARRIER_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "synchronize-two-ranks",
    title: "Synchronize two SPMD ranks",
    program: "pool = parpool(2);\nspmd\n    before = spmdIndex();\n    spmdBarrier();\n    after = spmdIndex();\nend",
    display_output: Some("both ranks continue after reaching the barrier"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(before{1} == 1 && after{1} == 1);\nassert(before{2} == 2 && after{2} == 2);\ndelete(pool);" },
}];

const LAB_BROADCAST_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "broadcast-typed-value",
    title: "Broadcast a typed value from the first lab",
    program: "pool = parpool(2);\nspmd\n    shared = labBroadcast(1, uint64([labindex 0x0020000000000001u64]));\nend",
    display_output: Some("both labs receive the first lab's uint64 vector"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "expected = uint64([1 0x0020000000000001u64]);\nassert(isequal(shared{1}, expected));\nassert(isequal(shared{2}, expected));\ndelete(pool);" },
}];
const SPMD_BROADCAST_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "broadcast-typed-value",
    title: "Broadcast a typed value from the first rank",
    program: "pool = parpool(2);\nspmd\n    shared = spmdBroadcast(1, uint64([spmdIndex() 0x0020000000000001u64]));\nend",
    display_output: Some("both ranks receive the first rank's uint64 vector"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "expected = uint64([1 0x0020000000000001u64]);\nassert(isequal(shared{1}, expected));\nassert(isequal(shared{2}, expected));\ndelete(pool);" },
}];

macro_rules! send_example {
    ($constant:ident, $send:literal, $receive:literal, $rank:literal, $noun:literal) => {
        const $constant: &[BuiltinExample] = &[BuiltinExample {
            id: "tagged-peer-message",
            title: concat!("Send a tagged value to the other ", $noun),
            program: concat!("pool = parpool(2);\nspmd\n    here = ", $rank, ";\n    peer = 3 - here;\n    ", $send, "(uint16(here), peer, 7);\n    received = ", $receive, "(peer, 7);\nend"),
            display_output: Some("the two workers exchange uint16 rank values"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: crate::BuiltinExampleFixture::None,
            requirements: crate::BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(received{1}, uint16(2)));\nassert(isequal(received{2}, uint16(1)));\ndelete(pool);" },
        }];
    };
}
send_example!(
    LAB_SEND_EXAMPLES,
    "labSend",
    "labReceive",
    "labindex",
    "lab"
);
send_example!(
    SPMD_SEND_EXAMPLES,
    "spmdSend",
    "spmdReceive",
    "spmdIndex()",
    "rank"
);

const LAB_RECEIVE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "receive-tagged-peer-message",
    title: "Receive a tagged value from the other lab",
    program: "pool = parpool(2);\nspmd\n    here = labindex;\n    peer = 3 - here;\n    labSend(uint16(here), peer, 7);\n    received = labReceive(peer, 7);\nend",
    display_output: Some("the two labs receive each other's uint16 rank values"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(received{1}, uint16(2)));\nassert(isequal(received{2}, uint16(1)));\ndelete(pool);" },
}];
const SPMD_RECEIVE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "receive-value-source-and-tag",
    title: "Receive a value with its source and tag",
    program: "pool = parpool(2);\nspmd\n    here = spmdIndex();\n    peer = 3 - here;\n    spmdSend(uint16(here), peer, 7);\n    [received, source, tag] = spmdReceive(peer, 7);\nend",
    display_output: Some("each rank receives the peer value, source rank, and tag"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(received{1}, uint16(2)));\nassert(isequal(received{2}, uint16(1)));\nassert(source{1} == 2 && source{2} == 1);\nassert(tag{1} == uint64(7) && tag{2} == uint64(7));\ndelete(pool);" },
}];

macro_rules! probe_example {
    ($constant:ident, $send:literal, $probe:literal, $receive:literal, $rank:literal) => {
        const $constant: &[BuiltinExample] = &[BuiltinExample {
            id: "probe-tagged-message",
            title: "Test for a tagged message before receiving it",
            program: concat!("pool = parpool(2);\nspmd\n    here = ", $rank, ";\n    ", $send, "(uint16(here), here, 11);\n    ready = ", $probe, "(here, 11);\n    received = ", $receive, "(here, 11);\nend"),
            display_output: Some("each worker observes and receives its queued message"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: crate::BuiltinExampleFixture::None,
            requirements: crate::BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(ready{1} && ready{2});\nassert(isequal(received{1}, uint16(1)));\nassert(isequal(received{2}, uint16(2)));\ndelete(pool);" },
        }];
    };
}
probe_example!(
    LAB_PROBE_EXAMPLES,
    "labSend",
    "labProbe",
    "labReceive",
    "labindex"
);
probe_example!(
    SPMD_PROBE_EXAMPLES,
    "spmdSend",
    "spmdProbe",
    "spmdReceive",
    "spmdIndex()"
);

macro_rules! send_receive_example {
    ($constant:ident, $function:literal, $rank:literal) => {
        const $constant: &[BuiltinExample] = &[BuiltinExample {
            id: "exchange-with-peer",
            title: "Exchange typed values between two workers",
            program: concat!("pool = parpool(2);\nspmd\n    here = ", $rank, ";\n    peer = 3 - here;\n    exchanged = ", $function, "(peer, peer, uint16(here), 13);\nend"),
            display_output: Some("each worker receives the peer's uint16 value"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: crate::BuiltinExampleFixture::None,
            requirements: crate::BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(exchanged{1}, uint16(2)));\nassert(isequal(exchanged{2}, uint16(1)));\ndelete(pool);" },
        }];
    };
}
send_receive_example!(LAB_SEND_RECEIVE_EXAMPLES, "labSendReceive", "labindex");
send_receive_example!(SPMD_SEND_RECEIVE_EXAMPLES, "spmdSendReceive", "spmdIndex()");

macro_rules! plus_example {
    ($constant:ident, $function:literal, $rank:literal) => {
        const $constant: &[BuiltinExample] = &[BuiltinExample {
            id: "sum-ranks",
            title: "Sum a typed value across two workers",
            program: concat!("pool = parpool(2);\nspmd\n    total = ", $function, "(uint32(", $rank, "));\nend"),
            display_output: Some("both workers receive uint32(3)"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: crate::BuiltinExampleFixture::None,
            requirements: crate::BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(total{1}, uint32(3)));\nassert(isequal(total{2}, uint32(3)));\ndelete(pool);" },
        }];
    };
}
plus_example!(GPLUS_EXAMPLES, "gplus", "labindex");
plus_example!(SPMD_PLUS_EXAMPLES, "spmdPlus", "spmdIndex()");

macro_rules! cat_example {
    ($constant:ident, $function:literal, $rank:literal) => {
        const $constant: &[BuiltinExample] = &[BuiltinExample {
            id: "concatenate-rank-values",
            title: "Concatenate typed rank values",
            program: concat!("pool = parpool(2);\nspmd\n    joined = ", $function, "(uint16(", $rank, "), 2);\nend"),
            display_output: Some("both workers receive uint16([1 2])"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: crate::BuiltinExampleFixture::None,
            requirements: crate::BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(joined{1}, uint16([1 2])));\nassert(isequal(joined{2}, uint16([1 2])));\ndelete(pool);" },
        }];
    };
}
cat_example!(GCAT_EXAMPLES, "gcat", "labindex");
cat_example!(SPMD_CAT_EXAMPLES, "spmdCat", "spmdIndex()");

macro_rules! reduce_example {
    ($constant:ident, $function:literal, $rank:literal) => {
        const $constant: &[BuiltinExample] = &[BuiltinExample {
            id: "reduce-with-plus",
            title: "Apply an associative reduction across two workers",
            program: concat!("pool = parpool(2);\nspmd\n    total = ", $function, "(@plus, uint32(", $rank, "));\nend"),
            display_output: Some("both workers receive uint32(3)"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: crate::BuiltinExampleFixture::None,
            requirements: crate::BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(total{1}, uint32(3)));\nassert(isequal(total{2}, uint32(3)));\ndelete(pool);" },
        }];
    };
}
reduce_example!(GOP_EXAMPLES, "gop", "labindex");
reduce_example!(SPMD_REDUCE_EXAMPLES, "spmdReduce", "spmdIndex()");

macro_rules! documentation {
    ($constant:ident, $title:literal, $summary:literal, $description:literal, $related:expr, $examples:expr) => {
        pub(in super::super) const $constant: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($title),
            slug: Some($title),
            summary: $summary,
            description: $description,
            keywords: &["parallel", "spmd", "collective", "distributed"],
            related: $related,
            sections: SECTIONS,
            examples: $examples,
            example_exemption: None,
            faqs: FAQS,
            links: LINKS,
            media: &[],
            evidence: EVIDENCE,
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Stable),
        };
    };
}

documentation!(
    LAB_BARRIER_DOCUMENTATION,
    "labBarrier",
    "Wait until every lab reaches the same barrier.",
    "`labBarrier` synchronizes every lab in the active SPMD gang before execution continues.",
    &["spmdBarrier", "labindex", "numlabs"],
    LAB_BARRIER_EXAMPLES
);
documentation!(
    SPMD_BARRIER_DOCUMENTATION,
    "spmdBarrier",
    "Wait until every SPMD rank reaches the same barrier.",
    "`spmdBarrier` synchronizes every rank in the active SPMD gang before execution continues.",
    &["labBarrier", "spmdIndex", "spmdSize"],
    SPMD_BARRIER_EXAMPLES
);
documentation!(
    LAB_BROADCAST_DOCUMENTATION,
    "labBroadcast",
    "Broadcast a value from one lab to every lab.",
    "`labBroadcast` returns the designated source lab's value on every lab in the active gang.",
    &["spmdBroadcast", "labSend", "labReceive"],
    LAB_BROADCAST_EXAMPLES
);
documentation!(
    SPMD_BROADCAST_DOCUMENTATION,
    "spmdBroadcast",
    "Broadcast a value from one SPMD rank to every rank.",
    "`spmdBroadcast` returns the designated source rank's value on every rank in the active gang.",
    &["labBroadcast", "spmdSend", "spmdReceive"],
    SPMD_BROADCAST_EXAMPLES
);
documentation!(
    LAB_SEND_DOCUMENTATION,
    "labSend",
    "Send a value to another lab.",
    "`labSend` queues a typed value for a destination lab, optionally under an exact message tag.",
    &["labReceive", "labProbe", "labSendReceive", "spmdSend"],
    LAB_SEND_EXAMPLES
);
documentation!(SPMD_SEND_DOCUMENTATION, "spmdSend", "Send a value to another SPMD rank.", "`spmdSend` queues a typed value for a destination rank, optionally under an exact message tag.", &["spmdReceive", "spmdProbe", "spmdSendReceive", "labSend"], SPMD_SEND_EXAMPLES);
documentation!(
    LAB_RECEIVE_DOCUMENTATION,
    "labReceive",
    "Receive a value from another lab.",
    "`labReceive` consumes the next message matching the requested source lab and tag.",
    &["labSend", "labProbe", "labSendReceive", "spmdReceive"],
    LAB_RECEIVE_EXAMPLES
);
documentation!(SPMD_RECEIVE_DOCUMENTATION, "spmdReceive", "Receive a value and optional message metadata from another SPMD rank.", "`spmdReceive` consumes the next matching message and can also return its source rank and exact tag.", &["spmdSend", "spmdProbe", "spmdSendReceive", "labReceive"], SPMD_RECEIVE_EXAMPLES);
documentation!(
    LAB_PROBE_DOCUMENTATION,
    "labProbe",
    "Test whether a matching lab message is available.",
    "`labProbe` checks for a matching source and tag without consuming the queued message.",
    &["labSend", "labReceive", "spmdProbe"],
    LAB_PROBE_EXAMPLES
);
documentation!(
    SPMD_PROBE_DOCUMENTATION,
    "spmdProbe",
    "Test whether a matching SPMD message is available.",
    "`spmdProbe` checks for a matching source and tag without consuming the queued message.",
    &["spmdSend", "spmdReceive", "labProbe"],
    SPMD_PROBE_EXAMPLES
);
documentation!(
    LAB_SEND_RECEIVE_DOCUMENTATION,
    "labSendReceive",
    "Exchange a value with another lab.",
    "`labSendReceive` performs one matched send and receive as a single ordered SPMD operation.",
    &["labSend", "labReceive", "spmdSendReceive"],
    LAB_SEND_RECEIVE_EXAMPLES
);
documentation!(
    SPMD_SEND_RECEIVE_DOCUMENTATION,
    "spmdSendReceive",
    "Exchange a value with another SPMD rank.",
    "`spmdSendReceive` performs one matched send and receive as a single ordered SPMD operation.",
    &["spmdSend", "spmdReceive", "labSendReceive"],
    SPMD_SEND_RECEIVE_EXAMPLES
);
documentation!(
    GPLUS_DOCUMENTATION,
    "gplus",
    "Sum values across the active labs.",
    "`gplus` applies the compatible typed addition semantics across the active lab gang.",
    &["spmdPlus", "gop", "gcat"],
    GPLUS_EXAMPLES
);
documentation!(
    SPMD_PLUS_DOCUMENTATION,
    "spmdPlus",
    "Sum values across the active SPMD ranks.",
    "`spmdPlus` applies typed addition across the active SPMD gang.",
    &["gplus", "spmdReduce", "spmdCat"],
    SPMD_PLUS_EXAMPLES
);
documentation!(
    GCAT_DOCUMENTATION,
    "gcat",
    "Concatenate values across the active labs.",
    "`gcat` concatenates lab values in rank order along a requested dimension.",
    &["spmdCat", "gplus", "gop"],
    GCAT_EXAMPLES
);
documentation!(
    SPMD_CAT_DOCUMENTATION,
    "spmdCat",
    "Concatenate values across the active SPMD ranks.",
    "`spmdCat` concatenates rank values in rank order along a requested dimension.",
    &["gcat", "spmdPlus", "spmdReduce"],
    SPMD_CAT_EXAMPLES
);
documentation!(
    GOP_DOCUMENTATION,
    "gop",
    "Reduce lab values with an associative function.",
    "`gop` applies an associative binary callable across values from the active lab gang.",
    &["spmdReduce", "gplus", "gcat"],
    GOP_EXAMPLES
);
documentation!(
    SPMD_REDUCE_DOCUMENTATION,
    "spmdReduce",
    "Reduce SPMD values with an associative function.",
    "`spmdReduce` applies an associative binary callable across values from the active SPMD gang.",
    &["gop", "spmdPlus", "spmdCat"],
    SPMD_REDUCE_EXAMPLES
);
