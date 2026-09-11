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
            label: "Partition and materialization service",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/parallel/distribution.rs"),
        },
        BuiltinDocumentationLink {
            label: "Distributed VM execution",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/interpreter/dispatch/parallel/distributed.rs"),
        },
    ],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Typed construction, local-part, and exact-index tests",
            location: "crates/runmat-core/src/tests.rs distributed value tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Redistribution and stale-generation tests",
            location: "crates/runmat-core/src/tests.rs distribution generation tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Native process-worker distributed test",
            location: "crates/runmat-cli/tests/parallel_execution.rs::spmd_executes_as_one_typed_multi_process_gang",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "A distributed value stores partition payloads in the execution service. The language value is a typed handle containing its global shape, value class, distribution scheme, owning pool, and generation. Exact integer classes, shapes, and supported complex or sparse representations are retained across partition and materialization boundaries.",
            "Driver construction partitions one complete input. Cooperative construction inside `spmd` can replicate one value, select a designated worker's value, or build from every worker's local contribution. The default build form validates class, rank, shape, layout, and codistributor agreement across the gang; `\"noCommunication\"` skips cross-worker validation only when the caller already guarantees consistency.",
            "Distributed values belong to one pool generation. Closing or resizing that pool retires local-part access, materialization, redistribution, and distributed builtin placement for the stale handle.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "Construction, redistribution, and materialization may cross the driver-worker boundary. `getLocalPart` reads only the selected worker partition, `globalIndices` computes ownership from the authoritative layout, and `getCodistributor` projects the resolved description without reading payloads.",
        ],
    },
];

const DISTRIBUTED_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "typed-driver-input",
        title: "Partition an exact typed driver value",
        program: "source = uint64([1 0x0020000000000001u64]);\nvalue = distributed(source);\nresult = gather(value)",
        display_output: Some("result retains both uint64 values exactly"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(iscodistributed(value));\nassert(isa(result, \"uint64\"));\nassert(isequal(result, source));\ndelete(gcp(\"nocreate\"));",
        },
    },
    BuiltinExample {
        id: "distributed-builtin",
        title: "Keep an admitted operation distributed",
        program: "value = distributed(int16([-7 2]));\nmapped = abs(value);\nresult = gather(mapped)",
        display_output: Some("result = int16([7 2])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(iscodistributed(mapped));\nassert(isequal(result, int16([7 2])));\ndelete(gcp(\"nocreate\"));",
        },
    },
];
const CODISTRIBUTED_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "client-default",
        title: "Create a codistributed value from driver data",
        program: "source = uint16([1 2 3 4]);\nvalue = codistributed(source);\nresult = gather(value)",
        display_output: Some("result = uint16([1 2 3 4])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(iscodistributed(value));\nassert(isequal(result, source));\ndelete(gcp(\"nocreate\"));",
        },
    },
    BuiltinExample {
        id: "client-codistributor",
        title: "Use an explicit partition description",
        program: "pool = parpool(2);\nsource = uint16([1 2 3 4]);\ncodist = codistributor1d(uint32(2), uint64([2 2]), uint64([1 4]));\nvalue = codistributed(source, codist);\nresult = gather(value)",
        display_output: Some("result = uint16([1 2 3 4])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "resolved = getCodistributor(value);\nassert(isequal(result, source));\nassert(resolved.Dimension == uint64(2));\ndelete(pool);",
        },
    },
];
const BUILD_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "worker-local-parts",
    title: "Build from one local partition per worker",
    program: "pool = parpool(2);\nspmd\n  local = uint64([2 * spmdIndex() - 1, 2 * spmdIndex()]);\n  codist = codistributor1d(uint32(2), uint64([2 2]), uint64([1 4]));\n  value = codistributed.build(local, codist);\nend\nresult = gather(value)",
    display_output: Some("result = uint64([1 2 3 4])"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(result, uint64([1 2 3 4])));\nassert(iscodistributed(value{1}));\nassert(iscodistributed(value{2}));\ndelete(pool);",
    },
}];
const REDISTRIBUTE_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "change-dimension",
        title: "Move partitions to another array dimension",
        program: "pool = parpool(2);\nsource = uint64([1 2; 3 4]);\nvalue = distributed(source);\ncodist = codistributor1d(uint32(2), uint64([1 1]), uint64([2 2]));\nmoved = redistribute(value, codist);\nresult = gather(moved)",
        display_output: Some("result equals source"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "resolved = getCodistributor(moved);\nassert(isequal(result, source));\nassert(resolved.Dimension == uint64(2));\ndelete(pool);",
        },
    },
    BuiltinExample {
        id: "block-cyclic",
        title: "Move a matrix to a block-cyclic grid",
        program: "pool = parpool(2);\nsource = uint16([1 2; 3 4]);\nvalue = distributed(source);\ncodist = codistributor2dbc(uint32([1 2]), uint64(1), \"row\", uint64([2 2]));\nmoved = redistribute(value, codist);\nresult = gather(moved)",
        display_output: Some("result equals source"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "resolved = getCodistributor(moved);\nassert(isequal(result, source));\nassert(all(resolved.WorkerGrid == uint64([1 2])));\ndelete(pool);",
        },
    },
];
const GET_CODISTRIBUTOR_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "resolved-description",
    title: "Inspect the resolved distribution",
    program: "pool = parpool(2);\nvalue = distributed(uint16([1 2 3 4]));\ncodist = getCodistributor(value)",
    display_output: Some("codist includes the resolved global size and partition"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isComplete(codist));\nassert(all(codist.GlobalSize == uint64([1 4])));\nassert(sum(codist.Partition) == uint64(4));\ndelete(pool);",
    },
}];
const GLOBAL_INDICES_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "owned-index-vector",
        title: "Read one worker's exact global indices",
        program: "pool = parpool(2);\nvalue = distributed(uint16([10 20 30 40]));\nindices = globalIndices(value, uint32(2), uint32(1))",
        display_output: Some("indices contains worker 1's uint64 column indices"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(indices, \"uint64\"));\nassert(isequal(indices, uint64([1 2])));\ndelete(pool);",
        },
    },
    BuiltinExample {
        id: "owned-endpoints",
        title: "Read the first and last owned index",
        program: "pool = parpool(2);\nvalue = distributed(uint16([10 20 30 40]));\n[first, last] = globalIndices(value, uint32(2), uint32(2))",
        display_output: Some("first = 3; last = 4"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(first == uint64(3));\nassert(last == uint64(4));\ndelete(pool);",
        },
    },
];
const GET_LOCAL_PART_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "driver-selected-part",
        title: "Read the current serial worker's partition",
        program: "source = uint64([1 0x0020000000000001u64]);\nvalue = distributed(source);\nlocal = getLocalPart(value)",
        display_output: Some("local retains the exact uint64 payload"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(local, source));\ndelete(gcp(\"nocreate\"));",
        },
    },
    BuiltinExample {
        id: "spmd-local-parts",
        title: "Read each worker's local partition",
        program: "pool = parpool(2);\nvalue = distributed(uint16([1 2 3 4]));\nspmd\n  local = getLocalPart(value);\nend",
        display_output: Some("local contains one partition per worker"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(local{1}, uint16([1 2])));\nassert(isequal(local{2}, uint16([3 4])));\ndelete(pool);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does the language value contain every partition?", answer: "No. It is a typed, generation-fenced handle; the execution service owns partition payloads." },
    BuiltinDocumentationFaq { question: "Are integer classes preserved?", answer: "Yes. Supported values retain their exact class, shape, and payload through partition, redistribution, local access, and gather." },
    BuiltinDocumentationFaq { question: "What happens after the pool is closed or resized?", answer: "The old generation is retired and distributed operations reject its stale handles." },
    BuiltinDocumentationFaq { question: "Does inspection gather the full value?", answer: "No. Codistributor and index inspection use handle metadata; local-part access reads only the selected partition." },
    BuiltinDocumentationFaq { question: "Where can I omit a lab argument?", answer: "Inside `spmd`, the active rank selects the local partition. Driver code supplies an explicit lab where the form requires one." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Parallel execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/execution/parallel"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "codistributor1d",
        target: BuiltinDocumentationLinkTarget::Builtin("codistributor1d"),
    },
];

macro_rules! documentation {
    ($constant:ident, $title:literal, $summary:literal, $description:literal, $keywords:expr, $related:expr, $examples:expr) => {
        pub(in super::super) const $constant: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($title),
            slug: Some($title),
            summary: $summary,
            description: $description,
            keywords: $keywords,
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
    DISTRIBUTED_DOCUMENTATION,
    "distributed",
    "Create a distributed array from a driver value.",
    "`distributed` partitions one complete typed value across the current or automatic pool.",
    &["parallel", "distributed", "array", "partition", "driver"],
    &["codistributed", "gather", "getLocalPart", "redistribute"],
    DISTRIBUTED_EXAMPLES
);
documentation!(CODISTRIBUTED_DOCUMENTATION, "codistributed", "Create a codistributed array from driver or worker-scoped data.", "`codistributed` creates a compatible distributed value from a complete client value, replicated worker value, or designated worker value.", &["parallel", "codistributed", "array", "worker", "partition"], &["codistributed.build", "distributed", "gather", "getLocalPart"], CODISTRIBUTED_EXAMPLES);
documentation!(CODISTRIBUTED_BUILD_DOCUMENTATION, "codistributed.build", "Build a codistributed array from worker-local partitions.", "`codistributed.build` cooperatively resolves and validates one local contribution from every worker in an SPMD gang.", &["parallel", "codistributed", "build", "local parts", "spmd"], &["codistributed", "gather", "getLocalPart", "globalIndices"], BUILD_EXAMPLES);
documentation!(REDISTRIBUTE_DOCUMENTATION, "redistribute", "Move a distributed array to another distribution layout.", "`redistribute` preserves the global typed value while replacing its resolved distribution scheme.", &["parallel", "distributed", "redistribute", "partition", "layout"], &["codistributor1d", "codistributor2dbc", "distributed", "gather"], REDISTRIBUTE_EXAMPLES);
documentation!(GET_CODISTRIBUTOR_DOCUMENTATION, "getCodistributor", "Return a distributed array's resolved codistributor.", "`getCodistributor` projects the authoritative distribution description without reading partition payloads.", &["parallel", "distributed", "codistributor", "inspection", "layout"], &["codistributor1d", "codistributor2dbc", "globalIndices", "redistribute"], GET_CODISTRIBUTOR_EXAMPLES);
documentation!(GLOBAL_INDICES_DOCUMENTATION, "globalIndices", "Return the exact global indices owned by a worker.", "`globalIndices` computes a worker's `uint64` ownership indices or first/last endpoints from the resolved layout.", &["parallel", "distributed", "indices", "ownership", "uint64"], &["getCodistributor", "getLocalPart", "redistribute"], GLOBAL_INDICES_EXAMPLES);
documentation!(
    GET_LOCAL_PART_DOCUMENTATION,
    "getLocalPart",
    "Return the partition local to the selected worker.",
    "`getLocalPart` reads one typed partition without gathering the full distributed value.",
    &[
        "parallel",
        "distributed",
        "local part",
        "worker",
        "partition"
    ],
    &["gather", "getCodistributor", "globalIndices"],
    GET_LOCAL_PART_EXAMPLES
);
