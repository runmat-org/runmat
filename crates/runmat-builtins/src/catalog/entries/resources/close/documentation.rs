use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "current-figure",
        title: "Close the current figure",
        program: "figure(7);\nassert(isgraphics(7));\nstatus = close\nassert(status == 1);\nassert(~isgraphics(7));",
        display_output: Some("status = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(status == 1); assert(~isgraphics(7));",
        },
    },
    BuiltinExample {
        id: "all-figures",
        title: "Close all figures",
        program: "figure(8);\nfigure(9);\nassert(isgraphics(8));\nassert(isgraphics(9));\nstatus = close('all')\nassert(status == 1);\nassert(~isgraphics(8));\nassert(~isgraphics(9));",
        display_output: Some("status = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(status == 1); assert(~isgraphics(8)); assert(~isgraphics(9));",
        },
    },
    BuiltinExample {
        id: "tcp-client",
        title: "Close a loopback TCP client",
        program: "client = tcpclient('__RUNMAT_LOOPBACK_HOST__', __RUNMAT_LOOPBACK_PORT__);\nwrite(client, uint8(9));\nstatus = close(client)\nassert(status == 1);\nassert(close(client) == 0);",
        display_output: Some("status = 1"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeLoopbackNetwork,
        fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
            id: BuiltinExampleFixtureId { local_name: "close-client" },
            scenario: BuiltinLoopbackScenario::Tcp(BuiltinTcpScenario { exchanges: &[BuiltinTcpExchange { client_bytes: &[9], server_bytes: b"" }] }),
            endpoint_substitutions: &[
                BuiltinEndpointSubstitution::LoopbackHost,
                BuiltinEndpointSubstitution::LoopbackPort,
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::NativeOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(status == 1); assert(close(client) == 0);",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("close"),
    slug: Some("close"),
    summary: "Close figures or a session-owned TCP resource.",
    description: "Bare close closes the current figure and succeeds when none exists. close('all') closes figures. A sealed TCP client or server, or the RunMat selectors 'clients' and 'servers', closes only resources owned by the installed TCP service.",
    keywords: &["close", "figure", "tcpclient", "tcpserver", "socket"],
    related: &["figure", "tcpclient", "tcpserver", "accept", "read", "write", "readline"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Graphics forms",
            paragraphs: &[
                "close, close(fig), and close('all') close the current, selected, or all figures. Successful and no-op graphics forms return scalar double 1. An explicit nonexistent figure raises its figure-handle diagnostic. CloseRequestFcn refusal is not implemented, so graphics status 0 is not currently produced.",
                "RunMat mode additionally accepts exact host integer figure selectors and separate variadic targets. Numeric GPU targets are rejected before provider access. A cell of figure targets is not supported.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "TCP forms",
            paragraphs: &[
                "close(client) and close(server) use the receiver's sealed session authority. Closing a server also closes clients accepted from it. Closing an already-closed resource returns double 0; closing one or more live resources returns double 1.",
                "close('clients') and close('servers') target the current session's TCP resources. Nested cells of TCP receivers are accepted in RunMat mode. The visible object class and properties are not socket authority; a forged or stale receiver cannot close another session's socket.",
                "TCP closure requires the installed native TCP service. Browser TCP targets report the typed unsupported-target diagnostic after input admission; graphics close remains available in browser rendering. Cancellation is checked before and after each close. If cancellation or a later target fails after an earlier socket has closed, the call reports an error but that closure remains in effect; close does not replay or roll back earlier targets.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Additional workflows",
            paragraphs: &[
                "A startup sequence such as clc; clear; close is safe with no open figures. For a loopback server, create s = tcpserver(...), accept a client, then close(s) to release both listener and accepted clients. Repeating close(s) returns 0.",
                "RunMat networking selectors can close all clients or servers in a session. Use close('clients') and close('servers'), not close('all'): the latter belongs to graphics. A mixed figure/TCP target container is not admitted.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can close be called twice?", answer: "A repeated TCP close returns 0; a repeated explicit figure-number close raises the missing-figure diagnostic. Bare close with no current figure succeeds as a no-op." },
        BuiltinDocumentationFaq { question: "Do GPU arrays require a kernel?", answer: "No. Figure and TCP targets are host-managed. Resident numeric figure targets reject without a provider download." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "figure", target: BuiltinDocumentationLinkTarget::Builtin("figure") },
        BuiltinDocumentationLink { label: "tcpclient", target: BuiltinDocumentationLinkTarget::Builtin("tcpclient") },
        BuiltinDocumentationLink { label: "tcpserver", target: BuiltinDocumentationLinkTarget::Builtin("tcpserver") },
        BuiltinDocumentationLink { label: "accept", target: BuiltinDocumentationLinkTarget::Builtin("accept") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Close dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/close/dispatch.rs") },
            BuiltinDocumentationLink { label: "Typed TCP close", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/net/close.rs") },
            BuiltinDocumentationLink { label: "Typed networking selector", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/resources/close/control.rs") },
            BuiltinDocumentationLink { label: "Graphics close", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/close.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Close routing and extensions", location: "builtins::close::tests" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
