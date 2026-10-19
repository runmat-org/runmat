use crate::*;

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "isolated-outbox",
    title: "Capture a message in an isolated local outbox",
    program: "status = sendmail('alice@example.test', 'Status', 'Ready');\nassert(status == 0);\ndisp(status);",
    display_output: Some("0"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Native,
    fixture: BuiltinExampleFixture::MailOutbox(BuiltinMailOutboxFixture {
        id: BuiltinExampleFixtureId { local_name: "mail-outbox" },
        expected_message: "To: alice@example.test\nSubject: Status\nContent-Type: text/plain; charset=utf-8\n\nReady\n",
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0);" },
}, BuiltinExample {
    id: "reject-numeric-mail-input",
    title: "Reject numeric mail arguments before transport selection",
    program: "sendmail(uint64(42), 'Status')",
    display_output: Some("sendmail: mail arguments must be text, not numeric or resident values"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::ExpectedError {
        identifier: "RunMat:sendmail:InvalidMailInput",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sendmail"),
    slug: Some("sendmail"),
    summary: "Create an email message using MATLAB sendmail-compatible arguments.",
    description: "`sendmail` creates a plain-text email message and sends it through a configured local command or writes it to an outbox capture directory.",
    keywords: &["sendmail", "email", "mail", "notification"],
    related: &[],
    sections: &[
        BuiltinDocumentationSection { heading: "Arguments and result", paragraphs: &[
            "Recipients may be a text scalar, string array, or cell array of text addresses. Subject is scalar text; the optional message defaults to an empty body. The fourth attachment argument is recognized but rejected because the current transports do not support attachments.",
            "A successful local delivery or outbox capture returns numeric zero. Direct integer, numeric-array, nested integer, and resident numeric mail arguments reject before provider access or transport effects; numeric character codes already in a character vector remain text.",
        ] },
        BuiltinDocumentationSection { heading: "Transport selection", paragraphs: &[
            "`RUNMAT_SENDMAIL_OUTBOX` writes an RFC822-style message in the configured directory. On native hosts, `RUNMAT_SENDMAIL_COMMAND` or an available `/usr/sbin/sendmail` supplies a local command transport.",
            "If neither route is selected, sendmail reads one `Internet` group snapshot from the installed application preference service, checking `SMTP_Server` then `SMTPServer`. Missing or failed preference storage is reported; it is not replaced by an empty session store. A configured SMTP server alone produces an error because direct TLS SMTP transport is unavailable in this build.",
        ] },
        BuiltinDocumentationSection { heading: "Placement and support", paragraphs: &[
            "Mail delivery is a host-side effect boundary, not a GPU kernel or fusion operation. Browser hosts have no local command route. The active host must supply the relevant application preferences and transport or outbox services.",
        ] },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Mail runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/http/compat.rs") },
            BuiltinDocumentationLink { label: "Recipient and transport controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/http/compat/sendmail_helpers.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Mail argument, transport, and preference tests", location: "builtins::io::http::compat::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
