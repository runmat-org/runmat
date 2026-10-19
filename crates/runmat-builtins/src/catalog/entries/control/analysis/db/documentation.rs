use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/db.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("db"), slug: Some("db"),
    summary: "Convert voltage magnitude or power to decibels.",
    description: "`db(y)` and `db(y, 'voltage')` compute `20*log10(abs(y))`. `db(y, 'power')` computes `10*log10(y)` for nonnegative real power. `db(y, R)` computes `10*log10(abs(y).^2 ./ R)` for finite positive real resistance. The resistance form uses implicit expansion.",
    keywords: &["db", "decibel", "voltage", "power", "resistance", "complex"],
    related: &["abs", "log10", "tf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Modes and values", paragraphs: &[
            "Voltage mode accepts real or complex numeric signal data and uses magnitude. Power mode accepts only real, nonnegative signal values; negative or complex values raise an error. Resistance must be finite, positive, and real, and may be an array broadcast-compatible with the signal. Zero signal values produce negative infinity.",
            "The result is real single if the signal or resistance has native single precision, and real double otherwise. Unary and text-mode results follow the signal shape; resistance-mode results have the broadcast shape. Integer, logical, complex-integer signals and typed-integer resistance are RunMat extensions gated by `compat = 'runmat'`; they are rejected in MATLAB compatibility mode.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Decibel conversion executes on the host and is a fusion boundary. Resident input arrays are gathered from their owning provider before conversion. There is no GPU kernel for `db`; the result is host-resident.",
            "The operation may suspend during a provider gather and may throw for invalid mode, signal, resistance, or incompatible array sizes. It has no workspace, filesystem, network, callback, user-interface, randomness, or time effect.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "voltage", title: "Voltage decibels", program: "y = db(10); fprintf('%.0f\\n', y);", display_output: Some("20"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y - 20) < 1e-10);" } },
        BuiltinExample { id: "power", title: "Power decibels", program: "p = db(100, 'power'); fprintf('%.0f\\n', p);", display_output: Some("20"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p - 20) < 1e-10);" } },
        BuiltinExample { id: "resistance", title: "Voltage across reference resistance", program: "p = db(10, 50); fprintf('%.4f\\n', p);", display_output: Some("3.0103"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p - 10*log10(2)) < 1e-6);" } },
        BuiltinExample { id: "complex", title: "Complex signal magnitude", program: "z = db(3 + 4i); fprintf('%.4f\\n', z);", display_output: Some("13.9794"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(z - 20*log10(5)) < 1e-6);" } },
        BuiltinExample { id: "single-precision", title: "Preserve single precision", program: "y = db(single([1 10]));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(y), 'single')); assert(isequal(size(y), [1 2]));" } },
        BuiltinExample { id: "integer-extension", title: "RunMat integer signal extension", program: "y = db(uint16([1 10]));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(y), 'double')); assert(abs(y(2) - 20) < 1e-10);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is the default mode?", answer: "Voltage mode, equivalent to `db(y, 'voltage')`." },
        BuiltinDocumentationFaq { question: "How are complex inputs handled?", answer: "Voltage and resistance modes use the complex magnitude and return real values. Power mode rejects complex input." },
        BuiltinDocumentationFaq { question: "What happens for zero input?", answer: "The logarithm yields negative infinity." },
        BuiltinDocumentationFaq { question: "Can resistance be an array?", answer: "Yes. It must be real, finite, positive, and broadcast-compatible with the signal." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "abs", target: BuiltinDocumentationLinkTarget::Builtin("abs") },
        BuiltinDocumentationLink { label: "log10", target: BuiltinDocumentationLinkTarget::Builtin("log10") },
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Decibel modes, precision, resistance broadcasting and extension gating", location: "crates/runmat-runtime/src/builtins/control/db.rs" }], notes: &["Runtime support and numeric placement are implemented; integrated native/browser/provider qualification remains outstanding."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
