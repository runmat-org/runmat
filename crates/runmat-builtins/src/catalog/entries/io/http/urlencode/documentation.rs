use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("urlencode"), slug: Some("urlencode"),
    summary: "Percent-encode text for use in URLs.", description: "`urlencode(text)` percent-encodes text for use in URLs. It is a RunMat language extension for scripts that need URL text utilities.",
    keywords: &["urlencode", "url", "percent encode", "web"], related: &[],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Accepts a string scalar or character row vector and returns encoded text.",
            "Preserves URL-safe characters and percent-encodes other UTF-8 bytes.",
            "Numeric values, including native integer values, are not text and are rejected before provider access.",
            "Singleton string arrays are accepted. Character input must have one row and only singleton dimensions after the second; N-D character pages are not flattened into a row.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "This text-only helper runs on the host. Numeric and provider-resident values are rejected before provider access.",
            "The result is a new character row, including for string scalar input. Its width is determined by the transformed text; no numeric or provider execution is involved.",
            "ASCII letters, digits, '-', '_', '.', and '~' are preserved; other UTF-8 bytes use uppercase hexadecimal percent escapes.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "percent-encoding-round-trip", title: "Round-trip URL text",
        program: "encoded = urlencode(\"a b/c\");\ndecoded = urldecode(encoded);",
        display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(encoded, 'a%20b%2Fc')); assert(isequal(decoded, 'a b/c'));" },
    }],
    example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB Language Extensions",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "URL text implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/http/compat.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "URL text compatibility tests", location: "builtins::io::http::compat::tests" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
