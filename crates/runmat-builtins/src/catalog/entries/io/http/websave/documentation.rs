use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("websave"),
    slug: Some("websave"),
    summary: "Download URL content into a file.",
    description: "`websave` downloads an HTTP or HTTPS response to a file and returns the saved filename.",
    keywords: &["websave"],
    related: &["weboptions", "webread"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Downloads HTTP or HTTPS URL content through the shared RunMat HTTP transport and writes it through the active filesystem provider.",
        "The filename and URL are host text values. Native integer and resident numeric values are rejected for these roles.",
        "Query names are text. Query values may be text, numeric, or logical scalars or vectors; native integers are formatted exactly and vectors use comma-separated formatting.",
        "An options structure immediately after the URL can supply `Timeout`, `UserAgent`, `HeaderFields`, and the paired `KeyName`/`KeyValue` HTTP header. Empty default key fields add no header or query parameter. The default timeout is 5 seconds, and an integer timeout is read exactly.",
        "RunMat also recognizes direct `Timeout` and `UserAgent` name-value pairs as request options.",
        "Rejects unpaired or unsupported option shapes instead of silently ignoring them.",
        ] },
        BuiltinDocumentationSection { heading: "Placement and support", paragraphs: &[
        "HTTP and file execution is host-side. Automatically resident values gather transparently; explicit gpuArray input requires RunMat mode.",
        "The network request and saved result are host-side. Automatically resident values can gather transparently; explicit gpuArray input requires RunMat mode.",
        "The request path gathers automatically resident query values through their owning provider. Explicit gpuArray input crosses the `websave-explicit-gpu-input` extension gate before any gather.",
        "HTTP is an effectful host boundary, not a GPU kernel or fusion operation. Browser custom User-Agent headers are restricted by the browser platform; the browser service does not promise to set them.",
        "No optional feature is required; the active host must install HTTP service capability.",
        ] },
        BuiltinDocumentationSection { heading: "Further examples", paragraphs: &[
        "The external API address below is illustrative. A runnable loopback example on this page checks exact integer query formatting and saved file contents against a fixed response.",
        "Download a file with a query value\nsaved = websave(\"report.json\", \"https://api.example.com/report\", \"id\", uint64(42))",
        ] },
    ],
    examples: super::examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does websave require a manual gather?", answer: "No manual gather is needed for automatically resident query values. Explicit gpuArray input is a named RunMat extension and is rejected before provider access in MATLAB-compatible mode." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "weboptions", target: BuiltinDocumentationLinkTarget::Builtin("weboptions") },
        BuiltinDocumentationLink { label: "webread", target: BuiltinDocumentationLinkTarget::Builtin("webread") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime HTTP implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/http/compat.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "HTTP domain unit tests", location: "builtins::io::http::compat::tests" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
