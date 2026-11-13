use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("weboptions"),
    slug: Some("weboptions"),
    summary: "Create options for HTTP requests.",
    description: "`weboptions` creates a host options structure for `webread`, `websave`, and `webwrite`.",
    keywords: &["weboptions", "http options", "timeout", "headers", "rest client"],
    related: &["webread", "webwrite", "websave", "jsondecode", "jsonencode"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and fields", paragraphs: &[
            "`weboptions()` returns defaults; case-insensitive name-value pairs override them. In RunMat mode, an existing options struct may be copied and overridden. The input struct is not mutated.",
            "The implemented fields are `ContentType`, `Timeout`, `HeaderFields`, `UserAgent`, `Username`, `Password`, `KeyName`, `KeyValue`, `RequestMethod`, `MediaType`, and the RunMat-only `QueryParameters`. Unknown names raise an error. `CharacterEncoding`, `ContentReader`, `ArrayFormat`, and `CertificateFilename` are not accepted by the current constructor or consumed by these HTTP operations.",
            "The default timeout is 5 seconds. `Timeout` accepts a positive numeric scalar up to 2147.483647 seconds or `Inf`; accepted native integer values are converted to bounded timeout seconds. `KeyValue` accepts a text, numeric, or logical scalar and retains exact native integer storage. When paired with nonempty `KeyName`, `webread` and `webwrite` emit one HTTP header with exact integer decimal text.",
            "`ContentType` accepts `auto`, `json`, `text`, or `binary` (with documented text/binary aliases). `RequestMethod` accepts `auto`, `get`, `post`, `put`, `patch`, or `delete`; the consuming operation decides whether a method is supported. `HeaderFields` accepts a struct or two-column cell array.",
            "Empty username and password leave basic authentication disabled. A nonempty password without a username is rejected. `HeaderFields` and `QueryParameters` can be cleared with an empty struct or an empty two-column cell array.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and compatibility", paragraphs: &[
            "Options construction performs no network request. `webread`, `webwrite`, or `websave` later use the active host HTTP service to make a request; a stored option value does not own a session or grant transport capability.",
            "Options are host metadata. Automatically resident values gather through their owning provider after pure option admission. Explicit gpuArray input requires the named RunMat extension before provider access. The resulting structure is host-resident and terminates fusion.",
            "Copying an input struct and using `QueryParameters` are separate RunMat extensions. Both are rejected in MATLAB-compatible mode; ordinary name-value construction remains available.",
            "This page's portable examples check the constructed fields without contacting a live endpoint. The `webread` and `webwrite` pages contain executable loopback examples that exercise these options against deterministic HTTP fixtures.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "timeout", title: "Set a request timeout", program: "opts = weboptions('Timeout', 10);", display_output: Some("opts.Timeout = 10"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(opts)); assert(opts.Timeout == 10);" } },
        BuiltinExample { id: "credentials", title: "Store basic authentication credentials", program: "opts = weboptions('Username', 'ada', 'Password', 'lovelace');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(opts.Username, 'ada')); assert(strcmp(opts.Password, 'lovelace'));" } },
        BuiltinExample { id: "json-body", title: "Configure JSON request and response", program: "opts = weboptions('ContentType', 'json', 'MediaType', 'application/json');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(opts.ContentType, 'json')); assert(strcmp(opts.MediaType, 'application/json'));" } },
        BuiltinExample { id: "headers", title: "Apply custom headers from a structure", program: "headers = struct('Accept', 'application/json');\nopts = weboptions('HeaderFields', headers);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(opts.HeaderFields.Accept, 'application/json'));" } },
        BuiltinExample { id: "copy-query", title: "Copy options and override with query parameters", program: "base = weboptions('ContentType', 'json');\nopts = weboptions(base, 'Timeout', 15, 'QueryParameters', struct('verbose', true));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(base.ContentType, 'json')); assert(base.Timeout == 5); assert(opts.Timeout == 15); assert(opts.QueryParameters.verbose);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which option names are supported in RunMat?", answer: "The constructor accepts `ContentType`, `Timeout`, `HeaderFields`, `UserAgent`, `Username`, `Password`, `KeyName`, `KeyValue`, `RequestMethod`, `MediaType`, and the RunMat-only `QueryParameters`. Unknown fields, including `CharacterEncoding`, `ContentReader`, `ArrayFormat`, and `CertificateFilename`, are rejected." },
        BuiltinDocumentationFaq { question: "What does `RequestMethod='auto'` mean?", answer: "`webread` treats `auto` as GET and `webwrite` treats it as POST. The consuming operation validates any explicit method it supports." },
        BuiltinDocumentationFaq { question: "How are empty usernames or passwords handled?", answer: "Empty values leave authentication disabled. A nonempty password without a username raises an error." },
        BuiltinDocumentationFaq { question: "Can I pass query parameters through the options struct?", answer: "Yes in RunMat mode: use a struct or two-column cell array for `QueryParameters`. MATLAB-compatible mode rejects this named extension." },
        BuiltinDocumentationFaq { question: "Do I need to manage GPU residency for options?", answer: "Automatically resident values gather through their owning provider. Explicit gpuArray values require RunMat mode; the returned struct is host-resident." },
        BuiltinDocumentationFaq { question: "Does `weboptions` mutate the input struct?", answer: "No. The RunMat copy-and-override form leaves the input struct unchanged." },
        BuiltinDocumentationFaq { question: "How can I clear headers or query parameters?", answer: "Use an empty struct or empty two-column cell array for the corresponding field." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "webread", target: BuiltinDocumentationLinkTarget::Builtin("webread") },
        BuiltinDocumentationLink { label: "webwrite", target: BuiltinDocumentationLinkTarget::Builtin("webwrite") },
        BuiltinDocumentationLink { label: "jsondecode", target: BuiltinDocumentationLinkTarget::Builtin("jsondecode") },
        BuiltinDocumentationLink { label: "jsonencode", target: BuiltinDocumentationLinkTarget::Builtin("jsonencode") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "HTTP options entrypoint", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/http/weboptions.rs") },
            BuiltinDocumentationLink { label: "HTTP options field validation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/http/weboptions/options.rs") },
            BuiltinDocumentationLink { label: "API-key header admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/http/api_key.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Options forms and gates", location: "builtins::io::http::weboptions::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "webread consumer", location: "builtins::io::http::weboptions::tests::webread_uses_weboptions_without_polluting_query" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "webwrite consumer", location: "builtins::io::http::weboptions::tests::webwrite_uses_weboptions_auto_request_method" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
