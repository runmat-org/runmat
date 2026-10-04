use crate::*;

// ZIP_STORED sample.txt containing "alpha\n", produced with a fixed ZIP
// timestamp and validated by the runtime archive reader in the example gate.
const SAMPLE_ZIP: &[u8] = &[
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0xec, 0x6e,
    0x60, 0x9f, 0x06, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x73, 0x61,
    0x6d, 0x70, 0x6c, 0x65, 0x2e, 0x74, 0x78, 0x74, 0x61, 0x6c, 0x70, 0x68, 0x61, 0x0a, 0x50, 0x4b,
    0x01, 0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0xec, 0x6e,
    0x60, 0x9f, 0x06, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x73, 0x61, 0x6d, 0x70,
    0x6c, 0x65, 0x2e, 0x74, 0x78, 0x74, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x01, 0x00, 0x38, 0x00, 0x00, 0x00, 0x2e, 0x00, 0x00, 0x00, 0x00, 0x00,
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("unzip"),
    slug: Some("unzip"),
    summary: "Extract files from ZIP archives in MATLAB and RunMat.",
    description: "`unzip(zipfilename)` extracts a local ZIP archive into the current folder. `unzip(zipfilename, outputfolder)` extracts beneath the requested destination. HTTP/HTTPS ZIP sources are also accepted. `filenames = unzip(...)` returns a column cell array of extracted regular-file paths.",
    keywords: &["unzip", "zip", "archive", "extract", "download", "url"],
    related: &["webread", "tempdir", "mkdir", "dir"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and outputs", paragraphs: &[
            "The archive and optional destination are scalar strings or one-row character arrays. The destination defaults to the current folder and is created when needed. Archive directory entries create folders but are not returned. A zero-output statement call still extracts while suppressing automatic output; at most one output may be requested.",
            "Each returned path is a character row, not a string scalar. The result is an N-by-1 cell array, including a 0-by-1 cell when no regular files were extracted.",
        ] },
        BuiltinDocumentationSection { heading: "Security and limits", paragraphs: &[
            "Absolute, user-relative, drive-prefixed, and parent-traversing entry names are rejected; existing symbolic-link path components are also rejected. Files are staged as temporary outputs before rename. Earlier entries may remain if a later entry fails.",
            "Compressed archives are limited to 512 MiB; extraction counts at most 10 GiB of written entry bytes. Deflate and stored entries are supported by the ZIP reader; malformed or unsupported archives raise `RunMat:unzip:Zip`.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Extraction is host filesystem I/O and a fusion boundary, not an accelerator kernel. ZIP source and folder controls must be host text; numeric gpuArray handles are rejected before provider download. URL retrieval uses the installed HTTP transport and its bounded archive stream; hosts without that capability reject the request before extraction. Native retrieval streams into a temporary archive. Browser retrieval checks each received chunk against the byte limit before appending it, then retains the complete accepted archive for extraction. The archive-byte limit is not a guarantee on total live memory.",
            "The active filesystem provider and HTTP transport determine availability. `~` path expansion reads the caller environment. A path is not a transferable file snapshot; distributed workers require an independently staged archive and destination.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "default-destination",
        title: "Extract into the current folder",
        program: "unzip('sample.zip');\ntext = fileread('sample.txt');\ntext",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "zip-stored-text" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "sample.zip", content: BuiltinFixtureContent::Bytes(SAMPLE_ZIP) }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isfile('sample.txt')); assert(strcmp(text, sprintf('alpha\\n')));" },
    }, BuiltinExample {
        id: "local-archive",
        title: "Extract a ZIP archive and inspect the returned files",
        program: "files = unzip('sample.zip', 'data');\ntext = fileread(files{1});\ntext",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "zip-stored-text" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File {
                relative_path: "sample.zip",
                content: BuiltinFixtureContent::Bytes(SAMPLE_ZIP),
            }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(files)); assert(isequal(size(files), [1 1])); assert(ischar(files{1})); assert(strcmp(text, sprintf('alpha\\n')));" },
    }, BuiltinExample {
        id: "http-archive",
        title: "Download and extract a ZIP archive",
        program: "files = unzip('__RUNMAT_HTTP_BASE_URL__/sample.zip', 'remote');\ntext = fileread(files{1});\ntext",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeLoopbackNetwork,
        fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
            id: BuiltinExampleFixtureId { local_name: "zip-http" },
            scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario {
                exchanges: &[BuiltinHttpExchange {
                    request: BuiltinHttpRequest { method: BuiltinHttpMethod::Get, path: "/sample.zip", body: None },
                    response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "application/zip" }], body: SAMPLE_ZIP },
                }],
            }),
            endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::NativeOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(files)); assert(isequal(size(files), [1 1])); assert(strcmp(text, sprintf('alpha\\n')));" },
    }, BuiltinExample {
        id: "returned-file-list",
        title: "Use the returned file list",
        program: "files = unzip('measurements.zip', 'measurements');\nfirstFile = files{1};\ntext = fileread(firstFile);\ntext",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "zip-measurements-text" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "measurements.zip", content: BuiltinFixtureContent::Bytes(SAMPLE_ZIP) }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(files)); assert(isequal(size(files), [1 1])); assert(ischar(firstFile)); assert(isfile(firstFile)); assert(strcmp(text, sprintf('alpha\\n')));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does unzip create the destination folder?", answer: "Yes. The output folder and archive subfolders are created when needed." },
        BuiltinDocumentationFaq { question: "Can unzip extract URLs?", answer: "Yes, HTTP and HTTPS archives are downloaded before extraction when the host HTTP transport is available." },
        BuiltinDocumentationFaq { question: "Are directory entries returned?", answer: "No. Only regular extracted file paths occur in the result cell." },
        BuiltinDocumentationFaq { question: "How are unsafe ZIP paths handled?", answer: "Absolute, parent-traversing, drive-prefixed and user-relative names are rejected before writing outside the destination." },
        BuiltinDocumentationFaq { question: "Does unzip run on the GPU?", answer: "No. ZIP extraction is host filesystem and network I/O." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "webread", target: BuiltinDocumentationLinkTarget::Builtin("webread") },
        BuiltinDocumentationLink { label: "tempdir", target: BuiltinDocumentationLinkTarget::Builtin("tempdir") },
        BuiltinDocumentationLink { label: "mkdir", target: BuiltinDocumentationLinkTarget::Builtin("mkdir") },
        BuiltinDocumentationLink { label: "dir", target: BuiltinDocumentationLinkTarget::Builtin("dir") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime archive domain", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/archive/unzip.rs") },
            BuiltinDocumentationLink { label: "Bounded archive HTTP input", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/archive/unzip/source/http.rs") },
            BuiltinDocumentationLink { label: "Browser HTTP stream", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/http/browser/stream.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Extraction, security, output and HTTP behavior", location: "builtins::io::archive::unzip::tests" }],
        notes: &[],
    },
    introduced: None,
    status: None,
};
