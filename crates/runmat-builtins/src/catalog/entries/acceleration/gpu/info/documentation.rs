use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gpuInfo"),
    slug: Some("gpuinfo"),
    summary: "Return a formatted status string for the active acceleration provider.",
    description: "`gpuInfo()` formats the same provider metadata projection used by `gpuDevice()` into a host `GPU[key=value, ...]` string. With no provider it returns `GPU[no provider]`.",
    keywords: &["gpuInfo", "gpu", "device info", "accelerate", "summary"],
    related: &["gpuDevice", "gpuArray", "gather", "arrayfun", "pagefun"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`gpuInfo()` accepts no inputs and returns one host string. It reads the provider registered for this call, without initializing or changing the provider, moving arrays, or enqueuing GPU kernels.",
            "The summary contains `device_id`, `index`, `name`, `vendor`, `precision`, and `supports_double`; `backend` and `memory_bytes` appear only when supplied by the provider. Numeric and logical fields are unquoted; string fields are single-quoted and embedded single quotes are doubled. Device IDs are assigned by the provider and should not be compared with a hard-coded zero.",
            "If no provider is registered, the result is exactly `GPU[no provider]`. Unlike `gpuDevice`, this absence is a valid result, not an error. Both builtins use the same typed metadata projection, but neither dispatches through the other builtin's public error identity.",
            "The field order follows the metadata struct's field iteration order; do not parse the summary as a stable machine protocol. To inspect selected provider execution precision, use `gpuDevice().precision` or `gpuDevice().supports_double` instead of searching text; these fields do not establish adapter-wide double-precision hardware capability."
        ]},
        BuiltinDocumentationSection { heading: "Examples", paragraphs: &[
            "Display provider status\ndisp(gpuInfo())",
            "Write a log line\nfprintf('Running on %s\\n', gpuInfo())",
            "Check for a missing provider\nstatus = gpuInfo();\nif strcmp(status, 'GPU[no provider]'); disp('Acceleration disabled'); end",
            "Inspect selected execution precision\ninfo = gpuDevice();\nif info.supports_double; disp('double'); else; disp('single'); end",
            "Display an optional memory hint\ninfo = gpuDevice();\nif isfield(info, 'memory_bytes'); disp(info.memory_bytes); else; disp('memory unknown'); end"
        ]},
        BuiltinDocumentationSection { heading: "Residency", paragraphs: &[
            "`gpuInfo` is a read-only metadata query and leaves existing host and resident arrays untouched. Use `gpuArray` and `gather` for explicit data transfer. The result itself is a host string."
        ]},
    ],
    examples: &[
        BuiltinExample { id: "display", title: "Display provider status", program: "status = gpuInfo(); disp(status);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(startsWith(status, 'GPU['));" } },
        BuiltinExample { id: "log", title: "Write a status line", program: "status = gpuInfo(); fprintf('Running on %s\\n', status);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(startsWith(status, 'GPU['));" } },
        BuiltinExample { id: "missing", title: "Handle an absent provider", program: "status = gpuInfo(); missing = strcmp(status, 'GPU[no provider]');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(missing));" } },
        BuiltinExample { id: "precision", title: "Inspect selected execution precision", program: "info = gpuDevice(); status = gpuInfo();", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(info.supports_double)); assert(strcmp(info.precision, 'double') == info.supports_double); assert(startsWith(status, 'GPU['));" } },
        BuiltinExample { id: "memory", title: "Inspect optional memory metadata", program: "info = gpuDevice(); status = gpuInfo(); if isfield(info, 'memory_bytes'); bytes = info.memory_bytes; else; bytes = uint64(0); end", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(bytes, 'uint64')); assert(startsWith(status, 'GPU['));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `gpuInfo` change GPU state?", answer: "No. It reads provider metadata and formats a host string; it performs no allocation, transfer, or kernel work." },
        BuiltinDocumentationFaq { question: "What happens without a provider?", answer: "It returns exactly `GPU[no provider]`, so callers can branch without catching an error." },
        BuiltinDocumentationFaq { question: "How is it different from `gpuDevice`?", answer: "`gpuDevice` returns a struct for programmatic inspection and errors when no provider exists. `gpuInfo` returns formatted text and has a valid no-provider result." },
        BuiltinDocumentationFaq { question: "Is field order stable?", answer: "It follows the current metadata struct's field iteration order, but the summary is for display rather than a stable protocol; use struct fields for programs." },
        BuiltinDocumentationFaq { question: "How are strings escaped?", answer: "Single quotes are doubled within quoted string field values, so a name such as `Ada'GPU` displays as `Ada''GPU`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "gpuDevice", target: BuiltinDocumentationLinkTarget::Documentation("./gpudevice") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Documentation("./gpuarray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Documentation("./gather") },
        BuiltinDocumentationLink { label: "arrayfun", target: BuiltinDocumentationLinkTarget::Documentation("./arrayfun") },
        BuiltinDocumentationLink { label: "pagefun", target: BuiltinDocumentationLinkTarget::Documentation("./pagefun") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/acceleration/gpu/gpuinfo.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Formatting and no-provider behavior", location: "builtins::acceleration::gpu::gpuinfo::tests" }], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
