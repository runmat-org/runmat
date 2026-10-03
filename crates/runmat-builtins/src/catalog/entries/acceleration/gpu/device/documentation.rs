use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gpuDevice"),
    slug: Some("gpudevice"),
    summary: "Query metadata for the active RunMat acceleration provider.",
    description: "`info = gpuDevice()` returns a host struct describing the active provider. This RunMat-only metadata facade is not MATLAB's GPUDevice object or device-selection/reset lifecycle.",
    keywords: &["gpuDevice", "gpu", "device info", "accelerate", "provider"],
    related: &["gpuInfo", "gpuArray", "gather", "arrayfun", "pagefun"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior and compatibility", paragraphs: &[
            "A registered acceleration provider is required. Without one, `gpuDevice` raises `RunMat:gpuDevice:NoProvider` and reports that no acceleration provider is registered.",
            "The returned struct always has `device_id` (uint32), `index` (uint32, currently 1), `name`, `vendor`, `precision` (`single` or `double`), and `supports_double` (logical). `backend` and `memory_bytes` (uint64) appear only when the provider reports them. Device IDs are provider-assigned, not a stable literal zero.",
            "`gpuDevice(1)` accepts the current device index; any other positive index reports that it is unavailable. Exact scalar signed and unsigned integer storage is accepted without conversion through double; a logical selector is separately compatibility-gated.",
            "`gpuDevice('reset')` and `gpuDevice([])` report that reset is unsupported. RunMat does not reset or switch providers. The provider-metadata result, and the logical selector, require their RunMat extension gates; MATLAB-compatible mode rejects the facade rather than impersonating a GPUDevice object.",
            "This query does not enqueue kernels, alter residency, or mutate GPU state. It reads the provider active at the call boundary; a later provider registration can change a later result. The returned struct remains an ordinary host value."
        ]},
        BuiltinDocumentationSection { heading: "Examples", paragraphs: &[
            "Inspect the active provider\ninfo = gpuDevice();\ndisp(info.name)",
            "Display vendor and optional backend\ninfo = gpuDevice();\ndisp(info.vendor);\nif isfield(info, 'backend'); disp(info.backend); end",
            "Branch on precision capability\ninfo = gpuDevice();\nif info.supports_double; disp('double'); else; disp('single'); end",
            "Select the currently active index\ninfo = gpuDevice(1);\nassert(info.index == uint32(1));",
            "Handle an absent provider\ntry; info = gpuDevice(); catch ex; disp(ex.message); end"
        ]},
        BuiltinDocumentationSection { heading: "Residency and diagnostics", paragraphs: &[
            "Neither the input selector nor the host struct result is a GPU kernel operand. Existing arrays remain where they are. Use `gpuArray` and `gather` to move array data explicitly.",
            "Invalid, nonpositive, nonintegral, out-of-range, or nonscalar selectors are rejected; reset-like arguments and unavailable indices have separate stable diagnostics. Arguments are classified before querying provider metadata where required by the runtime contract."
        ]},
    ],
    examples: &[
        BuiltinExample { id: "inspect", title: "Inspect active provider", program: "info = gpuDevice();", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(info)); assert(info.index == uint32(1)); assert(~isempty(info.name));" } },
        BuiltinExample { id: "vendor", title: "Show vendor and optional backend", program: "info = gpuDevice(); disp(info.vendor); if isfield(info, 'backend'); disp(info.backend); end", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(info.vendor));" } },
        BuiltinExample { id: "precision", title: "Check double support", program: "info = gpuDevice(); capable = info.supports_double;", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(capable)); assert(strcmp(info.precision, 'double') == capable);" } },
        BuiltinExample { id: "current-index", title: "Inspect current index", program: "info = gpuDevice(uint32(1));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(info.index == uint32(1));" } },
        BuiltinExample { id: "summary", title: "Format a status message", program: "info = gpuDevice(); status = gpuInfo();", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(info)); assert(startsWith(status, 'GPU['));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Must I call `gpuDevice` before GPU builtins?", answer: "No. It only inspects the provider already registered by the runtime; it does not initialize or select a GPU." },
        BuiltinDocumentationFaq { question: "Why are some fields absent?", answer: "Providers only report metadata they can supply. `backend` and `memory_bytes` are optional; the other documented fields are always projected." },
        BuiltinDocumentationFaq { question: "What if no provider is registered?", answer: "`gpuDevice` raises `RunMat:gpuDevice:NoProvider`. Use `gpuInfo()` if a nonthrowing status string is needed." },
        BuiltinDocumentationFaq { question: "Can I select or reset a device?", answer: "Only current index 1 is accepted by this RunMat facade. Other indices are unavailable, and reset requests are explicitly unsupported; this is not MATLAB's GPUDevice lifecycle." },
        BuiltinDocumentationFaq { question: "How do I get a short summary?", answer: "`gpuInfo()` formats the same shared metadata projection as a host string. It does not call `gpuDevice()` or inherit its no-provider error." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Documentation("./gpuarray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Documentation("./gather") },
        BuiltinDocumentationLink { label: "gpuInfo", target: BuiltinDocumentationLinkTarget::Documentation("./gpuinfo") },
        BuiltinDocumentationLink { label: "arrayfun", target: BuiltinDocumentationLinkTarget::Documentation("./arrayfun") },
        BuiltinDocumentationLink { label: "pagefun", target: BuiltinDocumentationLinkTarget::Documentation("./pagefun") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/acceleration/gpu/gpudevice.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Selector and metadata behavior", location: "builtins::acceleration::gpu::gpudevice::tests" }], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
