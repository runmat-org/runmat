use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gpuDevice"), slug: Some("gpudevice"),
    summary: "Inspect, select, synchronize and reset a session-owned GPUDevice.",
    description: "`device = gpuDevice()` returns a parallel.gpu.GPUDevice handle for the current session. It selects an available device without resetting an already selected device. `gpuDevice(ind)` physically resets this session's device allocations before selecting the indexed owner, including when it was already selected. `gpuDevice([])` resets and deselects. This is a typed object with read-only dependent properties, not the former provider metadata struct; use gpuInfo for a summary string.",
    keywords: &["gpuDevice", "gpu", "device info", "accelerate", "provider", "reset", "wait"],
    related: &["gpuInfo", "gpuArray", "gather", "arrayfun", "pagefun"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior and compatibility", paragraphs: &[
            "A selected RuntimeContext and a registered provider with genuine device-control capability are required. Device indices are positive exact host integers, numbered in registered device-ID order among this invocation's authorized, control-capable providers. Provider-assigned physical IDs are not device indices. Numeric indices are not widened through double; logical selectors remain a separately gated RunMat extension.",
            "Name is a UTF-16 character row; Index is a host double scalar. SupportsDouble reports the actual enabled device capability, not the current execution precision. DeviceAvailable reports backend availability, and DeviceSelected reads the requesting session's current selection. Public properties cannot be assigned; a handle from another session or a replaced class authority is rejected.",
            "TotalMemory and AvailableMemory are queried only from actual backend facts. WGPU does not expose those measurements here and property access reports an explicit unsupported diagnostic. max_buffer_size is an allocation limit, not VRAM; total CUDA memory is not free memory. Missing metrics are never fabricated as zero, an allocation limit, or a successful reset.",
            "`wait(device)` awaits the real queue-completion owner without retiring allocations. `reset(device)` and numeric selection await physical completion, fence ordinary session dispatch/publication, and retire this session's allocations through their existing owners. Previously inherited same-session allocations in other control-capable providers are included. Every relevant capability and execution lease is preflighted before the fence.",
            "RunMat's WGPU provider may be shared by sessions. Reset invalidates the requesting session's prior allocation epochs and cached/pool resources without destroying peer-session allocations or clearing the provider registry. An old GPU array from the reset session is no longer usable; a peer's array remains valid. This scoped device lifecycle is not a process-wide adapter reset or a claim that every CUDA-specific GPUDevice property is implemented.",
            "Reset is ordered retirement, not a transactional rollback. A later owner failure can follow earlier completed retirements; the session stays fenced with the retained original failure and epoch unchanged. The existing host task supervisor owns reset independently of its awaiting caller. Caller cancellation does not restore selection or discard GPU-completion claims; host shutdown before terminal disposition records a fenced failure. Native GPU waits run on the existing bounded device-progress worker, not the RuntimeContext/UI poll thread. Retirement yields between bounded batches.",
            "Only providers that expose actual control and allocation ownership participate. Unsupported backends are not silently treated as reset-capable. The catalog remains incomplete while backend/property coverage and native/browser physical qualification remain outstanding."
        ]},
        BuiltinDocumentationSection { heading: "Examples", paragraphs: &[
            "Inspect the current device\ndevice = gpuDevice();\ndisp(device.Name)",
            "Read live device availability\ndevice = gpuDevice();\nassert(device.DeviceAvailable);\nassert(device.DeviceSelected)",
            "Branch on actual double capability\ndevice = gpuDevice();\nif device.SupportsDouble; disp('double supported'); else; disp('double unavailable'); end",
            "Reset and select an indexed device\ndevice = gpuDevice(uint32(1));\nassert(device.Index == 1);\nwait(device)",
            "Keep the summary interface distinct\ndevice = gpuDevice();\nstatus = gpuInfo();\ndisp(status)"
        ]},
        BuiltinDocumentationSection { heading: "Residency and diagnostics", paragraphs: &[
            "The selector and GPUDevice handle remain on the host. Querying properties does not gather array payloads. Selection/reset changes allocation validity only after actual synchronized owner retirement; ordinary array transfer still uses gpuArray and gather.",
            "Invalid, nonpositive, fractional, out-of-range, nonscalar or resident selectors are rejected. An unavailable index, missing control/lease, unsupported measured property, foreign handle, and retained reset failure are explicit errors. Use reset(device), not the unsupported text selector gpuDevice('reset')."
        ]},
    ],
    examples: &[
        BuiltinExample { id: "inspect", title: "Inspect current device", program: "device = gpuDevice();", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(device, 'parallel.gpu.GPUDevice')); assert(device.Index >= 1); assert(~isempty(device.Name));" } },
        BuiltinExample { id: "vendor", title: "Read authoritative availability", program: "device = gpuDevice(); disp(device.Name); available = device.DeviceAvailable;", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(available)); assert(available); assert(device.DeviceSelected);" } },
        BuiltinExample { id: "precision", title: "Inspect actual double capability", program: "device = gpuDevice(); supports_double = device.SupportsDouble;", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(supports_double));" } },
        BuiltinExample { id: "current-index", title: "Reset and select an index", program: "device = gpuDevice(uint32(1)); wait(device); reset(device);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(device.Index == 1); assert(device.DeviceSelected);" } },
        BuiltinExample { id: "summary", title: "Keep summary and object interfaces distinct", program: "device = gpuDevice(); status = gpuInfo(); gpuDevice([]); selected_after = device.DeviceSelected;", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(device, 'parallel.gpu.GPUDevice')); assert(startsWith(status, 'GPU[')); assert(~selected_after);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Must I call gpuDevice before GPU builtins?", answer: "No. Existing inherited dispatch can use its admitted provider. Calling gpuDevice establishes explicit session selection; a deselected session does not fall back to the process default until it is explicitly selected again." },
        BuiltinDocumentationFaq { question: "Why do memory properties report unsupported?", answer: "A backend must supply actual measurements. WGPU buffer-size limits are not device memory, and total bytes are not free bytes." },
        BuiltinDocumentationFaq { question: "Does reset destroy another session's arrays?", answer: "No. WGPU retires the requesting session's prior epochs through their original allocation owners. It synchronizes shared queue work without globally clearing the device registry or invalidating peer scopes." },
        BuiltinDocumentationFaq { question: "What happens when reset fails or its caller is cancelled?", answer: "The host supervisor retains the reset task through caller cancellation. A physical/retirement failure leaves the session fenced with the original epoch and failure; completed earlier retirements are not rolled back or relabeled transactional." },
        BuiltinDocumentationFaq { question: "How do I get a short summary?", answer: "gpuInfo remains a separate metadata-summary string interface and does not manufacture a GPUDevice object." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Documentation("./gpuarray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Documentation("./gather") },
        BuiltinDocumentationLink { label: "gpuInfo", target: BuiltinDocumentationLinkTarget::Documentation("./gpuinfo") },
        BuiltinDocumentationLink { label: "arrayfun", target: BuiltinDocumentationLinkTarget::Documentation("./arrayfun") },
        BuiltinDocumentationLink { label: "pagefun", target: BuiltinDocumentationLinkTarget::Documentation("./pagefun") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/acceleration/gpu/gpudevice.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed handle, selectors and session lifecycle", location: "builtins::acceleration::gpu::gpudevice::tests" }], notes: &["Source implementation is not physical native/browser qualification evidence."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
