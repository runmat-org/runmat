use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("fcontour"), slug: Some("fcontour"), summary: "Plot contour lines of a scalar function.", description: concat!(
    "`fcontour` evaluates a function handle over a rectangular two-variable domain ",
    "and renders contour lines from the sampled values.",
),
 keywords: &["fcontour", "function contour", "contour plot", "matlab fcontour"], related: &["contour", "contourf", "fsurf"],
 sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`fcontour(f)` samples `f(x,y)` on the default `[-5 5 -5 5]` domain.", concat!(
    "`fcontour(f, [xmin xmax ymin ymax])` uses an explicit rectangular domain; a ",
    "two-element vector applies the same limits to both axes.",
), "`fcontour(ax, ___)` renders into the requested axes and returns a function-contour graphics handle.", concat!(
    "Function-handle outputs must be scalar real numeric values at every sampled ",
    "point; non-finite samples are preserved as contour discontinuities.",
), concat!(
    "Typed integer domain vectors, property scalars, and documented level vectors ",
    "are accepted for all eight integer classes.",
)] },
BuiltinDocumentationSection { heading: "Options", paragraphs: &[concat!(
    "`'MeshDensity'` controls the number of samples along each domain dimension. ",
    "RunMat accepts exact integer counts from 3 through the resource ceiling of 400.",
), concat!(
    "`'LevelList'` accepts a nonempty real numeric row or column vector. An integer ",
    "scalar selects one explicit level, and descending level order is accepted.",
), concat!(
    "`'LevelStep'` accepts a nonnegative finite real numeric scalar. Zero selects ",
    "automatic levels and a positive integer crosses one checked contour-geometry ",
    "conversion boundary.",
), concat!(
    "`'LineWidth'` accepts a positive finite real numeric scalar and crosses one ",
    "checked renderer-property conversion boundary.",
), concat!(
    "`'Fill'` accepts logical values or the exact numeric scalar values zero and ",
    "one. One adds filled contour bands beneath the contour lines.",
), concat!(
    "`'LineColor'` accepts the documented text forms. Typed-integer RGB is a ",
    "separately gated RunMat extension; the undocumented `'Color'` alias is not ",
    "accepted by `fcontour`.",
), "`'DisplayName'` sets the function-contour graphics handle display name."] },
BuiltinDocumentationSection { heading: "Integer semantics", paragraphs: &[concat!(
    "`xyinterval` must be a two- or four-element row or column vector. Bounds are ",
    "validated before the deliberate binary64 graphics-domain conversion.",
), concat!(
    "All eight built-in integer classes are accepted for `xyinterval`, ",
    "`MeshDensity`, `LevelList`, `LevelStep`, `LineWidth`, and numeric `Fill`.",
), concat!(
    "Typed-integer `LineColor` and contour-style positional integer level ",
    "specifications are RunMat extensions and are rejected unless RunMat extension ",
    "mode is enabled.",
), concat!(
    "Resident numeric arguments are also a RunMat extension; the compatibility gate ",
    "runs before provider gathering.",
)] },
BuiltinDocumentationSection { heading: "Compatibility notes", paragraphs: &[concat!(
    "RunMat currently invokes the callback once per sampled point. First-class ",
    "function-contour object fidelity, matrix callback evaluation, symbolic ",
    "expressions, and adaptive sampling remain deferred broader compatibility work.",
), concat!(
    "The RunMat-only positional level shorthand is not part of the documented ",
    "`fcontour` compatibility surface. Use `LevelList` or `LevelStep` for portable ",
    "code.",
)] },
BuiltinDocumentationSection { heading: "Rendering and residency", paragraphs: &[concat!(
    "`fcontour` samples arbitrary function handles on the host, then uses RunMat's ",
    "existing contour renderer. There is no provider callback kernel because MATLAB ",
    "callbacks are not GPU kernels.",
), concat!(
    "`fcontour` is a callback-sampling builtin. It creates a new sampled contour ",
    "plot and does not preserve input residency, because the input is a MATLAB ",
    "function handle rather than a numeric provider buffer.",
), "Callback sampling runs on the host.", "The resulting contour uses the existing `runmat-plot` contour rendering path."] },
BuiltinDocumentationSection { heading: "Numeric provider precision frontier", paragraphs: &[concat!(
    "fcontour declares no direct numeric provider precision list; host ",
    "generation/sampling or owner-aware gather and installed graphics rendering are ",
    "separate boundaries.",
)] },
BuiltinDocumentationSection { heading: "Coordinate broadcasting", paragraphs: &["none"] },
BuiltinDocumentationSection { heading: "Runtime contract and boundaries", paragraphs: &[concat!(
    "Graphics results are host double handles; fill/fill3 may return a row of ",
    "handles for independently created polygon columns/groups. Sphere instead ",
    "supports zero through three requested outputs: statement form plots, while ",
    "output forms return host double coordinate grids. Handle aliases, ",
    "value-selected options and exact graphics receiver roles are resolved by ",
    "Runtime, not guessed from scalar facts.",
), concat!(
    "The existing numerical and geometry owners validate coordinates, integer ",
    "controls, shapes and properties. Single/double and all eight integer source ",
    "classes retain their native storage wherever the object has authoritative ",
    "source properties; client geometry and renderer buffers are deliberate floating ",
    "boundaries, not a promise of exact wide-integer pixel positions. There is no ",
    "sparse/complex plotting implementation or numeric broadcasting promise here. ",
    "Logical, scalar, vector, matrix, rank and empty admission follows each ",
    "documented owner below rather than a universal coercion.",
), concat!(
    "RuntimeOwned preparation retains the original BuiltinArguments through ",
    "suspension and completion. Eligible direct GPU routes require the live owning ",
    "provider, admitted precision/storage/shape, compatible exports and installed ",
    "plotting device. NoAttempt means pre-effect inapplicability, not permission to ",
    "swallow export, allocation, cancellation or emission errors. Existing provider ",
    "errors are propagated and ingress errors are terminal Never replay. ",
    "Source-property queries and figure persistence may require explicit owner-aware ",
    "readback; surface property snapshots and quiver3/patch construction are not ",
    "zero-copy routes.",
), concat!(
    "Canonical UserInterface capability and host-callback effects reflect scene ",
    "creation/presentation; function-sampling entries additionally declare Unknown ",
    "effects for arbitrary user callbacks and require the Process execution stack. ",
    "NativeAndWasm means the Runtime entry is portable, not that an absent graphics ",
    "host is emulated. Portable Runtime linkage introduces no separate native ",
    "numerical library artifact. Rendering is a fusion boundary and distributed ",
    "execution is unsupported; no numeric provider kernel or fusion primitive is ",
    "invented.",
), concat!(
    "Chart scratch, metadata, retained render state and the sphere coordinate ",
    "builder do not yet have complete finite host allocation ownership. The shared ",
    "canonical HostAllocationReceipt policy guard refuses Managed work before ",
    "extraction/gather/callbacks or scene mutation, preserves ordinary Unbounded ",
    "routes, and checks captured cancellation generation and policy before ",
    "publication. This is an explicit L25 gap, not allocation completion. Long ",
    "synchronous geometry/copy loops, renderer/provider-private allocation ",
    "accounting, full MATLAB property parity, physical performance and actual ",
    "example execution remain independently required.",
)] },
BuiltinDocumentationSection { heading: "Supported representations and remaining forms", paragraphs: &[concat!(
    "Forms are a function handle, optional two/four-element domain, leading axes, ",
    "and name/value controls. Default domain is [-5,5] in each dimension, default ",
    "density 71. MeshDensity is an exact whole count 3..400. LevelList permits a ",
    "nonempty row/column vector, including descending order and a single integer ",
    "level. LevelStep zero chooses automatic levels; a positive finite step crosses ",
    "one checked geometry boundary. Fill accepts logical or exact numeric zero/one. ",
    "DisplayName and existing line styles remain supported.",
), concat!(
    "The callback is evaluated once per sampled point using the actual invocation ",
    "context and rooted result. Cancellation/policy is checked around each callback ",
    "suspension and before publication. The retained function-contour object owns ",
    "its callback through the existing session callback owner, not an ambient ",
    "replacement registry. Numeric results must be scalar real; nonfinite samples ",
    "represent discontinuities. Callback failures are not resampled.",
), concat!(
    "All eight integer classes follow the retained domain/control capability ",
    "declarations. Resident arguments, positional level shorthand and integer RGB ",
    "colors are separately gated RunMat extensions before provider access. The ",
    "undocumented Color/Levels aliases are rejected. Adaptive sampling, symbolic ",
    "expressions, vectorized matrix callbacks and full MATLAB function-object ",
    "fidelity remain unsupported rather than emulated by a fake fixture.",
)] }],
 examples: &[BuiltinExample { id: "retained-1", title: "Plot function contours over the default domain", program: "fcontour(@(x,y) sin(x) + cos(y));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: concat!(
    "witness = findobj(gca, 'Type', 'functioncontour'); assert(numel(witness) >= 1); ",
    "assert(get(witness(1),'MeshDensity') == 71); ",
    "assert(isequal(get(witness(1),'XRange'), [-5 5]));",
) } },
BuiltinExample { id: "retained-2", title: "Set the domain, levels, and mesh density", program: "h = fcontour(@(x,y) x.^2 - y.^2, [-2 2 -1 1], 'LevelList', [-1 0 1], 'MeshDensity', 75);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: concat!(
    "assert(strcmp(get(h,'Type'),'functioncontour')); assert(get(h,'MeshDensity') == ",
    "75); assert(isequal(get(h,'XRange'),[-2 2])); ",
    "assert(isequal(get(h,'YRange'),[-1 1]));",
) } }],
 example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "Does `fcontour` evaluate the function handle on the GPU?", answer: concat!(
    "No. MATLAB function handles can contain arbitrary user code, so RunMat samples ",
    "them on the host and then hands the sampled grid to the contour renderer.",
) },
BuiltinDocumentationFaq { question: "What happens if the function returns a vector?", answer: concat!(
    "RunMat raises `RunMat:fcontour:EvaluationFailed`; each sampled point must ",
    "produce one scalar numeric value.",
) }],
 links: &[BuiltinDocumentationLink { label: "contour", target: BuiltinDocumentationLinkTarget::Builtin("contour") },
BuiltinDocumentationLink { label: "contourf", target: BuiltinDocumentationLinkTarget::Builtin("contourf") },
BuiltinDocumentationLink { label: "fsurf", target: BuiltinDocumentationLinkTarget::Builtin("fsurf") }],
 media: &[BuiltinDocumentationMedia { kind: BuiltinDocumentationMediaKind::Image, url: "https://web.runmatstatic.com/builtin-image/runmat-matlab-plot-contour.webp", alt: "fcontour" }],
 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/fcontour.rs`", target: BuiltinDocumentationLinkTarget::Source(concat!(
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",
    "s/plotting/ops/fcontour.rs",
)) }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "unit", location: "builtins::plotting::fcontour::tests" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "integration", location: "runmat-vm --test plotting::fcontour_dispatches_function_handle_contour_through_vm" }], notes: &[] },
 introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
