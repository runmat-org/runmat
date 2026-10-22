use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("polarplot"),
 slug: Some("polarplot"),
 summary: "Create 2-D line plots in polar coordinates.",
 description: concat!(
    "`polarplot` converts angular/radial data to cartesian line coordinates and ",

    "marks the active axes as polar so the renderer draws concentric radial rings ",

    "and angular spokes.",

),

 keywords: &["polarplot",
 "polar plot",
 "polar coordinates",
 "theta rho",
 "matlab polarplot",
 "line plot"],
 related: &["plot",
 "hold",
 "legend",
 "set",
 "get"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "Real integer coordinates are accepted before the explicit floating geometry ",

    "boundary; complex integer `Z` data is a separately gated RunMat-mode extension ",

    "that must be exactly representable before rendering.",

),
 concat!(
    "`polarplot(theta, rho)` interprets theta values as radians and requires theta ",

    "and rho to have the same number of elements.",

),
 concat!(
    "`polarplot(rho)` creates implicit theta values over `[0, 2*pi]` with the same ",

    "number of samples as rho.",

),
 "`polarplot(Z)` interprets complex input as `theta = angle(Z)` and `rho = abs(Z)`.",
 concat!(
    "Matrix rho or Z inputs create one polar line per column. A vector theta can be ",

    "paired with a rho matrix when its length matches each column.",

),
 concat!(
    "Multiple theta/rho pairs in one call create multiple line objects and return ",

    "the handle for the first created series.",

),
 concat!(
    "LineSpec strings and common line name/value pairs such as `LineWidth`, `Color`, ",

    "`Marker`, and `DisplayName` share the same parser used by `plot`. A final ",

    "all-name/value group applies to every line created by that call.",

),
 "Empty data creates an empty line object, matching MATLAB graphics-object behavior.",
 concat!(
    "The target axes are switched to polar mode with equal x/y scaling and symmetric ",

    "cartesian limits based on the largest radius.",

)] },

BuiltinDocumentationSection { heading: "Options",
 paragraphs: &["Use MATLAB-style line specs such as `'r--'`, `'bo-'`, and `':'` after a polar series.",
 "Use `hold on` before additional `polarplot` calls to overlay more polar lines on the same axes.",
 "Use `set(h, ...)` on the returned line handle to update line properties after plotting."] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "polarplot is a rendering sink. Real GPU inputs are gathered before ",

    "polar-to-cartesian conversion; the resulting line uses the normal plotting ",

    "renderer.",

)] },

BuiltinDocumentationSection { heading: "Numeric provider precision frontier",
 paragraphs: &[concat!(
    "polarplot declares no direct numeric provider precision list; host ",

    "generation/sampling or owner-aware gather and installed graphics rendering are ",

    "separate boundaries.",

)] },

BuiltinDocumentationSection { heading: "Coordinate broadcasting",
 paragraphs: &["none"] },

BuiltinDocumentationSection { heading: "Runtime contract and boundaries",
 paragraphs: &[concat!(
    "Catalog default bindings retain the actual RuntimeContext and original rooted ",

    "BuiltinArguments across parsing, provider readback, delegated calls and ",

    "completion. Value-selected axes, properties and output counts remain Runtime ",

    "decisions. Constructor results are host double graphics handles, not numerical ",

    "GPU arrays; hist instead returns double counts/centres, plotyy up to three ",

    "distinct handle outputs, and plotmatrix up to five.",

),
 concat!(
    "Rendering is a fusion boundary and requires the installed graphics host/device. ",

    "A direct compatible renderer route is not an elementwise or reduction provider ",

    "kernel. Resident buffers retain their owning provider and existing ",

    "export/error/completion protocol; an export, readback, allocation, cancellation ",

    "or emission failure is an error, not permission to replay an attempted ",

    "operation. Source data and renderer caches have distinct lifetimes. No new ",

    "distributed chart implementation is declared.",

),
 concat!(
    "The existing canonical CoordinateWork allocation policy refuses Managed calls ",

    "before unowned chart scratch, metadata, readback or scene mutation. It checks ",

    "captured cancellation generation and policy before publication and on ",

    "completion; ingress failures are Never replay. This is an explicit missing ",

    "finite owner, not an allocation estimate, algorithm completion or L25 closure. ",

    "Provider-private allocations, renderer lifetime accounting, synchronous long ",

    "geometry/copy cooperation, measured throughput and physical presentation remain ",

    "independent qualification gates.",

),
 concat!(
    "Native Single/Double/integer source ownership is distinguished from explicit ",

    "binary64/f32 graphics conversion. The renderer does not promise exact ",

    "wide-integer pixel positions. Logical, complex, sparse, scalar, vector, matrix, ",

    "empty and higher-rank forms follow the identity-specific parser rather than a ",

    "universal implicit cast. Sparse/complex rejection is not an integer ",

    "not-applicable audit. Full MATLAB properties, browser/native parity and ",

    "empty-shape conformance are not inferred from migration or historical test ",

    "references.",

)] },

BuiltinDocumentationSection { heading: "Supported representations and remaining forms",
 paragraphs: &[concat!(
    "Real theta/rho pairs, rho-only implicit angles, complex Z and multiple series ",

    "share the existing polar owner. Implicit theta spans 0..2*pi. Compatible matrix ",

    "columns expand into separate lines, including a matching theta or rho vector ",

    "reused per column. The returned scalar is the first created line handle. ",

    "LineSpec and common line properties, final group name/value application and ",

    "optional full axes target retain their current semantics.",

),
 concat!(
    "Complex scalar/tensor Z becomes angle/magnitude; typed complex integer Z keeps ",

    "its exactness/compatibility extension. Real coordinates and polar conversion ",

    "use floating geometry rather than native polar source storage. Real GPU inputs ",

    "gather through the owning provider. Empty input follows the existing empty-line ",

    "owner, not an invented provider fill; advanced ",

    "ThetaLim/RLim/ThetaDir/ThetaZeroLocation parity remains absent. Cartesian line ",

    "properties, not original Z, are authoritative after conversion.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Plot a polar curve from theta and rho",
 program: "theta = linspace(0, 2*pi, 200);\nrho = abs(sin(2*theta).*cos(2*theta));\nh = polarplot(theta, rho);",
 display_output: Some("% h is a numeric handle for the first polar line"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "observed=get(h,'XData'); expected=rho.*cos(theta); ",

    "assert(numel(observed)==numel(expected)); ",

    "assert(max(abs(observed(:)-expected(:)))<1e-4); observed=get(h,'YData'); ",

    "expected=rho.*sin(theta); assert(numel(observed)==numel(expected)); ",

    "assert(max(abs(observed(:)-expected(:)))<1e-4); ",

) } },

BuiltinExample { id: "retained-2",
 title: "Use complex data",
 program: "z = exp(1i*linspace(0, 2*pi, 100));\npolarplot(z, 'r--');",
 display_output: Some("% theta comes from angle(z), rho comes from abs(z)"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','line'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'XData'); expected=real(z); ",

    "assert(numel(observed)==numel(expected)); ",

    "assert(max(abs(observed(:)-expected(:)))<1e-4); ",

    "observed=get(witness(1),'YData'); expected=imag(z); ",

    "assert(numel(observed)==numel(expected)); ",

    "assert(max(abs(observed(:)-expected(:)))<1e-4); ",

) } },

BuiltinExample { id: "retained-3",
 title: "Overlay multiple polar series",
 program: concat!(
    "theta = linspace(0, 2*pi, 200);\npolarplot(theta, abs(sin(theta)), 'b-');\nhold ",

    "on;\npolarplot(theta, abs(cos(theta)), 'r--', 'LineWidth', 2);",

),
 display_output: Some("% both lines share the same polar axes"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','line'); assert(numel(witness)>=1); ",

    "assert(numel(witness)==2); widths=[get(witness(1),'LineWidth'),get(witness(2),'L",

    "ineWidth')]; assert(any(widths==2));",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "Does polarplot store polar data internally?",
 answer: concat!(
    "The line object stores cartesian x/y coordinates after conversion. The axes ",

    "metadata records that the axes are polar, which controls grid rendering and ",

    "equal aspect behavior.",

) },

BuiltinDocumentationFaq { question: "Are advanced MATLAB polar axes properties supported?",
 answer: concat!(
    "The initial implementation supports the common plotting forms, line styles, and ",

    "polar grid rendering. Advanced axes properties such as `ThetaLim`, `RLim`, ",

    "`ThetaDir`, and `ThetaZeroLocation` are not yet separate builtins/properties.",

) }],

 links: &[BuiltinDocumentationLink { label: "plot",
 target: BuiltinDocumentationLinkTarget::Builtin("plot") },

BuiltinDocumentationLink { label: "hold",
 target: BuiltinDocumentationLinkTarget::Builtin("hold") },

BuiltinDocumentationLink { label: "legend",
 target: BuiltinDocumentationLinkTarget::Builtin("legend") },

BuiltinDocumentationLink { label: "set",
 target: BuiltinDocumentationLinkTarget::Builtin("set") },

BuiltinDocumentationLink { label: "get",
 target: BuiltinDocumentationLinkTarget::Builtin("get") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::polarplot::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "integration",
 location: "crates/runmat-vm/tests/plotting.rs" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
