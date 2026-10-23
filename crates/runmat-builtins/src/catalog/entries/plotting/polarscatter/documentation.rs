use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,

 title: Some("polarscatter"),
 slug: Some("polarscatter"),
 summary: "Create scatter charts in polar coordinates.",
 description: concat!(
    "`polarscatter` converts angular/radial point data to cartesian scatter ",

    "coordinates, marks the active axes as polar, and preserves the original ",

    "`ThetaData` and `RData` on returned scatter handles.",

),

 keywords: &["polarscatter",
 "polar scatter",
 "polar coordinates",
 "theta rho",
 "matlab polarscatter",
 "marker plot"],
 related: &["scatter",
 "polarplot",
 "hold",
 "get",
 "set"],

 sections: &[BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[concat!(
    "All eight integer classes are accepted for coordinates, marker sizes, and ",

    "numeric color data before the explicit floating renderer boundary; integer ",

    "table-variable selectors remain structural.",

),
 concat!(
    "`polarscatter(theta, rho)` interprets theta values as radians and requires ",

    "theta and rho to have compatible element counts.",

),
 concat!(
    "Vector theta with matrix rho, matrix theta with vector rho, and matching ",

    "theta/rho matrices create one scatter object per matrix column.",

),
 concat!(
    "`polarscatter(theta, rho, sz)` accepts scalar, vector, or matrix marker-size ",

    "data. Matrix sizes are sliced by series when theta/rho expands to multiple ",

    "scatter objects.",

),
 concat!(
    "`polarscatter(theta, rho, sz, c)` accepts the same positional color forms as ",

    "`scatter`, including color names, RGB triplets, scalar color values, and RGB ",

    "matrices.",

),
 concat!(
    "Marker strings, the `filled` flag, and common scatter name/value options share ",

    "the parser used by `scatter`.",

),
 concat!(
    "The returned value is a scalar scatter handle for one series or a row-vector of ",

    "scatter handles for multiple series.",

),
 concat!(
    "`get(h, 'ThetaData')` and `get(h, 'RData')` return the stored polar source ",

    "coordinates. Setting either property recomputes the cartesian `XData` and ",

    "`YData` used by the renderer.",

),
 concat!(
    "The target axes are switched to polar mode with equal x/y scaling and symmetric ",

    "cartesian limits based on the largest radius.",

)] },

BuiltinDocumentationSection { heading: "Options",
 paragraphs: &["Use `polarscatter(ax, theta, rho, ...)` to target existing axes.",
 "Use marker symbols such as `'o'`, `'+'`, and `'x'` after the size/color arguments.",
 concat!(
    "Use `hold on` before additional `polarscatter` calls to overlay more polar ",

    "scatter objects on the same axes.",

),
 "Use `set(h, ...)` on returned handles to update scatter properties after plotting."] },

BuiltinDocumentationSection { heading: "Rendering and residency",
 paragraphs: &[concat!(
    "polarscatter is a rendering sink. Real GPU inputs are gathered before ",

    "polar-to-cartesian conversion; the resulting scatter objects use the normal ",

    "plotting renderer.",

)] },

BuiltinDocumentationSection { heading: "Numeric provider precision frontier",
 paragraphs: &[concat!(
    "polarscatter declares no direct numeric provider precision list; host ",

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
    "Theta/rho vectors or compatible vector/matrix columns, optional marker ",

    "sizes/colors, shared scatter markers/filled/styles and optional full axes ",

    "target produce one scatter handle or a row of handles. Empty coordinate series ",

    "reject. Matching scalar/vector/matrix marker-size and numeric color columns are ",

    "selected per series through the borrowed native selector; no whole input ",

    "payload is materialized per column and invalid selection returns a typed error ",

    "rather than expect panic.",

),
 concat!(
    "Polar source ThetaData/RData is stored as the existing floating vectors; ",

    "setting them recomputes Cartesian display coordinates. Native integer ",

    "coordinates and sizes cross the documented renderer boundary rather than ",

    "promising exact wide-integer polar storage. SizeData is reconstructed from ",

    "floating marker diameter and therefore uses a numerical tolerance in examples. ",

    "Color-name/table-variable/full MATLAB scatter frontiers are not silently ",

    "certified by this numeric parser. GPU coordinates use owning-provider host ",

    "polar conversion; there is no native provider polar-scatter arithmetic kernel.",

)] }],

 examples: &[BuiltinExample { id: "retained-1",
 title: "Create a polar scatter chart",
 program: "theta = linspace(0, 2*pi, 20);\nrho = abs(sin(theta));\nh = polarscatter(theta, rho);",
 display_output: Some("% h is a scatter handle with ThetaData and RData"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "observed=get(h,'ThetaData'); expected=theta; ",

    "assert(isequal(observed(:),expected(:))); observed=get(h,'RData'); ",

    "expected=rho; assert(isequal(observed(:),expected(:))); ",

) } },

BuiltinExample { id: "retained-2",
 title: "Use filled markers with sizes and colors",
 program: concat!(
    "theta = pi/4:pi/4:2*pi;\nrho = [19 6 12 18 16 11 15 15];\nsz = 100*[6 15 20 3 15 ",

    "3 6 40];\nc = [1 2 2 2 1 1 2 1];\npolarscatter(theta, rho, sz, c, 'filled');",

),
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "witness=findobj(gca,'Type','scatter'); assert(numel(witness)>=1); ",

    "observed=get(witness(1),'ThetaData'); expected=theta; ",

    "assert(isequal(observed(:),expected(:))); observed=get(witness(1),'RData'); ",

    "expected=rho; assert(isequal(observed(:),expected(:))); ",

    "observed=get(witness(1),'SizeData'); expected=sz; ",

    "assert(numel(observed)==numel(expected)); ",

    "assert(max(abs(observed(:)-expected(:)))<1e-4); ",

) } },

BuiltinExample { id: "retained-3",
 title: "Create multiple scatter series from a matrix",
 program: concat!(
    "theta = [0 pi/6 pi/2];\nrho = [1 2; 2 4; 5 6];\nps = polarscatter(theta, rho, [50 ",

    "100; 75 125; 100 150]);",

),
 display_output: Some("% ps is a row-vector with one handle for each rho column"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::BrowserGraphics,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Figure { minimum_figures: 1,
 assertions: concat!(
    "assert(isequal(size(ps),[1 2])); observed=get(ps(1),'RData'); ",

    "expected=rho(:,1); assert(isequal(observed(:),expected(:))); ",

    "observed=get(ps(2),'RData'); expected=rho(:,2); ",

    "assert(isequal(observed(:),expected(:))); observed=get(ps(2),'SizeData'); ",

    "expected=[100;125;150]; assert(numel(observed)==numel(expected)); ",

    "assert(max(abs(observed(:)-expected(:)))<1e-4); ",

) } }],

 example_exemption: None,

 faqs: &[BuiltinDocumentationFaq { question: "Does polarscatter keep the original polar data?",
 answer: concat!(
    "Yes. The renderer uses cartesian `XData` and `YData`, but the scatter handle ",

    "also stores `ThetaData` and `RData` for MATLAB-style inspection and updates.",

) },

BuiltinDocumentationFaq { question: "Are table-based polarscatter forms supported?",
 answer: concat!(
    "The numeric and axes-targeted plotting forms are supported. Table variable ",

    "forms depend on the broader table plotting surface and are not treated as ",

    "complete by this builtin alone.",

) }],

 links: &[BuiltinDocumentationLink { label: "scatter",
 target: BuiltinDocumentationLinkTarget::Builtin("scatter") },

BuiltinDocumentationLink { label: "polarplot",
 target: BuiltinDocumentationLinkTarget::Builtin("polarplot") },

BuiltinDocumentationLink { label: "hold",
 target: BuiltinDocumentationLinkTarget::Builtin("hold") },

BuiltinDocumentationLink { label: "get",
 target: BuiltinDocumentationLinkTarget::Builtin("get") },

BuiltinDocumentationLink { label: "set",
 target: BuiltinDocumentationLinkTarget::Builtin("set") }],

 media: &[],

 evidence: BuiltinDocumentationEvidence { implementation: &[],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "unit",
 location: "builtins::plotting::polarscatter::tests" },

BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "integration",
 location: "crates/runmat-vm/tests/plotting.rs" }],
 notes: &[] },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),

};
