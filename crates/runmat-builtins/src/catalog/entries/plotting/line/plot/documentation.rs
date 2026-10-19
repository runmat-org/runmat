use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority:BuiltinDocumentationAuthority::Catalog,

 title:Some("plot"),
  slug:Some("plot"),
  summary:"Create 2-D line plots in MATLAB and RunMat.",

 description:"`plot` creates 2-D line charts from `plot(y)`, `plot(x, y)`, and multi-series call forms. Returned handle behavior and plotting-property integration follow MATLAB semantics.",
  keywords:&["plot",
 "line plot",
 "2-D plotting",
 "plot x y",
 "matlab plot",
 "gpu plotting",
 "runmat plotting"],

 related:&["semilogx",
 "semilogy",
 "loglog",
 "scatter",
 "stairs",
 "legend",
 "get",
 "set"],

 sections:&[BuiltinDocumentationSection { heading:"Behavior",
  paragraphs:&["`plot(y)` uses `1:n` as the implicit x-axis, matching MATLAB shorthand behavior.",
 "`plot(x, y)` requires the same element count in both inputs. Row and column vectors are both accepted.",
 "Multiple data pairs in one call create multiple line objects and return the handle for the first created series, matching common MATLAB plotting conventions.",
 "The returned handle works with `get` and `set`, so line color, width, display names, and marker properties can be inspected or updated after plotting.",
 "Plotting is subplot-local. If the active axes come from `subplot`, the line is attached to that subplot rather than globally mutating every axes in the figure.",
 "RunMat treats `plot` as a rendering sink: fusion stops at the builtin, but GPU residency is preserved on the happy path when the shared plotting device is installed."] },
 BuiltinDocumentationSection { heading:"Options",
  paragraphs:&["Style strings like `'-'`, `'--'`, `':'`, `'o'`, `'x'`, and combined forms such as `'r--o'` are accepted in MATLAB-style plot calls, including per-series forms such as `plot(x1,y1,'r',x2,y2,'b')`.",
 "Name/value property updates can be supplied inline or applied after plotting through `set(h, ...)`, where `h` is the returned line handle. Inline line-spec tokens may appear before or after name/value pairs for a series.",
 "`HandleVisibility` accepts `'on'`, `'off'`, and `'callback'`; `'off'` keeps the line visible but removes it from automatic legend entries.",
 "Display names set during or after plotting are used by `legend` when labels are not passed explicitly."] },
 BuiltinDocumentationSection { heading:"GPU residency",
  paragraphs:&["Plot is a rendering sink. GPU inputs can remain resident when the shared plotting device is available; otherwise RunMat gathers once before rendering.",
 "`plot` preserves GPU residency when the plotting context can consume exported GPU buffers directly. If the active plotting path cannot use the input buffers, RunMat gathers once before rendering while keeping the resulting figure semantics unchanged.",
 "On the GPU happy path, RunMat reuses exported plot input buffers and emits renderer-ready line geometry without a full host round-trip.",
 "Marker-heavy or otherwise unsupported combinations can fall back to host materialization while preserving the same visible result and handle semantics."] },
 BuiltinDocumentationSection { heading:"Execution",
  paragraphs:&["plot is a rendering sink, not an elementwise or reduction fusion primitive. Floating GPU line paths accept single and double data without broadcasting. Native integer coordinate data remain authoritative in line source properties; rendering crosses an explicit floating geometry boundary. Integer resident inputs gather through the existing typed owner because direct WGPU lines are floating-only.",
 "Zero requested outputs still constructs the plot. At most one numeric line handle is published. The first created series supplies that handle; additional series remain accessible through the existing graphics property owners."] }],

 examples:&[BuiltinExample {
 id:"example-1",
  title:"Plot paired x/y data as a basic line chart",
  program:"x = linspace(0, 2*pi, 200);\ny = sin(x);\nh = plot(x, y);",

 display_output:Some("% h is a numeric handle for the first line object"),

 compatibility:BuiltinExampleCompatibility::Matlab,

 harness:BuiltinExampleHarness::BrowserGraphics,
  fixture:BuiltinExampleFixture::None,

 requirements:BuiltinExampleRequirements::NONE,

 verification:BuiltinExampleVerification::Figure { minimum_figures:1,
  assertions:"assert(strcmp(get(h,'Type'),'line')); assert(isequal(get(h,'XData'),x)); assert(isequal(get(h,'YData'),y));" }
},
 BuiltinExample {
 id:"example-2",
  title:"Use MATLAB-style y-only shorthand",
  program:"samples = [1 4 2 5 3];\nplot(samples);",

 display_output:Some("% RunMat uses 1:length(samples) for the x-axis"),

 compatibility:BuiltinExampleCompatibility::Matlab,

 harness:BuiltinExampleHarness::BrowserGraphics,
  fixture:BuiltinExampleFixture::None,

 requirements:BuiltinExampleRequirements::NONE,

 verification:BuiltinExampleVerification::Figure { minimum_figures:1,
  assertions:"children = get(gca(),'Children'); assert(numel(children)==1); assert(isequal(get(children(1),'XData'),1:5)); assert(isequal(get(children(1),'YData'),samples));" }
},
 BuiltinExample {
 id:"example-3",
  title:"Create multiple series and drive the legend from display names",
  program:"x = linspace(0, 2*pi, 200);\nh1 = plot(x, sin(x));\nset(h1, 'DisplayName', 'sin(x)', 'LineWidth', 2);\nhold on;\nh2 = plot(x, cos(x));\nset(h2, 'DisplayName', 'cos(x)');\nlegend;",

 display_output:Some("% The legend uses the line DisplayName properties"),

 compatibility:BuiltinExampleCompatibility::Matlab,

 harness:BuiltinExampleHarness::BrowserGraphics,
  fixture:BuiltinExampleFixture::None,

 requirements:BuiltinExampleRequirements::NONE,

 verification:BuiltinExampleVerification::Figure { minimum_figures:1,
  assertions:"assert(strcmp(get(h1,'DisplayName'),'sin(x)')); assert(strcmp(get(h2,'DisplayName'),'cos(x)')); assert(get(h1,'LineWidth')==2);" }
},
 BuiltinExample {
 id:"example-4",
  title:"Use per-series LineSpec and line properties",
  program:"x = 1:3;\ny = [1 2 3];\nplot(x, y, 'r', x, y + 1, 'b');\nh = plot(x, y, 'LineWidth', 1.2, 'b', 'HandleVisibility', 'off');",

 display_output:Some("% The first call creates red and blue series; h is visible but excluded from automatic legends"),

 compatibility:BuiltinExampleCompatibility::Matlab,

 harness:BuiltinExampleHarness::BrowserGraphics,
  fixture:BuiltinExampleFixture::None,

 requirements:BuiltinExampleRequirements::NONE,

 verification:BuiltinExampleVerification::Figure { minimum_figures:1,
  assertions:"assert(strcmp(get(h,'HandleVisibility'),'off')); assert(get(h,'LineWidth')==1.2);" }
},
 BuiltinExample {
 id:"example-5",
  title:"Inspect and update a plotted line through handles",
  program:"x = 0:0.1:1;\nh = plot(x, x.^2);\nget(h, 'Type')\nset(h, 'Color', 'r', 'Marker', 'o', 'MarkerSize', 6);",

 display_output:Some("ans =\n    'line'"),

 compatibility:BuiltinExampleCompatibility::Matlab,

 harness:BuiltinExampleHarness::BrowserGraphics,
  fixture:BuiltinExampleFixture::None,

 requirements:BuiltinExampleRequirements::NONE,

 verification:BuiltinExampleVerification::Figure { minimum_figures:1,
  assertions:"assert(strcmp(get(h,'Type'),'line')); assert(strcmp(get(h,'Marker'),'o')); assert(get(h,'MarkerSize')==6);" }
},
 BuiltinExample {
 id:"example-6",
  title:"Damped oscillation with envelope",
  program:"t = linspace(0, 4*pi, 500);\nenvelope = exp(-0.3*t);\nsignal = envelope .* sin(5*t);\n\nh1 = plot(t, signal);\nset(h1, 'DisplayName', 'Damped signal', 'LineWidth', 1.5);\nhold on;\nh2 = plot(t, envelope, '--');\nset(h2, 'DisplayName', 'Envelope', 'LineWidth', 1.5, 'Color', [0.85 0.33 0.1]);\nh3 = plot(t, -envelope, '--');\nset(h3, 'DisplayName', '-Envelope', 'LineWidth', 1.5, 'Color', [0.85 0.33 0.1]);\nhold off;\n\ntitle('Damped Oscillation');\nxlabel('Time (s)');\nylabel('Amplitude');\nlegend;\ngrid on;",

 display_output:None,

 compatibility:BuiltinExampleCompatibility::Matlab,

 harness:BuiltinExampleHarness::BrowserGraphics,
  fixture:BuiltinExampleFixture::None,

 requirements:BuiltinExampleRequirements::NONE,

 verification:BuiltinExampleVerification::Figure { minimum_figures:1,
  assertions:"assert(strcmp(get(h1,'DisplayName'),'Damped signal')); assert(strcmp(get(h2,'DisplayName'),'Envelope')); assert(strcmp(get(h3,'DisplayName'),'-Envelope')); assert(isequal(get(h1,'YData'),signal));" }
}],
  example_exemption:None,

 faqs:&[BuiltinDocumentationFaq {question:"How do I plot multiple series on the same axes?",
 answer:"Call `hold on` between `plot` calls. Each subsequent `plot` adds a new line to the current axes instead of replacing the previous one.\n\n```matlab\nx = linspace(0, 2*pi, 200);\nplot(x, sin(x));\nhold on;\nplot(x, cos(x));\nlegend('sin', 'cos');\n```\n\nYou can also pass multiple x/y pairs in a single `plot` call, e.g. `plot(x, sin(x), x, cos(x))`, though you lose per-series handle control that way."},
 BuiltinDocumentationFaq {question:"What line styles and markers does plot accept?",
 answer:"Style strings combine a color code, a line style, and a marker in any order. Line styles: `'-'` (solid), `'--'` (dashed), `':'` (dotted), `'-.'` (dash-dot). Markers: `'o'`, `'x'`, `'+'`, `'*'`, `'s'` (square), `'d'` (diamond), `'^'`, `'v'`. Colors: `'r'`, `'g'`, `'b'`, `'k'`, `'m'`, `'c'`, `'y'`. For example, `plot(x, y, 'r--o')` draws a red dashed line with circle markers."},
 BuiltinDocumentationFaq {question:"When should I use plot vs scatter?",
 answer:"`plot` draws connected lines between points and is the right choice for continuous data or time-series. `scatter` draws unconnected markers and supports per-point size and color encoding through `SizeData` and `CData`. If you just want a line chart, use `plot`. If you need to encode a third variable via marker size or color, use `scatter`."}],

 links:&[BuiltinDocumentationLink {label:"semilogx",
 target:BuiltinDocumentationLinkTarget::Builtin("semilogx")},
 BuiltinDocumentationLink {label:"semilogy",
 target:BuiltinDocumentationLinkTarget::Builtin("semilogy")},
 BuiltinDocumentationLink {label:"loglog",
 target:BuiltinDocumentationLinkTarget::Builtin("loglog")},
 BuiltinDocumentationLink {label:"scatter",
 target:BuiltinDocumentationLinkTarget::Builtin("scatter")},
 BuiltinDocumentationLink {label:"stairs",
 target:BuiltinDocumentationLinkTarget::Builtin("stairs")},
 BuiltinDocumentationLink {label:"legend",
 target:BuiltinDocumentationLinkTarget::Builtin("legend")},
 BuiltinDocumentationLink {label:"get",
 target:BuiltinDocumentationLinkTarget::Builtin("get")},
 BuiltinDocumentationLink {label:"set",
 target:BuiltinDocumentationLinkTarget::Builtin("set")},
 BuiltinDocumentationLink {label:"Choosing the right plot type",
 target:BuiltinDocumentationLinkTarget::External("/docs/plotting/choosing-the-right-plot-type")},
 BuiltinDocumentationLink {label:"Styling plots and axes",
 target:BuiltinDocumentationLinkTarget::External("/docs/plotting/styling-plots-and-axes")},
 BuiltinDocumentationLink {label:"Plot replay and export",
 target:BuiltinDocumentationLinkTarget::External("/docs/plotting/plot-replay-and-export")},
 BuiltinDocumentationLink {label:"Complete plotting guide",
 target:BuiltinDocumentationLinkTarget::External("/blog/matlab-plotting-guide")}],

 media:&[BuiltinDocumentationMedia {kind:BuiltinDocumentationMediaKind::Image,
 url:"https://web.runmatstatic.com/builtin-image/runmat-matlab-plot-damped-oscillation.webp",
 alt:"Damped oscillation with envelope"}],

 evidence:BuiltinDocumentationEvidence {
 implementation:&[BuiltinDocumentationLink {label:"`crates/runmat-runtime/src/builtins/plotting/ops/plot.rs`",
 target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/plot.rs")}],

 verification:&[
 BuiltinEvidenceReference {kind:BuiltinEvidenceKind::UnitTest,
 label:"Plot runtime owner",
 location:"builtins::plotting::plot::tests"},

 BuiltinEvidenceReference {kind:BuiltinEvidenceKind::IntegrationTest,
 label:"Renderer integration",
 location:"runmat-plot/tests/renderer_tests.rs"}
 ],

 notes:&["Original six programs and embedded FAQ program retained byte-for-byte. Source declarations name existing test owners, not execution receipts. Canonical placement and fusion declarations replace obsolete Runtime metadata registrations; executable shared-device buffer export and line construction remain at the existing Runtime implementation owner."]
 },

 introduced:None,
 status:Some(BuiltinDocumentationStatus::Stable)
};
