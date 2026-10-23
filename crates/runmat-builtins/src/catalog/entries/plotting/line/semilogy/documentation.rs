use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("semilogy"),
    slug: Some("semilogy"),
    summary:
        "Plot data with a logarithmic y-axis using MATLAB-compatible `semilogy` call patterns.",
    description: concat!(
        "`semilogy` creates line plots like `plot` and configures the active axes to ",
        "logarithmic scaling on Y. It follows MATLAB-compatible argument forms, axes ",
        "targeting, and handle behavior.",
    ),
    keywords: &[
        "semilogy",
        "logarithmic y-axis",
        "semilog plot",
        "matlab semilogy",
        "exponential plot",
    ],
    related: &["plot", "semilogx", "loglog", "axis"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The returned value is a numeric line handle.",
                "Only the y-axis is switched to log scale; the x-axis remains linear.",
                "Y log-mode state is subplot-local and does not leak into other subplot axes.",
                "Positive y-values are the meaningful domain for a logarithmic y-axis.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Runtime contract and boundaries",
            paragraphs: &[
                concat!(
    "The result is a host graphics handle, not a numeric coordinate array. ",
    "Coordinate parsing, native source storage, series expansion, style validation ",
    "and graphics state use the existing plotting owners. Numeric handle aliases are ",
    "resolved at Runtime, so static inference does not guess that a scalar ",
    "coordinate denotes axes. The line, bar, barh and area constructors may return a ",
    "row of handles; the other identities return the first created handle. More than ",
    "one language output is rejected before chart work.",
),
                concat!(
    "Single and double source data use their native storage where the chart has ",
    "typed source properties; all eight integer classes use the declared ",
    "identity-specific coordinate/control rules. Rendering is an explicit floating ",
    "geometry boundary. In particular errorbar, bar, barh and area convert client ",
    "graphics data to floating geometry; this does not certify exact integer-valued ",
    "rendered pixels. Complex, logical, sparse, empty and matrix admission is ",
    "decided by each existing numerical/shape parser, not by this shared ingress. ",
    "Unsupported representations raise the existing parser diagnostic rather than ",
    "being assumed fully MATLAB compatible.",
),
                concat!(
    "Direct GPU rendering is conditional on the actual owning provider, live ",
    "physical input leases, floating class/storage, supported shape/style, ",
    "compatible shared device exports and native plotting feature availability. A ",
    "pre-effect unavailable direct route may take the existing client path; a failed ",
    "or partially completed export/reduction/emission is terminal and never ",
    "permission to replay on the host. Source-property reads and figure persistence ",
    "can perform later explicit readback. There is no elementwise/reduction fusion, ",
    "distributed execution or portable numeric provider kernel for these graphics ",
    "sinks.",
),
                concat!(
    "User-interface capability and host-callback effects describe graphics ",
    "creation/presentation, not numeric purity. Portable Runtime linking and an ",
    "ordinary execution stack allow Native and Wasm ingress; actual presentation ",
    "requires the installed graphics host. Retained chart data, style metadata, ",
    "renderer buffers and serialization are separate lifetimes. Current chart ",
    "scratch/state has no complete finite allocation owner: Managed policy is ",
    "rejected before argument extraction, gather or chart mutation through the ",
    "existing HostAllocationReceipt authority. Ordinary Unbounded resident and host ",
    "routes remain available. Issuer/policy/cancellation are fenced before ",
    "publication and errors carry Never replay. This refusal, unbounded host loops, ",
    "provider/renderer-private allocations, full property parity and measured ",
    "performance remain open obligations; Incomplete/Partial is not a completion ",
    "certificate.",
),
            ],
        },
        BuiltinDocumentationSection {
            heading: "Supported representations and remaining forms",
            paragraphs: &[
                concat!(
    "Coordinates use real single/double Tensor storage or native integer storage. ",
    "Scalar logical coordinates pass through the existing NumericInput double ",
    "conversion; logical arrays, complex arrays, sparse arrays, categorical and ",
    "datetime coordinate objects do not have a NumericInput representation and are ",
    "not implemented by this chart owner.",
),
                concat!(
    "Y-only calls, explicit X/Y pairs, repeated series, optional leading axes, ",
    "LineSpec and shared line name/value controls delegate to the canonical plot ",
    "owner with the actual RuntimeContext. Implicit indices are one-based doubles. ",
    "Scalar numeric Y is normalized by NumericInput; empty Y produces matching empty ",
    "coordinates. Host matrix expansion preserves native single/double/integer class ",
    "and exact selected values, with no whole-matrix materialization per selected ",
    "row or column. Log scaling changes display coordinates, not the source ",
    "properties. Nonpositive values break logarithmic rendered segments rather than ",
    "changing stored coordinates.",
),
                concat!(
                    "The owning call explicitly sets X linear and Y logarithmic on its selected ",
                    "subplot axes after plotting succeeds. It does not silently log-transform or ",
                    "overwrite XData/YData.",
                ),
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Plot an exponential decay on a logarithmic y-axis",
            program: "t = linspace(0, 10, 200);\nsemilogy(t, exp(-t));",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
    "assert(strcmp(get(gca, 'XScale'), 'linear')); assert(strcmp(get(gca, 'YScale'), ",
    "'log')); witness = findobj(gca, 'Type', 'line'); assert(numel(witness) == 1); ",
    "observed = get(witness, 'YData'); expected = exp(-t); ",
    "assert(isequal(observed(:), expected(:))); ",
),
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Compare multiple decays on the same axes",
            program: concat!(
                "t = linspace(0, 10, 200);\nh1 = semilogy(t, exp(-t));\nset(h1, 'DisplayName', ",
                "'exp(-t)');\nhold on;\nh2 = semilogy(t, exp(-0.3*t));\nset(h2, 'DisplayName', ",
                "'exp(-0.3t)');\nlegend;",
            ),
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
    "assert(strcmp(get(gca, 'XScale'), 'linear')); assert(strcmp(get(gca, 'YScale'), ",
    "'log')); assert(strcmp(get(h1, 'DisplayName'), 'exp(-t)')); ",
    "assert(strcmp(get(h2, 'DisplayName'), 'exp(-0.3t)')); observed = get(h2, ",
    "'YData'); expected = exp(-0.3*t); assert(isequal(observed(:), expected(:))); ",
),
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Inspect the y-axis scale after plotting",
            program: "t = 0:0.1:4;\nsemilogy(t, exp(t));\nax = gca;\nget(ax, 'YScale')",
            display_output: Some("ans =\n    'log'"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
    "assert(strcmp(get(gca, 'XScale'), 'linear')); assert(strcmp(get(gca, 'YScale'), ",
    "'log')); witness = findobj(gca, 'Type', 'line'); assert(numel(witness) == 1); ",
    "observed = get(witness, 'YData'); expected = exp(t); ",
    "assert(isequal(observed(:), expected(:))); ",
),
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "When is semilogy the right choice over a linear plot?",
            answer: concat!(
                "Whenever the interesting structure lives in multiplicative changes along Y — ",
                "exponential decay, growth rates, signal attenuation. On a log y-axis, `exp(-t)` ",
                "becomes a straight line, which makes decay constants easy to read directly from ",
                "the slope.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "How do I visualize exponential decay and read the time constant?",
            answer: concat!(
                "Plot with `semilogy` and the decay appears as a straight line. The slope of ",
                "that line is the decay rate. For `exp(-t/tau)`, the signal drops by a factor of ",
                "`e` every `tau` units along the x-axis.\n\n```matlab\nt = linspace(0, 10, ",
                "200);\nsemilogy(t, exp(-t/3)); % tau = 3\ngrid on;\n```",
            ),
        },
        BuiltinDocumentationFaq {
            question: "What about zero or negative y-values on a log y-axis?",
            answer: concat!(
                "They're excluded from rendering since log of zero or negative numbers is ",
                "undefined. The data handle still stores them, but only positive y-values ",
                "produce visible points. If your data crosses zero, consider splitting into ",
                "positive and negative series or using a linear axis.",
            ),
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "plot",
            target: BuiltinDocumentationLinkTarget::Builtin("plot"),
        },
        BuiltinDocumentationLink {
            label: "semilogx",
            target: BuiltinDocumentationLinkTarget::Builtin("semilogx"),
        },
        BuiltinDocumentationLink {
            label: "loglog",
            target: BuiltinDocumentationLinkTarget::Builtin("loglog"),
        },
        BuiltinDocumentationLink {
            label: "axis",
            target: BuiltinDocumentationLinkTarget::Builtin("axis"),
        },
        BuiltinDocumentationLink {
            label: "Choosing the right plot type",
            target: BuiltinDocumentationLinkTarget::Documentation(
                "/docs/plotting/choosing-the-right-plot-type",
            ),
        },
        BuiltinDocumentationLink {
            label: "Styling plots and axes",
            target: BuiltinDocumentationLinkTarget::Documentation(
                "/docs/plotting/styling-plots-and-axes",
            ),
        },
        BuiltinDocumentationLink {
            label: "Plot replay and export",
            target: BuiltinDocumentationLinkTarget::Documentation(
                "/docs/plotting/plot-replay-and-export",
            ),
        },
        BuiltinDocumentationLink {
            label: "Complete plotting guide",
            target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/plotting/ops/semilogy.rs`",
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",
                "s/plotting/ops/semilogy.rs",
            )),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::plotting::semilogy::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
