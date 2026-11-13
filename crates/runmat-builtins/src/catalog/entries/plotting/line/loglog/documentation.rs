use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("loglog"),
    slug: Some("loglog"),
    summary: "Plot data on logarithmic x and y axes in MATLAB and RunMat.",
    description: concat!(
        "`loglog(x, y, ...)` creates line plots and sets both axes to logarithmic scale. ",
        "Argument handling, returned handles, and axis-scaling behavior follow MATLAB ",
        "semantics.",
    ),
    keywords: &[
        "loglog",
        "log-log plot",
        "power law plot",
        "matlab loglog",
        "scaling analysis",
    ],
    related: &["plot", "semilogx", "semilogy", "axis"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The returned value is a numeric line handle.",
                "Both x and y axes are switched to log scale on the active axes.",
                "Axes scale state remains subplot-local.",
                "Logarithmic axes are meaningful for positive-domain data on both x and y.",
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
    "The owning call explicitly sets both axes logarithmic on its selected subplot ",
    "axes after plotting succeeds. Each neighboring subplot retains its own scale ",
    "state.",
),
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Plot a power-law curve on logarithmic axes",
            program: "x = logspace(0, 3, 200);\nloglog(x, x.^(-1.5));",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
    "assert(strcmp(get(gca, 'XScale'), 'log')); assert(strcmp(get(gca, 'YScale'), ",
    "'log')); witness = findobj(gca, 'Type', 'line'); assert(numel(witness) == 1); ",
    "observed = get(witness, 'YData'); expected = x.^(-1.5); ",
    "assert(isequal(observed(:), expected(:))); ",
),
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Compare multiple scaling laws",
            program: concat!(
                "x = logspace(0, 3, 200);\nh1 = loglog(x, x.^(-1));\nset(h1, 'DisplayName', ",
                "'x^{-1}');\nhold on;\nh2 = loglog(x, x.^(-2));\nset(h2, 'DisplayName', ",
                "'x^{-2}');\nlegend;",
            ),
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
    "assert(strcmp(get(gca, 'XScale'), 'log')); assert(strcmp(get(gca, 'YScale'), ",
    "'log')); assert(strcmp(get(h1, 'DisplayName'), 'x^{-1}')); ",
    "assert(strcmp(get(h2, 'DisplayName'), 'x^{-2}')); observed = get(h2, 'YData'); ",
    "expected = x.^(-2); assert(isequal(observed(:), expected(:))); ",
),
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Inspect both axes after plotting",
            program: concat!(
                "x = logspace(0, 2, 50);\nloglog(x, x.^2);\nax = gca;\nxscale = get(ax, ",
                "'XScale');\nyscale = get(ax, 'YScale');",
            ),
            display_output: Some("% Both axes report 'log'"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: concat!(
                    "assert(strcmp(xscale, 'log')); assert(strcmp(yscale, 'log')); witness = ",
                    "findobj(gca, 'Type', 'line'); assert(numel(witness) == 1); observed = ",
                    "get(witness, 'YData'); expected = x.^2; assert(isequal(observed(:), ",
                    "expected(:))); ",
                ),
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Why do power-law relationships appear as straight lines on a log-log plot?",
            answer: concat!(
                "Taking the log of both sides of `y = a * x^b` gives `log(y) = log(a) + b * ",
                "log(x)`, which is linear in log-space. The slope of the line equals the ",
                "exponent `b`, and the y-intercept gives `log(a)`. That's why log-log is the ",
                "standard way to identify and compare power laws.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "How is loglog different from semilogx or semilogy?",
            answer: concat!(
                "`semilogx` only logs the x-axis (linear y), `semilogy` only logs the y-axis ",
                "(linear x), and `loglog` logs both. Use `loglog` when both dimensions span ",
                "orders of magnitude — spectrum plots, scaling benchmarks, fractal dimension ",
                "analysis.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Can I mix loglog with other scale modes in a subplot layout?",
            answer: concat!(
                "Yes. Axes scale state is subplot-local, so `loglog` in one panel doesn't affect ",
                "neighbors.\n\n```matlab\nsubplot(1, 2, 1);\nloglog(x, x.^(-2));\nsubplot(1, 2, ",
                "2);\nplot(x, x.^(-2)); % linear axes here\n```",
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
            label: "semilogy",
            target: BuiltinDocumentationLinkTarget::Builtin("semilogy"),
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
            label: "`crates/runmat-runtime/src/builtins/plotting/ops/loglog.rs`",
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtin",
                "s/plotting/ops/loglog.rs",
            )),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::plotting::loglog::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
