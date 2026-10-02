//! Public least-squares-line behavior, source links, and visible examples.
use crate::*;

const OWNER: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Least-squares line runtime",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/lsline.rs"),
};
const DOMAIN: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Shared reference-line graphics domain",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/reference_lines/mod.rs"),
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lsline"), slug: Some("lsline"),
    summary: "Add least-squares fit lines to scatter-style plots.",
    description: "`lsline` appends least-squares fit lines to eligible scatter-style series in the current or specified axes. It returns the created line handle or handles when requested.",
    keywords: &["lsline", "least squares", "fit line", "statistics", "plotting"],
    related: &["refline", "scatter", "plot"],
    sections: &[
        BuiltinDocumentationSection { heading: "Eligible data and fit", paragraphs: &[
            "`lsline()` scans the current axes; `lsline(ax)` targets only that axes. It fits each visible scatter plot and marker-only line plot separately, ignoring connected lines, surfaces, bars, and other graphics objects.",
            "Non-finite X/Y pairs are skipped. A vertical or constant-X eligible series produces a horizontal line at the mean Y value. No eligible data yields an empty 0-by-0 handle array.",
        ] },
        BuiltinDocumentationSection { heading: "Graphics handles and execution", paragraphs: &[
            "The returned line handles support `get` and `set` for `XData`, `YData`, `Color`, `LineStyle`, `LineWidth`, and `DisplayName`.",
            "This is host graphics work, not a numeric GPU kernel. It reads existing graphics objects, creates line objects, and may notify graphics observers in a desktop or browser host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "lsline-scatter", title: "Fit a scatter plot",
            program: "x = [1 2 3 4]; y = [1.2 1.9 3.1 3.8]; scatter(x,y); h = lsline;",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isgraphics(h)); assert(isequal(get(h,'XData'),[1 4])); fit_y = get(h,'YData'); assert(numel(fit_y)==2); assert(all(abs(fit_y-[1.15 3.85])<1e-10));" },
        },
        BuiltinExample { id: "lsline-target-axes", title: "Target a specific axes",
            program: "ax = subplot(1,2,2); scatter(ax,[1 2 3],[2 4 6]); h = lsline(ax);",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isgraphics(h)); assert(isequal(get(h,'YData'),[2 6]));" },
        },
        BuiltinExample { id: "lsline-inspect-data", title: "Inspect the generated line",
            program: "scatter([1 2 3],[2 4 6]); h = lsline; get(h,'YData')",
            display_output: Some("ans =\n     2     6"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(get(h,'YData'),[2 6]));" },
        },
        BuiltinExample { id: "lsline-invalid-arguments", title: "Reject non-axes extra arguments",
            program: "lsline(1,2)",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:lsline:InvalidArgument" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is lsline related to refline?", answer: "`lsline` fits eligible scatter-style plots. `refline` does the same without coefficients and can also draw a supplied slope and intercept." },
        BuiltinDocumentationFaq { question: "Does lsline fit every plot in the axes?", answer: "No. It fits scatter and marker-only line plots, not connected lines, surfaces, bars, or other graphics objects." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "refline", target: BuiltinDocumentationLinkTarget::Builtin("refline") },
        BuiltinDocumentationLink { label: "scatter", target: BuiltinDocumentationLinkTarget::Builtin("scatter") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        OWNER, DOMAIN,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[OWNER, DOMAIN],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Fitted line geometry, axes targeting and diagnostics", location: "crates/runmat-runtime/src/builtins/stats/summary/lsline.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Statistics and graphics VM integration", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &["The fit scans current host graphics state; no resident numeric computation is claimed."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
