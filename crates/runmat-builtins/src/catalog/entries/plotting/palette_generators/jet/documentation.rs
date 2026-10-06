use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "jet() returns a normalized host double RGB matrix. Its row count is the active axes colormap length, or 256 when no active axes length exists.",
            "jet(m) returns exactly m rows and three columns. m accepts a real integral single or double scalar or any of the eight exact host integer classes, including zero; negative, fractional, nonfinite, nonscalar and over-limit values reject.",
            "Generating an RGB matrix does not change a figure or axes palette. Pass the result to colormap to update graphics state.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and placement",
        paragraphs: &[
            "Length controls must already be host values. Resident device or remote controls reject without downloading or invoking a numerical provider.",
            "RGB generation uses the existing host ColorMap sampling algorithm and produces a dense double m-by-3 result. There is no array-provider kernel or fusion path; the result can be transferred by a later, separately admitted operation.",
            "The zero-argument form reads the current graphics colormap length but does not create or mutate graphics state. Explicit-length generation is deterministic for a given m.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "matrix",
        title: "Generate an RGB matrix",
        program: "c = jet(8);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(c),[8 3])); assert(all(c(:)>=0)); assert(all(c(:)<=1));",
        },
    },
    BuiltinExample {
        id: "graphics",
        title: "Apply the palette to a figure",
        program: "imagesc([1 2; 3 4]);\ncolormap(jet(8));\ncolorbar;",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(isequal(size(colormap()),[8 3]));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("jet"),
    slug: Some("jet"),
    summary: "Return the jet RGB palette as an m-by-3 host double array.",
    description: "The named jet palette generator samples RunMat's existing graphics palette into a normalized RGB matrix. It does not change graphics state until passed to colormap.",
    keywords: &["jet", "colormap", "RGB palette", "plotting"],
    related: &["colormap", "colorbar"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What size is jet()?",
            answer: "It uses the current axes colormap length when available, otherwise 256 rows, and always has three RGB columns.",
        },
        BuiltinDocumentationFaq {
            question: "Can an integer specify the palette length?",
            answer: "Yes. All eight host integer classes are accepted exactly as structural row counts within the existing allocation guard; the output is double.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "colormap",
            target: BuiltinDocumentationLinkTarget::Builtin("colormap"),
        },
        BuiltinDocumentationLink {
            label: "colorbar",
            target: BuiltinDocumentationLinkTarget::Builtin("colorbar"),
        },
        BuiltinDocumentationLink {
            label: "Palette implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-plot/src/plots/surface.rs"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Length admission and host RGB materialization",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/colormap_arrays.rs"),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Named palette matrix, range, length and graphics-state tests",
                location: "builtins::plotting::colormap_arrays::tests",
            },
        ],
        notes: &[
            "Interpreter, JIT and AOT share the runtime binding. Source tests do not prove compiled or browser presentation execution; the BrowserGraphics example requires an actual graphics host.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
