use crate::*;
mod sections;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("groot"),
    summary: "Return the root graphics object handle.",
    description: "`groot` returns the MATLAB-compatible root graphics object handle, numeric handle `0`. Root properties expose global plotting state such as the current figure, open figure children, screen metadata, units, and show-hidden-handles mode.",
    slug: Some("groot"),
    keywords: &[
        "groot",
        "root graphics object",
        "graphics root",
        "plotting",
    ],
    related: &[
        "figure",
        "gcf",
        "get",
        "set",
    ],
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is `groot` the same value as MATLAB's root object?",
            answer: "RunMat currently returns the opaque numeric root identity zero. MATLAB returns a graphics root object; numeric encoding is a graphics-representation frontier, not a claim of identical public value class.",
        },
        BuiltinDocumentationFaq {
            question: "Does setting root defaults restyle all future graphics objects?",
            answer: "RunMat stores and returns common scalar, text, string-array, and numeric-array `default...` root properties for compatibility. Full automatic application of every MATLAB root default is handled incrementally as each graphics property area is implemented.",
        },
        BuiltinDocumentationFaq {
            question: "Does `groot` gather GPU arrays?",
            answer: "No. `groot()` only returns constant host-double zero. Separate get/set operations inspect or mutate root graphics metadata.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "figure", target: BuiltinDocumentationLinkTarget::Builtin("figure") },
        BuiltinDocumentationLink { label: "gcf", target: BuiltinDocumentationLinkTarget::Builtin("gcf") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/groot.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/groot.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::plotting::groot::tests", location: "builtins::plotting::groot::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "runmat-vm plotting::groot_supports_root_handle_properties_and_dot_access", location: "runmat-vm plotting::groot_supports_root_handle_properties_and_dot_access" },
        ],
        notes: &[
            "Source-only migration: listed preexisting test locations and independently asserted examples are not claims of fresh execution. BrowserGraphics needs actual graphics-host qualification; headless semantic tests do not prove Desktop presentation.",
            "Legacy GPU/fusion hints are normalized to the canonical typed host/fusion-boundary contract. No optional feature gate is added; no provider precision, native OpenGL availability, or finite allocation completion is inferred from prose.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
