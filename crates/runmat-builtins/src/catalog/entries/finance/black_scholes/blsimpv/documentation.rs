use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/finance/black_scholes.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("blsimpv"), slug: Some("blsimpv"),
    summary: "Compute Black-Scholes implied volatility from observed European option prices.",
    description: "`blsimpv(Price,Strike,Rate,Time,Value)` finds annualized volatility that reproduces an observed European option value in the Black-Scholes pricing model.",
    keywords: &["blsimpv", "Black-Scholes", "implied volatility", "option", "finance"],
    related: &["blsprice"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and options", paragraphs: &[
            "The five required numeric inputs allow scalar expansion over matching nonscalar shapes. The result has the expanded shape and is computed in binary64 on the host.",
            "Name-value options are `Limit`, `Yield`, `Tolerance`, `Class`, and `Method`. `Class` selects call or put, accepting logical values (`true` for call), scalar text, string arrays, or cell arrays containing `call`/`put` values. Empty optional values use their defaults.",
            "`Method` accepts `search` and `jackel2016`. The default `jackel2016` option uses the current robust scalar solver without applying caller `Limit`/`Tolerance` overrides; `search` uses the supplied bounds and tolerance. A `search` result is NaN if no volatility exists in `[0, Limit]`.",
            "The public compatibility target documents double numeric inputs. Native-single, fixed-width integer, and logical pricing inputs are RunMat-only extensions; wide integers enter binary64 and may round. This does not assert unverified MATLAB parity for other classes.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Root finding runs on the host. Resident arguments are gathered through their owning provider before solving. There is no direct GPU root-finding hook; this is a fusion boundary. The function does not mutate its arguments.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "call", title: "Recover call implied volatility",
            program: "Volatility = blsimpv(100, 95, 0.10, 0.25, 13.6953); fprintf('Volatility = %.4f\\n', Volatility);",
            display_output: Some("Volatility = 0.5000"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(Volatility)); assert(abs(Volatility-0.5)<1e-3);" },
        },
        BuiltinExample { id: "put-yield", title: "Put implied volatility with yield",
            program: "Volatility = blsimpv(910, 980, 0.02, 0.25, 90.4683, 'Yield', 0.025, 'Class', 'put'); fprintf('Volatility = %.4f\\n', Volatility);",
            display_output: Some("Volatility = 0.2500"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(Volatility)); assert(abs(Volatility-0.25)<1e-3);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What happens when the requested volatility is outside the search interval?", answer: "With `Method='search'`, the result is NaN when no root exists in `[0, Limit]`. The default method does not apply caller `Limit`/`Tolerance` overrides." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "blsprice", target: BuiltinDocumentationLinkTarget::Builtin("blsprice") },
        BuiltinDocumentationLink { label: "MathWorks blsimpv documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/blsimpv.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Class, yield, method, root, broadcast, and host-gather tests", location: "crates/runmat-runtime/src/builtins/finance/black_scholes.rs" }],
        notes: &["The six legacy behavior bullets, both visible examples, source, and related link are retained. Numeric-class parity beyond double is not asserted."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
