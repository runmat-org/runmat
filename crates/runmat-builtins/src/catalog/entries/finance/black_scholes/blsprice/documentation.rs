use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/finance/black_scholes.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("blsprice"),
    slug: Some("blsprice"),
    summary: "Compute Black-Scholes European call and put option prices.",
    description: "`blsprice(Price,Strike,Rate,Time,Volatility)` prices European call and put options under the Black-Scholes model. Optional `Yield` is the continuously compounded dividend or foreign risk-free yield and defaults to zero.",
    keywords: &["blsprice", "Black-Scholes", "option", "call", "put", "finance"],
    related: &["blsimpv"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs and outputs",
            paragraphs: &[
                "With one output, `blsprice` returns the call price; `[Call, Put]` requests both. A zero-output request produces no output. More than two outputs are rejected before pricing or provider work.",
                "The five required inputs and optional `Yield` can be scalars or arrays with MATLAB-compatible implicit-expansion dimensions. Empty `Yield` selects zero. Invalid financial domains and NaN inputs produce elementwise NaN rather than aborting the entire array calculation.",
                "Double inputs produce double call and put prices. Single inputs produce single prices unless a nonscalar double array is also present; scalar double parameters, whether represented as scalar values or 1-by-1 arrays, can accompany single inputs. With no single input the result is double. This input-class rule is independent of provider choice and is RunMat behavior, not a claim about MATLAB's undocumented output-class combinations.",
                "Single-result pricing rounds inputs to single at the calculation boundary. A finite double scalar that would overflow to infinity or underflow to zero in single is rejected before provider work. Fixed-width integer and logical pricing inputs use the double route; a wide integer may round when it enters binary64. The public compatibility target documents double inputs. Single, fixed-width integer, and logical pricing inputs are RunMat-only extensions, admitted before provider work.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU and execution",
            paragraphs: &[
                "Resident real gpuArray inputs can use the owning provider's `black_scholes_price` hook. Host scalar or tensor arguments in a mixed call become temporary inputs for that same provider. A provider without the hook uses the host pricing implementation. Resident results from a provider are checked for owner, shape, and physical precision before they are returned.",
                "The provider hook accepts physical single or double inputs only when its precision matches the input-derived result class. A mixed call with a nonscalar double array cannot silently narrow to a single provider; it falls back to host double pricing if that provider cannot execute double. Single is available only in RunMat compatibility mode. This is a custom provider operation, not a fused elementwise kernel. The function can suspend for fallback gathers and does not mutate its arguments.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "stock-option", title: "European stock option",
            program: "[Call, Put] = blsprice(100, 95, 0.10, 0.25, 0.50); fprintf('Call = %.4f, Put = %.4f\\n', Call, Put);",
            display_output: Some("Call = 13.6953, Put = 6.3497"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(Call) && isscalar(Put)); assert(abs(Call-13.6953)<5e-4); assert(abs(Put-6.3497)<5e-4);" },
        },
        BuiltinExample {
            id: "index-yield", title: "Stock index option with yield",
            program: "[Call, Put] = blsprice(910, 980, 0.02, 0.25, 0.25, 0.025); fprintf('Call = %.4f, Put = %.4f\\n', Call, Put);",
            display_output: Some("Call = 19.6863, Put = 90.4683"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(Call) && isscalar(Put)); assert(abs(Call-19.6863)<5e-4); assert(abs(Put-90.4683)<5e-4);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can a GPU-resident call remain resident?", answer: "Yes when the input owner implements the pricing hook and returns valid owned outputs. Otherwise the operation falls back to the host implementation; there is no generic GPU pricing kernel." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "blsimpv", target: BuiltinDocumentationLinkTarget::Builtin("blsimpv") },
        BuiltinDocumentationLink { label: "MathWorks blsprice documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/finance/blsprice.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pricing, broadcasts, outputs, and resident provider ownership", location: "crates/runmat-runtime/src/builtins/finance/black_scholes.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Provider hook, temporary uploads, and output validation", location: "crates/runmat-runtime/src/builtins/finance/black_scholes/resident.rs" },
        ],
        notes: &["The previous reference's source, related link, provider behavior, pricing examples, and all four behavior bullets are retained. Native/browser/provider qualification remains an integrated gate."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
