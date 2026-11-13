use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalSampling.canonicalPoints",
    summary: "Sample a generated signal at an explicit list of coordinates.",
    description: concat!(
        "`fea.SignalSampling.canonicalPoints(points)` returns an `fea.SignalSampling` that ",
        "evaluates the signal exactly at the coordinates in `points`. Use it to place samples",
        " densely around fast transients and sparsely elsewhere.",
    ),
    keywords: &["fea", "signal", "sampling", "points", "nonuniform"],
    related: &["fea.SignalSampling.uniform", "fea.SignalSampling.adaptive", "fea.ForceSignal.fromFunction", "fea.PressureSignal.fromFunction", "fea.TemperatureSignal.fromFunction"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`points` must be a nonempty real row or column vector with at least two finite, ",
                    "strictly increasing values. Duplicates or decreasing values fail with ",
                    "`RunMat:fea:InvalidInput`. Integer arrays are accepted and converted once to ",
                    "binary64. The stored policy is `sampling.canonical_points`, a row vector.",
                ),
                super::SAMPLING_ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::SAMPLING_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "impact-points",
            "Sample densely around an impact at 10 ms",
            "sampling = fea.SignalSampling.canonicalPoints([0 0.009 0.01 0.011 0.02 0.1]);",
            concat!(
                "assert(strcmp(class(sampling), 'fea.SignalSampling'));\n",
                "assert(numel(sampling.sampling.canonical_points) == 6);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Must the points be uniformly spaced?",
            "No. Any strictly increasing sequence of at least two finite coordinates is accepted.",
        ),
    ],
    links: &[
        page("RunMat function integration", "/docs/simulation/language/runmat-function-integration"),
        page("Run a transient study with a generated input", "/docs/simulation/language/guides/transient-study-with-generated-inputs"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/generator_policies.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/signal_sampling/tests.rs::signal_sampling_identities_conform",
        ),
        unit_test(
            "Generator policies are typed, portable, and bounded",
            "crates/runmat-runtime/src/builtins/fea/generator_policies.rs::generator_policies_are_typed_portable_and_validate_bounds",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SIGNAL_SAMPLING_CANONICAL_POINTS_CATALOG_ENTRY,
    name: "fea.SignalSampling.canonicalPoints",
    result: classes::SIGNAL_SAMPLING,
    signature: "sampling = fea.SignalSampling.canonicalPoints(points)",
    inputs: [
        (
            "points",
            NumericArray,
            "Strictly increasing finite sample coordinates in the domain's unit; at least two.",
            PolicyArg::RealVector,
        ),
    ],
    output: ("sampling", "Typed fea.SignalSampling."),
    integer_capabilities: &integer::POINT_SAMPLING_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
