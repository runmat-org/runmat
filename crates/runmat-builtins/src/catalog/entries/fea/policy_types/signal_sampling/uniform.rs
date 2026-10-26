use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalSampling.uniform",
    summary: "Sample a generated signal at evenly spaced points including both endpoints.",
    description: concat!(
        "`fea.SignalSampling.uniform(start, end, sampleCount)` returns an ",
        "`fea.SignalSampling` that evaluates the signal at `sampleCount` evenly spaced ",
        "coordinates from `start` to `end`, inclusive. The spacing is `(end - start) / ",
        "(sampleCount - 1)`.",
    ),
    keywords: &["fea", "signal", "sampling", "uniform", "generator"],
    related: &["fea.SignalSampling.canonicalPoints", "fea.SignalSampling.adaptive", "fea.SignalDomain.time", "fea.ForceSignal.fromFunction", "fea.PressureSignal.fromFunction", "fea.TemperatureSignal.fromFunction"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`start` and `end` are finite real scalars with `start < end`; `sampleCount` is a ",
                    "nonnegative integer value of at least 2. Integer classes are accepted for all three:",
                    " coordinates are converted once to binary64 and the count stays exact. The stored ",
                    "policy is `sampling.uniform` with fields `start`, `end`, and `sample_count`.",
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
            "half-second-101",
            "Sample 0 to 0.5 s every 5 ms",
            "sampling = fea.SignalSampling.uniform(0, 0.5, 101);",
            concat!(
                "assert(strcmp(class(sampling), 'fea.SignalSampling'));\n",
                "assert(sampling.sampling.uniform.sample_count == 101);\n",
                "assert(sampling.sampling.uniform.start == 0);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Does sampleCount count intervals or points?",
            "Points. `uniform(0, 1, 11)` samples 0, 0.1, ..., 1.0.",
        ),
        faq(
            "Can start equal end?",
            concat!(
                "No. The interval must have positive length; use a constant signal for a single ",
                "value.",
            ),
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
    FEA_SIGNAL_SAMPLING_UNIFORM_CATALOG_ENTRY,
    name: "fea.SignalSampling.uniform",
    result: classes::SIGNAL_SAMPLING,
    signature: "sampling = fea.SignalSampling.uniform(start, end, sampleCount)",
    inputs: [
        (
            "start",
            NumericScalar,
            "Finite first sample coordinate in the domain's unit.",
            PolicyArg::RealScalar,
        ),
        (
            "end",
            NumericScalar,
            "Finite last sample coordinate; must be greater than start.",
            PolicyArg::RealScalar,
        ),
        (
            "sampleCount",
            IntegerScalar,
            "Number of samples including both endpoints; at least 2.",
            PolicyArg::Count,
        ),
    ],
    output: ("sampling", "Typed fea.SignalSampling."),
    integer_capabilities: &integer::UNIFORM_SAMPLING_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
