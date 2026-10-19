use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SignalSampling.adaptive",
    summary: "Declare error-bounded adaptive sampling of a generated signal over an interval.",
    description: concat!(
        "`fea.SignalSampling.adaptive(start, end, maxAbsoluteError, maxSamples)` returns an ",
        "`fea.SignalSampling` that asks for samples on `[start, end]` placed until linear ",
        "interpolation error is at most `maxAbsoluteError`, using no more than `maxSamples` ",
        "samples.",
    ),
    keywords: &["fea", "signal", "sampling", "adaptive", "error bound"],
    related: &["fea.SignalSampling.uniform", "fea.SignalSampling.canonicalPoints", "fea.ForceSignal.fromFunction", "fea.PressureSignal.fromFunction", "fea.TemperatureSignal.fromFunction"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`start < end` must be finite, `maxAbsoluteError` must be finite and positive, and ",
                    "`maxSamples` must be an integer value of at least 2. The error target is in the ",
                    "canonical SI unit of the signal it samples (newtons for a force signal, pascals for ",
                    "pressure). The stored policy is `sampling.adaptive` with fields `start`, `end`, ",
                    "`max_absolute_error`, and `max_samples`.",
                ),
                concat!(
                    "The policy is validated and portable, and it participates in the generator's ",
                    "identity. Materializing a callable signal with adaptive sampling requires a bounded ",
                    "refinement executor that is not yet available, so a study whose generated signal ",
                    "uses this policy currently fails when the signal is materialized. Use `uniform` or ",
                    "`canonicalPoints` for signals that must run today.",
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
            "adaptive-force",
            "Declare adaptive sampling with a 1 N error bound",
            "sampling = fea.SignalSampling.adaptive(0, 2, 1.0, 400);",
            concat!(
                "assert(strcmp(class(sampling), 'fea.SignalSampling'));\n",
                "assert(sampling.sampling.adaptive.max_absolute_error == 1.0);\n",
                "assert(sampling.sampling.adaptive.max_samples == 400);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Can I run a study with adaptive sampling today?",
            concat!(
                "Not yet. The policy can be authored, validated, and saved, but materialization fails",
                " until the bounded refinement executor ships. Use `uniform` or `canonicalPoints` ",
                "meanwhile.",
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
    notes: &[
        concat!(
            "Adaptive callable materialization is rejected by the analysis execution bridge ",
            "(`adaptive callable materialization requires the bounded refinement executor`).",
        ),
    ],
});

policy_entry!(
    FEA_SIGNAL_SAMPLING_ADAPTIVE_CATALOG_ENTRY,
    name: "fea.SignalSampling.adaptive",
    result: classes::SIGNAL_SAMPLING,
    signature: "sampling = fea.SignalSampling.adaptive(start, end, maxAbsoluteError, maxSamples)",
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
            "maxAbsoluteError",
            NumericScalar,
            "Positive interpolation-error target in the signal's canonical SI unit.",
            PolicyArg::RealScalar,
        ),
        (
            "maxSamples",
            IntegerScalar,
            "Sample budget; at least 2.",
            PolicyArg::Count,
        ),
    ],
    output: ("sampling", "Typed fea.SignalSampling."),
    integer_capabilities: &integer::ADAPTIVE_SAMPLING_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
