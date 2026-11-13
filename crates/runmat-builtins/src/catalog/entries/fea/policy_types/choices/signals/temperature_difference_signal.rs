//! `fea.TemperatureDifferenceSignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::SCALAR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_kelvin {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`value` is a finite real scalar in kelvin; arrays are rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance.",
            ],
        ),
        USE_TEMPERATURE_DIFFERENCE_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant temperature-difference signal",
            "signal = fea.TemperatureDifferenceSignal.constantKelvin(\"delta_t\", 40);",
            "assert(strcmp(class(signal), 'fea.TemperatureDifferenceSignal'));\nassert(strcmp(signal.signal.temperature_difference.signal_id, 'delta_t'));\nassert(strcmp(signal.signal.temperature_difference.shape, 'scalar'));\nassert(isfield(signal.signal.temperature_difference.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.TemperatureDifferenceSignal.fromTimeSamples",
        "fea.TemperatureDifferenceSignal.fromFunction",
        "fea.TemperatureDifferenceSignal.fromTimeKnots",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TemperatureDifferenceSignal.constantKelvin",
        summary: "Create a constant temperature-difference signal in kelvin.",
        description: "`fea.TemperatureDifferenceSignal.constantKelvin(id, value)` returns a `fea.TemperatureDifferenceSignal` whose value is the same scalar in kelvin (K) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "temperature-difference",
            "kelvin",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.TemperatureDifferenceSignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    policy_entry!(
        CATALOG_ENTRY,
        name: "fea.TemperatureDifferenceSignal.constantKelvin",
        result: crate::fea_contract::policy_types::classes::TEMPERATURE_DIFFERENCE_SIGNAL,
        signature: "signal = fea.TemperatureDifferenceSignal.constantKelvin(id, value)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("value", NumericScalar, "Finite real scalar in kelvin.", PolicyArg::RealScalar),
        ],
        output: ("signal", "Typed fea.TemperatureDifferenceSignal object."),
        integer_capabilities: &SCALAR_CONSTANT_CAPABILITIES,
        integer_audit: None,
        documentation: DOCUMENTATION,
    );
}

pub(super) mod from_time_samples {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`times` is a finite real row or column vector in seconds with at least two strictly increasing entries.",
                "`values` is a vector with exactly one kelvin sample per time.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_TEMPERATURE_DIFFERENCE_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a temperature-difference history",
            "signal = fea.TemperatureDifferenceSignal.fromTimeSamples(\"delta_ramp\", [0 30 60], [0 20 40], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.TemperatureDifferenceSignal'));\ntable = signal.signal.temperature_difference.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 3);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.TemperatureDifferenceSignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.TemperatureDifferenceSignal.constantKelvin",
        "fea.TemperatureDifferenceSignal.fromFunction",
        "fea.TemperatureDifferenceSignal.fromTimeKnots",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TemperatureDifferenceSignal.fromTimeSamples",
        summary: "Create a temperature-difference history from time samples in kelvin.",
        description: "`fea.TemperatureDifferenceSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.TemperatureDifferenceSignal` tabulated over time in seconds, with one scalar in kelvin per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "temperature-difference",
            "kelvin",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.TemperatureDifferenceSignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    time_samples_entry!(
        CATALOG_ENTRY,
        name: "fea.TemperatureDifferenceSignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::TEMPERATURE_DIFFERENCE_SIGNAL,
        shape: Scalar,
        signature: "signal = fea.TemperatureDifferenceSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
        documentation: DOCUMENTATION,
    );
}

pub(super) mod from_function {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`callable` is a function handle or closure bound to the executing program; anonymous functions and program functions qualify, builtins such as `@sin` and workspace-only handles do not. The callable and its captured values are frozen at construction.",
                "`domain` is a `fea.SignalDomain` (time, frequency, or study parameter) and `sampling` a `fea.SignalSampling` (uniform, adaptive, or explicit points). Frequency-domain sampling cannot include negative frequencies.",
                "Options: `Arguments` is a cell array of extra arguments (finite real scalars or vectors, exact integers, logical scalars, or text); `Interpolation` defaults to `fea.SignalInterpolation.linear()` and `Extrapolation` to `fea.SignalExtrapolation.reject()`. Unknown or repeated option names are rejected.",
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one value per abscissa in kelvin, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_TEMPERATURE_DIFFERENCE_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a temperature-difference signal from a closure",
            "generator = @(t, rise) rise*min(t/60, 1);\nsignal = fea.TemperatureDifferenceSignal.fromFunction(\"delta_t_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41), \"Arguments\", {40});",
            "assert(strcmp(class(signal), 'fea.TemperatureDifferenceSignal'));\ngenerated = signal.signal.temperature_difference.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "When is the function evaluated?",
            "Not at construction. RunMat freezes the callable and its captures, then evaluates it at the sampling abscissae when the study runs; edits to workspace variables afterward do not change the signal.",
        ),
        UNITS_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.TemperatureDifferenceSignal.constantKelvin",
        "fea.TemperatureDifferenceSignal.fromTimeSamples",
        "fea.TemperatureDifferenceSignal.fromTimeKnots",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TemperatureDifferenceSignal.fromFunction",
        summary: "Create a temperature-difference signal from a RunMat function or closure.",
        description: "`fea.TemperatureDifferenceSignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.TemperatureDifferenceSignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce kelvin values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "temperature-difference",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: FUNCTION_LINKS,
        implementation: FUNCTION_IMPLEMENTATION,
        verification: FUNCTION_VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.TemperatureDifferenceSignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    function_signal_entry!(
        CATALOG_ENTRY,
        name: "fea.TemperatureDifferenceSignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::TEMPERATURE_DIFFERENCE_SIGNAL,
        signature: "signal = fea.TemperatureDifferenceSignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_kelvin::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
