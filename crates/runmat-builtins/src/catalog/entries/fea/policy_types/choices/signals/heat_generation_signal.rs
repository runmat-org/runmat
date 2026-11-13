//! `fea.HeatGenerationSignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::SCALAR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_watts_per_cubic_meter {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`value` is a finite real scalar in watts per cubic meter; arrays are rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance.",
            ],
        ),
        USE_HEAT_GENERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant volumetric heat-generation signal",
            "signal = fea.HeatGenerationSignal.constantWattsPerCubicMeter(\"coil_heating\", 2.5e5);",
            "assert(strcmp(class(signal), 'fea.HeatGenerationSignal'));\nassert(strcmp(signal.signal.heat_generation_density.signal_id, 'coil_heating'));\nassert(strcmp(signal.signal.heat_generation_density.shape, 'scalar'));\nassert(isfield(signal.signal.heat_generation_density.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.HeatGenerationSignal.fromTimeSamples",
        "fea.HeatGenerationSignal.fromFunction",
        "fea.HeatGenerationSignal.fromTimeKnots",
        "fea.ThermalExcitation.heatGeneration",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.HeatGenerationSignal.constantWattsPerCubicMeter",
        summary: "Create a constant volumetric heat-generation signal in watts per cubic meter.",
        description: "`fea.HeatGenerationSignal.constantWattsPerCubicMeter(id, value)` returns a `fea.HeatGenerationSignal` whose value is the same scalar in watts per cubic meter (W/m^3) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "volumetric heat-generation",
            "watts per cubic meter",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[],
    });

    policy_entry!(
        CATALOG_ENTRY,
        name: "fea.HeatGenerationSignal.constantWattsPerCubicMeter",
        result: crate::fea_contract::policy_types::classes::HEAT_GENERATION_SIGNAL,
        signature: "signal = fea.HeatGenerationSignal.constantWattsPerCubicMeter(id, value)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("value", NumericScalar, "Finite real scalar in watts per cubic meter.", PolicyArg::RealScalar),
        ],
        output: ("signal", "Typed fea.HeatGenerationSignal object."),
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
                "`values` is a vector with exactly one watts per cubic meter sample per time.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_HEAT_GENERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a volumetric heat-generation history",
            "signal = fea.HeatGenerationSignal.fromTimeSamples(\"duty_cycle\", [0 10 20 30], [0 2.5e5 2.5e5 0], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.HeatGenerationSignal'));\ntable = signal.signal.heat_generation_density.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 4);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.HeatGenerationSignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.HeatGenerationSignal.constantWattsPerCubicMeter",
        "fea.HeatGenerationSignal.fromFunction",
        "fea.HeatGenerationSignal.fromTimeKnots",
        "fea.ThermalExcitation.heatGeneration",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.HeatGenerationSignal.fromTimeSamples",
        summary: "Create a volumetric heat-generation history from time samples in watts per cubic meter.",
        description: "`fea.HeatGenerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.HeatGenerationSignal` tabulated over time in seconds, with one scalar in watts per cubic meter per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "volumetric heat-generation",
            "watts per cubic meter",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[],
    });

    time_samples_entry!(
        CATALOG_ENTRY,
        name: "fea.HeatGenerationSignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::HEAT_GENERATION_SIGNAL,
        shape: Scalar,
        signature: "signal = fea.HeatGenerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
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
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one value per abscissa in watts per cubic meter, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_HEAT_GENERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a volumetric heat-generation signal from a closure",
            "generator = @(t, peak) peak*(mod(t, 20) < 10);\nsignal = fea.HeatGenerationSignal.fromFunction(\"coil_heating_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41), \"Arguments\", {2.5e5});",
            "assert(strcmp(class(signal), 'fea.HeatGenerationSignal'));\ngenerated = signal.signal.heat_generation_density.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
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
        "fea.HeatGenerationSignal.constantWattsPerCubicMeter",
        "fea.HeatGenerationSignal.fromTimeSamples",
        "fea.HeatGenerationSignal.fromTimeKnots",
        "fea.ThermalExcitation.heatGeneration",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.HeatGenerationSignal.fromFunction",
        summary: "Create a volumetric heat-generation signal from a RunMat function or closure.",
        description: "`fea.HeatGenerationSignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.HeatGenerationSignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce watts per cubic meter values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "volumetric heat-generation",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: FUNCTION_LINKS,
        implementation: FUNCTION_IMPLEMENTATION,
        verification: FUNCTION_VERIFICATION,
        notes: &[],
    });

    function_signal_entry!(
        CATALOG_ENTRY,
        name: "fea.HeatGenerationSignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::HEAT_GENERATION_SIGNAL,
        signature: "signal = fea.HeatGenerationSignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_watts_per_cubic_meter::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
