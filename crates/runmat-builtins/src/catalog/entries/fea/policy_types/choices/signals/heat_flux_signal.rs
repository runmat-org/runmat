//! `fea.HeatFluxSignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::SCALAR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_watts_per_square_meter {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`value` is a finite real scalar in watts per square meter; arrays are rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance.",
            ],
        ),
        USE_HEAT_FLUX_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant heat-flux signal",
            "signal = fea.HeatFluxSignal.constantWattsPerSquareMeter(\"solar_load\", 1361);",
            "assert(strcmp(class(signal), 'fea.HeatFluxSignal'));\nassert(strcmp(signal.signal.heat_flux.signal_id, 'solar_load'));\nassert(strcmp(signal.signal.heat_flux.shape, 'scalar'));\nassert(isfield(signal.signal.heat_flux.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.HeatFluxSignal.fromTimeSamples",
        "fea.HeatFluxSignal.fromFunction",
        "fea.HeatFluxSignal.fromTimeKnots",
        "fea.ThermalExcitation.heatFlux",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.HeatFluxSignal.constantWattsPerSquareMeter",
        summary: "Create a constant heat-flux signal in watts per square meter.",
        description: "`fea.HeatFluxSignal.constantWattsPerSquareMeter(id, value)` returns a `fea.HeatFluxSignal` whose value is the same scalar in watts per square meter (W/m^2) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "heat-flux",
            "watts per square meter",
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
        name: "fea.HeatFluxSignal.constantWattsPerSquareMeter",
        result: crate::fea_contract::policy_types::classes::HEAT_FLUX_SIGNAL,
        signature: "signal = fea.HeatFluxSignal.constantWattsPerSquareMeter(id, value)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("value", NumericScalar, "Finite real scalar in watts per square meter.", PolicyArg::RealScalar),
        ],
        output: ("signal", "Typed fea.HeatFluxSignal object."),
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
                "`values` is a vector with exactly one watts per square meter sample per time.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_HEAT_FLUX_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a heat-flux history",
            "signal = fea.HeatFluxSignal.fromTimeSamples(\"flux_burst\", [0 1 2 3], [0 5000 5000 0], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.HeatFluxSignal'));\ntable = signal.signal.heat_flux.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 4);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.HeatFluxSignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.HeatFluxSignal.constantWattsPerSquareMeter",
        "fea.HeatFluxSignal.fromFunction",
        "fea.HeatFluxSignal.fromTimeKnots",
        "fea.ThermalExcitation.heatFlux",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.HeatFluxSignal.fromTimeSamples",
        summary: "Create a heat-flux history from time samples in watts per square meter.",
        description: "`fea.HeatFluxSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.HeatFluxSignal` tabulated over time in seconds, with one scalar in watts per square meter per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "heat-flux",
            "watts per square meter",
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
        name: "fea.HeatFluxSignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::HEAT_FLUX_SIGNAL,
        shape: Scalar,
        signature: "signal = fea.HeatFluxSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
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
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one value per abscissa in watts per square meter, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_HEAT_FLUX_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a heat-flux signal from a closure",
            "generator = @(t, peak) peak*max(sin(2*pi*t/86400), 0);\nsignal = fea.HeatFluxSignal.fromFunction(\"solar_load_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41), \"Arguments\", {1000});",
            "assert(strcmp(class(signal), 'fea.HeatFluxSignal'));\ngenerated = signal.signal.heat_flux.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
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
        "fea.HeatFluxSignal.constantWattsPerSquareMeter",
        "fea.HeatFluxSignal.fromTimeSamples",
        "fea.HeatFluxSignal.fromTimeKnots",
        "fea.ThermalExcitation.heatFlux",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.HeatFluxSignal.fromFunction",
        summary: "Create a heat-flux signal from a RunMat function or closure.",
        description: "`fea.HeatFluxSignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.HeatFluxSignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce watts per square meter values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "heat-flux",
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
        name: "fea.HeatFluxSignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::HEAT_FLUX_SIGNAL,
        signature: "signal = fea.HeatFluxSignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_watts_per_square_meter::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
