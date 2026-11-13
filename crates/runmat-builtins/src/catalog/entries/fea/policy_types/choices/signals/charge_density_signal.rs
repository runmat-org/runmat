//! `fea.ChargeDensitySignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::SCALAR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_coulombs_per_cubic_meter {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`value` is a finite real scalar in coulombs per cubic meter; arrays are rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance.",
            ],
        ),
        USE_CHARGE_DENSITY_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant charge-density signal",
            "signal = fea.ChargeDensitySignal.constantCoulombsPerCubicMeter(\"space_charge\", 1e-6);",
            "assert(strcmp(class(signal), 'fea.ChargeDensitySignal'));\nassert(strcmp(signal.signal.electric_charge_density.signal_id, 'space_charge'));\nassert(strcmp(signal.signal.electric_charge_density.shape, 'scalar'));\nassert(isfield(signal.signal.electric_charge_density.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.ChargeDensitySignal.fromTimeSamples",
        "fea.ChargeDensitySignal.fromFunction",
        "fea.ChargeDensitySignal.fromTimeKnots",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ChargeDensitySignal.constantCoulombsPerCubicMeter",
        summary: "Create a constant charge-density signal in coulombs per cubic meter.",
        description: "`fea.ChargeDensitySignal.constantCoulombsPerCubicMeter(id, value)` returns a `fea.ChargeDensitySignal` whose value is the same scalar in coulombs per cubic meter (C/m^3) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "charge-density",
            "coulombs per cubic meter",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.ChargeDensitySignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    policy_entry!(
        CATALOG_ENTRY,
        name: "fea.ChargeDensitySignal.constantCoulombsPerCubicMeter",
        result: crate::fea_contract::policy_types::classes::CHARGE_DENSITY_SIGNAL,
        signature: "signal = fea.ChargeDensitySignal.constantCoulombsPerCubicMeter(id, value)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("value", NumericScalar, "Finite real scalar in coulombs per cubic meter.", PolicyArg::RealScalar),
        ],
        output: ("signal", "Typed fea.ChargeDensitySignal object."),
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
                "`values` is a vector with exactly one coulombs per cubic meter sample per time.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_CHARGE_DENSITY_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a charge-density history",
            "signal = fea.ChargeDensitySignal.fromTimeSamples(\"charge_build\", [0 1 2], [0 5e-7 1e-6], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.ChargeDensitySignal'));\ntable = signal.signal.electric_charge_density.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 3);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.ChargeDensitySignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.ChargeDensitySignal.constantCoulombsPerCubicMeter",
        "fea.ChargeDensitySignal.fromFunction",
        "fea.ChargeDensitySignal.fromTimeKnots",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ChargeDensitySignal.fromTimeSamples",
        summary: "Create a charge-density history from time samples in coulombs per cubic meter.",
        description: "`fea.ChargeDensitySignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.ChargeDensitySignal` tabulated over time in seconds, with one scalar in coulombs per cubic meter per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "charge-density",
            "coulombs per cubic meter",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.ChargeDensitySignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    time_samples_entry!(
        CATALOG_ENTRY,
        name: "fea.ChargeDensitySignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::CHARGE_DENSITY_SIGNAL,
        shape: Scalar,
        signature: "signal = fea.ChargeDensitySignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
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
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one value per abscissa in coulombs per cubic meter, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_CHARGE_DENSITY_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a charge-density signal from a closure",
            "generator = @(t, peak) peak*(1 - exp(-t));\nsignal = fea.ChargeDensitySignal.fromFunction(\"space_charge_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41), \"Arguments\", {1e-6});",
            "assert(strcmp(class(signal), 'fea.ChargeDensitySignal'));\ngenerated = signal.signal.electric_charge_density.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
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
        "fea.ChargeDensitySignal.constantCoulombsPerCubicMeter",
        "fea.ChargeDensitySignal.fromTimeSamples",
        "fea.ChargeDensitySignal.fromTimeKnots",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ChargeDensitySignal.fromFunction",
        summary: "Create a charge-density signal from a RunMat function or closure.",
        description: "`fea.ChargeDensitySignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.ChargeDensitySignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce coulombs per cubic meter values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "charge-density",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: FUNCTION_LINKS,
        implementation: FUNCTION_IMPLEMENTATION,
        verification: FUNCTION_VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.ChargeDensitySignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    function_signal_entry!(
        CATALOG_ENTRY,
        name: "fea.ChargeDensitySignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::CHARGE_DENSITY_SIGNAL,
        signature: "signal = fea.ChargeDensitySignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_coulombs_per_cubic_meter::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
