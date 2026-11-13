//! `fea.ElectricPotentialSignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::SCALAR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_volts {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`value` is a finite real scalar in volts; arrays are rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance.",
            ],
        ),
        USE_ELECTRIC_POTENTIAL_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant electric-potential signal",
            "signal = fea.ElectricPotentialSignal.constantVolts(\"supply\", 12);",
            "assert(strcmp(class(signal), 'fea.ElectricPotentialSignal'));\nassert(strcmp(signal.signal.electric_potential.signal_id, 'supply'));\nassert(strcmp(signal.signal.electric_potential.shape, 'scalar'));\nassert(isfield(signal.signal.electric_potential.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.ElectricPotentialSignal.fromTimeSamples",
        "fea.ElectricPotentialSignal.fromFunction",
        "fea.ElectricPotentialSignal.fromTimeKnots",
        "fea.ElectricalExcitation.terminalVoltage",
        "fea.ElectricalInitialCondition.potential",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ElectricPotentialSignal.constantVolts",
        summary: "Create a constant electric-potential signal in volts.",
        description: "`fea.ElectricPotentialSignal.constantVolts(id, value)` returns a `fea.ElectricPotentialSignal` whose value is the same scalar in volts (V) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "electric-potential",
            "volts",
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
        name: "fea.ElectricPotentialSignal.constantVolts",
        result: crate::fea_contract::policy_types::classes::ELECTRIC_POTENTIAL_SIGNAL,
        signature: "signal = fea.ElectricPotentialSignal.constantVolts(id, value)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("value", NumericScalar, "Finite real scalar in volts.", PolicyArg::RealScalar),
        ],
        output: ("signal", "Typed fea.ElectricPotentialSignal object."),
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
                "`values` is a vector with exactly one volts sample per time.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_ELECTRIC_POTENTIAL_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a electric-potential history",
            "signal = fea.ElectricPotentialSignal.fromTimeSamples(\"switching\", [0 0.001 0.002 0.003], [0 12 12 0], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.ElectricPotentialSignal'));\ntable = signal.signal.electric_potential.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 4);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.ElectricPotentialSignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.ElectricPotentialSignal.constantVolts",
        "fea.ElectricPotentialSignal.fromFunction",
        "fea.ElectricPotentialSignal.fromTimeKnots",
        "fea.ElectricalExcitation.terminalVoltage",
        "fea.ElectricalInitialCondition.potential",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ElectricPotentialSignal.fromTimeSamples",
        summary: "Create a electric-potential history from time samples in volts.",
        description: "`fea.ElectricPotentialSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.ElectricPotentialSignal` tabulated over time in seconds, with one scalar in volts per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "electric-potential",
            "volts",
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
        name: "fea.ElectricPotentialSignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::ELECTRIC_POTENTIAL_SIGNAL,
        shape: Scalar,
        signature: "signal = fea.ElectricPotentialSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
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
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one value per abscissa in volts, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_ELECTRIC_POTENTIAL_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a electric-potential signal from a closure",
            "generator = @(t, amplitude) amplitude*sin(2*pi*60*t);\nsignal = fea.ElectricPotentialSignal.fromFunction(\"supply_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41), \"Arguments\", {170});",
            "assert(strcmp(class(signal), 'fea.ElectricPotentialSignal'));\ngenerated = signal.signal.electric_potential.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
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
        "fea.ElectricPotentialSignal.constantVolts",
        "fea.ElectricPotentialSignal.fromTimeSamples",
        "fea.ElectricPotentialSignal.fromTimeKnots",
        "fea.ElectricalExcitation.terminalVoltage",
        "fea.ElectricalInitialCondition.potential",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ElectricPotentialSignal.fromFunction",
        summary: "Create a electric-potential signal from a RunMat function or closure.",
        description: "`fea.ElectricPotentialSignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.ElectricPotentialSignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce volts values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "electric-potential",
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
        name: "fea.ElectricPotentialSignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::ELECTRIC_POTENTIAL_SIGNAL,
        signature: "signal = fea.ElectricPotentialSignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_volts::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
