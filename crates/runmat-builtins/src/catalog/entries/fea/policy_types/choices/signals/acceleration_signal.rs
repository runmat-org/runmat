//! `fea.AccelerationSignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::VECTOR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_meters_per_second_squared {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`components` is a finite real 1-by-3 or 3-by-1 vector of global-frame components in meters per second squared; any other shape is rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance; vector signals carry the global coordinate frame.",
            ],
        ),
        USE_ACCELERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant acceleration signal",
            "signal = fea.AccelerationSignal.constantMetersPerSecondSquared(\"gravity\", [0 0 -9.80665]);",
            "assert(strcmp(class(signal), 'fea.AccelerationSignal'));\nassert(strcmp(signal.signal.acceleration.signal_id, 'gravity'));\nassert(strcmp(signal.signal.acceleration.shape, 'vector3'));\nassert(isfield(signal.signal.acceleration.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.AccelerationSignal.fromTimeSamples",
        "fea.AccelerationSignal.fromFunction",
        "fea.AccelerationSignal.fromTimeKnots",
        "fea.StructuralMotion.acceleration",
        "fea.SpectrumConstruction.fromAccelerationHistoryExact",
        "fea.SpectrumConstruction.fromAccelerationHistoryRecursive",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AccelerationSignal.constantMetersPerSecondSquared",
        summary: "Create a constant acceleration signal in meters per second squared.",
        description: "`fea.AccelerationSignal.constantMetersPerSecondSquared(id, components)` returns a `fea.AccelerationSignal` whose value is the same three-component global vector in meters per second squared (m/s^2) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "acceleration",
            "meters per second squared",
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
        name: "fea.AccelerationSignal.constantMetersPerSecondSquared",
        result: crate::fea_contract::policy_types::classes::ACCELERATION_SIGNAL,
        signature: "signal = fea.AccelerationSignal.constantMetersPerSecondSquared(id, components)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("components", NumericArray, "Finite 1-by-3 or 3-by-1 real vector of global components in meters per second squared.", PolicyArg::RealVector3),
        ],
        output: ("signal", "Typed fea.AccelerationSignal object."),
        integer_capabilities: &VECTOR_CONSTANT_CAPABILITIES,
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
                "`values` holds one three-component sample per time as an N-by-3 or 3-by-N matrix in meters per second squared; a 3-by-3 matrix is read as one sample per row.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_ACCELERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a acceleration history",
            "signal = fea.AccelerationSignal.fromTimeSamples(\"base_shake\", [0 0.01 0.02 0.03], [0 0 0; 0 0 4.9; 0 0 -4.9; 0 0 0], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.AccelerationSignal'));\ntable = signal.signal.acceleration.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 4);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.AccelerationSignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.AccelerationSignal.constantMetersPerSecondSquared",
        "fea.AccelerationSignal.fromFunction",
        "fea.AccelerationSignal.fromTimeKnots",
        "fea.StructuralMotion.acceleration",
        "fea.SpectrumConstruction.fromAccelerationHistoryExact",
        "fea.SpectrumConstruction.fromAccelerationHistoryRecursive",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AccelerationSignal.fromTimeSamples",
        summary: "Create a acceleration history from time samples in meters per second squared.",
        description: "`fea.AccelerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.AccelerationSignal` tabulated over time in seconds, with one three-component global vector in meters per second squared per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "acceleration",
            "meters per second squared",
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
        name: "fea.AccelerationSignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::ACCELERATION_SIGNAL,
        shape: Vector3,
        signature: "signal = fea.AccelerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
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
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one three-component value per abscissa in meters per second squared, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_ACCELERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a acceleration signal from a closure",
            "generator = @(t) [zeros(numel(t), 2), -9.81*sin(2*pi*5*t(:))];\nsignal = fea.AccelerationSignal.fromFunction(\"gravity_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41));",
            "assert(strcmp(class(signal), 'fea.AccelerationSignal'));\ngenerated = signal.signal.acceleration.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
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
        "fea.AccelerationSignal.constantMetersPerSecondSquared",
        "fea.AccelerationSignal.fromTimeSamples",
        "fea.AccelerationSignal.fromTimeKnots",
        "fea.StructuralMotion.acceleration",
        "fea.SpectrumConstruction.fromAccelerationHistoryExact",
        "fea.SpectrumConstruction.fromAccelerationHistoryRecursive",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AccelerationSignal.fromFunction",
        summary: "Create a acceleration signal from a RunMat function or closure.",
        description: "`fea.AccelerationSignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.AccelerationSignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce meters per second squared values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "acceleration",
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
        name: "fea.AccelerationSignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::ACCELERATION_SIGNAL,
        signature: "signal = fea.AccelerationSignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_meters_per_second_squared::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
