//! `fea.AngularAccelerationSignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::VECTOR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_radians_per_second_squared {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`components` is a finite real 1-by-3 or 3-by-1 vector of global-frame components in radians per second squared; any other shape is rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance; vector signals carry the global coordinate frame.",
            ],
        ),
        USE_ANGULAR_ACCELERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant angular-acceleration signal",
            "signal = fea.AngularAccelerationSignal.constantRadiansPerSecondSquared(\"spin_up_rate\", [0 0 50]);",
            "assert(strcmp(class(signal), 'fea.AngularAccelerationSignal'));\nassert(strcmp(signal.signal.angular_acceleration.signal_id, 'spin_up_rate'));\nassert(strcmp(signal.signal.angular_acceleration.shape, 'vector3'));\nassert(isfield(signal.signal.angular_acceleration.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.AngularAccelerationSignal.fromTimeSamples",
        "fea.AngularAccelerationSignal.fromFunction",
        "fea.AngularAccelerationSignal.fromTimeKnots",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AngularAccelerationSignal.constantRadiansPerSecondSquared",
        summary: "Create a constant angular-acceleration signal in radians per second squared.",
        description: "`fea.AngularAccelerationSignal.constantRadiansPerSecondSquared(id, components)` returns a `fea.AngularAccelerationSignal` whose value is the same three-component global vector in radians per second squared (rad/s^2) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "angular-acceleration",
            "radians per second squared",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.AngularAccelerationSignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    policy_entry!(
        CATALOG_ENTRY,
        name: "fea.AngularAccelerationSignal.constantRadiansPerSecondSquared",
        result: crate::fea_contract::policy_types::classes::ANGULAR_ACCELERATION_SIGNAL,
        signature: "signal = fea.AngularAccelerationSignal.constantRadiansPerSecondSquared(id, components)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("components", NumericArray, "Finite 1-by-3 or 3-by-1 real vector of global components in radians per second squared.", PolicyArg::RealVector3),
        ],
        output: ("signal", "Typed fea.AngularAccelerationSignal object."),
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
                "`values` holds one three-component sample per time as an N-by-3 or 3-by-N matrix in radians per second squared; a 3-by-3 matrix is read as one sample per row.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_ANGULAR_ACCELERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a angular-acceleration history",
            "signal = fea.AngularAccelerationSignal.fromTimeSamples(\"brake\", [0 0.5 1.0 1.5], [0 0 0; 0 0 -40; 0 0 -40; 0 0 0], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.AngularAccelerationSignal'));\ntable = signal.signal.angular_acceleration.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 4);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.AngularAccelerationSignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.AngularAccelerationSignal.constantRadiansPerSecondSquared",
        "fea.AngularAccelerationSignal.fromFunction",
        "fea.AngularAccelerationSignal.fromTimeKnots",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AngularAccelerationSignal.fromTimeSamples",
        summary: "Create a angular-acceleration history from time samples in radians per second squared.",
        description: "`fea.AngularAccelerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.AngularAccelerationSignal` tabulated over time in seconds, with one three-component global vector in radians per second squared per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "angular-acceleration",
            "radians per second squared",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.AngularAccelerationSignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    time_samples_entry!(
        CATALOG_ENTRY,
        name: "fea.AngularAccelerationSignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::ANGULAR_ACCELERATION_SIGNAL,
        shape: Vector3,
        signature: "signal = fea.AngularAccelerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
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
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one three-component value per abscissa in radians per second squared, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_ANGULAR_ACCELERATION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a angular-acceleration signal from a closure",
            "generator = @(t) [zeros(numel(t), 2), 50*cos(2*pi*t(:))];\nsignal = fea.AngularAccelerationSignal.fromFunction(\"spin_up_rate_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41));",
            "assert(strcmp(class(signal), 'fea.AngularAccelerationSignal'));\ngenerated = signal.signal.angular_acceleration.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
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
        "fea.AngularAccelerationSignal.constantRadiansPerSecondSquared",
        "fea.AngularAccelerationSignal.fromTimeSamples",
        "fea.AngularAccelerationSignal.fromTimeKnots",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AngularAccelerationSignal.fromFunction",
        summary: "Create a angular-acceleration signal from a RunMat function or closure.",
        description: "`fea.AngularAccelerationSignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.AngularAccelerationSignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce radians per second squared values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "angular-acceleration",
        ],
        related: RELATED,
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: FUNCTION_LINKS,
        implementation: FUNCTION_IMPLEMENTATION,
        verification: FUNCTION_VERIFICATION,
        notes: &[
            "No RunMat constructor consumes `fea.AngularAccelerationSignal` yet; it is accepted by `fea.model` Signals only.",
        ],
    });

    function_signal_entry!(
        CATALOG_ENTRY,
        name: "fea.AngularAccelerationSignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::ANGULAR_ACCELERATION_SIGNAL,
        signature: "signal = fea.AngularAccelerationSignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_radians_per_second_squared::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
