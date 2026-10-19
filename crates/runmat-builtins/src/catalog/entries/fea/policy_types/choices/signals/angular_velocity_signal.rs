//! `fea.AngularVelocitySignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::VECTOR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_radians_per_second {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`components` is a finite real 1-by-3 or 3-by-1 vector of global-frame components in radians per second; any other shape is rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance; vector signals carry the global coordinate frame.",
            ],
        ),
        USE_ANGULAR_VELOCITY_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant angular-velocity signal",
            "signal = fea.AngularVelocitySignal.constantRadiansPerSecond(\"rotor_spin\", [0 0 314.16]);",
            "assert(strcmp(class(signal), 'fea.AngularVelocitySignal'));\nassert(strcmp(signal.signal.angular_velocity.signal_id, 'rotor_spin'));\nassert(strcmp(signal.signal.angular_velocity.shape, 'vector3'));\nassert(isfield(signal.signal.angular_velocity.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.AngularVelocitySignal.fromTimeSamples",
        "fea.AngularVelocitySignal.fromFunction",
        "fea.AngularVelocitySignal.fromTimeKnots",
        "fea.StructuralInitialCondition.angularVelocity",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AngularVelocitySignal.constantRadiansPerSecond",
        summary: "Create a constant angular-velocity signal in radians per second.",
        description: "`fea.AngularVelocitySignal.constantRadiansPerSecond(id, components)` returns a `fea.AngularVelocitySignal` whose value is the same three-component global vector in radians per second (rad/s) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "angular-velocity",
            "radians per second",
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
        name: "fea.AngularVelocitySignal.constantRadiansPerSecond",
        result: crate::fea_contract::policy_types::classes::ANGULAR_VELOCITY_SIGNAL,
        signature: "signal = fea.AngularVelocitySignal.constantRadiansPerSecond(id, components)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("components", NumericArray, "Finite 1-by-3 or 3-by-1 real vector of global components in radians per second.", PolicyArg::RealVector3),
        ],
        output: ("signal", "Typed fea.AngularVelocitySignal object."),
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
                "`values` holds one three-component sample per time as an N-by-3 or 3-by-N matrix in radians per second; a 3-by-3 matrix is read as one sample per row.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_ANGULAR_VELOCITY_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a angular-velocity history",
            "signal = fea.AngularVelocitySignal.fromTimeSamples(\"spin_up\", [0 1 2 3], [0 0 0; 0 0 100; 0 0 200; 0 0 314.16], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.AngularVelocitySignal'));\ntable = signal.signal.angular_velocity.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 4);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.AngularVelocitySignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.AngularVelocitySignal.constantRadiansPerSecond",
        "fea.AngularVelocitySignal.fromFunction",
        "fea.AngularVelocitySignal.fromTimeKnots",
        "fea.StructuralInitialCondition.angularVelocity",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AngularVelocitySignal.fromTimeSamples",
        summary: "Create a angular-velocity history from time samples in radians per second.",
        description: "`fea.AngularVelocitySignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.AngularVelocitySignal` tabulated over time in seconds, with one three-component global vector in radians per second per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "angular-velocity",
            "radians per second",
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
        name: "fea.AngularVelocitySignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::ANGULAR_VELOCITY_SIGNAL,
        shape: Vector3,
        signature: "signal = fea.AngularVelocitySignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
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
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one three-component value per abscissa in radians per second, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_ANGULAR_VELOCITY_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a angular-velocity signal from a closure",
            "generator = @(t) [zeros(numel(t), 2), 314.16*min(t(:)/3, 1)];\nsignal = fea.AngularVelocitySignal.fromFunction(\"rotor_spin_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41));",
            "assert(strcmp(class(signal), 'fea.AngularVelocitySignal'));\ngenerated = signal.signal.angular_velocity.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
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
        "fea.AngularVelocitySignal.constantRadiansPerSecond",
        "fea.AngularVelocitySignal.fromTimeSamples",
        "fea.AngularVelocitySignal.fromTimeKnots",
        "fea.StructuralInitialCondition.angularVelocity",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AngularVelocitySignal.fromFunction",
        summary: "Create a angular-velocity signal from a RunMat function or closure.",
        description: "`fea.AngularVelocitySignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.AngularVelocitySignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce radians per second values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "angular-velocity",
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
        name: "fea.AngularVelocitySignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::ANGULAR_VELOCITY_SIGNAL,
        signature: "signal = fea.AngularVelocitySignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_radians_per_second::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
