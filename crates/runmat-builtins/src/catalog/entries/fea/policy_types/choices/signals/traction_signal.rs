//! `fea.TractionSignal` constructors: constant, time-sampled, and callable-backed.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_choices::entry::function_signal_entry;
use crate::fea_contract::policy_choices::entry::time_samples_entry;
use crate::fea_contract::policy_choices::integer::VECTOR_CONSTANT_CAPABILITIES;
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::*;

pub(super) mod constant_pascals {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "`components` is a finite real 1-by-3 or 3-by-1 vector of global-frame components in pascals; any other shape is rejected.",
                "`id` must be nonempty text; excitations, initial conditions, and constraints refer to the signal through it. Integer classes are accepted and converted once to binary64; logical, complex, nonfinite, and device-resident values are rejected before the object is created.",
                "The object stores a constant representation with real64 precision and authored provenance; vector signals carry the global coordinate frame.",
            ],
        ),
        USE_TRACTION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "constant",
            "Create a constant traction signal",
            "signal = fea.TractionSignal.constantPascals(\"shear_load\", [5e4 0 0]);",
            "assert(strcmp(class(signal), 'fea.TractionSignal'));\nassert(strcmp(signal.signal.traction.signal_id, 'shear_load'));\nassert(strcmp(signal.signal.traction.shape, 'vector3'));\nassert(isfield(signal.signal.traction.representation, 'constant'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, INTEGER_FAQ];
    const RELATED: &[&str] = &[
        "fea.TractionSignal.fromTimeSamples",
        "fea.TractionSignal.fromFunction",
        "fea.TractionSignal.fromTimeKnots",
        "fea.StructuralExcitation.traction",
        "fea.model",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TractionSignal.constantPascals",
        summary: "Create a constant traction signal in pascals.",
        description: "`fea.TractionSignal.constantPascals(id, components)` returns a `fea.TractionSignal` whose value is the same three-component global vector in pascals (Pa) at every sample.",
        keywords: &[
            "fea",
            "signal",
            "constant",
            "traction",
            "pascals",
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
        name: "fea.TractionSignal.constantPascals",
        result: crate::fea_contract::policy_types::classes::TRACTION_SIGNAL,
        signature: "signal = fea.TractionSignal.constantPascals(id, components)",
        inputs: [
            ("id", StringScalar, "Nonempty signal identifier.", PolicyArg::Text),
            ("components", NumericArray, "Finite 1-by-3 or 3-by-1 real vector of global components in pascals.", PolicyArg::RealVector3),
        ],
        output: ("signal", "Typed fea.TractionSignal object."),
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
                "`values` holds one three-component sample per time as an N-by-3 or 3-by-N matrix in pascals; a 3-by-3 matrix is read as one sample per row.",
                "`interpolation` (`fea.SignalInterpolation`) decides values between samples and `extrapolation` (`fea.SignalExtrapolation`) decides values outside the first and last time; both are required. The object stores continuous time knots, real64 precision, authored provenance, and the sample count as evidence.",
            ],
        ),
        USE_TRACTION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "history",
            "Tabulate a traction history",
            "signal = fea.TractionSignal.fromTimeSamples(\"shear_history\", [0 0.1 0.2 0.3], [0 0 0; 2.5e4 0 0; 5e4 0 0; 5e4 0 0], ...\n    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());",
            "assert(strcmp(class(signal), 'fea.TractionSignal'));\ntable = signal.signal.traction.representation.tabulated;\nassert(strcmp(table.domain, 'time_seconds'));\nassert(numel(table.knots) == 4);\nassert(strcmp(table.interpolation, 'linear'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "How do I add a step change?",
            "Use `fea.TractionSignal.fromTimeKnots` with `fea.SignalKnot.jump`; time samples are always continuous knots.",
        ),
        UNITS_FAQ,
        INTEGER_FAQ,
    ];
    const RELATED: &[&str] = &[
        "fea.TractionSignal.constantPascals",
        "fea.TractionSignal.fromFunction",
        "fea.TractionSignal.fromTimeKnots",
        "fea.StructuralExcitation.traction",
        "fea.model",
        "fea.SignalInterpolation.linear",
        "fea.SignalExtrapolation.reject",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TractionSignal.fromTimeSamples",
        summary: "Create a traction history from time samples in pascals.",
        description: "`fea.TractionSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)` returns a `fea.TractionSignal` tabulated over time in seconds, with one three-component global vector in pascals per time sample.",
        keywords: &[
            "fea",
            "signal",
            "time history",
            "samples",
            "traction",
            "pascals",
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
        name: "fea.TractionSignal.fromTimeSamples",
        class: crate::fea_contract::policy_types::classes::TRACTION_SIGNAL,
        shape: Vector3,
        signature: "signal = fea.TractionSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
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
                "Construction does not call the function. When the study runs, RunMat evaluates the frozen callable with the sampling abscissae as its first argument followed by the `Arguments` entries, expects one three-component value per abscissa in pascals, and replaces the generator with a tabulated signal that records generated provenance. Evaluation must finish within the wall-time ceiling of the steps' accuracy plans.",
            ],
        ),
        USE_TRACTION_SIGNAL,
        ERRORS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "closure",
            "Generate a traction signal from a closure",
            "generator = @(t) [5e4*sin(2*pi*10*t(:)), zeros(numel(t), 2)];\nsignal = fea.TractionSignal.fromFunction(\"shear_load_generated\", generator, fea.SignalDomain.time(), ...\n    fea.SignalSampling.uniform(0, 0.2, 41));",
            "assert(strcmp(class(signal), 'fea.TractionSignal'));\ngenerated = signal.signal.traction.representation.generator;\nassert(strcmp(generated.interpolation, 'linear'));\nassert(strcmp(generated.extrapolation, 'reject'));",
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
        "fea.TractionSignal.constantPascals",
        "fea.TractionSignal.fromTimeSamples",
        "fea.TractionSignal.fromTimeKnots",
        "fea.StructuralExcitation.traction",
        "fea.model",
        "fea.SignalSampling.uniform",
        "fea.SignalDomain.time",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.TractionSignal.fromFunction",
        summary: "Create a traction signal from a RunMat function or closure.",
        description: "`fea.TractionSignal.fromFunction(id, callable, domain, sampling, Name, Value, ...)` returns a `fea.TractionSignal` backed by a frozen RunMat callable that RunMat evaluates at run time to produce pascals values.",
        keywords: &[
            "fea",
            "signal",
            "function",
            "closure",
            "generator",
            "traction",
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
        name: "fea.TractionSignal.fromFunction",
        class: crate::fea_contract::policy_types::classes::TRACTION_SIGNAL,
        signature: "signal = fea.TractionSignal.fromFunction(id, callable, domain, sampling, Name, Value)",
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constant_pascals::CATALOG_ENTRY,
    &from_time_samples::CATALOG_ENTRY,
    &from_function::CATALOG_ENTRY,
];
