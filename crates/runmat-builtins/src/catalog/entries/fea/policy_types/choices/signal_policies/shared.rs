//! Documentation facts shared by every identity of this policy family.

use crate::fea_contract::policy_types::documentation::{page, section, source, unit_test};
use crate::*;

pub(super) const USE_TABULATED: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Pass interpolation and extrapolation policies as the fourth and fifth arguments of any `fea.<Quantity>Signal.fromTimeSamples` or `fea.<Quantity>Signal.fromTimeKnots` constructor, or as the `Interpolation` and `Extrapolation` options of `fea.<Quantity>Signal.fromFunction` (which default to `linear` and `reject`). A callable-backed signal keeps its policies when RunMat realizes it into a table at run time.",
        "The signal is listed on the model with `fea.model(..., \"Signals\", {...})` and referenced by excitations such as `fea.StructuralExcitation.force`, by initial conditions, and by support motion. The policy is applied whenever a step samples the signal, for example at each time increment of a transient step.",
        "Support motion built with `fea.StructuralMotion.acceleration` admits only time-domain acceleration signals with `linear` interpolation and `reject` extrapolation, so the motion and its integrated velocity and displacement stay derivative-consistent.",
    ],
);
pub(super) const USE_SAMPLE_APPROACH: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Pass the approach as the fifth argument of an initial-condition constructor such as `fea.ThermalInitialCondition.temperature(id, target, signal, abscissa, sampleApproach, beforeStepId)` or `fea.StructuralInitialCondition.velocity(...)`. The initial condition reads one value from the signal at `abscissa` and applies it before the step named by `beforeStepId`.",
        "The signal is sampled when the study is realized for planning or running, not when the initial condition is constructed. An invalid one-sided request fails that realization before any solve starts.",
    ],
);
pub(super) const USE_CONTINUITY: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Pass the continuity as the fourth argument of `fea.SignalKnot.jump(abscissa, leftLimit, rightLimit, continuity)`, then build the signal from its knots with `fea.<Quantity>Signal.fromTimeKnots(id, knots, interpolation, extrapolation)`. Jumps are allowed only at interior knots.",
        "Interpolation inside the neighboring intervals always uses the jump's left limit as the incoming value and its right limit as the outgoing value. The continuity choice decides only the value assigned exactly at the jump abscissa, which is what an `at` sample (for example an initial condition sampled at the jump time) reads.",
    ],
);
pub(super) const LINKS: &[BuiltinDocumentationLink] = &[
    page("Loads and signals", "/docs/simulation/loads"),
    page(
        "Initial conditions",
        "/docs/simulation/constraints/initial-conditions",
    ),
];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    source(
        "Signal policy constructors",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/signal_policies.rs",
    ),
    source(
        "Signal sampling",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-analysis/fea/src/excitation/sample.rs",
    ),
];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] = &[
    unit_test(
        "Closed, typed, portable signal policies",
        "crates/runmat-runtime/src/builtins/fea/signal_policies.rs::tests::policies_are_closed_typed_and_portable",
    ),
    unit_test(
        "Time samples keep their closed policies",
        "crates/runmat-runtime/src/builtins/fea/signals.rs::tests::time_samples_preserve_vector_samples_and_closed_policies",
    ),
    unit_test(
        "Catalog contract conformance",
        "crates/runmat-builtins/src/catalog/entries/fea/policy_types/choices/signal_policies/tests.rs::signal_policy_choices_conform",
    ),
];
