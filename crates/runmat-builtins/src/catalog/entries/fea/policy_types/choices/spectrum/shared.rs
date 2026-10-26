//! Documentation facts shared by every identity of this policy family.

use crate::fea_contract::policy_types::documentation::{page, section, source, unit_test};
use crate::*;

pub(super) const USE_SPECTRUM: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Every response-spectrum policy is an argument of `fea.ResponseSpectrum.new(id, axis, ordinates, damping, response, envelope, interpolation, extrapolation, direction, modalCombination, directionalCombination, truncation, rigidResponse, construction, evidence, provenanceDigest)`. The spectrum is listed on the model with `fea.model(..., \"ResponseSpectra\", {...})` and selected by a modal response-spectrum step through `fea.IndependentDomain.responseSpectra(spectra)`; the step consumes a modal basis from an upstream modal extraction.",
        "`fea.ResponseSpectrum.new` validates the policies together before it returns: the interpolation must match the axis, log-log interpolation needs positive ordinates, missing-mass correction needs an acceleration spectrum, and rigid-response separation needs an acceleration spectrum with complete quadratic combination.",
    ],
);
pub(super) const LINKS: &[BuiltinDocumentationLink] = &[
    page(
        "Structural analysis: response spectra",
        "/docs/simulation/physics/structural",
    ),
    page(
        "Structural loads and support motion",
        "/docs/simulation/loads/structural",
    ),
];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    source(
        "Response-spectrum policy constructors",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/response_spectrum_policies/choices.rs",
    ),
    source(
        "Modal and directional combination",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-analysis/fea/src/solve/dynamics/response_spectrum/combine.rs",
    ),
    source(
        "Spectrum validation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-analysis/core/src/problem/response_spectrum.rs",
    ),
];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] = &[
    unit_test(
        "Typed spectrum round trip",
        "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::tests::typed_spectrum_round_trips_portably",
    ),
    unit_test(
        "Catalog contract conformance",
        "crates/runmat-builtins/src/catalog/entries/fea/policy_types/choices/spectrum/tests.rs::spectrum_policy_choices_conform",
    ),
];
