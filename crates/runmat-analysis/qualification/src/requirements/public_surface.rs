//! Public command surface: studies run, language forms, readable results and published validation.
use crate::requirement::*;
const STUDY: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [ExecutionEnvelope::PublicStudy]
);
const CHECK: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::MeshIndependent],
    [
        ExecutionEnvelope::SourceCheck,
        ExecutionEnvelope::PublicStudy
    ]
);
const FAMILIES: &[Envelope] = cells!(
    [
        Formulation::LinearStructural,
        Formulation::ModalStructural,
        Formulation::LinearStructuralDynamics,
        Formulation::StructuralStability,
        Formulation::NonlinearStructural,
        Formulation::PenaltyCoulombContact,
        Formulation::ThermalSteady,
        Formulation::ThermalTransient,
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell,
        Formulation::ElectromagneticTransient,
        Formulation::SaturatingMagnetics,
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction,
        Formulation::AcousticHarmonic,
        Formulation::AcousticModal,
        Formulation::AcousticTransient,
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalTetrahedra],
    [
        ExecutionEnvelope::PublicStudy,
        ExecutionEnvelope::RemoteExecution
    ]
);
const COUPLED: &[Envelope] = cells!(
    [
        Formulation::Thermoelastic,
        Formulation::Electrothermal,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction
    ],
    [MeshOrder::CanonicalTetrahedra],
    [
        ExecutionEnvelope::PublicStudy,
        ExecutionEnvelope::RemoteExecution
    ]
);
const TRANSIENT: &[Envelope] = cells!(
    [Formulation::LinearStructuralDynamics],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::PublicStudy,
        ExecutionEnvelope::RemoteExecution
    ]
);
const PRACTICAL: &[Envelope] = cells!(
    [
        Formulation::StructuralStability,
        Formulation::LinearStructuralDynamics
    ],
    [MeshOrder::Tet10],
    [ExecutionEnvelope::PublicStudy]
);
const CONDUCTION: &[Envelope] = cells!(
    [Formulation::ThermalSteady],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [ExecutionEnvelope::PublicStudy]
);
const SWEEPS: &[Envelope] = cells!(
    [
        Formulation::ThermalSteady,
        Formulation::ConjugateHeatTransfer
    ],
    [MeshOrder::CanonicalTetrahedra],
    [
        ExecutionEnvelope::PublicStudy,
        ExecutionEnvelope::RemoteExecution
    ]
);
const BRACKET: &[Envelope] = cells!(
    [Formulation::LinearStructural],
    [MeshOrder::Tet10],
    [ExecutionEnvelope::PublicStudy]
);
const DESKTOP: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::CanonicalMixedTopology],
    [
        ExecutionEnvelope::DesktopIntegration,
        ExecutionEnvelope::PublicStudy
    ]
);
const REGISTER: &[Envelope] = cells!(
    [Formulation::SharedInfrastructure],
    [MeshOrder::MeshIndependent],
    [ExecutionEnvelope::SourceCheck]
);
const FEA_PUBLIC_STUDIES_001: Requirement = Requirement {
    id: "FEA-PUBLIC-STUDIES-001",
    title: "Published studies run from the public command",
    stages: &[Stage::PublicSurface],
    acceptance: "Every published example and website study runs end to end from `runmat check|plan|run <file>.fea` on the release build, locally and, for every physics family, remotely, publishing the fields and outputs the page shows, with no maturity switch, no hidden preparation step and no family rejected as unsupported.",
    owners: &["crates/runmat-cli/src/commands/fea/run.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "docs_examples_run",
            "Every file in docs/simulation/examples and docs/fea/examples completes with `runmat run` on the release build; the generated catalog records pass and wall time for each.",
            EvidenceKind::PublicRun,
            STUDY,
        ),
        obligation(
            "each_family_runs_local_and_remote",
            "For each physics family named on the site, its worked example runs locally and with `--cluster` and publishes the same fields.",
            EvidenceKind::PublicRun,
            FAMILIES,
        ),
        obligation(
            "website_hero_studies_run",
            "Each hero study in oct-13-2026/website/physics/studies (structural cantilever and fork, heat sink, induction hardening, busbar, current sensor, electrode, muffler, cabin modes, speaker, room, cold plate and its sweep, enclosure, carotid, static mixer, journal bearing, filter) runs from its full `.fea` file, not only its `.today.fea` variant, and publishes every output its page quotes.",
            EvidenceKind::PublicRun,
            STUDY,
        ),
        obligation(
            "chained_and_coupled_studies_run",
            "Thermal into preload into buckling or prestressed dynamics, electrothermal, conjugate heat transfer and fluid-structure studies are admitted and run from the public command with converged coupling and interface residuals published.",
            EvidenceKind::PublicRun,
            COUPLED,
        ),
        obligation(
            "transient_structural_runs",
            "Direct and modal transient structural studies, including a study driven by a generated load, run on Tet4 and Tet10 without a transient block bound failure, a stack overflow or a codec limit at 1,001 or more signal samples.",
            EvidenceKind::PublicRun,
            TRANSIENT,
        ),
        obligation(
            "buckling_and_random_at_practical_size",
            "Buckling and random-response studies run on the 4,655-node cantilever within the default publication budget, producing the load factor and RMS response the page quotes.",
            EvidenceKind::PublicRun,
            PRACTICAL,
        ),
        obligation(
            "published_conduction_energy_gate",
            "The published steady conduction example passes its energy gate and publishes heat flux consistent with k times the temperature difference over the length.",
            EvidenceKind::PublicRun,
            CONDUCTION,
        ),
        obligation(
            "parameter_sweeps_with_assessments",
            "A study sweeps a typed input (convection film coefficient from still air to forced air; cold-plate mass flow) and evaluates an assessment per case (peak temperature below 85 C; lowest pump power meeting the limit), publishing per-case verdicts in authored order.",
            EvidenceKind::PublicRun,
            SWEEPS,
        ),
        obligation(
            "bracket_static_study",
            "The demo bracket STEP with steel, 2 kN on the top face and both mounting holes fixed meshes and solves from the public command and publishes von Mises stress, peak stress and peak displacement.",
            EvidenceKind::PublicRun,
            BRACKET,
        ),
    ],
};
const FEA_PUBLIC_LANGUAGE_001: Requirement = Requirement {
    id: "FEA-PUBLIC-LANGUAGE-001",
    title: "Authored language surface the published studies rely on",
    stages: &[Stage::PublicSurface, Stage::Authoring],
    acceptance: "The `.fea` language accepts, checks and lowers every construct the published studies use, reports located diagnostics, and never accepts at check what run then rejects. New forms the pages show are admitted with checked dimensions and documented exactly.",
    owners: &["crates/runmat-analysis/language/src/semantic/units.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "located_diagnostics",
            "`runmat check` on a `.fea` file prints `error[code]: message`, the file:line:col location, the source line with a caret underline and label, the expected and found units, trims leading indentation, suppresses cascade errors from a failed dependency, and ends with the `checked ...: N error(s), M warning(s)` summary; `--json` carries the same spans.",
            EvidenceKind::SemanticRealization,
            CHECK,
        ),
        obligation(
            "default_load_case",
            "A model with loads and no `load_case` applies them in one default case and `runmat check` accepts it; a model with several loads and an explicit case set is unchanged.",
            EvidenceKind::SemanticRealization,
            CHECK,
        ),
        obligation(
            "unit_prefixes",
            "Unit prefixes kW, MW, kA, kV, mA, MN, mN, kHz and the other SI prefixes parse with correct scale and dimension, and `degC` is accepted with the documented offset rule.",
            EvidenceKind::SemanticRealization,
            CHECK,
        ),
        obligation(
            "check_matches_run",
            "`runmat check` validates procedure members, state references and name selectors so that a study it accepts cannot fail at run for those reasons.",
            EvidenceKind::RejectionInvariant,
            CHECK,
        ),
        obligation(
            "output_expression_forms",
            "Output forms the pages show check with dimensions, lower and run: `pressure_drop(analysis, from =, to =)`, `transmission_loss(analysis, incident =, transmitted =)`, `sound_power(analysis, through =)`, `interpolate(output, at = [x, y, z])`, `mean(field, over = time)`, `max(field, over = region)`, field names such as `field.face.wall_shear_stress`, and arithmetic between outputs of different analyses in one study, `oscillatory_shear_index(...)`, `coefficient_of_variation(...)` over a region, and the `lubrication` outputs `film_pressure` and `friction_torque`.",
            EvidenceKind::SemanticRealization,
            CHECK,
        ),
        obligation(
            "unit_and_source_forms",
            "`rpm` is a unit, units with fractional powers (such as Pa*s^0.45 for a power-law consistency) parse with exact dimensions, and total power on a body (`heat_generation on <body> = 2 W`) is accepted alongside W/m^3 and converts to the correct volumetric source.",
            EvidenceKind::SemanticRealization,
            CHECK,
        ),
        obligation(
            "input_forms",
            "Typed input forms the pages show are admitted: `input film : heat_transfer_coefficient`, frequency tables and generated one-third-octave center sets, `samples = file \"...csv\"` with `interpolation = periodic_spline`, impedance tables over frequency with a `delany_bazley` model, `normal_velocity` read from a retained state of another analysis, `species <name> { diffusivity }` with `fluid steady { transport = [...] }`, `porous { law = darcy_forchheimer; permeability; inertial_coefficient; porosity }`, fluid laws `carreau` and `power_law`, the `lubrication` family (`lubrication.journal`, `lubrication.pressure`, `lubrication steady { cavitation = elrod }`), and `mass_flow_rate` as a sweep input type.",
            EvidenceKind::SemanticRealization,
            CHECK,
        ),
        obligation(
            "browser_check",
            "`.fea` files type-check in the browser with the same diagnostics as the CLI, and the browser never runs or meshes silently.",
            EvidenceKind::SemanticRealization,
            CHECK,
        ),
        obligation(
            "matlab_generated_signals",
            "`import function` of a MATLAB function that generates a load over time checks, solves a transient with it and returns the response shown on the runtime page.",
            EvidenceKind::PublicRun,
            STUDY,
        ),
    ],
};
const FEA_PUBLIC_RESULTS_001: Requirement = Requirement {
    id: "FEA-PUBLIC-RESULTS-001",
    title: "Results readable, verdicted and viewable",
    stages: &[Stage::PublicSurface],
    acceptance: "Results of every published study are readable from MATLAB code, carry passed, failed or indeterminate verdicts against stated criteria that can be re-assessed without re-solving, reopen cold, and capture into the saved scenes the pages show.",
    owners: &["crates/runmat-cli/src/commands/fea/result_presentation.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "scripts_run_and_read",
            "A MATLAB script runs a `.fea` study by path with `fea.run`, reads fields with `fea.results` and `fea.field`, and compares the peak with beam theory, printing the three differences the runtime page shows.",
            EvidenceKind::PublicRun,
            STUDY,
        ),
        obligation(
            "verdicts_and_reassess",
            "Each published example shows a verdict of passed, failed or indeterminate against its stated criteria with evidence in the results pane and CLI; an unconverged case returns indeterminate; `runmat reassess` re-evaluates criteria without re-solving.",
            EvidenceKind::AccuracyAcceptance,
            STUDY,
        ),
        obligation(
            "closed_form_beside_field",
            "The field result is shown beside a closed-form estimate for bending, torsion and conduction cases, computed from the study's own parameters.",
            EvidenceKind::IndependentReference,
            STUDY,
        ),
        obligation(
            "saved_scene_capture",
            "`runmat snapshot` captures each hero study into a saved scene with the views the page shows: fields with cut planes and caps, field lines, arrows, isosurfaces, per-body transparency, AC-cycle playback, one view per frequency or mode, transient playback and streamlines, and the scene replays faithfully after save (NaN, colormaps, legends, units).",
            EvidenceKind::PublicationIntegrity,
            DESKTOP,
        ),
        obligation(
            "cold_reopen_of_results",
            "Guide and hero results save and cold-reopen in Desktop and the graphics-only web viewer with identical fields and views.",
            EvidenceKind::ColdReplay,
            DESKTOP,
        ),
        obligation(
            "repeated_eigenvalue_modes",
            "Repeated eigenvalues from symmetric sections do not degrade a modal run or block its snapshot.",
            EvidenceKind::IndependentReference,
            STUDY,
        ),
        obligation(
            "provenance_recorded",
            "Every result records the requested and actual backend, precision, solver method, preconditioner and any fallback stage, and a fallback never reports the requested backend as used.",
            EvidenceKind::PublicationIntegrity,
            STUDY,
        ),
        obligation(
            "desktop_study_surface",
            "The Desktop study editor, geometry viewer, results pane and agent pane operate on the published studies in the released app, with the same checks and results as the CLI.",
            EvidenceKind::PublicRun,
            DESKTOP,
        ),
        obligation(
            "result_digest_determinism",
            "The same study yields identical result digests across runs, thread counts and worker counts, with the guarantee and its scope (same build, hardware class) stated.",
            EvidenceKind::ExecutionEquivalence,
            STUDY,
        ),
    ],
};
const FEA_PUBLIC_VALIDATION_001: Requirement = Requirement {
    id: "FEA-PUBLIC-VALIDATION-001",
    title: "Validation published, docs true, no maturity surface",
    stages: &[Stage::PublicSurface, Stage::Release],
    acceptance: "Every closed-form and benchmark check the site quotes is filled from a run and published with misses and convergence, results are compared with an open solver on named parts, the simulation docs and status match the code, and no experimental or maturity switch remains anywhere: language, admission, kernels, examples, docs or file names.",
    owners: &["crates/runmat-analysis/fea/src/results/accuracy/tests.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "published_benchmark_tables",
            "For every physics a table of named benchmark cases (analytic, manufactured and published) is generated from runs and published with the reference value, the computed value, the difference, mesh convergence and wall time, misses included; no value is typed by hand.",
            EvidenceKind::IndependentReference,
            STUDY,
        ),
        obligation(
            "open_solver_comparison",
            "Formula, RunMat and an open solver agree within stated tolerance side by side on five real parts, with the method and any disagreement published.",
            EvidenceKind::IndependentReference,
            STUDY,
        ),
        obligation(
            "status_and_docs_match_code",
            "The simulation docs, status tables and per-physics pages are generated from or checked against the admission registry and runner gate, and every row in the docs-drift list is closed; branch features are documented in the change that adds them.",
            EvidenceKind::SemanticRealization,
            STUDY,
        ),
        obligation(
            "guides_published_with_runs",
            "The heat-sink and tuning-fork guides publish with their projects, archives and HTTP checks from accepted runs.",
            EvidenceKind::PublicRun,
            STUDY,
        ),
        obligation(
            "guide_per_family",
            "Each shippable physics family has a guide with a runnable example, including electrostatics, conduction, magnetostatics, harmonic Maxwell, acoustics and flow.",
            EvidenceKind::PublicRun,
            STUDY,
        ),
        obligation(
            "no_maturity_surface",
            "No `formulation = experimental` switch, maturity gate in admission or kernels, 'production qualification unsupported' rejection, or the word experimental in docs, examples and file names remains; a finished capability simply runs.",
            EvidenceKind::RejectionInvariant,
            CHECK,
        ),
    ],
};
const FEA_RELEASE_REGISTER_001: Requirement = Requirement {
    id: "FEA-RELEASE-REGISTER-001",
    title: "The register defines done and maintains itself",
    stages: &[Stage::Release],
    acceptance: "The register, its witness tags and its scope inventory are the definition of done: witnesses are declared at the tests, validated in CI, run to produce a generated evidence catalog, and every scope row the sources claim maps to an obligation or an explicit exclusion.",
    owners: &["crates/runmat-analysis/qualification/src/witness.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "witness_tags_valid",
            "Every witness tag in the repository names an existing obligation, claims only envelopes that obligation requires, carries a non-empty boundary and sits on an existing test function; a stale or broken tag fails the qualification tests.",
            EvidenceKind::PublicationIntegrity,
            REGISTER,
        ),
        obligation(
            "evidence_catalog_generated",
            "The evidence catalog (markdown and JSON, per requirement, obligation and envelope with PASS, FAIL or OPEN and the test locator) is generated by running the tagged tests, never hand-edited, and a cell is PASS only when every tagged test for it passed in that run.",
            EvidenceKind::PublicationIntegrity,
            REGISTER,
        ),
        obligation(
            "scope_inventory_complete",
            "Every capability the website plans, end states, claim rows and FEA scope documents state has an inventory row that maps to existing obligations or to an explicit exclusion with its reason; no row is unmapped and every mapped obligation exists.",
            EvidenceKind::PublicationIntegrity,
            REGISTER,
        ),
        obligation(
            "register_integrity",
            "Requirement and obligation declarations are structurally valid: unique, non-empty, with distinct required envelopes drawn from the declared vocabulary, and owning sources exist.",
            EvidenceKind::PublicationIntegrity,
            REGISTER,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_PUBLIC_STUDIES_001,
    FEA_PUBLIC_LANGUAGE_001,
    FEA_PUBLIC_RESULTS_001,
    FEA_PUBLIC_VALIDATION_001,
    FEA_RELEASE_REGISTER_001,
];
