//! `fea.RequiredAccuracyEvidence` choices for estimators, convergence, and balances.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod a_posteriori_estimator {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "An a-posteriori estimator computes an error indicator from the solved fields themselves, such as a recovery-based stress error estimate.",
                "Structural adaptive refinement produces it for the recovery contribution.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with recovery or spatial-discretization criteria when a single solve must carry its own error estimate.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "a-posteriori-estimator",
            "Use aPosterioriEstimator in an accuracy plan",
            "value = fea.RequiredAccuracyEvidence.aPosterioriEstimator();\ncriterion = fea.AccuracyCriterion.new(\"a_posteriori_estimator\", fea.AccuracyTarget.observation(\"tip_displacement\"), fea.ErrorContribution.recovery(), fea.AccuracyTolerance.relative(1e-3), value, true);",
            "assert(strcmp(class(value), 'fea.RequiredAccuracyEvidence'));\nassert(strcmp(value.value, 'a_posteriori_estimator'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.AccuracyCriterion.new",
        "fea.ErrorContribution.spatialDiscretization",
        "fea.ErrorContribution.algebraic",
        "fea.AccuracyTolerance.relative",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.RequiredAccuracyEvidence.aPosterioriEstimator",
        summary: "Require an a-posteriori error estimate.",
        description: "`fea.RequiredAccuracyEvidence.aPosterioriEstimator()` returns the closed `fea.RequiredAccuracyEvidence` value `a_posteriori_estimator`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "RequiredAccuracyEvidence",
            "aPosterioriEstimator",
            "evidence",
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
        name: "fea.RequiredAccuracyEvidence.aPosterioriEstimator",
        result: crate::fea_contract::policy_types::classes::REQUIRED_ACCURACY_EVIDENCE,
        signature: "value = fea.RequiredAccuracyEvidence.aPosterioriEstimator()",
        inputs: [],
        output: ("value", "Typed fea.RequiredAccuracyEvidence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod convergence_study {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A convergence study compares the target quantity across refinements (mesh levels, modal basis size, frequency density) and measures its change.",
                "Structural adaptive refinement, modal response-spectrum, and random-response steps produce it for their respective contributions.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for spatial, modal-truncation, frequency-sampling, and recovery criteria.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "convergence-study",
            "Use convergenceStudy in an accuracy plan",
            "value = fea.RequiredAccuracyEvidence.convergenceStudy();\ncriterion = fea.AccuracyCriterion.new(\"convergence_study\", fea.AccuracyTarget.observation(\"tip_displacement\"), fea.ErrorContribution.spatialDiscretization(), fea.AccuracyTolerance.relative(1e-3), value, true);",
            "assert(strcmp(class(value), 'fea.RequiredAccuracyEvidence'));\nassert(strcmp(value.value, 'convergence_study'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.AccuracyCriterion.new",
        "fea.ErrorContribution.spatialDiscretization",
        "fea.ErrorContribution.algebraic",
        "fea.AccuracyTolerance.relative",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.RequiredAccuracyEvidence.convergenceStudy",
        summary: "Require evidence from a convergence study.",
        description: "`fea.RequiredAccuracyEvidence.convergenceStudy()` returns the closed `fea.RequiredAccuracyEvidence` value `convergence_study`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "RequiredAccuracyEvidence",
            "convergenceStudy",
            "evidence",
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
        name: "fea.RequiredAccuracyEvidence.convergenceStudy",
        result: crate::fea_contract::policy_types::classes::REQUIRED_ACCURACY_EVIDENCE,
        signature: "value = fea.RequiredAccuracyEvidence.convergenceStudy()",
        inputs: [],
        output: ("value", "Typed fea.RequiredAccuracyEvidence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod conservation_balance {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A conservation balance compares what enters and leaves the model (forces and reactions, heat in and out, current, mass flow) and reports the relative imbalance.",
                "Thermal, electrical, magnetostatic, Maxwell harmonic, acoustic harmonic, and steady fluid solves produce it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with the algebraic contribution at the global-equilibrium target as a physical cross-check of the linear solve.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "conservation-balance",
            "Use conservationBalance in an accuracy plan",
            "value = fea.RequiredAccuracyEvidence.conservationBalance();\ncriterion = fea.AccuracyCriterion.new(\"conservation_balance\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), fea.AccuracyTolerance.relative(1e-3), value, true);",
            "assert(strcmp(class(value), 'fea.RequiredAccuracyEvidence'));\nassert(strcmp(value.value, 'conservation_balance'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.AccuracyCriterion.new",
        "fea.ErrorContribution.spatialDiscretization",
        "fea.ErrorContribution.algebraic",
        "fea.AccuracyTolerance.relative",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.RequiredAccuracyEvidence.conservationBalance",
        summary: "Require a conservation balance check.",
        description: "`fea.RequiredAccuracyEvidence.conservationBalance()` returns the closed `fea.RequiredAccuracyEvidence` value `conservation_balance`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "RequiredAccuracyEvidence",
            "conservationBalance",
            "evidence",
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
        name: "fea.RequiredAccuracyEvidence.conservationBalance",
        result: crate::fea_contract::policy_types::classes::REQUIRED_ACCURACY_EVIDENCE,
        signature: "value = fea.RequiredAccuracyEvidence.conservationBalance()",
        inputs: [],
        output: ("value", "Typed fea.RequiredAccuracyEvidence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod residual_bound {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A residual bound is the relative norm of the remaining equation residual after the solve.",
                "Linear and nonlinear solves across structural, thermal, electromagnetic, acoustic, and fluid physics produce it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with algebraic or nonlinear criteria at the global-equilibrium target.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "residual-bound",
            "Use residualBound in an accuracy plan",
            "value = fea.RequiredAccuracyEvidence.residualBound();\ncriterion = fea.AccuracyCriterion.new(\"residual_bound\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), fea.AccuracyTolerance.relative(1e-3), value, true);",
            "assert(strcmp(class(value), 'fea.RequiredAccuracyEvidence'));\nassert(strcmp(value.value, 'residual_bound'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.AccuracyCriterion.new",
        "fea.ErrorContribution.spatialDiscretization",
        "fea.ErrorContribution.algebraic",
        "fea.AccuracyTolerance.relative",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.RequiredAccuracyEvidence.residualBound",
        summary: "Require a bound on the equation residual.",
        description: "`fea.RequiredAccuracyEvidence.residualBound()` returns the closed `fea.RequiredAccuracyEvidence` value `residual_bound`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "RequiredAccuracyEvidence",
            "residualBound",
            "evidence",
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
        name: "fea.RequiredAccuracyEvidence.residualBound",
        result: crate::fea_contract::policy_types::classes::REQUIRED_ACCURACY_EVIDENCE,
        signature: "value = fea.RequiredAccuracyEvidence.residualBound()",
        inputs: [],
        output: ("value", "Typed fea.RequiredAccuracyEvidence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &a_posteriori_estimator::CATALOG_ENTRY,
    &convergence_study::CATALOG_ENTRY,
    &conservation_balance::CATALOG_ENTRY,
    &residual_bound::CATALOG_ENTRY,
];
