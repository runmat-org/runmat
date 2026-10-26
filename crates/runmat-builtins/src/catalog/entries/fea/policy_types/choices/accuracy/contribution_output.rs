//! `fea.ErrorContribution` choices for recovery, sampling, precision, and equivalence error.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod recovery {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Recovery error comes from computing derived fields (stress, flux, strain) from the primary solution, including averaging and projection to nodes.",
                "Structural adaptive refinement reports an a-posteriori recovery estimator and a recovery convergence study.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it on stress or flux observations that drive acceptance.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "recovery",
            "Use recovery in an accuracy plan",
            "value = fea.ErrorContribution.recovery();\ncriterion = fea.AccuracyCriterion.new(\"recovery\", fea.AccuracyTarget.observation(\"tip_displacement\"), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.aPosterioriEstimator(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'recovery'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Which accuracy targets accept this contribution?",
            "Global equilibrium accepts algebraic, nonlinear, and coupling error; interface closure accepts coupling, interface-projection, and field-transfer error; the numerical-result target accepts everything except observation sampling and execution equivalence; observation and quantity targets accept everything except execution equivalence; the execution-equivalence target accepts only execution-equivalence error.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.AccuracyCriterion.new",
        "fea.AccuracyTarget.globalEquilibrium",
        "fea.AccuracyTarget.numericalResult",
        "fea.AccuracyTolerance.relative",
        "fea.RequiredAccuracyEvidence.residualBound",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ErrorContribution.recovery",
        summary: "Name error from recovering derived quantities such as stress.",
        description: "`fea.ErrorContribution.recovery()` returns the closed `fea.ErrorContribution` value `recovery`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "recovery",
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
        name: "fea.ErrorContribution.recovery",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.recovery()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod observation_sampling {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Observation-sampling error comes from retaining an observation only at sampled times, frequencies, or positions, which can miss a peak between samples.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it on history and probe observations whose peaks matter. The numerical-result target does not accept it.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "observation-sampling",
            "Use observationSampling in an accuracy plan",
            "value = fea.ErrorContribution.observationSampling();\ncriterion = fea.AccuracyCriterion.new(\"observation_sampling\", fea.AccuracyTarget.observation(\"tip_displacement\"), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'observation_sampling'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Which accuracy targets accept this contribution?",
            "Global equilibrium accepts algebraic, nonlinear, and coupling error; interface closure accepts coupling, interface-projection, and field-transfer error; the numerical-result target accepts everything except observation sampling and execution equivalence; observation and quantity targets accept everything except execution equivalence; the execution-equivalence target accepts only execution-equivalence error.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.AccuracyCriterion.new",
        "fea.AccuracyTarget.globalEquilibrium",
        "fea.AccuracyTarget.numericalResult",
        "fea.AccuracyTolerance.relative",
        "fea.RequiredAccuracyEvidence.residualBound",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ErrorContribution.observationSampling",
        summary: "Name error from sampling an observation at finitely many points.",
        description: "`fea.ErrorContribution.observationSampling()` returns the closed `fea.ErrorContribution` value `observation_sampling`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "observationSampling",
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
        name: "fea.ErrorContribution.observationSampling",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.observationSampling()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod precision {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Precision error is the effect of finite floating-point precision and reduced-precision storage on the result.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when results are stored or computed in reduced precision.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "precision",
            "Use precision in an accuracy plan",
            "value = fea.ErrorContribution.precision();\ncriterion = fea.AccuracyCriterion.new(\"precision\", fea.AccuracyTarget.numericalResult(), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.referenceComparison(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'precision'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Which accuracy targets accept this contribution?",
            "Global equilibrium accepts algebraic, nonlinear, and coupling error; interface closure accepts coupling, interface-projection, and field-transfer error; the numerical-result target accepts everything except observation sampling and execution equivalence; observation and quantity targets accept everything except execution equivalence; the execution-equivalence target accepts only execution-equivalence error.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.AccuracyCriterion.new",
        "fea.AccuracyTarget.globalEquilibrium",
        "fea.AccuracyTarget.numericalResult",
        "fea.AccuracyTolerance.relative",
        "fea.RequiredAccuracyEvidence.residualBound",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ErrorContribution.precision",
        summary: "Name error from floating-point precision.",
        description: "`fea.ErrorContribution.precision()` returns the closed `fea.ErrorContribution` value `precision`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "precision",
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
        name: "fea.ErrorContribution.precision",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.precision()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod execution_equivalence {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Execution-equivalence error is the difference between results of the same study run on different placements (local, GPU, or cluster), governed by the plan's numerical-equivalence policy.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it only with the execution-equivalence target, which accepts no other contribution.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "execution-equivalence",
            "Use executionEquivalence in an accuracy plan",
            "value = fea.ErrorContribution.executionEquivalence();\ncriterion = fea.AccuracyCriterion.new(\"execution_equivalence\", fea.AccuracyTarget.executionEquivalence(), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.referenceComparison(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'execution_equivalence'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Which accuracy targets accept this contribution?",
            "Global equilibrium accepts algebraic, nonlinear, and coupling error; interface closure accepts coupling, interface-projection, and field-transfer error; the numerical-result target accepts everything except observation sampling and execution equivalence; observation and quantity targets accept everything except execution equivalence; the execution-equivalence target accepts only execution-equivalence error.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.AccuracyCriterion.new",
        "fea.AccuracyTarget.globalEquilibrium",
        "fea.AccuracyTarget.numericalResult",
        "fea.AccuracyTolerance.relative",
        "fea.RequiredAccuracyEvidence.residualBound",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.ErrorContribution.executionEquivalence",
        summary: "Name differences between equivalent executions of the same study.",
        description: "`fea.ErrorContribution.executionEquivalence()` returns the closed `fea.ErrorContribution` value `execution_equivalence`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "executionEquivalence",
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
        name: "fea.ErrorContribution.executionEquivalence",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.executionEquivalence()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &recovery::CATALOG_ENTRY,
    &observation_sampling::CATALOG_ENTRY,
    &precision::CATALOG_ENTRY,
    &execution_equivalence::CATALOG_ENTRY,
];
