//! `fea.ErrorContribution` choices for solver and time/frequency error.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod algebraic {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Algebraic error is the residual left by the linear solver in the assembled equilibrium equations.",
                "Linear structural, thermal, electrical, magnetostatic, Maxwell harmonic, acoustic harmonic, and steady fluid solves report residual-bound evidence for it, and most also report a conservation balance.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with a residual-bound or conservation-balance criterion at the global-equilibrium target; it is the most widely produced evidence in RunMat.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "algebraic",
            "Use algebraic in an accuracy plan",
            "value = fea.ErrorContribution.algebraic();\ncriterion = fea.AccuracyCriterion.new(\"algebraic\", fea.AccuracyTarget.globalEquilibrium(), value, fea.AccuracyTolerance.relative(1e-8), fea.RequiredAccuracyEvidence.residualBound(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'algebraic'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.algebraic",
        summary: "Name error from solving the discrete linear system inexactly.",
        description: "`fea.ErrorContribution.algebraic()` returns the closed `fea.ErrorContribution` value `algebraic`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "algebraic",
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
        name: "fea.ErrorContribution.algebraic",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.algebraic()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod nonlinear {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Nonlinear error is the equilibrium residual and pending Newton correction left when nonlinear iterations stop.",
                "Nonlinear solid and nonlinear thermal solves report residual-bound evidence for it; it measures nonlinear closure only, not mesh or load-step accuracy.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for nonlinear solid, contact, and nonlinear thermal steps at the global-equilibrium target.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "nonlinear",
            "Use nonlinear in an accuracy plan",
            "value = fea.ErrorContribution.nonlinear();\ncriterion = fea.AccuracyCriterion.new(\"nonlinear\", fea.AccuracyTarget.globalEquilibrium(), value, fea.AccuracyTolerance.relative(1e-6), fea.RequiredAccuracyEvidence.residualBound(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'nonlinear'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.nonlinear",
        summary: "Name error from stopping nonlinear iterations before exact equilibrium.",
        description: "`fea.ErrorContribution.nonlinear()` returns the closed `fea.ErrorContribution` value `nonlinear`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "nonlinear",
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
        name: "fea.ErrorContribution.nonlinear",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.nonlinear()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod temporal {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Temporal error comes from the time step and integrator in transient procedures.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for transient histories where the time step is an accuracy decision rather than a stability limit.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "temporal",
            "Use temporal in an accuracy plan",
            "value = fea.ErrorContribution.temporal();\ncriterion = fea.AccuracyCriterion.new(\"temporal\", fea.AccuracyTarget.observation(\"tip_displacement\"), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'temporal'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.temporal",
        summary: "Name error from time discretization.",
        description: "`fea.ErrorContribution.temporal()` returns the closed `fea.ErrorContribution` value `temporal`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "temporal",
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
        name: "fea.ErrorContribution.temporal",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.temporal()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod frequency_sampling {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Frequency-sampling error comes from integrating or enveloping a frequency response with finitely many frequency points.",
                "Random-response steps report a frequency-sampling convergence study on the numerical result.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for random-response steps, whose variance integrates a PSD over frequency.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "frequency-sampling",
            "Use frequencySampling in an accuracy plan",
            "value = fea.ErrorContribution.frequencySampling();\ncriterion = fea.AccuracyCriterion.new(\"frequency_sampling\", fea.AccuracyTarget.numericalResult(), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'frequency_sampling'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.frequencySampling",
        summary: "Name error from sampling a frequency range at discrete points.",
        description: "`fea.ErrorContribution.frequencySampling()` returns the closed `fea.ErrorContribution` value `frequency_sampling`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "frequencySampling",
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
        name: "fea.ErrorContribution.frequencySampling",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.frequencySampling()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod modal_truncation {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Modal-truncation error is the response missing because modes above the retained set are left out.",
                "Modal response-spectrum steps report a modal-truncation convergence study on the numerical result.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for modal response-spectrum and modal superposition steps, alongside the mass-ratio requirement of `fea.ModalTruncation.new`.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "modal-truncation",
            "Use modalTruncation in an accuracy plan",
            "value = fea.ErrorContribution.modalTruncation();\ncriterion = fea.AccuracyCriterion.new(\"modal_truncation\", fea.AccuracyTarget.numericalResult(), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'modal_truncation'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.modalTruncation",
        summary: "Name error from truncating the modal basis.",
        description: "`fea.ErrorContribution.modalTruncation()` returns the closed `fea.ErrorContribution` value `modal_truncation`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "modalTruncation",
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
        name: "fea.ErrorContribution.modalTruncation",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.modalTruncation()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &algebraic::CATALOG_ENTRY,
    &nonlinear::CATALOG_ENTRY,
    &temporal::CATALOG_ENTRY,
    &frequency_sampling::CATALOG_ENTRY,
    &modal_truncation::CATALOG_ENTRY,
];
