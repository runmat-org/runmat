//! `fea.ErrorContribution` choices for stochastic, coupling, and transfer error.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod stochastic {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Stochastic error is the sampling uncertainty of statistics estimated from a finite number of random realizations.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with statistical-confidence evidence for sampled uncertainty studies.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "stochastic",
            "Use stochastic in an accuracy plan",
            "value = fea.ErrorContribution.stochastic();\ncriterion = fea.AccuracyCriterion.new(\"stochastic\", fea.AccuracyTarget.numericalResult(), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.statisticalConfidence(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'stochastic'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.stochastic",
        summary: "Name error from finite random sampling.",
        description: "`fea.ErrorContribution.stochastic()` returns the closed `fea.ErrorContribution` value `stochastic`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "stochastic",
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
        name: "fea.ErrorContribution.stochastic",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.stochastic()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod coupling {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Coupling error is the mismatch left when partitioned multiphysics iterations stop before the coupled fields agree.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it at the global-equilibrium or interface-closure target for coupled steps.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "coupling",
            "Use coupling in an accuracy plan",
            "value = fea.ErrorContribution.coupling();\ncriterion = fea.AccuracyCriterion.new(\"coupling\", fea.AccuracyTarget.interfaceClosure(), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.transferBound(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'coupling'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.coupling",
        summary: "Name error from incomplete coupling between physics fields.",
        description: "`fea.ErrorContribution.coupling()` returns the closed `fea.ErrorContribution` value `coupling`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "coupling",
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
        name: "fea.ErrorContribution.coupling",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.coupling()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod interface_projection {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Interface-projection error comes from mapping loads or states between non-matching meshes on a shared interface.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it at the interface-closure target for fluid-structure or conjugate heat transfer interfaces.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "interface-projection",
            "Use interfaceProjection in an accuracy plan",
            "value = fea.ErrorContribution.interfaceProjection();\ncriterion = fea.AccuracyCriterion.new(\"interface_projection\", fea.AccuracyTarget.interfaceClosure(), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.transferBound(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'interface_projection'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.interfaceProjection",
        summary: "Name error from projecting fields across non-matching interfaces.",
        description: "`fea.ErrorContribution.interfaceProjection()` returns the closed `fea.ErrorContribution` value `interface_projection`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "interfaceProjection",
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
        name: "fea.ErrorContribution.interfaceProjection",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.interfaceProjection()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod field_transfer {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Field-transfer error comes from moving a solved field onto another mesh, for example after adaptive refinement or between sequence steps.",
                "Structural adaptive refinement reports transfer-bound evidence for fields carried between refinement levels.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with transfer-bound evidence whenever a result depends on a transferred state.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "field-transfer",
            "Use fieldTransfer in an accuracy plan",
            "value = fea.ErrorContribution.fieldTransfer();\ncriterion = fea.AccuracyCriterion.new(\"field_transfer\", fea.AccuracyTarget.interfaceClosure(), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.transferBound(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'field_transfer'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.fieldTransfer",
        summary: "Name error from transferring a field between meshes or steps.",
        description: "`fea.ErrorContribution.fieldTransfer()` returns the closed `fea.ErrorContribution` value `field_transfer`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "fieldTransfer",
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
        name: "fea.ErrorContribution.fieldTransfer",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.fieldTransfer()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod signal_interpolation {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Signal-interpolation error is the deviation between an input history and the interpolated table the solver samples.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when a coarsely tabulated or generated load history drives the governing response.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "signal-interpolation",
            "Use signalInterpolation in an accuracy plan",
            "value = fea.ErrorContribution.signalInterpolation();\ncriterion = fea.AccuracyCriterion.new(\"signal_interpolation\", fea.AccuracyTarget.observation(\"tip_displacement\"), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'signal_interpolation'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.signalInterpolation",
        summary: "Name error from interpolating tabulated input signals.",
        description: "`fea.ErrorContribution.signalInterpolation()` returns the closed `fea.ErrorContribution` value `signal_interpolation`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "signalInterpolation",
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
        name: "fea.ErrorContribution.signalInterpolation",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.signalInterpolation()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &stochastic::CATALOG_ENTRY,
    &coupling::CATALOG_ENTRY,
    &interface_projection::CATALOG_ENTRY,
    &field_transfer::CATALOG_ENTRY,
    &signal_interpolation::CATALOG_ENTRY,
];
