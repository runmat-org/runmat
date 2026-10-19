//! `fea.RequiredAccuracyEvidence` choices for reference, statistical, and transfer evidence.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod reference_comparison {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A reference comparison measures the deviation from an analytical, manufactured, or independently computed reference result.",
                "No current RunMat solver produces it automatically; criteria requiring it are indeterminate.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for verification studies and execution-equivalence checks.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "reference-comparison",
            "Use referenceComparison in an accuracy plan",
            "value = fea.RequiredAccuracyEvidence.referenceComparison();\ncriterion = fea.AccuracyCriterion.new(\"reference_comparison\", fea.AccuracyTarget.numericalResult(), fea.ErrorContribution.precision(), fea.AccuracyTolerance.relative(1e-3), value, true);",
            "assert(strcmp(class(value), 'fea.RequiredAccuracyEvidence'));\nassert(strcmp(value.value, 'reference_comparison'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.RequiredAccuracyEvidence.referenceComparison",
        summary: "Require comparison against a reference solution.",
        description: "`fea.RequiredAccuracyEvidence.referenceComparison()` returns the closed `fea.RequiredAccuracyEvidence` value `reference_comparison`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "RequiredAccuracyEvidence",
            "referenceComparison",
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
        name: "fea.RequiredAccuracyEvidence.referenceComparison",
        result: crate::fea_contract::policy_types::classes::REQUIRED_ACCURACY_EVIDENCE,
        signature: "value = fea.RequiredAccuracyEvidence.referenceComparison()",
        inputs: [],
        output: ("value", "Typed fea.RequiredAccuracyEvidence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod statistical_confidence {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A statistical confidence bound expresses the sampling uncertainty of an estimated statistic at a stated confidence level.",
                "No current RunMat solver produces it; criteria requiring it are indeterminate.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with the stochastic contribution.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "statistical-confidence",
            "Use statisticalConfidence in an accuracy plan",
            "value = fea.RequiredAccuracyEvidence.statisticalConfidence();\ncriterion = fea.AccuracyCriterion.new(\"statistical_confidence\", fea.AccuracyTarget.numericalResult(), fea.ErrorContribution.stochastic(), fea.AccuracyTolerance.relative(1e-3), value, true);",
            "assert(strcmp(class(value), 'fea.RequiredAccuracyEvidence'));\nassert(strcmp(value.value, 'statistical_confidence'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.RequiredAccuracyEvidence.statisticalConfidence",
        summary: "Require a statistical confidence bound.",
        description: "`fea.RequiredAccuracyEvidence.statisticalConfidence()` returns the closed `fea.RequiredAccuracyEvidence` value `statistical_confidence`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "RequiredAccuracyEvidence",
            "statisticalConfidence",
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
        name: "fea.RequiredAccuracyEvidence.statisticalConfidence",
        result: crate::fea_contract::policy_types::classes::REQUIRED_ACCURACY_EVIDENCE,
        signature: "value = fea.RequiredAccuracyEvidence.statisticalConfidence()",
        inputs: [],
        output: ("value", "Typed fea.RequiredAccuracyEvidence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod transfer_bound {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "A transfer bound limits the error introduced when a field is mapped between meshes or steps.",
                "Structural adaptive refinement produces it for field transfer between refinement levels.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with field-transfer, interface-projection, or coupling criteria.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "transfer-bound",
            "Use transferBound in an accuracy plan",
            "value = fea.RequiredAccuracyEvidence.transferBound();\ncriterion = fea.AccuracyCriterion.new(\"transfer_bound\", fea.AccuracyTarget.interfaceClosure(), fea.ErrorContribution.fieldTransfer(), fea.AccuracyTolerance.relative(1e-3), value, true);",
            "assert(strcmp(class(value), 'fea.RequiredAccuracyEvidence'));\nassert(strcmp(value.value, 'transfer_bound'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.RequiredAccuracyEvidence.transferBound",
        summary: "Require a bound on field-transfer error.",
        description: "`fea.RequiredAccuracyEvidence.transferBound()` returns the closed `fea.RequiredAccuracyEvidence` value `transfer_bound`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "RequiredAccuracyEvidence",
            "transferBound",
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
        name: "fea.RequiredAccuracyEvidence.transferBound",
        result: crate::fea_contract::policy_types::classes::REQUIRED_ACCURACY_EVIDENCE,
        signature: "value = fea.RequiredAccuracyEvidence.transferBound()",
        inputs: [],
        output: ("value", "Typed fea.RequiredAccuracyEvidence value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &reference_comparison::CATALOG_ENTRY,
    &statistical_confidence::CATALOG_ENTRY,
    &transfer_bound::CATALOG_ENTRY,
];
