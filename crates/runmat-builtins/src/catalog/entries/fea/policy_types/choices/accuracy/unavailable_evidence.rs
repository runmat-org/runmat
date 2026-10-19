//! `fea.UnavailableEvidencePolicy` choices: missing evidence.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod fail {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "When a mandatory criterion has no matching evidence, the plan is indeterminate and publication is blocked, for production and experimental formulations alike.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for production acceptance, where an unverified result must not be published.",
            ],
        ),
        USE_PLAN,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "fail",
            "Use fail in an accuracy plan",
            "value = fea.UnavailableEvidencePolicy.fail();\ncriterion = fea.AccuracyCriterion.new(\"equilibrium\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), ...\n    fea.AccuracyTolerance.relative(1e-8), fea.RequiredAccuracyEvidence.residualBound(), true);\nplan = fea.AccuracyPlan.new(\"acceptance\", {criterion}, value, fea.AccuracyLimitAction.fail(), ...\n    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), fea.AccuracyResourceCeiling.new(0, 0, 100, 600000, 2000000000));",
            "assert(strcmp(class(value), 'fea.UnavailableEvidencePolicy'));\nassert(strcmp(value.value, 'fail'));\nassert(strcmp(class(plan), 'fea.AccuracyPlan'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.UnavailableEvidencePolicy.markIndeterminateExperimental",
        "fea.AccuracyPlan.new",
        "fea.AccuracyResourceCeiling.new",
        "fea.AccuracyPlan.experimentalBaseline",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.UnavailableEvidencePolicy.fail",
        summary: "Block publication when required evidence is missing.",
        description: "`fea.UnavailableEvidencePolicy.fail()` returns the closed `fea.UnavailableEvidencePolicy` value `fail`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "UnavailableEvidencePolicy",
            "fail",
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
        name: "fea.UnavailableEvidencePolicy.fail",
        result: crate::fea_contract::policy_types::classes::UNAVAILABLE_EVIDENCE_POLICY,
        signature: "value = fea.UnavailableEvidencePolicy.fail()",
        inputs: [],
        output: ("value", "Typed fea.UnavailableEvidencePolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod mark_indeterminate_experimental {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "For an experimental formulation, an indeterminate plan (a mandatory criterion without evidence) still publishes, and the result is marked degraded. A production formulation is still blocked.",
                "It is the only unavailable-evidence policy that permits the `warnExperimental` limit action, and it is what `fea.AccuracyPlan.experimentalBaseline(id)` uses.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it while developing or diagnosing a formulation whose evidence producers are not complete yet.",
            ],
        ),
        USE_PLAN,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "mark-indeterminate-experimental",
            "Use markIndeterminateExperimental in an accuracy plan",
            "value = fea.UnavailableEvidencePolicy.markIndeterminateExperimental();\ncriterion = fea.AccuracyCriterion.new(\"equilibrium\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), ...\n    fea.AccuracyTolerance.relative(1e-8), fea.RequiredAccuracyEvidence.residualBound(), true);\nplan = fea.AccuracyPlan.new(\"acceptance\", {criterion}, value, fea.AccuracyLimitAction.warnExperimental(), ...\n    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), fea.AccuracyResourceCeiling.new(0, 0, 100, 600000, 2000000000));",
            "assert(strcmp(class(value), 'fea.UnavailableEvidencePolicy'));\nassert(strcmp(value.value, 'mark_indeterminate_experimental'));\nassert(strcmp(class(plan), 'fea.AccuracyPlan'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Does this make an experimental result publishable as production?",
            "No. Formulation maturity and publication status stay independent; this policy only lets an explicitly experimental result publish as degraded.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.UnavailableEvidencePolicy.fail",
        "fea.AccuracyPlan.new",
        "fea.AccuracyResourceCeiling.new",
        "fea.AccuracyPlan.experimentalBaseline",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.UnavailableEvidencePolicy.markIndeterminateExperimental",
        summary: "Publish an indeterminate experimental result as degraded instead of failing.",
        description: "`fea.UnavailableEvidencePolicy.markIndeterminateExperimental()` returns the closed `fea.UnavailableEvidencePolicy` value `mark_indeterminate_experimental`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "UnavailableEvidencePolicy",
            "markIndeterminateExperimental",
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
        name: "fea.UnavailableEvidencePolicy.markIndeterminateExperimental",
        result: crate::fea_contract::policy_types::classes::UNAVAILABLE_EVIDENCE_POLICY,
        signature: "value = fea.UnavailableEvidencePolicy.markIndeterminateExperimental()",
        inputs: [],
        output: ("value", "Typed fea.UnavailableEvidencePolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &fail::CATALOG_ENTRY,
    &mark_indeterminate_experimental::CATALOG_ENTRY,
];
