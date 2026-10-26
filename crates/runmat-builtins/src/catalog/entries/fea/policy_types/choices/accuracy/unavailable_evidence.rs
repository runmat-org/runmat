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
                "When a mandatory criterion has no matching evidence, the plan is indeterminate and publication is blocked.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it for acceptance where an unverified result must not be published.",
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
        "fea.UnavailableEvidencePolicy.markIndeterminate",
        "fea.AccuracyPlan.new",
        "fea.AccuracyResourceCeiling.new",
        "fea.AccuracyPlan.baseline",
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

pub(super) mod mark_indeterminate {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "An indeterminate plan (a mandatory criterion without evidence) still publishes, with its accuracy status reported as indeterminate and the result marked degraded. An unsatisfied plan is still blocked.",
                "It is what `fea.AccuracyPlan.baseline(id)` uses.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when a study should still publish its fields while reporting that a criterion could not be evaluated. Use `fail` when every mandatory criterion must be established before anything publishes.",
            ],
        ),
        USE_PLAN,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "mark-indeterminate",
            "Use markIndeterminate in an accuracy plan",
            "value = fea.UnavailableEvidencePolicy.markIndeterminate();\ncriterion = fea.AccuracyCriterion.new(\"equilibrium\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), ...\n    fea.AccuracyTolerance.relative(1e-8), fea.RequiredAccuracyEvidence.residualBound(), true);\nplan = fea.AccuracyPlan.new(\"acceptance\", {criterion}, value, fea.AccuracyLimitAction.fail(), ...\n    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), fea.AccuracyResourceCeiling.new(0, 0, 100, 600000, 2000000000));",
            "assert(strcmp(class(value), 'fea.UnavailableEvidencePolicy'));\nassert(strcmp(value.value, 'mark_indeterminate'));\nassert(strcmp(class(plan), 'fea.AccuracyPlan'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[
        faq(
            "Does this let a result that misses its tolerance publish?",
            "No. It only governs criteria whose evidence is unavailable. A criterion that is evaluated and not satisfied blocks publication.",
        ),
    ];
    const RELATED: &[&str] = &[
        "fea.UnavailableEvidencePolicy.fail",
        "fea.AccuracyPlan.new",
        "fea.AccuracyResourceCeiling.new",
        "fea.AccuracyPlan.baseline",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.UnavailableEvidencePolicy.markIndeterminate",
        summary: "Publish a result whose evidence is unavailable as indeterminate instead of failing.",
        description: "`fea.UnavailableEvidencePolicy.markIndeterminate()` returns the closed `fea.UnavailableEvidencePolicy` value `mark_indeterminate`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "UnavailableEvidencePolicy",
            "markIndeterminate",
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
        name: "fea.UnavailableEvidencePolicy.markIndeterminate",
        result: crate::fea_contract::policy_types::classes::UNAVAILABLE_EVIDENCE_POLICY,
        signature: "value = fea.UnavailableEvidencePolicy.markIndeterminate()",
        inputs: [],
        output: ("value", "Typed fea.UnavailableEvidencePolicy value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&fail::CATALOG_ENTRY, &mark_indeterminate::CATALOG_ENTRY];
