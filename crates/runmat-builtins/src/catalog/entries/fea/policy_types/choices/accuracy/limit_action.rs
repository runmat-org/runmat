//! `fea.AccuracyLimitAction` choices: unsatisfied accuracy.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod refine {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "When a criterion is not satisfied, structural adaptive refinement continues with another refinement cycle until the plan is satisfied or `maximumRefinementCycles` from `fea.AccuracyResourceCeiling.new` is exhausted; then the result is rejected.",
                "Plan validation requires `maximumRefinementCycles > 0` with this action.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with spatial-discretization criteria on adaptive structural studies.",
            ],
        ),
        USE_PLAN,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "refine",
            "Use refine in an accuracy plan",
            "value = fea.AccuracyLimitAction.refine();\ncriterion = fea.AccuracyCriterion.new(\"equilibrium\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), ...\n    fea.AccuracyTolerance.relative(1e-8), fea.RequiredAccuracyEvidence.residualBound(), true);\nplan = fea.AccuracyPlan.new(\"acceptance\", {criterion}, fea.UnavailableEvidencePolicy.fail(), value, ...\n    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), fea.AccuracyResourceCeiling.new(3, 0, 100, 600000, 2000000000));",
            "assert(strcmp(class(value), 'fea.AccuracyLimitAction'));\nassert(strcmp(value.value, 'refine'));\nassert(strcmp(class(plan), 'fea.AccuracyPlan'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.AccuracyLimitAction.tightenSolver",
        "fea.AccuracyLimitAction.extendSampling",
        "fea.AccuracyLimitAction.fail",
        "fea.AccuracyPlan.new",
        "fea.AccuracyResourceCeiling.new",
        "fea.AccuracyPlan.baseline",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AccuracyLimitAction.refine",
        summary: "Refine the discretization when accuracy is not met.",
        description: "`fea.AccuracyLimitAction.refine()` returns the closed `fea.AccuracyLimitAction` value `refine`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "AccuracyLimitAction",
            "refine",
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
        name: "fea.AccuracyLimitAction.refine",
        result: crate::fea_contract::policy_types::classes::ACCURACY_LIMIT_ACTION,
        signature: "value = fea.AccuracyLimitAction.refine()",
        inputs: [],
        output: ("value", "Typed fea.AccuracyLimitAction value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod tighten_solver {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Plan validation requires `maximumSolverTightenings > 0` with this action. RunMat records the action in the plan, but no current procedure performs automatic solver tightening: an unsatisfied plan is not published.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to state the intended remedy for algebraic or nonlinear criteria.",
            ],
        ),
        USE_PLAN,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "tighten-solver",
            "Use tightenSolver in an accuracy plan",
            "value = fea.AccuracyLimitAction.tightenSolver();\ncriterion = fea.AccuracyCriterion.new(\"equilibrium\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), ...\n    fea.AccuracyTolerance.relative(1e-8), fea.RequiredAccuracyEvidence.residualBound(), true);\nplan = fea.AccuracyPlan.new(\"acceptance\", {criterion}, fea.UnavailableEvidencePolicy.fail(), value, ...\n    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), fea.AccuracyResourceCeiling.new(0, 2, 100, 600000, 2000000000));",
            "assert(strcmp(class(value), 'fea.AccuracyLimitAction'));\nassert(strcmp(value.value, 'tighten_solver'));\nassert(strcmp(class(plan), 'fea.AccuracyPlan'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.AccuracyLimitAction.refine",
        "fea.AccuracyLimitAction.extendSampling",
        "fea.AccuracyLimitAction.fail",
        "fea.AccuracyPlan.new",
        "fea.AccuracyResourceCeiling.new",
        "fea.AccuracyPlan.baseline",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AccuracyLimitAction.tightenSolver",
        summary: "Request tighter solver tolerances when accuracy is not met.",
        description: "`fea.AccuracyLimitAction.tightenSolver()` returns the closed `fea.AccuracyLimitAction` value `tighten_solver`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "AccuracyLimitAction",
            "tightenSolver",
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
        name: "fea.AccuracyLimitAction.tightenSolver",
        result: crate::fea_contract::policy_types::classes::ACCURACY_LIMIT_ACTION,
        signature: "value = fea.AccuracyLimitAction.tightenSolver()",
        inputs: [],
        output: ("value", "Typed fea.AccuracyLimitAction value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod extend_sampling {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Plan validation requires a positive `maximumSamples` with this action. RunMat records the action, but no current procedure extends frequency or time sampling automatically: an unsatisfied plan is not published.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to state the intended remedy for frequency-sampling or observation-sampling criteria.",
            ],
        ),
        USE_PLAN,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "extend-sampling",
            "Use extendSampling in an accuracy plan",
            "value = fea.AccuracyLimitAction.extendSampling();\ncriterion = fea.AccuracyCriterion.new(\"equilibrium\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), ...\n    fea.AccuracyTolerance.relative(1e-8), fea.RequiredAccuracyEvidence.residualBound(), true);\nplan = fea.AccuracyPlan.new(\"acceptance\", {criterion}, fea.UnavailableEvidencePolicy.fail(), value, ...\n    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), fea.AccuracyResourceCeiling.new(0, 0, 100, 600000, 2000000000));",
            "assert(strcmp(class(value), 'fea.AccuracyLimitAction'));\nassert(strcmp(value.value, 'extend_sampling'));\nassert(strcmp(class(plan), 'fea.AccuracyPlan'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.AccuracyLimitAction.refine",
        "fea.AccuracyLimitAction.tightenSolver",
        "fea.AccuracyLimitAction.fail",
        "fea.AccuracyPlan.new",
        "fea.AccuracyResourceCeiling.new",
        "fea.AccuracyPlan.baseline",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AccuracyLimitAction.extendSampling",
        summary: "Request additional sampling when accuracy is not met.",
        description: "`fea.AccuracyLimitAction.extendSampling()` returns the closed `fea.AccuracyLimitAction` value `extend_sampling`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "AccuracyLimitAction",
            "extendSampling",
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
        name: "fea.AccuracyLimitAction.extendSampling",
        result: crate::fea_contract::policy_types::classes::ACCURACY_LIMIT_ACTION,
        signature: "value = fea.AccuracyLimitAction.extendSampling()",
        inputs: [],
        output: ("value", "Typed fea.AccuracyLimitAction value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod fail {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &["An unsatisfied plan blocks publication. No ceiling is required."],
        ),
        section(
            "When to use it",
            &["Use it for acceptance with a fixed discretization."],
        ),
        USE_PLAN,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "fail",
            "Use fail in an accuracy plan",
            "value = fea.AccuracyLimitAction.fail();\ncriterion = fea.AccuracyCriterion.new(\"equilibrium\", fea.AccuracyTarget.globalEquilibrium(), fea.ErrorContribution.algebraic(), ...\n    fea.AccuracyTolerance.relative(1e-8), fea.RequiredAccuracyEvidence.residualBound(), true);\nplan = fea.AccuracyPlan.new(\"acceptance\", {criterion}, fea.UnavailableEvidencePolicy.fail(), value, ...\n    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), fea.AccuracyResourceCeiling.new(0, 0, 100, 600000, 2000000000));",
            "assert(strcmp(class(value), 'fea.AccuracyLimitAction'));\nassert(strcmp(value.value, 'fail'));\nassert(strcmp(class(plan), 'fea.AccuracyPlan'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.AccuracyLimitAction.refine",
        "fea.AccuracyLimitAction.tightenSolver",
        "fea.AccuracyLimitAction.extendSampling",
        "fea.AccuracyPlan.new",
        "fea.AccuracyResourceCeiling.new",
        "fea.AccuracyPlan.baseline",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.AccuracyLimitAction.fail",
        summary: "Fail when accuracy is not met.",
        description: "`fea.AccuracyLimitAction.fail()` returns the closed `fea.AccuracyLimitAction` value `fail`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "AccuracyLimitAction",
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
        name: "fea.AccuracyLimitAction.fail",
        result: crate::fea_contract::policy_types::classes::ACCURACY_LIMIT_ACTION,
        signature: "value = fea.AccuracyLimitAction.fail()",
        inputs: [],
        output: ("value", "Typed fea.AccuracyLimitAction value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &refine::CATALOG_ENTRY,
    &tighten_solver::CATALOG_ENTRY,
    &extend_sampling::CATALOG_ENTRY,
    &fail::CATALOG_ENTRY,
];
