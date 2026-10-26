//! `fea.ErrorContribution` choices for geometry and discretization error.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, faq, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod geometry_admission {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Geometry-admission error is the difference between the design intent and the admitted analysis geometry: tolerance-based healing, defeaturing, and facet approximation of CAD surfaces.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to bound how far an imported or healed geometry may deviate before a result is trusted, typically against an observation near the affected feature.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "geometry-admission",
            "Use geometryAdmission in an accuracy plan",
            "value = fea.ErrorContribution.geometryAdmission();\ncriterion = fea.AccuracyCriterion.new(\"geometry_admission\", fea.AccuracyTarget.observation(\"tip_displacement\"), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'geometry_admission'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.geometryAdmission",
        summary: "Name error from admitting or healing the source geometry.",
        description: "`fea.ErrorContribution.geometryAdmission()` returns the closed `fea.ErrorContribution` value `geometry_admission`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "geometryAdmission",
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
        name: "fea.ErrorContribution.geometryAdmission",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.geometryAdmission()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod boundary_approximation {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Boundary-approximation error comes from representing curved boundaries with element faces and from projecting boundary loads and constraints onto them.",
                "No current RunMat solver produces evidence for this contribution, so a criterion naming it is reported indeterminate and the plan's unavailable-evidence policy applies.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when stresses or fluxes on curved surfaces drive acceptance.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "boundary-approximation",
            "Use boundaryApproximation in an accuracy plan",
            "value = fea.ErrorContribution.boundaryApproximation();\ncriterion = fea.AccuracyCriterion.new(\"boundary_approximation\", fea.AccuracyTarget.observation(\"tip_displacement\"), value, fea.AccuracyTolerance.relative(1e-3), fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'boundary_approximation'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.boundaryApproximation",
        summary: "Name error from approximating curved boundaries and boundary data.",
        description: "`fea.ErrorContribution.boundaryApproximation()` returns the closed `fea.ErrorContribution` value `boundary_approximation`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "boundaryApproximation",
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
        name: "fea.ErrorContribution.boundaryApproximation",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.boundaryApproximation()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod spatial_discretization {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Spatial-discretization error is the difference between the discrete and exact solution caused by mesh size and element order.",
                "Structural adaptive refinement produces convergence-study evidence for this contribution on the paired target quantity; it is convergence evidence, not an exact error bound.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with a convergence-study criterion on the quantity you report; structural adaptive refinement drives the mesh until it is met.",
            ],
        ),
        USE_CRITERION,
        EVIDENCE_PRODUCERS,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "spatial-discretization",
            "Use spatialDiscretization in an accuracy plan",
            "value = fea.ErrorContribution.spatialDiscretization();\ncriterion = fea.AccuracyCriterion.new(\"spatial_discretization\", fea.AccuracyTarget.observation(\"tip_displacement\"), value, fea.AccuracyTolerance.relative(0.01), fea.RequiredAccuracyEvidence.convergenceStudy(), true);",
            "assert(strcmp(class(value), 'fea.ErrorContribution'));\nassert(strcmp(value.value, 'spatial_discretization'));\nassert(strcmp(class(criterion), 'fea.AccuracyCriterion'));",
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
        title: "fea.ErrorContribution.spatialDiscretization",
        summary: "Name error from the finite-element mesh.",
        description: "`fea.ErrorContribution.spatialDiscretization()` returns the closed `fea.ErrorContribution` value `spatial_discretization`. It takes no arguments.",
        keywords: &[
            "fea",
            "accuracy",
            "ErrorContribution",
            "spatialDiscretization",
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
        name: "fea.ErrorContribution.spatialDiscretization",
        result: crate::fea_contract::policy_types::classes::ERROR_CONTRIBUTION,
        signature: "value = fea.ErrorContribution.spatialDiscretization()",
        inputs: [],
        output: ("value", "Typed fea.ErrorContribution value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &geometry_admission::CATALOG_ENTRY,
    &boundary_approximation::CATALOG_ENTRY,
    &spatial_discretization::CATALOG_ENTRY,
];
