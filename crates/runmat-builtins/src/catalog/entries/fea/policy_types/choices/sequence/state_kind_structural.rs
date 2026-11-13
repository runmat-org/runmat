//! `fea.StateKind` choices for structural solution states.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod structural_primary {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Structural primary state is the solved displacement (and rotation) field of a structural step, sufficient to restart or continue from it.",
                "Structural, thermomechanical, and fluid-structure steps can publish it. As a prestress input it supplies a deformed reference; as an initial guess it seeds a structural solve.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it as the canonical output of a static step and as the source of an initial guess or continuation.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "structural-primary",
            "Use structuralPrimary in a sequence",
            "value = fea.StateKind.structuralPrimary();\noutput = fea.StateOutput.new(\"structural_primary\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'structural_primary'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateOutput.new",
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateKind.structuralPrimary",
        summary: "Declare the complete structural primary solution state.",
        description: "`fea.StateKind.structuralPrimary()` returns the closed `fea.StateKind` value `structural_primary`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "structuralPrimary",
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
        name: "fea.StateKind.structuralPrimary",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.structuralPrimary()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod structural_kinematics {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Structural kinematic state carries the motion of a structure at the end of a step, including rates needed to start a dynamic step.",
                "Structural, thermomechanical, and fluid-structure steps can publish it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to hand the end state of a transient step to a following dynamic step.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "structural-kinematics",
            "Use structuralKinematics in a sequence",
            "value = fea.StateKind.structuralKinematics();\noutput = fea.StateOutput.new(\"structural_kinematics\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'structural_kinematics'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateOutput.new",
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateKind.structuralKinematics",
        summary: "Declare structural kinematic state (displacement, velocity, acceleration).",
        description: "`fea.StateKind.structuralKinematics()` returns the closed `fea.StateKind` value `structural_kinematics`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "structuralKinematics",
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
        name: "fea.StateKind.structuralKinematics",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.structuralKinematics()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod structural_stress_strain {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Structural stress-strain state is the converged stress and strain field of a structural step.",
                "Structural, thermomechanical, and fluid-structure steps can publish it; a consumer can use it as prestress.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it when a later step needs the stress field itself, for example as prestress.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "structural-stress-strain",
            "Use structuralStressStrain in a sequence",
            "value = fea.StateKind.structuralStressStrain();\noutput = fea.StateOutput.new(\"structural_stress_strain\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'structural_stress_strain'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateOutput.new",
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateKind.structuralStressStrain",
        summary: "Declare structural stress and strain state.",
        description: "`fea.StateKind.structuralStressStrain()` returns the closed `fea.StateKind` value `structural_stress_strain`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "structuralStressStrain",
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
        name: "fea.StateKind.structuralStressStrain",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.structuralStressStrain()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod geometric_stiffness {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Geometric-stiffness state is the stress-dependent stiffness of a preloaded structure.",
                "Structural and thermomechanical steps can publish it. Prestressed modal extraction, eigenvalue buckling, and prestressed harmonic response require it as an explicit prestress input.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to connect a preload step to a prestressed modal, buckling, or harmonic step.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "geometric-stiffness",
            "Use geometricStiffness in a sequence",
            "value = fea.StateKind.geometricStiffness();\noutput = fea.StateOutput.new(\"geometric_stiffness\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'geometric_stiffness'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
        ),
    ];
    const FAQS: &[BuiltinDocumentationFaq] = &[];
    const RELATED: &[&str] = &[
        "fea.StateOutput.new",
        "fea.StateInput.upstream",
        "fea.StateInput.initialConditions",
        "fea.ProcedureStep.configure",
    ];
    const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructorDocumentationDefinition {
        title: "fea.StateKind.geometricStiffness",
        summary: "Declare geometric-stiffness state from a preloaded structure.",
        description: "`fea.StateKind.geometricStiffness()` returns the closed `fea.StateKind` value `geometric_stiffness`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "geometricStiffness",
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
        name: "fea.StateKind.geometricStiffness",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.geometricStiffness()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &structural_primary::CATALOG_ENTRY,
    &structural_kinematics::CATALOG_ENTRY,
    &structural_stress_strain::CATALOG_ENTRY,
    &geometric_stiffness::CATALOG_ENTRY,
];
