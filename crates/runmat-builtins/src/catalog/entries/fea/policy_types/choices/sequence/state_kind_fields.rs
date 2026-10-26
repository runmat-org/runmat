//! `fea.StateKind` choices for electromagnetic, acoustic, and fluid states.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod electromagnetic_static {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Static electromagnetic state is a solved electrostatic, current-flow, or magnetostatic field.",
                "Electromagnetic and electrothermal steps can publish it; thermal steps can consume it, for example for Joule heating.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to pass a current-flow solution to a thermal step.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "electromagnetic-static",
            "Use electromagneticStatic in a sequence",
            "value = fea.StateKind.electromagneticStatic();\noutput = fea.StateOutput.new(\"electromagnetic_static\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'electromagnetic_static'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateKind.electromagneticStatic",
        summary: "Declare a static electromagnetic state.",
        description: "`fea.StateKind.electromagneticStatic()` returns the closed `fea.StateKind` value `electromagnetic_static`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "electromagneticStatic",
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
        name: "fea.StateKind.electromagneticStatic",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.electromagneticStatic()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod electromagnetic_complex {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Complex electromagnetic state is a solved time-harmonic electromagnetic field stored as complex phasors.",
                "Electromagnetic and electrothermal steps can publish it; thermal steps can consume it for induced heating.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to pass a harmonic electromagnetic solution to a thermal step.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "electromagnetic-complex",
            "Use electromagneticComplex in a sequence",
            "value = fea.StateKind.electromagneticComplex();\noutput = fea.StateOutput.new(\"electromagnetic_complex\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'electromagnetic_complex'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateKind.electromagneticComplex",
        summary: "Declare a complex (time-harmonic) electromagnetic state.",
        description: "`fea.StateKind.electromagneticComplex()` returns the closed `fea.StateKind` value `electromagnetic_complex`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "electromagneticComplex",
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
        name: "fea.StateKind.electromagneticComplex",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.electromagneticComplex()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod acoustic_complex {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Complex acoustic state is a solved acoustic pressure field stored as complex phasors.",
                "Acoustic steps can publish it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to retain a harmonic acoustic solution for later steps.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "acoustic-complex",
            "Use acousticComplex in a sequence",
            "value = fea.StateKind.acousticComplex();\noutput = fea.StateOutput.new(\"acoustic_complex\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'acoustic_complex'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateKind.acousticComplex",
        summary: "Declare a complex (time-harmonic) acoustic state.",
        description: "`fea.StateKind.acousticComplex()` returns the closed `fea.StateKind` value `acoustic_complex`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "acousticComplex",
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
        name: "fea.StateKind.acousticComplex",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.acousticComplex()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod fluid {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Fluid state is a solved velocity and pressure field.",
                "Fluid, conjugate heat transfer, and fluid-structure steps can publish it; thermal steps can consume it. Steady fluid execution requires one complete fluid output kept until dependents complete.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to hand a flow field to a thermal or coupled step.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "fluid",
            "Use fluid in a sequence",
            "value = fea.StateKind.fluid();\noutput = fea.StateOutput.new(\"fluid\", value, fea.StateCompleteness.complete(), fea.StatePersistence.untilDependentsComplete());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'fluid'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateKind.fluid",
        summary: "Declare a fluid flow state.",
        description: "`fea.StateKind.fluid()` returns the closed `fea.StateKind` value `fluid`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "fluid",
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
        name: "fea.StateKind.fluid",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.fluid()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &electromagnetic_static::CATALOG_ENTRY,
    &electromagnetic_complex::CATALOG_ENTRY,
    &acoustic_complex::CATALOG_ENTRY,
    &fluid::CATALOG_ENTRY,
];
