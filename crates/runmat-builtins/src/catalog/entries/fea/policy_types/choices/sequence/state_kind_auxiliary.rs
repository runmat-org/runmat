//! `fea.StateKind` choices for meshes, bases, interfaces, and integrators.

use super::shared::*;
use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, section};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::integer::NO_ARGUMENT_AUDIT;
use crate::*;

pub(super) mod moving_mesh {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Moving-mesh state is the mesh motion of an arbitrary Lagrangian-Eulerian fluid or fluid-structure step.",
                "Structural, fluid, and fluid-structure steps can publish it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it to continue a fluid-structure analysis with the deformed fluid mesh.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "moving-mesh",
            "Use movingMesh in a sequence",
            "value = fea.StateKind.movingMesh();\noutput = fea.StateOutput.new(\"moving_mesh\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'moving_mesh'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateKind.movingMesh",
        summary: "Declare a moving-mesh state.",
        description: "`fea.StateKind.movingMesh()` returns the closed `fea.StateKind` value `moving_mesh`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "movingMesh",
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
        name: "fea.StateKind.movingMesh",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.movingMesh()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod modal_basis {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Modal-basis state is a set of extracted eigenmodes with their frequencies and modal masses.",
                "Structural steps can publish it; modal extraction requires exactly one complete, publishable modal-basis output. Modal-reduced steps (modal harmonic, response spectrum, random response) require a modal-basis input with use `modalBasis`.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it on a modal extraction step and on the modal-reduced steps that consume its modes.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "modal-basis",
            "Use modalBasis in a sequence",
            "value = fea.StateKind.modalBasis();\noutput = fea.StateOutput.new(\"modal_basis\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'modal_basis'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateKind.modalBasis",
        summary: "Declare a modal basis (eigenpairs) state.",
        description: "`fea.StateKind.modalBasis()` returns the closed `fea.StateKind` value `modal_basis`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "modalBasis",
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
        name: "fea.StateKind.modalBasis",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.modalBasis()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod coupled_interface {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Coupled-interface state is the exchanged data on a multiphysics interface, such as tractions and displacements or heat flux and temperature.",
                "Coupled steps can publish it; structural, thermal, and fluid steps can consume it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with state use `coupledInterface` to pass interface data between partitioned steps.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "coupled-interface",
            "Use coupledInterface in a sequence",
            "value = fea.StateKind.coupledInterface();\noutput = fea.StateOutput.new(\"coupled_interface\", value, fea.StateCompleteness.complete(), fea.StatePersistence.canonical());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'coupled_interface'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateKind.coupledInterface",
        summary: "Declare coupled-interface state.",
        description: "`fea.StateKind.coupledInterface()` returns the closed `fea.StateKind` value `coupled_interface`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "coupledInterface",
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
        name: "fea.StateKind.coupledInterface",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.coupledInterface()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) mod integrator {
    use super::*;

    const SECTIONS: &[BuiltinDocumentationSection] = &[
        section(
            "Behavior",
            &[
                "Integrator state is the internal history of a time integrator (previous steps and rates) needed to continue a transient exactly.",
                "Every physics family can publish it.",
            ],
        ),
        section(
            "When to use it",
            &[
                "Use it with checkpoint completeness to resume a long transient.",
            ],
        ),
        USE_STATE_OUTPUT,
    ];
    const EXAMPLES: &[BuiltinExample] = &[
        example(
            "integrator",
            "Use integrator in a sequence",
            "value = fea.StateKind.integrator();\noutput = fea.StateOutput.new(\"integrator\", value, fea.StateCompleteness.checkpoint(), fea.StatePersistence.checkpointOnly());",
            "assert(strcmp(class(value), 'fea.StateKind'));\nassert(strcmp(value.value, 'integrator'));\nassert(strcmp(class(output), 'fea.StateOutput'));",
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
        title: "fea.StateKind.integrator",
        summary: "Declare time-integrator state.",
        description: "`fea.StateKind.integrator()` returns the closed `fea.StateKind` value `integrator`. It takes no arguments.",
        keywords: &[
            "fea",
            "sequence",
            "state",
            "StateKind",
            "integrator",
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
        name: "fea.StateKind.integrator",
        result: crate::fea_contract::study_types::classes::FEA_STATE_KIND_CLASS,
        signature: "value = fea.StateKind.integrator()",
        inputs: [],
        output: ("value", "Typed fea.StateKind value."),
        integer_capabilities: &[],
        integer_audit: Some(&NO_ARGUMENT_AUDIT),
        documentation: DOCUMENTATION,
    );
}

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &moving_mesh::CATALOG_ENTRY,
    &modal_basis::CATALOG_ENTRY,
    &coupled_interface::CATALOG_ENTRY,
    &integrator::CATALOG_ENTRY,
];
