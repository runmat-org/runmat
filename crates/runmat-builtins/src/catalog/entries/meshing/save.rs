//! `meshing.save` catalog identity.

use super::shared::*;
use crate::catalog::argument_error;
use crate::*;
use runmat_types::{
    CallInference, CallRequest, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    arity(
        &mut diagnostics,
        "SAVE",
        name,
        request.arguments.len(),
        3,
        None,
    );
    require_mesh(&mut diagnostics, "SAVE", name, request, 0);
    require_text(&mut diagnostics, "SAVE", name, request, 1);
    require_text(&mut diagnostics, "SAVE", name, request, 2);
    options(&mut diagnostics, "SAVE", name, request, 3, &["force"]);
    if request.arguments.len() > 5 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MESHING-SAVE-OPTION",
            "meshing.save accepts at most one Force option",
            5,
        ));
    }
    if let Some(force) = request.arguments.get(4) {
        if !matches!(force.kind, ValueKindFact::Logical | ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MESHING-SAVE-FORCE",
                "Force must be logical",
                4,
            ));
        }
    }
    if let (Some(first), Some(second)) = (
        request.literals.literal_string_at(1),
        request.literals.literal_string_at(2),
    ) {
        if first == second {
            diagnostics.push(argument_error(
                "RM-CATALOG-MESHING-SAVE-PATHS",
                "mesh and evidence paths must be distinct",
                2,
            ));
        }
    }
    let mut saved = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    saved.residency = ResidencyFact::Host;
    finish(entry, request, saved, diagnostics)
}

const EXAMPLES: &[BuiltinExample] = &[
    native_example(
        "save-pair",
        "Save a mesh and its evidence for an archive",
        concat!(
            box_mesh_setup!(),
            "saved = meshing.save(solverMesh, 'artifacts/box.solver-mesh.cbor', 'artifacts/box.meshing-evidence.cbor');",
        ),
        "assert(saved);\nassert(isfile('artifacts/box.solver-mesh.cbor'));\nassert(isfile('artifacts/box.meshing-evidence.cbor'));",
    ),
    native_example(
        "protected-destinations",
        "Existing files are replaced only with Force",
        concat!(
            box_mesh_setup!(),
            "meshing.save(solverMesh, 'box.solver-mesh.cbor', 'box.meshing-evidence.cbor');\n",
            "try\n",
            "    meshing.save(solverMesh, 'box.solver-mesh.cbor', 'box.meshing-evidence.cbor');\n",
            "    protected = false;\n",
            "catch err\n",
            "    protected = strcmp(err.identifier, 'RunMat:meshing:InvalidInput');\n",
            "end\n",
            "replaced = meshing.save(solverMesh, 'box.solver-mesh.cbor', 'box.meshing-evidence.cbor', 'Force', true);",
        ),
        "assert(protected);\nassert(replaced);",
    ),
];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.save",
    summary: "Save a solver mesh and its evidence as a canonical artifact pair.",
    description: "`saved = meshing.save(mesh, meshPath, evidencePath)` writes the canonical solver-mesh artifact and its meshing evidence to two files and returns logical `true`; `meshing.save(..., 'Force', true)` replaces existing files.",
    keywords: &["meshing", "save", "artifact", "evidence", "cbor", "archive"],
    related: &["meshing.load", "meshing.generate", "meshing.evidence"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`meshPath` and `evidencePath` are scalar text and must differ. Missing parent directories are created. The two artifacts are the canonical CBOR encodings, identical to the files `runmat mesh` writes for the same mesh, and they can be read back with `meshing.load`. Region identity is part of the mesh artifact, so no other file is written.",
                "Existing destinations are protected: if either file exists, the call fails and none is changed, unless the only option, `Force`, is `true`. `Force` must be logical; numeric `1` is rejected.",
                "All files are staged before any destination is replaced. If publishing one fails, the files already replaced are rolled back, so the set on disk always matches. The call returns `true` only after every file is in place.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Keep both files with analysis results. The mesh alone does not record how it was generated; the evidence binds it to the geometry revision, request, seed, and measured stage checks, and `meshing.load` admits the mesh only together with that evidence.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "Typical causes are identical paths, an existing destination without `Force`, a non-logical `Force` value, an unknown option, or a filesystem failure."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "Why are there two files?",
        answer: "The mesh artifact holds the discretization; the evidence artifact holds the request, geometry binding, and stage checks that prove the mesh is valid. They are saved and loaded as a pair.",
    }],
    implementation: &[
        BUILTINS_SOURCE,
        BuiltinDocumentationLink {
            label: "Atomic artifact-pair publication",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/meshing/persistence.rs",
            ),
        },
    ],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Save and load round-trip only the canonical artifacts",
            location: "crates/runmat-runtime/src/builtins/meshing/mod.rs::save_and_load_round_trip_only_the_canonical_artifacts",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Pair writes protect and replace existing destinations",
            location: "crates/runmat-runtime/src/meshing/persistence.rs::pair_write_protects_and_replaces_existing_destinations",
        },
    ],
});

pub const MESHING_SAVE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "meshing.save",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_SAVE_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Save,
    integer_capabilities: &[],
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_SAVE_CATALOG_ENTRY];
