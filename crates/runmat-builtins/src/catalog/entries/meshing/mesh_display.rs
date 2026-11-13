//! `mesh(solverMesh, ...)`: the method-only display binding for `meshing.Mesh`.

use super::shared::*;
use crate::catalog::argument_error;
use crate::*;
use runmat_types::{CallInference, CallRequest, ValueKindFact};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let name = "mesh";
    let mut diagnostics = Vec::new();
    arity(
        &mut diagnostics,
        "MESH",
        name,
        request.arguments.len(),
        1,
        None,
    );
    require_mesh(&mut diagnostics, "MESH", name, request, 0);
    options(
        &mut diagnostics,
        "MESH",
        name,
        request,
        1,
        &["edges", "regions", "section"],
    );
    for index in (1..request.arguments.len()).step_by(2) {
        let (Some(option), Some(value)) = (
            request.literals.literal_string_at(index),
            request.arguments.get(index + 1),
        ) else {
            continue;
        };
        let valid = match option.as_str() {
            "edges" => matches!(value.kind, ValueKindFact::Logical | ValueKindFact::Unknown),
            "regions" => matches!(
                value.kind,
                ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Cell(_)
                    | ValueKindFact::Unknown
            ),
            "section" => {
                matches!(
                    value.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Unknown
                ) && !matches!(value.shape.element_count(), Some(count) if count != 4)
            }
            _ => true,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-MESHING-MESH-VALUE",
                "mesh options: Edges is logical, Regions is text, Section is a four-element plane",
                index + 1,
            ));
        }
    }
    finish(entry, request, double_scalar(), diagnostics)
}

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "show-mesh",
        title: "Display a solver mesh",
        program: concat!(box_mesh_setup!(), "h = mesh(solverMesh);"),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_FIXTURE,
        requirements: NATIVE_MESHING_PLOT_CORE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(isnumeric(h) && isscalar(h));",
        },
    },
    BuiltinExample {
        id: "section-view",
        title: "Cut a section through the middle of the part",
        program: concat!(
            box_mesh_setup!(),
            "regions = meshing.regions(solverMesh);\n",
            "body = regions{1};\n",
            "nodes = meshing.nodes(solverMesh);\n",
            "x = nodes.coordinates;\n",
            "xMid = (min(x(:, 1)) + max(x(:, 1))) / 2;\n",
            "h = mesh(solverMesh, 'Regions', body.id, 'Section', [1 0 0 -xMid], 'Edges', true);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_FIXTURE,
        requirements: NATIVE_MESHING_PLOT_CORE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(isnumeric(h) && isscalar(h));\nassert(abs(xMid - 0.5) < 1e-9);",
        },
    },
];

// Hidden method-only binding: no public title or slug (user syntax is `mesh(solverMesh)`).
const DOCUMENTATION: BuiltinDocumentation = hidden(documentation(MeshingDocumentation {
    title: "__runmat_meshing_mesh",
    summary: "Display a solver mesh with `mesh(solverMesh)`, optionally by region or as a section.",
    description: "`mesh(solverMesh)` and `mesh(solverMesh, Name, Value, ...)` draw the boundary of a `meshing.Mesh` and return the figure handle. RunMat's object dispatch routes `mesh` on a `meshing.Mesh` to this method-only binding; numeric `mesh(X, Y, Z)` is unchanged.",
    keywords: &["meshing", "mesh", "display", "section", "regions", "plot"],
    related: &["meshing.generate", "meshing.regions", "meshing.nodes", "geometry.load"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The figure shows the solver mesh's boundary faces. Option names are case-insensitive. `Edges` (logical, default `true`) draws element edges. `Regions` takes one region id or a string array or cell of ids from `meshing.regions(solverMesh)` and shows only boundary faces of elements in those regions; an unknown id is an error.",
                "`Section` takes a plane `[a b c d]` in meters, meaning `a*x + b*y + c*z + d = 0`, with a finite nonzero normal `[a b c]`. The view keeps the side where `a*x + b*y + c*z + d <= 0` and caps the cut with a highlighted section face, so `[1 0 0 -xMid]` shows the half with `x <= xMid`. Integer-class plane values are converted to `double`.",
                "Display requires a host built with plotting support; without it the call fails. It never modifies the mesh.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Inspect the solver mesh before solving: check density near holes and fillets, confirm that each region was meshed, and cut sections to see interior element sizes. `mesh(part)` on the `geometry.Asset` shows the CAD tessellation instead, which is not the solver discretization.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "Typical causes are options that are not Name, Value pairs, an unknown option, a non-logical `Edges`, an empty or unknown region id, or a section plane that is not four finite values with a nonzero normal."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "Where do region ids for the Regions option come from?",
        answer: "From the `id` field of `meshing.regions(solverMesh)`. Material names are not region ids.",
    }],
    implementation: &[
        BUILTINS_SOURCE,
        BuiltinDocumentationLink {
            label: "Solver-mesh figure",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/meshing/visualization.rs",
            ),
        },
    ],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Solver-mesh display accepts region and section options",
            location: "crates/runmat-runtime/src/builtins/meshing/mod.rs::solver_mesh_display_accepts_region_and_section_options",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Standard mesh dispatches solver meshes through the object protocol",
            location: "crates/runmat-runtime/src/builtins/meshing/mod.rs::standard_mesh_dispatches_solver_meshes_through_object_protocol",
        },
    ],
}));

const fn hidden(mut documentation: BuiltinDocumentation) -> BuiltinDocumentation {
    documentation.title = None;
    documentation.slug = None;
    documentation
}

pub const MESHING_MESH_METHOD_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "__runmat_meshing_mesh",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_METHOD_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Display,
    integer_capabilities: &MESHING_METHOD_INTEGER_CAPABILITIES,
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_MESH_METHOD_CATALOG_ENTRY];
