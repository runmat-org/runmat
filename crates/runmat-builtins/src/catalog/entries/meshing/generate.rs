//! `meshing.generate` catalog identity.

use super::shared::*;
use crate::catalog::argument_error;
use crate::*;
use runmat_types::{CallInference, CallRequest, ResidencyFact, ValueKindFact};

const OPTIONS: &[&str] = &[
    "targetsize",
    "maximumdeviation",
    "maximumgradingratio",
    "elementorder",
    "maximumelements",
    "seed",
];

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    arity(
        &mut diagnostics,
        "GENERATE",
        name,
        request.arguments.len(),
        1,
        None,
    );
    if let Some(geometry) = request.arguments.first() {
        let possible_asset = !matches!(geometry.residency, ResidencyFact::Device { .. })
            && match &geometry.kind {
                ValueKindFact::Unknown => true,
                ValueKindFact::Object(object) => object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|class| class.is(crate::GEOMETRY_ASSET_CLASS)),
                _ => false,
            };
        if !possible_asset {
            diagnostics.push(argument_error(
                "RM-CATALOG-MESHING-GENERATE-GEOMETRY",
                "meshing.generate requires a geometry.Asset from geometry.load",
                0,
            ));
        }
    }
    options(&mut diagnostics, "GENERATE", name, request, 1, OPTIONS);
    for index in (1..request.arguments.len()).step_by(2) {
        let is_order = request
            .literals
            .literal_string_at(index)
            .is_some_and(|option| option.replace(['_', ' '], "") == "elementorder");
        if is_order {
            if let Some(order) = request.literals.literal_string_at(index + 1) {
                if order != "tet4" && order != "tet10" {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-MESHING-GENERATE-ORDER",
                        "ElementOrder must be 'tet4' or 'tet10'",
                        index + 1,
                    ));
                }
            }
        }
    }
    finish(entry, request, mesh_result(), diagnostics)
}

const EXAMPLES: &[BuiltinExample] = &[
    native_example(
        "coarse-box",
        "Generate a linear mesh of a 1 m x 2 m x 3 m block",
        concat!(
            "part = geometry.load('box.brep', 'Units', 'm');\n",
            "solverMesh = meshing.generate(part, 'TargetSize', 0.5, 'MaximumDeviation', 0.05, ...\n",
            "    'ElementOrder', 'tet4', 'Seed', 42);\n",
            "solverMesh.element_count",
        ),
        "assert(isa(solverMesh, 'meshing.Mesh'));\nassert(isa(solverMesh.element_count, 'uint64'));\nassert(solverMesh.element_count > 0);\nassert(numel(solverMesh.canonical_digest) == 64);",
    ),
    native_example(
        "quadratic-reproducible",
        "Generate quadratic tetrahedra reproducibly",
        concat!(
            "part = geometry.load('box.brep', 'Units', 'm');\n",
            "settings = {'TargetSize', 0.5, 'MaximumDeviation', 0.05, 'ElementOrder', 'tet10', ...\n",
            "    'MaximumElements', uint64(100000), 'Seed', uint64(7)};\n",
            "first = meshing.generate(part, settings{:});\n",
            "second = meshing.generate(part, settings{:});\n",
            "elements = meshing.elements(first);",
        ),
        "assert(strcmp(first.canonical_digest, second.canonical_digest));\nassert(size(elements.connectivity, 2) == 10);\nassert(strcmp(elements.order, 'tet10'));",
    ),
];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.generate",
    summary: "Generate a validated tetrahedral solver mesh from exact CAD geometry.",
    description: "`mesh = meshing.generate(geometry)` and `mesh = meshing.generate(geometry, Name, Value, ...)` discretize a `geometry.Asset` into linear or quadratic tetrahedra and return an immutable `meshing.Mesh` whose solver artifact and stage evidence have passed joint validation.",
    keywords: &["meshing", "mesh", "generate", "tetrahedra", "tet4", "tet10", "solver", "cad", "fea"],
    related: &[
        "geometry.load",
        "meshing.save",
        "meshing.load",
        "meshing.quality",
        "meshing.evidence",
        "meshing.regions",
        "fea.study",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`geometry` must be a `geometry.Asset` from `geometry.load` of exact CAD (STEP, IGES, or BREP) containing a closed solid. Meshing re-reads the source file recorded in the asset, checks it against the asset's SHA-256 so that the mesh always matches the geometry you loaded, and uses the asset's units. Tessellated formats such as STL can be displayed but not meshed for analysis.",
                "Options are Name, Value pairs; names are case-insensitive and ignore underscores and spaces. `TargetSize` is the global target edge length in meters (default `0.01`). `MaximumDeviation` is the maximum chordal distance in meters between the discrete boundary and the exact curves and surfaces (default `1e-4`); curved geometry may need edges much shorter than `TargetSize` to meet it. `MaximumGradingRatio` limits how quickly element size may change between neighbors (default `1.3`; lower is smoother but needs more elements). `ElementOrder` is `'tet4'` (default) or `'tet10'`. `MaximumElements` is a hard element ceiling (default `10000000`), and `Seed` is the deterministic tie-breaking seed (default `0`). A repeated option uses its last value.",
                "`TargetSize`, `MaximumDeviation`, and `MaximumGradingRatio` must be finite and positive. `MaximumElements` must be a positive integer and `Seed` a nonnegative integer; both accept integer classes up to `uint64` exactly, or a `double` holding an exact integer. Other budgets use canonical limits: 10,000,000 nodes, 512 MiB of memory, 512 MiB of scratch space, a 512 MiB artifact, and one hour of wall time.",
                "Tet10 adds midside nodes, projects boundary midside nodes onto the exact CAD surface, and validates positive curved Jacobians. Boundary faces are Tri3 for Tet4 and Tri6 for Tet10.",
                "Generation either returns a mesh that passed topology, geometry, sizing, quality, provenance, resource, and serialization checks, or fails without returning a mesh. A budget that cannot be met fails rather than truncating or coarsening the mesh. The same geometry, options, and seed produce the same canonical digest, so a mesh can be regenerated and compared exactly.",
                "The result's public properties are `canonical_digest` (64 hexadecimal characters), `node_count`, `element_count`, and `boundary_face_count` (each `uint64`). Use `meshing.nodes`, `meshing.elements`, `meshing.boundaryFaces`, `meshing.regions`, `meshing.quality`, and `meshing.evidence` to inspect it, and `mesh(solverMesh)` to display it.",
                "Meshing runs through the native meshing service that RunMat hosts install, using parallel native workers. A host without that service, including the browser, fails with a message saying the host cannot execute meshing.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Attach the mesh to a study with `fea.study(id, part, model, \"Mesh\", solverMesh)` using the same `geometry.Asset`; the model's regions are resolved onto the mesh through the persistent CAD identities the mesher records. Save the mesh and its evidence with `meshing.save` when results must be reproducible from the exact discretization.",
                "Materials and boundary conditions are not meshing options. Assign them on geometry regions in the model; meshing only discretizes the geometry and preserves its regions.",
                "Check mesh convergence for engineering conclusions: reduce `TargetSize` and `MaximumDeviation` systematically and confirm that the quantities of interest stop changing.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[
                "`RunMat:meshing:InvalidInput` reports every failure: a first argument that is not a `geometry.Asset`, malformed options, an unsupported element order, out-of-range values, a host without meshing, a changed or unreadable source file, and meshing failures such as invalid geometry, unsatisfiable constraints, unreachable quality, or an exceeded budget; the message names the cause and stage.",
            ],
        },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Which element order should I use?",
            answer: "Use `tet10` for stress, bending, and modal work, where linear tetrahedra are too stiff, and for curved geometry. Use `tet4` for temperature fields and quick checks.",
        },
        BuiltinDocumentationFaq {
            question: "My mesh fails with a budget error. Can RunMat coarsen it automatically?",
            answer: "No. A budget failure means the requested sizing cannot be met within the limit. Raise `MaximumElements` or relax `TargetSize` or `MaximumDeviation` deliberately; the mesher never silently reduces quality.",
        },
        BuiltinDocumentationFaq {
            question: "Are the size options in model units?",
            answer: "No. `TargetSize` and `MaximumDeviation` are always meters, whatever units the CAD file uses.",
        },
    ],
    implementation: &[BUILTINS_SOURCE, OPTIONS_SOURCE, OBJECT_SOURCE],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Generation controls preserve wide integers",
            location: "crates/runmat-runtime/src/builtins/meshing/options.rs::generation_controls_preserve_wide_integers",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Generation controls reject negative integers",
            location: "crates/runmat-runtime/src/builtins/meshing/options.rs::generation_controls_reject_negative_integers",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Generation controls reject doubles above the u64 domain",
            location: "crates/runmat-runtime/src/builtins/meshing/options.rs::generation_controls_reject_double_above_the_u64_domain",
        },
    ],
});

pub const MESHING_GENERATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "meshing.generate",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_GENERATE_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Generate,
    integer_capabilities: &MESHING_GENERATE_INTEGER_CAPABILITIES,
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_GENERATE_CATALOG_ENTRY];
