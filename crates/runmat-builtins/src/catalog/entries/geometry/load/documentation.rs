use super::super::example_fixtures::{BOX_BREP, NATIVE_CAD, TRIANGLE_STL};
use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("geometry.load"),
    slug: Some("geometry.load"),
    summary: "Import a geometry file as a typed geometry.Asset.",
    description: "`asset = geometry.load(path)` reads a local geometry file, imports it, and returns a host `geometry.Asset`; `asset = geometry.load(path, 'Units', unit)` also sets the length unit of the asset's coordinates.",
    keywords: &["geometry", "load", "import", "STEP", "IGES", "BREP", "STL", "OBJ", "CAD", "units"],
    related: &[
        "geometry.inspect",
        "geometry.listRegions",
        "geometry.region",
        "geometry.meshes",
        "meshing.generate",
        "fea.model",
        "fea.study",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Input and result",
            paragraphs: &[
                "`path` is scalar text. The result is a `geometry.Asset` with source, revision, unit, mesh, region, and mapping metadata; the imported payload is retained so that `geometry.region`, `geometry.listRegions`, `geometry.meshes`, `mesh(asset)`, and `meshing.generate` can use it. The asset records the SHA-256 of the bytes it was imported from.",
                "STEP, IGES, and BREP import as exact CAD through the native OCCT importer, with one region per face (`face_000001`, `face_000002`, ...). Without OCCT, STEP is imported metadata-only: exchange facts are available but there is no topology or render mesh. STL, OBJ, PLY, and glTF import through the portable mesh importer as a single `region_default` region.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Units",
            paragraphs: &[
                "The only option is `'Units'`, which sets the length unit of the asset's coordinates: `'m'` (the default), `'mm'`, or `'in'`. The longer spellings `'meter'`, `'meters'`, `'metre'`, `'metres'`, `'millimeter'`, `'millimeters'`, `'millimetre'`, `'millimetres'`, `'inch'`, and `'inches'` are also accepted, and matching is case-insensitive. Give the option at most once.",
                "Formats that do not declare a unit (BREP, STL, OBJ, PLY, and glTF) are read as being in that unit. STEP files declare their own units, and the declared unit always governs how the file is read; the geometry is then expressed in the requested unit. The unit in effect is reported in `asset.units` as `'meter'`, `'millimeter'`, or `'inch'`.",
                "Meshing and analysis work in SI units. `meshing.generate` re-reads the source with the asset's unit, so a BREP or other undeclared-unit file modeled in millimeters must be loaded with `'Units', 'mm'` to mesh at the right scale.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "The asset is the geometry of a script-built study: select regions with `geometry.region`, assemble a model with `fea.model`, generate a solver mesh with `meshing.generate`, and bind them with `fea.study`, always on the same asset so that region references and the mesh share one geometry revision.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Effects and errors",
            paragraphs: &[
                "Reading the file can suspend and requires filesystem access. The import checks cancellation, and a failed import returns no asset. Browser execution needs an available filesystem bridge and a supported importer.",
                "`RunMat:geometry:load:InvalidPath` reports a path that is not scalar text and an invalid option: options that are not Name, Value pairs, a name other than `Units`, a repeated `Units`, or an unsupported unit. `RunMat:geometry:load:IoFailure` reports a file that cannot be read, `RunMat:geometry:load:OperationFailed` an importer rejection, `RunMat:geometry:load:Internal` a result that cannot be converted, and `RunMat:geometry:load:TooManyOutputs` a request for more than one output.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "triangle",
            title: "Import a triangle STL",
            program: "asset = geometry.load('part.stl');\nclass(asset)",
            display_output: Some("ans = geometry.Asset"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: TRIANGLE_STL,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(asset,'geometry.Asset')); assert(~isempty(asset.regions)); assert(strcmp(asset.units, 'meter'));",
            },
        },
        BuiltinExample {
            id: "millimeter-mesh",
            title: "Import an STL modeled in millimeters",
            program: "asset = geometry.load('part.stl', 'Units', 'mm');\nasset.units",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: TRIANGLE_STL,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(asset, 'geometry.Asset')); assert(strcmp(asset.units, 'millimeter'));",
            },
        },
        BuiltinExample {
            id: "exact-cad",
            title: "Import an exact CAD solid and list its faces",
            program: "part = geometry.load('box.brep', 'Units', 'm');\nfirstFace = part.regions{1};\nfirstFace.region_id",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BOX_BREP,
            requirements: NATIVE_CAD,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(part.regions) == 6); assert(strcmp(firstFace.region_id, 'face_000001')); assert(strcmp(part.units, 'meter'));",
            },
        },
        BuiltinExample {
            id: "unsupported-unit",
            title: "Unsupported units are rejected before the file is imported",
            program: "asset = geometry.load('part.stl', 'Units', 'ft');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: TRIANGLE_STL,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:geometry:load:InvalidPath",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does load always need native CAD support?",
            answer: "No. Mesh formats such as STL import portably. STEP, IGES, and BREP need the native OCCT importer for exact topology, which analysis and meshing require.",
        },
        BuiltinDocumentationFaq {
            question: "My STEP file is in millimeters. Do I need 'Units'?",
            answer: "No. STEP declares its units and they are always honored. `'Units'` matters for files without a declaration, such as BREP and STL.",
        },
    ],
    links: &[BuiltinDocumentationLink {
        label: "Geometry",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/geometry"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Geometry import builtin",
                target: BuiltinDocumentationLinkTarget::Source(
                    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/entrypoints.rs",
                ),
            },
            BuiltinDocumentationLink {
                label: "Units option decoding",
                target: BuiltinDocumentationLinkTarget::Source(
                    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/display.rs",
                ),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Canonical asset without native CAD",
                location: "crates/runmat-runtime/src/builtins/geometry/tests.rs::geometry_load_returns_canonical_asset_without_native_cad",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "File read errors keep the I/O identifier",
                location: "crates/runmat-runtime/src/builtins/geometry/tests.rs::geometry_file_read_error_retains_io_identifier",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Output count is rejected before the file is read",
                location: "crates/runmat-runtime/src/builtins/geometry/tests.rs::geometry_output_count_is_rejected_before_file_read",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
