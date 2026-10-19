use super::super::example_fixtures::{BOX_BREP, NATIVE_CAD, TRIANGLE_STL};
use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("geometry.region"),
    slug: Some("geometry.region"),
    summary: "Resolve one region of a loaded geometry asset by ID, name, or tag.",
    description: "`region = geometry.region(asset, selector)` finds exactly one region of a `geometry.Asset` and returns it as an immutable `geometry.Region` bound to that asset's identity and revision.",
    keywords: &["geometry", "region", "selector", "face", "solid", "boundary", "fea"],
    related: &[
        "geometry.load",
        "geometry.listRegions",
        "fea.PropertyRegionTarget.solid",
        "fea.StructuralConstraint.fixed",
        "fea.materialAssignment",
        "fea.model",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Matching is exact and case-sensitive. If `selector` equals a region ID it selects that region, even when the ID itself contains a colon. Otherwise a prefix restricts the field that is matched: `id:` or `region:` match the region ID, `name:` the imported name, and `tag:` the imported tag. Without a prefix the selector may match any of the three. Exactly one region must match; any other prefix, an empty selector or value, no match, or several matches is an error, and an ambiguous match suggests an `id:` selector.",
                "The result has the properties `geometry_id`, `geometry_revision` (a `uint32`), `region_id`, and `name`, plus `tag` when the imported region has one. It is an immutable value; it records which asset revision it came from so that model constructors can check it against the model's geometry.",
                "Region IDs come from the importer. Exact CAD imports (STEP, IGES, BREP) publish one region per face, with IDs such as `face_000001`, names such as `Face 1`, and the tag `occt_face`. Mesh formats such as STL publish a single `region_default` region named `Default Region` with the tag `mesh_default`. Use `geometry.listRegions` or `asset.regions` to see what a file contains before choosing selectors.",
                "The call works on an asset that is already loaded. It is pure: it reads no files and does not suspend.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Regions are how a script-built model names where materials, constraints, and loads apply. Pass a face region to a boundary constructor such as `fea.StructuralConstraint.fixed(\"support\", geometry.region(part, \"face_000001\"))`, or a solid region to `fea.PropertyRegionTarget.solid` for a `fea.materialAssignment`, then assemble them with `fea.model` on the same asset.",
                "Prefer `id:` selectors in scripts that must stay stable: names and tags come from the CAD author and need not be unique.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[
                "`RunMat:geometry:region:InvalidRegion` reports an empty selector or value, an unsupported prefix, a selector that matches no region, or one that matches several regions; the message names the selector and the match count.",
                "`RunMat:geometry:region:InvalidInput` reports a call that does not pass exactly a `geometry.Asset` and a scalar text selector.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "stl-default-region",
            title: "Select the default region of an STL by ID and by tag",
            program: "part = geometry.load('part.stl');\nbody = geometry.region(part, 'region_default');\nsame = geometry.region(part, 'tag:mesh_default');\nbody.name",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: TRIANGLE_STL,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(body, 'geometry.Region'));\nassert(strcmp(body.region_id, 'region_default'));\nassert(strcmp(same.region_id, body.region_id));\nassert(isa(body.geometry_revision, 'uint32'));",
            },
        },
        BuiltinExample {
            id: "cad-face",
            title: "Select a face of an exact CAD solid for a support",
            program: "part = geometry.load('box.brep', 'Units', 'm');\nsupportFace = geometry.region(part, 'face_000001');\nsameFace = geometry.region(part, 'name:Face 1');\nsupportFace.tag",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BOX_BREP,
            requirements: NATIVE_CAD,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(supportFace, 'geometry.Region'));\nassert(strcmp(supportFace.region_id, 'face_000001'));\nassert(strcmp(sameFace.region_id, 'face_000001'));\nassert(strcmp(supportFace.tag, 'occt_face'));",
            },
        },
        BuiltinExample {
            id: "missing-region",
            title: "A selector that matches nothing is an error",
            program: "part = geometry.load('part.stl');\nregion = geometry.region(part, 'name:Top');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: TRIANGLE_STL,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:geometry:region:InvalidRegion",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Two faces share a name. How do I pick one?",
            answer: "Use its region ID, for example `geometry.region(part, 'id:face_000004')`. A name or tag that matches several regions is rejected rather than resolved arbitrarily.",
        },
        BuiltinDocumentationFaq {
            question: "Does a region stay valid if I reload the file?",
            answer: "The region records the asset's `geometry_id` and `geometry_revision`. Resolve regions from the asset you pass to the model so the two always agree.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "Geometry",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/geometry"),
        },
        BuiltinDocumentationLink {
            label: "Models",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Region selector resolution",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/region.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "ID, name, and tag resolve to a revision-bound value",
                location: "crates/runmat-runtime/src/builtins/geometry/region.rs::resolves_id_name_and_tag_to_a_revision_bound_value",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Missing, ambiguous, and unknown-prefix selectors are rejected",
                location: "crates/runmat-runtime/src/builtins/geometry/region.rs::rejects_missing_ambiguous_and_unknown_prefixed_selectors",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
