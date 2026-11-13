mod connected;
mod cube;
mod layered;
mod solver_mesh;

pub use cube::{
    elevate_planar_tet4_fixture, quadratic_slab_bar_mesh, quadratic_slab_mesh, slab_bar_mesh,
    slab_mesh,
};
pub use layered::layered_box_mesh;

pub use connected::{
    connected_quadratic_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh,
    elevate_connected_tet4_fixture,
};
pub use solver_mesh::{
    canonical_tetrahedron_meshing_evidence, canonical_tetrahedron_solver_mesh,
    three_region_tetrahedron_solver_mesh, three_region_tetrahedron_solver_mesh_with_order,
};

use runmat_geometry_core::{
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
};
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

/// The occurrence of a named entity in every meshing-owned fixture mesh.
///
/// Fixture meshes place each entity under the `root` assembly occurrence. Analysis fixtures
/// select these entities only through [`fixture_selection`], the canonical
/// `persistent-entity:sha256:` selector of the same occurrence.
pub fn fixture_entity(kind: PersistentEntityKind, source: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, source.into(), vec!["root".into()])
}

/// Canonical analysis selector for [`fixture_entity`].
pub fn fixture_selection(kind: PersistentEntityKind, source: &str) -> String {
    let entity = fixture_entity(kind, source);
    let selection = entity
        .analysis_selection_id()
        .expect("fixture entity names are valid persistent identities");
    fixture_selection_index()
        .lock()
        .expect("fixture selection index is not poisoned")
        .insert(selection.clone(), entity);
    selection
}

/// The fixture entity a canonical selector returned by [`fixture_selection`] denotes.
///
/// Fixture mesh builders that shape a mesh around an authored model use this to place the exact
/// entity a target selects; selectors that no fixture produced return `None`.
pub fn fixture_entity_for_selection(selection: &str) -> Option<PersistentEntityId> {
    fixture_selection_index()
        .lock()
        .expect("fixture selection index is not poisoned")
        .get(selection)
        .cloned()
}

fn fixture_selection_index(
) -> &'static std::sync::Mutex<std::collections::BTreeMap<String, PersistentEntityId>> {
    static INDEX: std::sync::OnceLock<
        std::sync::Mutex<std::collections::BTreeMap<String, PersistentEntityId>>,
    > = std::sync::OnceLock::new();
    INDEX.get_or_init(Default::default)
}

/// Canonical selector of the fixture solid named `source`.
pub fn fixture_solid(source: &str) -> String {
    fixture_selection(PersistentEntityKind::Solid, source)
}

/// Canonical selector of the fixture mesh region named `source`.
pub fn fixture_region(source: &str) -> String {
    fixture_selection(PersistentEntityKind::Region, source)
}

/// Canonical selector of the fixture face named `source`.
pub fn fixture_face(source: &str) -> String {
    fixture_selection(PersistentEntityKind::Face, source)
}

/// Canonical face selector for a fixture boundary name. Point ids (`node:`) and selectors that
/// are already canonical pass through unchanged.
pub fn fixture_boundary_selection(name: &str) -> String {
    if name.starts_with("node:") || name.starts_with("persistent-entity:sha256:") {
        name.to_owned()
    } else {
        fixture_face(name)
    }
}

/// Canonical selector of the fixture edge named `source`.
pub fn fixture_edge(source: &str) -> String {
    fixture_selection(PersistentEntityKind::Edge, source)
}

/// Canonical selector of the fixture vertex named `source`.
pub fn fixture_vertex(source: &str) -> String {
    fixture_selection(PersistentEntityKind::Vertex, source)
}

pub fn through_hole_plate_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_through_hole_plate".to_string(),
        source: GeometrySource {
            unit_evidence: Default::default(),
            path: "/fixtures/generic_through_hole_plate.step".to_string(),
            sha256: "generic-through-hole-plate".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            exchange: None,
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        coordinate_units: None,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "through_hole_plate_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 16,
            element_count: 32,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "through_hole_plate_surface",
            vec![
                [0.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [3.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
                [1.0, 2.0, 1.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            vec![
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
                [8, 13, 9],
                [8, 12, 13],
                [9, 14, 10],
                [9, 13, 14],
                [10, 15, 11],
                [10, 14, 15],
                [11, 12, 8],
                [11, 15, 12],
                [0, 8, 9],
                [0, 9, 1],
                [1, 9, 10],
                [1, 10, 2],
                [2, 10, 11],
                [2, 11, 3],
                [3, 11, 8],
                [3, 8, 0],
                [4, 5, 13],
                [4, 13, 12],
                [5, 6, 14],
                [5, 14, 13],
                [6, 7, 15],
                [6, 15, 14],
                [7, 4, 12],
                [7, 12, 15],
            ],
        )],
        regions: vec![Region {
            region_id: "body".to_string(),
            name: "body".into(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "body",
            "through_hole_plate_surface",
            32,
        )],
        diagnostics: Vec::new(),
    }
}

pub fn split_material_through_hole_plate_geometry() -> GeometryAsset {
    let mut geometry = through_hole_plate_geometry();
    geometry.geometry_id = "geo_root_meshing_split_material_through_hole_plate".to_string();
    geometry.source.path = "/fixtures/generic_split_material_through_hole_plate.step".to_string();
    geometry.source.sha256 = "generic-split-material-through-hole-plate".to_string();
    geometry.regions = vec![
        Region {
            region_id: "region_base".to_string(),
            name: "base".into(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        },
        Region {
            region_id: "region_cap".to_string(),
            name: "cap".into(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        },
    ];
    geometry.region_entity_mappings = vec![
        RegionEntityMapping::new(
            "region_base",
            "through_hole_plate_surface",
            EntityKind::Face,
            vec![
                EntityIdRange::new(0, 4),
                EntityIdRange::new(8, 4),
                EntityIdRange::new(16, 4),
                EntityIdRange::new(24, 4),
            ],
        ),
        RegionEntityMapping::new(
            "region_cap",
            "through_hole_plate_surface",
            EntityKind::Face,
            vec![
                EntityIdRange::new(4, 4),
                EntityIdRange::new(12, 4),
                EntityIdRange::new(20, 4),
                EntityIdRange::new(28, 4),
            ],
        ),
    ];
    geometry
}
