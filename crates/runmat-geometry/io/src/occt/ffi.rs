#[cxx::bridge(namespace = "runmat_geometry_io::occt_backend")]
pub(crate) mod ffi {
    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum OcctCadFormat {
        Step,
        Iges,
        Brep,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctImportOptions {
        linear_deflection: f64,
        angular_deflection: f64,
        relative_deflection: bool,
        max_triangles: u64,
    }

    #[derive(Debug, Clone)]
    struct OcctFaceSemanticPayload {
        face_id: u64,
        label_entry: String,
        label_name: String,
        label_kind: String,
        owner_entries: Vec<String>,
        owner_names: Vec<String>,
        owner_kinds: Vec<String>,
        layer_names: Vec<String>,
        color_type: String,
        color_hex_rgba: String,
        material_label_entry: String,
        material_name: String,
        material_description: String,
        material_density: String,
        material_density_name: String,
        material_density_value_type: String,
    }

    #[derive(Debug, Clone)]
    struct OcctAssemblyNodePayload {
        node_id: String,
        parent_node_id: String,
        label: String,
    }

    #[derive(Debug, Clone)]
    struct OcctImportPayload {
        backend: String,
        format_name: String,
        vertices: Vec<f64>,
        triangles: Vec<u32>,
        triangle_face_ids: Vec<u64>,
        face_ids: Vec<u64>,
        face_names: Vec<String>,
        face_semantics: Vec<OcctFaceSemanticPayload>,
        assembly_nodes: Vec<OcctAssemblyNodePayload>,
        warnings: Vec<String>,
    }

    unsafe extern "C++" {
        include!("runmat-geometry-io/src/occt/occt_bridge.hxx");

        fn import_cad_bytes(
            path: &str,
            bytes: &[u8],
            format: OcctCadFormat,
            options: OcctImportOptions,
        ) -> Result<OcctImportPayload>;
    }
}
