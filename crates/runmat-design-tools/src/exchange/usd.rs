//! A part or assembly as a USD layer in its text form (`.usda`, Universal Scene Description
//! 1.0, written from the public file-format documentation): one `Mesh` prim per placed body,
//! in world coordinates, under one root `Xform` named after the file. The stage is Z-up in
//! metres (`upAxis = "Z"`, `metersPerUnit = 1`), the frame the design runtime regenerates in, so
//! a viewer shows the model the right way up and at true size.

use super::Mesh;
use std::fmt::Write as _;

/// A USD identifier from `name`: letters, digits and underscores, never starting with a digit.
fn identifier(name: &str) -> String {
    let mut id: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if id.is_empty() || id.starts_with(|c: char| c.is_ascii_digit()) {
        id.insert(0, '_');
    }
    id
}

/// The `.usda` text of `meshes` under a root named `name`.
pub fn usda(meshes: &[Mesh], name: &str) -> String {
    let root = identifier(name);
    let mut out = String::new();
    let _ = writeln!(out, "#usda 1.0\n(\n    defaultPrim = \"{root}\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n");
    let _ = writeln!(out, "def Xform \"{root}\"\n{{");
    let mut taken = std::collections::BTreeSet::new();
    for (i, mesh) in meshes.iter().enumerate() {
        let mut prim = identifier(&mesh.name);
        if !taken.insert(prim.clone()) {
            prim = format!("{prim}_{i}");
            taken.insert(prim.clone());
        }
        let points = mesh
            .positions
            .iter()
            .map(|p| format!("({:.6}, {:.6}, {:.6})", p[0], p[1], p[2]))
            .collect::<Vec<_>>()
            .join(", ");
        let counts = vec!["3"; mesh.triangles.len()].join(", ");
        let indices = mesh
            .triangles
            .iter()
            .flat_map(|t| t.iter())
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "    def Mesh \"{prim}\"\n    {{");
        let _ = writeln!(out, "        int[] faceVertexCounts = [{counts}]");
        let _ = writeln!(out, "        int[] faceVertexIndices = [{indices}]");
        let _ = writeln!(out, "        point3f[] points = [{points}]");
        let _ = writeln!(out, "        uniform token subdivisionScheme = \"none\"");
        let _ = writeln!(out, "    }}");
    }
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_triangle_writes_as_one_mesh_prim_under_the_file_root() {
        let mesh = Mesh {
            name: "1 plate".into(),
            material: None,
            positions: vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        let text = usda(&[mesh], "rig");
        assert!(text.starts_with("#usda 1.0"), "{text}");
        assert!(text.contains("defaultPrim = \"rig\""), "{text}");
        assert!(text.contains("def Mesh \"_1_plate\""), "{text}");
        assert!(
            text.contains("int[] faceVertexIndices = [0, 1, 2]"),
            "{text}"
        );
    }
}
