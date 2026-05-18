#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryFormat {
    Stl,
    Step,
    Obj,
    Ply,
    Gltf,
    Unknown,
}

pub fn detect_geometry_format(path: &str, bytes: &[u8]) -> GeometryFormat {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".stl") {
        return GeometryFormat::Stl;
    }
    if lower.ends_with(".step") || lower.ends_with(".stp") {
        return GeometryFormat::Step;
    }
    if lower.ends_with(".obj") {
        return GeometryFormat::Obj;
    }
    if lower.ends_with(".ply") {
        return GeometryFormat::Ply;
    }
    if lower.ends_with(".gltf") || lower.ends_with(".glb") {
        return GeometryFormat::Gltf;
    }

    let header = String::from_utf8_lossy(&bytes[..bytes.len().min(512)]).to_ascii_lowercase();
    if header.starts_with("solid") {
        return GeometryFormat::Stl;
    }
    if header.contains("iso-10303-21") {
        return GeometryFormat::Step;
    }
    if header.starts_with("ply\n") || header.starts_with("ply\r\n") {
        return GeometryFormat::Ply;
    }
    if header.contains("\"asset\"") && header.contains("\"version\"") && header.contains("2.0") {
        return GeometryFormat::Gltf;
    }
    if looks_like_obj(&header) {
        return GeometryFormat::Obj;
    }
    GeometryFormat::Unknown
}

fn looks_like_obj(header: &str) -> bool {
    let mut vertex_lines = 0usize;
    let mut face_lines = 0usize;
    for line in header.lines().take(32) {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("v ") {
            vertex_lines += 1;
        } else if trimmed.starts_with("f ") {
            face_lines += 1;
        }
    }
    vertex_lines >= 3 && face_lines >= 1
}

#[cfg(test)]
mod tests {
    use super::{detect_geometry_format, GeometryFormat};

    #[test]
    fn step_detected_from_extension() {
        let format = detect_geometry_format("/model.step", b"not-needed");
        assert_eq!(format, GeometryFormat::Step);
    }

    #[test]
    fn step_detected_from_header_without_extension() {
        let format = detect_geometry_format("/model.dat", b"ISO-10303-21;\nHEADER;\n");
        assert_eq!(format, GeometryFormat::Step);
    }

    #[test]
    fn obj_detected_from_header_without_extension() {
        let format = detect_geometry_format(
            "/model.dat",
            b"# mesh\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        );
        assert_eq!(format, GeometryFormat::Obj);
    }

    #[test]
    fn ply_detected_from_header_without_extension() {
        let format = detect_geometry_format(
            "/model.dat",
            b"ply\nformat ascii 1.0\nelement vertex 3\nend_header\n",
        );
        assert_eq!(format, GeometryFormat::Ply);
    }
}
