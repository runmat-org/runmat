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

    let header = String::from_utf8_lossy(&bytes[..bytes.len().min(64)]).to_ascii_lowercase();
    if header.starts_with("solid") {
        return GeometryFormat::Stl;
    }
    if header.contains("iso-10303-21") {
        return GeometryFormat::Step;
    }
    GeometryFormat::Unknown
}
