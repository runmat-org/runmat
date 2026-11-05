//! Validate actual generated source independently of adapter availability.
use super::*;
use crate::gpu::ScalarType;

#[test]
fn plotting_shader_sources_parse_and_validate_for_every_declared_variant() {
    let pairs = [
        ("area", area::F32, area::F64),
        ("bar", bar::F32, bar::F64),
        ("contour", contour::F32, contour::F64),
        ("contour-fill", contour_fill::F32, contour_fill::F64),
        ("errorbar", errorbar::F32, errorbar::F64),
        ("image", image::F32, image::F64),
        ("line", line::F32, line::F64),
        ("marker", line::MARKER_F32, line::MARKER_F64),
        ("line3", line3::F32, line3::F64),
        ("quiver", quiver::F32, quiver::F64),
        ("scatter2", scatter2::F32, scatter2::F64),
        ("scatter3", scatter3::F32, scatter3::F64),
        ("stairs", stairs::F32, stairs::F64),
        ("stem", stem::F32, stem::F64),
        ("surface", surface::F32, surface::F64),
    ];
    let mut failures = Vec::new();
    let mut variants = 0;
    for workgroup in [64, 256] {
        for (name, f32, f64) in pairs {
            for (precision, template) in [("f32", f32), ("f64", f64)] {
                validate(
                    &format!("{name}/{precision}/{workgroup}"),
                    &compute_source(template, workgroup),
                    &mut failures,
                );
                variants += 1;
            }
        }
        for (name, scalar) in [
            ("histogram/f32", ScalarType::F32),
            ("histogram/f64", ScalarType::F64),
        ] {
            validate(
                &format!("{name}/{workgroup}"),
                &histogram::source(scalar, workgroup),
                &mut failures,
            );
            variants += 1;
        }
    }
    for (name, source) in [
        ("vertex/grid", vertex::GRID_PLANE),
        ("vertex/image", vertex::IMAGE_DIRECT),
        ("vertex/line", vertex::LINE),
        ("vertex/line-direct", vertex::LINE_DIRECT),
        ("vertex/point", vertex::POINT),
        ("vertex/billboard", vertex::POINT_BILLBOARD),
        ("vertex/point-direct", vertex::POINT_DIRECT),
        ("vertex/triangle", vertex::TRIANGLE),
    ] {
        validate(name, source, &mut failures);
        variants += 1;
    }
    for (depth_tested, reversed_z) in [(true, true), (true, false), (false, true)] {
        validate(
            &format!("vertex/edge-stroke depth={depth_tested} reversed={reversed_z}"),
            &vertex::edge_stroke::source(depth_tested, reversed_z),
            &mut failures,
        );
        variants += 1;
    }
    assert_eq!(variants, 75, "the declared shader inventory changed");
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

fn validate(name: &str, source: &str, failures: &mut Vec<String>) {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(error) => {
            failures.push(format!("{name}: {}", error.emit_to_string(source)));
            return;
        }
    };
    if let Err(error) = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    {
        failures.push(format!("{name}: {}", error.emit_to_string(source)));
    }
}
