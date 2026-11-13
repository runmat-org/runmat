//! Face-ID drawing callouts for agent images, shared by browser and desktop.
//! Leader endpoints must lie on a visible triangle in the final observation.
use crate::{
    core::{Camera, DepthMode, PipelineType},
    GeometryScene,
};
use font8x8::{UnicodeFonts, BASIC_FONTS};
use image::{Rgba, RgbaImage};
use std::io::Cursor;

type Point = [f32; 3];
type Triangle = [Point; 3];
fn project(matrix: glam::Mat4, point: [f32; 3], width: u32, height: u32) -> Option<Point> {
    let p = matrix * glam::Vec3::from_array(point).extend(1.0);
    if !p.is_finite() || p.w <= 0.0 {
        return None;
    }
    let p = p.truncate() / p.w;
    (p.z >= 0.0 && p.z <= 1.0).then_some([
        (p.x + 1.0) * width as f32 * 0.5,
        (1.0 - p.y) * height as f32 * 0.5,
        p.z,
    ])
}
fn area(t: &Triangle) -> f32 {
    (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])
}
fn depth(t: &Triangle, p: Point) -> Option<f32> {
    let a = area(t);
    if a.abs() < 0.001 {
        return None;
    }
    let u = ((t[1][0] - p[0]) * (t[2][1] - p[1]) - (t[1][1] - p[1]) * (t[2][0] - p[0])) / a;
    let v = ((t[2][0] - p[0]) * (t[0][1] - p[1]) - (t[2][1] - p[1]) * (t[0][0] - p[0])) / a;
    let w = 1.0 - u - v;
    (u >= -0.0001 && v >= -0.0001 && w >= -0.0001)
        .then_some(u * t[0][2] + v * t[1][2] + w * t[2][2])
}
fn projected(matrix: glam::Mat4, t: Triangle, width: u32, height: u32) -> Option<Triangle> {
    Some([
        project(matrix, t[0], width, height)?,
        project(matrix, t[1], width, height)?,
        project(matrix, t[2], width, height)?,
    ])
}
fn put(image: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
        image.put_pixel(x as u32, y as u32, color);
    }
}
fn line(image: &mut RgbaImage, from: [i32; 2], to: [i32; 2], color: Rgba<u8>) {
    let steps = (to[0] - from[0]).abs().max((to[1] - from[1]).abs()).max(1);
    for i in 0..=steps {
        put(
            image,
            from[0] + (to[0] - from[0]) * i / steps,
            from[1] + (to[1] - from[1]) * i / steps,
            color,
        );
    }
}
fn text(image: &mut RgbaImage, at: [i32; 2], label: &str, scale: i32) {
    for (index, c) in label.chars().enumerate() {
        if let Some(bitmap) = BASIC_FONTS.get(c) {
            for (y, row) in bitmap.iter().enumerate() {
                for x in 0..8 {
                    if row & (1 << x) != 0 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                put(
                                    image,
                                    at[0] + (index as i32 * 8 + x) * scale + dx,
                                    at[1] + y as i32 * scale + dy,
                                    Rgba([245, 247, 250, 255]),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
pub(super) fn annotate_png(
    png: Vec<u8>,
    scene: &GeometryScene,
    camera: &Camera,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    if scene.observation_callouts.is_empty() || width < 256 || height < 256 {
        return Ok(png);
    }
    let mut camera = camera.clone();
    let matrix = camera.view_proj_matrix();
    let mut surfaces = Vec::new();
    for chunk in &scene.chunks {
        if !chunk.visible || chunk.render_data.pipeline_type != PipelineType::Triangles {
            continue;
        }
        let data = &chunk.render_data;
        for n in 0..chunk.triangle_count() {
            let vertex = |corner| {
                let at = n * 3 + corner;
                let i = data.indices.as_ref().map_or(at, |ids| ids[at] as usize);
                data.vertices[i].position
            };
            if let Some(t) = projected(matrix, [vertex(0), vertex(1), vertex(2)], width, height) {
                surfaces.push(t);
            }
        }
    }
    let mut labels = Vec::new();
    for callout in &scene.observation_callouts {
        let mut candidates = callout
            .triangles
            .iter()
            .filter_map(|t| projected(matrix, *t, width, height))
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| area(b).abs().total_cmp(&area(a).abs()));
        let anchor = candidates
            .iter()
            .filter(|t| area(t).abs() > 0.5)
            .find_map(|t| {
                let p = [
                    (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                    (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                    (t[0][2] + t[1][2] + t[2][2]) / 3.0,
                ];
                if p[0] < 0.0 || p[1] < 0.0 || p[0] >= width as f32 || p[1] >= height as f32 {
                    return None;
                }
                let depths = surfaces.iter().filter_map(|t| depth(t, p));
                let nearest = match camera.depth_mode {
                    DepthMode::Standard => depths.fold(f32::INFINITY, f32::min),
                    DepthMode::ReversedZ => depths.fold(f32::NEG_INFINITY, f32::max),
                };
                // Also rejects a source face removed by visibility or sectioning.
                ((nearest - p[2]).abs() < 0.00001).then_some(p)
            });
        let name = callout
            .region_id
            .strip_prefix("face_")
            .and_then(|s| s.parse::<u64>().ok())
            .map(|id| format!("Face {id}"))
            .unwrap_or_else(|| callout.region_id.chars().take(36).collect());
        labels.push((anchor, name, callout.color));
    }
    labels.sort_by(|a, b| {
        a.0.map_or(f32::INFINITY, |p| p[1])
            .total_cmp(&b.0.map_or(f32::INFINITY, |p| p[1]))
    });
    let mut image = image::load_from_memory(&png)
        .map_err(|e| e.to_string())?
        .into_rgba8();
    let scale = if width >= 800 { 2 } else { 1 };
    let row = 8 * scale + 12;
    // Two compact rails with monotonic row placement keep labels from overlapping.
    let side_of =
        |anchor: Option<Point>| usize::from(anchor.is_none_or(|p| p[0] >= width as f32 / 2.0));
    let mut remaining = [0i32; 2];
    for (anchor, _, _) in &labels {
        remaining[side_of(*anchor)] += 1;
    }
    let mut next = [12i32; 2];
    for (anchor, name, color) in labels {
        let side = side_of(anchor);
        let name = if anchor.is_some() {
            name
        } else {
            format!("{name} (not visible)")
        };
        let box_width = (name.chars().count() as i32 * 8 * scale + 12).min(width as i32 - 16);
        let last_start = height as i32 - row - 8 - (remaining[side] - 1) * (row + 6);
        let y = anchor
            .map_or(next[side], |p| (p[1] as i32 - row / 2).max(next[side]))
            .min(last_start.max(next[side]));
        remaining[side] -= 1;
        if y + row >= height as i32 {
            continue;
        }
        next[side] = y + row + 6;
        let x = if side == 0 {
            8
        } else {
            width as i32 - box_width - 8
        };
        let ink = Rgba([
            (color[0] * 255.0) as u8,
            (color[1] * 255.0) as u8,
            (color[2] * 255.0) as u8,
            255,
        ]);
        if let Some(p) = anchor {
            let endpoint = [if side == 0 { x + box_width } else { x }, y + row / 2];
            line(&mut image, [p[0] as i32, p[1] as i32], endpoint, ink);
            for dy in -2..=2 {
                for dx in -2..=2 {
                    put(&mut image, p[0] as i32 + dx, p[1] as i32 + dy, ink);
                }
            }
        }
        for dy in 0..row {
            for dx in 0..box_width {
                put(
                    &mut image,
                    x + dx,
                    y + dy,
                    if dy == 0 || dy == row - 1 {
                        ink
                    } else {
                        Rgba([22, 25, 31, 255])
                    },
                );
            }
        }
        text(&mut image, [x + 6, y + 6], &name, scale);
    }
    let mut output = Cursor::new(Vec::new());
    image
        .write_to(&mut output, image::ImageOutputFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(output.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry_scene::GeometrySceneCallout, GeometrySceneChunk};

    #[test]
    fn leaders_point_only_to_visible_faces_in_both_depth_modes() {
        let front = [[-1.0, -1.0, 0.5], [1.0, -1.0, 0.5], [0.0, 1.0, 0.5]];
        let back = front.map(|[x, y, _]| [x, y, 0.0]);
        let vertices = front
            .into_iter()
            .map(|p| crate::geometry_scene_vertex(p, [0.5; 4], [0.0, 0.0, 1.0]))
            .collect();
        let mut scene = GeometryScene::new(
            "callout-test",
            1,
            vec![GeometrySceneChunk::indexed_triangles(
                "front",
                vertices,
                vec![0, 1, 2],
                Default::default(),
            )],
        );
        for mode in [DepthMode::Standard, DepthMode::ReversedZ] {
            let mut camera = Camera::new();
            camera.position = glam::Vec3::new(0.0, 0.0, 4.0);
            camera.target = glam::Vec3::ZERO;
            camera.up = glam::Vec3::Y;
            camera.aspect_ratio = 1.0;
            camera.depth_mode = mode;
            camera.mark_dirty();
            for (triangle, visible) in [(front, true), (back, false)] {
                scene.observation_callouts = vec![GeometrySceneCallout {
                    region_id: "face_8".into(),
                    color: [1.0, 0.0, 0.0, 1.0],
                    triangles: vec![triangle],
                }];
                let image = RgbaImage::from_pixel(512, 512, Rgba([255; 4]));
                let mut png = Cursor::new(Vec::new());
                image
                    .write_to(&mut png, image::ImageOutputFormat::Png)
                    .unwrap();
                let annotated = annotate_png(png.into_inner(), &scene, &camera, 512, 512).unwrap();
                let decoded = image::load_from_memory(&annotated).unwrap().into_rgba8();
                let mut camera_copy = camera.clone();
                let anchor = project(
                    camera_copy.view_proj_matrix(),
                    [0.0, -1.0 / 3.0, triangle[0][2]],
                    512,
                    512,
                )
                .unwrap();
                assert_eq!(
                    decoded.get_pixel(anchor[0] as u32, anchor[1] as u32).0 == [255, 0, 0, 255],
                    visible,
                    "mode={mode:?}"
                );
                if let Ok(directory) = std::env::var("RUNMAT_CALLOUT_TEST_OUTPUT") {
                    std::fs::write(
                        format!("{directory}/callout-{mode:?}-{visible}.png"),
                        annotated,
                    )
                    .unwrap();
                }
            }
        }
    }
    #[test]
    fn depth_only_accepts_points_on_the_triangle_and_rejects_degenerate_faces() {
        let t = [[0.0, 0.0, 0.2], [10.0, 0.0, 0.4], [0.0, 10.0, 0.6]];
        assert!((depth(&t, [2.0, 2.0, 0.0]).unwrap() - 0.32).abs() < 0.00001);
        assert!(depth(&t, [8.0, 8.0, 0.0]).is_none());
        assert!(depth(&[[0.0; 3]; 3], [0.0; 3]).is_none());
    }
}
