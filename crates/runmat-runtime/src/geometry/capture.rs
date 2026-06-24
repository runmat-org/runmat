use runmat_geometry_core::GeometryAsset;
use runmat_geometry_ops::compute_stats;

use super::{GeometryCaptureViewResult, GeometryCaptureViewSpec, GeometryViewCaptureAdapter};

pub(super) static DEFAULT_SVG_CAPTURE_ADAPTER: DefaultSvgCaptureAdapter = DefaultSvgCaptureAdapter;

pub(super) struct DefaultSvgCaptureAdapter;

impl GeometryViewCaptureAdapter for DefaultSvgCaptureAdapter {
    fn adapter_name(&self) -> &'static str {
        "runtime-svg-summary"
    }

    fn capture(
        &self,
        asset: &GeometryAsset,
        view_spec: &GeometryCaptureViewSpec,
    ) -> Result<GeometryCaptureViewResult, String> {
        if !view_spec.format.eq_ignore_ascii_case("svg") {
            return Err(format!(
                "unsupported capture format '{}' for default adapter",
                view_spec.format
            ));
        }

        let stats = compute_stats(asset);
        let width = view_spec.width;
        let height = view_spec.height;

        let svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\
<defs><linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\"><stop offset=\"0%\" stop-color=\"#0f172a\"/>\
<stop offset=\"100%\" stop-color=\"#1e293b\"/></linearGradient></defs>\
<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"url(#bg)\"/>\
<g fill=\"none\" stroke=\"#7dd3fc\" stroke-width=\"2\" opacity=\"0.85\">\
<path d=\"M 48 {h2} L {w2} 40 L {w3} {h2} L {w2} {h3} Z\"/>\
<path d=\"M 48 {h2} L 48 {h3} L {w2} {h4} L {w2} 40\" opacity=\"0.65\"/>\
<path d=\"M {w3} {h2} L {w3} {h3} L {w2} {h4}\" opacity=\"0.65\"/>\
</g>\
<g fill=\"#e2e8f0\" font-family=\"ui-monospace, SFMono-Regular, Menlo, monospace\">\
<text x=\"28\" y=\"{t1}\" font-size=\"20\">Geometry Snapshot</text>\
<text x=\"28\" y=\"{t2}\" font-size=\"12\" fill=\"#94a3b8\">id: {geometry_id}</text>\
<text x=\"28\" y=\"{t3}\" font-size=\"12\" fill=\"#94a3b8\">units: {units}</text>\
<text x=\"28\" y=\"{t4}\" font-size=\"12\" fill=\"#cbd5e1\">meshes: {mesh_count}  vertices: {vertices}  elements: {elements}</text>\
<text x=\"28\" y=\"{t5}\" font-size=\"12\" fill=\"#cbd5e1\">regions: {regions}  revision: {revision}</text>\
</g></svg>",
            h2 = (height as f32 * 0.5) as u32,
            h3 = (height as f32 * 0.74) as u32,
            h4 = (height as f32 * 0.88) as u32,
            w2 = (width as f32 * 0.56) as u32,
            w3 = (width as f32 * 0.88) as u32,
            t1 = (height as f32 * 0.14) as u32,
            t2 = (height as f32 * 0.22) as u32,
            t3 = (height as f32 * 0.28) as u32,
            t4 = (height as f32 * 0.90) as u32,
            t5 = (height as f32 * 0.95) as u32,
            geometry_id = xml_escape(&asset.geometry_id),
            units = xml_escape(&format!("{:?}", asset.units)).to_lowercase(),
            mesh_count = stats.mesh_count,
            vertices = stats.total_vertices,
            elements = stats.total_elements,
            regions = stats.region_count,
            revision = asset.revision,
        );

        Ok(GeometryCaptureViewResult {
            format: "svg".to_string(),
            width,
            height,
            payload: svg.into_bytes(),
        })
    }
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
