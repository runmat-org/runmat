use super::*;
use crate::core::{RenderData, Vertex};
use crate::plots::{ContourPlot, LinePlot, ReferenceLine, ReferenceLineOrientation, StairsPlot};
use glam::{Vec3, Vec4};

fn rendered_extent(items: &[(usize, RenderData)]) -> Option<BoundingBox> {
    items
        .iter()
        .filter_map(|(_, data)| data.bounds)
        .reduce(|a, b| a.union(&b))
}

#[test]
fn mixed_series_contour_and_reference_extents_match_render_contract() {
    let mut figure = Figure::new();
    figure.add_line_plot(LinePlot::new(vec![0., 4.], vec![1., 2.]).unwrap());
    figure.add_contour_plot(
        ContourPlot::from_vertices(
            vec![
                Vertex::new(Vec3::new(-1., -1., 0.), Vec4::ONE),
                Vertex::new(Vec3::new(5., 3., 0.), Vec4::ONE),
            ],
            0.,
            BoundingBox::new(Vec3::new(-1., -1., 0.), Vec3::new(5., 3., 0.)),
        )
        .with_line_width(5.),
    );
    figure.add_reference_line_on_axes(
        ReferenceLine::new(ReferenceLineOrientation::Vertical, 8.).unwrap(),
        0,
    );
    let prepared = figure.prepare_axes_extents(&[(600, 400)]);
    let items =
        figure.render_data_with_axes_with_viewport_and_gpu(Some((600, 400)), None, None, None);
    let expected = rendered_extent(&items).unwrap();
    let actual = prepared.bounds[0].unwrap();
    assert_eq!(actual.min, expected.min);
    assert_eq!(actual.max, expected.max);
    assert!(actual.min.y < -1.);
}

#[test]
fn reference_only_extents_keep_existing_ranges_and_manual_limits() {
    for orientation in [
        ReferenceLineOrientation::Horizontal,
        ReferenceLineOrientation::Vertical,
    ] {
        let mut figure = Figure::new();
        figure.set_axes_limits(0, Some((-20., 40.)), Some((-5., 15.)));
        figure.add_reference_line_on_axes(ReferenceLine::new(orientation, 3.).unwrap(), 0);
        let prepared = figure.prepare_axes_extents(&[(600, 400)]);
        let items =
            figure.render_data_with_axes_with_viewport_and_gpu(Some((600, 400)), None, None, None);
        let expected = rendered_extent(&items).unwrap();
        assert_eq!(prepared.bounds[0].unwrap().min, expected.min);
        assert_eq!(prepared.bounds[0].unwrap().max, expected.max);
    }
}

#[test]
fn mixed_line_stairs_use_same_manual_display_scale() {
    let mut figure = Figure::new();
    figure.add_line_plot(LinePlot::new(vec![0., 1.], vec![0., 0.]).unwrap());
    let mut stairs = StairsPlot::new(vec![0., 1.], vec![0., 0.]).unwrap();
    stairs.line_width = 4.;
    figure.add_stairs_plot(stairs);
    let items = figure.render_data_with_axes_with_viewport_and_gpu(
        Some((100, 100)),
        None,
        Some(&[Some((-10., 10., -10., 10.))]),
        None,
    );
    let stairs = &items[1].1;
    let min = stairs
        .vertices
        .iter()
        .map(|v| v.position[1])
        .fold(f32::INFINITY, f32::min);
    let max = stairs
        .vertices
        .iter()
        .map(|v| v.position[1])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (max - min - 0.8).abs() < 1e-5,
        "manual scale stroke width: {min}..{max}"
    );
    assert_eq!(items[0].1.bounds.unwrap().max.x, 1.);
}

#[test]
fn every_plot_variant_prepares_the_existing_render_extent_policy() {
    use crate::plots::*;
    let points = vec![
        Vec3::new(-2., -1., 0.),
        Vec3::new(3., -1., 1.),
        Vec3::new(0., 4., 2.),
    ];
    let vertices: Vec<_> = points.iter().map(|p| Vertex::new(*p, Vec4::ONE)).collect();
    let bounds = BoundingBox::from_points(&points);
    let surface =
        || SurfacePlot::new(vec![0., 2.], vec![0., 3.], vec![vec![1., 2.], vec![3., 4.]]).unwrap();
    let variants = vec![
        PlotElement::Line(LinePlot::new(vec![-2., 3.], vec![4., -1.]).unwrap()),
        PlotElement::Scatter(ScatterPlot::new(vec![-2., 3.], vec![4., -1.]).unwrap()),
        PlotElement::Bar(BarChart::new(vec!["a".into(), "b".into()], vec![-2., 3.]).unwrap()),
        PlotElement::ErrorBar(Box::new(
            ErrorBar::new_vertical(vec![0., 2.], vec![1., 3.], vec![1., 2.], vec![3., 4.]).unwrap(),
        )),
        PlotElement::Stairs(StairsPlot::new(vec![0., 2.], vec![-1., 3.]).unwrap()),
        PlotElement::Stem(StemPlot::new(vec![0., 2.], vec![-1., 3.]).unwrap()),
        PlotElement::Area(AreaPlot::new(vec![0., 2.], vec![-1., 3.]).unwrap()),
        PlotElement::Quiver(
            QuiverPlot::new(vec![0., 2.], vec![-1., 3.], vec![1., 2.], vec![3., 4.]).unwrap(),
        ),
        PlotElement::Pie(PieChart::new(vec![1., 2.], None).unwrap()),
        PlotElement::Surface(surface()),
        PlotElement::Surface(surface().with_image_mode(true)),
        PlotElement::Surface(surface().with_flatten_z(true)),
        PlotElement::Mesh(Box::new(
            MeshPlot::new(points.clone(), vec![[0, 1, 2]]).unwrap(),
        )),
        PlotElement::Patch(PatchPlot::new(points.clone(), vec![vec![0, 1, 2]]).unwrap()),
        PlotElement::Line3(Line3Plot::new(vec![0., 1.], vec![2., 3.], vec![-2., 4.]).unwrap()),
        PlotElement::Scatter3(Scatter3Plot::new(points).unwrap()),
        PlotElement::Contour(
            ContourPlot::from_vertices(vertices.clone(), 0., bounds).with_line_width(4.),
        ),
        PlotElement::ContourFill(ContourFillPlot::from_vertices(vertices, bounds)),
        PlotElement::ReferenceLine(
            ReferenceLine::new(ReferenceLineOrientation::Horizontal, 4.).unwrap(),
        ),
    ];
    for (index, plot) in variants.into_iter().enumerate() {
        for kind in [
            super::super::AxesKind::Cartesian,
            super::super::AxesKind::Polar,
        ] {
            let mut figure = Figure::new();
            figure.set_axes_kind(0, kind);
            figure.add_plot_element_on_axes(plot.clone(), 0);
            let prepared = figure.prepare_axes_extents(&[(600, 400)]).bounds[0];
            let rendered = figure.render_data_with_axes_with_viewport_and_gpu(
                Some((600, 400)),
                None,
                None,
                None,
            );
            match (prepared, rendered_extent(&rendered)) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.min, b.min, "variant {index}, {kind:?}");
                    assert_eq!(a.max, b.max, "variant {index}, {kind:?}");
                }
                (a, b) => panic!("variant {index}, {kind:?}: prepared={a:?}, rendered={b:?}"),
            }
        }
    }
}
