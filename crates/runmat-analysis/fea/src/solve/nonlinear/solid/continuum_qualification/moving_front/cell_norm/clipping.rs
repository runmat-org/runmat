//! Independent convex tetrahedron clipping for this unit-cube reference only.
type Point = [f64; 3];
type Tet = [Point; 4];

fn unique(points: impl IntoIterator<Item = Point>) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::new();
    for point in points {
        if out.iter().all(|p| {
            p.iter()
                .zip(point)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                > 1e-28
        }) {
            out.push(point);
        }
    }
    out
}

pub(super) fn clip(tet: Tet, plane: f64, left: bool) -> Vec<Tet> {
    // Analytic load factors can place a front on a mesh plane within arithmetic
    // roundoff. Resolve coincident vertices before clipping, rather than generating
    // a sliver whose cut face collapses under duplicate-point roundoff. This is a
    // coordinate arithmetic bound for the unit fixture, not a physics-error budget.
    let plane = tet
        .iter()
        .map(|p| p[0])
        .find(|x| (x - plane).abs() <= 32.0 * f64::EPSILON)
        .unwrap_or(plane);
    let inside = |p: Point| if left { p[0] <= plane } else { p[0] >= plane };
    if tet.iter().all(|&p| inside(p)) {
        return vec![tet];
    }
    if tet.iter().all(|&p| !inside(p) || p[0] == plane) {
        return Vec::new();
    }
    let intersection = |a: Point, b: Point| {
        let t = (plane - a[0]) / (b[0] - a[0]);
        [plane, a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
    };
    let mut faces = Vec::new();
    let mut cut = Vec::new();
    for indices in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
        let vertices = indices.map(|i| tet[i]);
        let mut polygon = Vec::new();
        for edge in 0..3 {
            let a = vertices[edge];
            let b = vertices[(edge + 1) % 3];
            if inside(a) {
                polygon.push(a);
                if a[0] == plane {
                    cut.push(a);
                }
            }
            if inside(a) != inside(b) {
                let p = intersection(a, b);
                polygon.push(p);
                cut.push(p);
            }
        }
        let polygon = unique(polygon);
        if polygon.len() >= 3 {
            faces.push(polygon);
        }
    }
    let mut cut = unique(cut);
    assert!(cut.len() >= 3);
    let center_y = cut.iter().map(|p| p[1]).sum::<f64>() / cut.len() as f64;
    let center_z = cut.iter().map(|p| p[2]).sum::<f64>() / cut.len() as f64;
    cut.sort_by(|a, b| {
        (a[2] - center_z)
            .atan2(a[1] - center_y)
            .total_cmp(&(b[2] - center_z).atan2(b[1] - center_y))
    });
    faces.push(cut);
    let vertices = unique(faces.iter().flatten().copied());
    let center = std::array::from_fn(|axis| {
        vertices.iter().map(|p| p[axis]).sum::<f64>() / vertices.len() as f64
    });
    faces
        .into_iter()
        .flat_map(|face| (1..face.len() - 1).map(move |i| [center, face[0], face[i], face[i + 1]]))
        .collect()
}
