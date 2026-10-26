//! Independent divergence-theorem recovery check and error decomposition; no extra solve.
use super::*;
use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

pub(super) struct Errors {
    pub recovery_max: f64,
    pub galerkin_rms: f64,
    pub interpolation_rms: f64,
    pub cross_mean: f64,
    pub total_rms: f64,
    pub independent_total_rms: f64,
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

pub(super) fn measure(
    result: &CanonicalAcousticResponse,
    cells: usize,
    quadratic: bool,
    frequency: f64,
    amplitude: [f64; 2],
    rotated: bool,
    direction: [f64; 3],
) -> Errors {
    let (model, mesh) = fixture(cells, quadratic, frequency, amplitude, rotated);
    let properties = crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
    let omega = std::f64::consts::TAU * frequency;
    let assembly = AcousticAssembly::assemble(
        result.projection.clone(),
        &properties,
        runmat_types::physical::AngularFrequency::new(omega),
    )
    .unwrap();
    let c = model.materials[0]
        .acoustic
        .as_ref()
        .unwrap()
        .speed_of_sound_m_per_s
        .canonical_value();
    let k = omega / c;
    let exact_nodes = result
        .projection
        .nodes
        .iter()
        .map(|node| {
            let x = dot(node.coordinates_m.canonical_value(), direction);
            multiply(amplitude, [(k * x).cos(), -(k * x).sin()])
        })
        .collect::<Vec<_>>();
    let real = exact_nodes.iter().map(|p| p[0]).collect::<Vec<_>>();
    let imaginary = exact_nodes.iter().map(|p| p[1]).collect::<Vec<_>>();
    let recovered = super::super::super::recovery::recover(&assembly, &real, &imaginary).unwrap();
    let mut errors = Errors {
        recovery_max: 0.0,
        galerkin_rms: 0.0,
        interpolation_rms: 0.0,
        cross_mean: 0.0,
        total_rms: 0.0,
        independent_total_rms: 0.0,
    };
    for (index, element) in result.projection.elements.iter().enumerate() {
        let corners: [[f64; 3]; 4] = std::array::from_fn(|i| {
            result.projection.nodes[element.node_indices[i]]
                .coordinates_m
                .canonical_value()
        });
        let volume = dot(
            subtract(corners[1], corners[0]),
            cross(
                subtract(corners[2], corners[0]),
                subtract(corners[3], corners[0]),
            ),
        ) / 6.0;
        assert!(volume.is_finite() && volume > 0.0);
        let mut integral = [[0.0; 3]; 2];
        for omitted in 0..4 {
            let face = (0..4).filter(|i| *i != omitted).collect::<Vec<_>>();
            let origin = corners[face[0]];
            let mut area = cross(
                subtract(corners[face[1]], origin),
                subtract(corners[face[2]], origin),
            )
            .map(|x| x * 0.5);
            if dot(area, subtract(corners[omitted], origin)) > 0.0 {
                area = area.map(|x| -x);
            }
            // Exact integral of an affine Tri3 or quadratic Tri6 nodal interpolant.
            let locals = if quadratic {
                TETRAHEDRON_MIDSIDE_EDGE_CORNERS
                    .into_iter()
                    .enumerate()
                    .filter(|(_, ends)| ends.iter().all(|i| face.contains(i)))
                    .map(|(edge, _)| edge + 4)
                    .collect::<Vec<_>>()
            } else {
                face
            };
            assert_eq!(locals.len(), 3);
            for local in locals {
                let p = exact_nodes[element.node_indices[local]];
                for (component, pressure) in p.iter().enumerate() {
                    for (axis, value) in area.iter().enumerate() {
                        integral[component][axis] += pressure * value / 3.0;
                    }
                }
            }
        }
        let rho = assembly.density.canonical_values()[index];
        let face_velocity = [
            integral[1].map(|g| -g / (volume * omega * rho)),
            integral[0].map(|g| g / (volume * omega * rho)),
        ];
        let reference = multiply(amplitude, mean_wave(corners.map(|p| dot(p, direction)), k));
        for (axis, axial_direction) in direction.iter().enumerate() {
            let interpolation = [
                recovered.velocity[index].real[axis],
                recovered.velocity[index].imaginary[axis],
            ];
            let actual = [
                result.element_velocity_m_per_s.canonical_values()[index].real[axis],
                result.element_velocity_m_per_s.canonical_values()[index].imaginary[axis],
            ];
            for (component, interpolated) in interpolation.iter().enumerate() {
                let exact = axial_direction * reference[component] / (rho * c);
                let galerkin = actual[component] - interpolated;
                let interpolation_error = interpolated - exact;
                errors.recovery_max = errors
                    .recovery_max
                    .max((interpolated - face_velocity[component][axis]).abs());
                errors.galerkin_rms += galerkin.powi(2);
                errors.interpolation_rms += interpolation_error.powi(2);
                errors.cross_mean += 2.0 * galerkin * interpolation_error;
                errors.total_rms += (actual[component] - exact).powi(2);
                // Independent divergence-theorem mean-gradient reconstruction of v_I.
                // Combining that with v_h-v_I must recover the original total error.
                errors.independent_total_rms +=
                    (galerkin + face_velocity[component][axis] - exact).powi(2);
            }
        }
    }
    let count = result.projection.elements.len() as f64;
    let reconstructed = errors.galerkin_rms + errors.interpolation_rms + errors.cross_mean;
    let scale = errors.galerkin_rms + errors.interpolation_rms + errors.cross_mean.abs();
    assert!(
        (errors.total_rms - reconstructed).abs() <= 128.0 * f64::EPSILON * scale,
        "complex mean-velocity error decomposition does not close: total={} reconstructed={reconstructed} scale={scale}",
        errors.total_rms
    );
    errors.cross_mean /= count;
    errors.total_rms = (errors.total_rms / count).sqrt();
    errors.independent_total_rms = (errors.independent_total_rms / count).sqrt();
    errors.galerkin_rms = (errors.galerkin_rms / count).sqrt();
    errors.interpolation_rms = (errors.interpolation_rms / count).sqrt();
    errors
}
