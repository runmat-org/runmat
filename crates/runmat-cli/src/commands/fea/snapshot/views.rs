use anyhow::Result;
use serde::Serialize;
use std::path::Path;
mod camera;
mod legend;
mod memory;
mod presentation;
mod profiles;
pub(super) use legend::authored_bound_units;
pub(super) use memory::peak_resident_memory;
pub(super) use presentation::Presentation;

/// What a view's legend is written in terms of: the unit each assessment states its bound in
/// (by assessment id) and each run's assessment results (by run id).
#[derive(Default)]
pub(super) struct LegendTerms {
    pub display_units: std::collections::BTreeMap<String, String>,
    pub assessments: std::collections::BTreeMap<
        String,
        Vec<runmat_runtime::analysis::assessment::PublishedStudyAssessment>,
    >,
}
pub(super) use profiles::CaptureProfile;

#[derive(Serialize)]
pub(super) struct View {
    pub id: String,
    /// The run whose original fields this view shows.
    pub run_id: String,
    pub label: String,
    pub field: Option<String>,
    /// The sample of `field` shown (a frequency, a mode, a time), when the field has several.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<runmat_analysis_fea::results::field::RecoveredFieldSample>,
    /// The published field that displaces the shape, when it is not `field` itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deformation_field: Option<String>,
    pub scalar_scale: f64,
    pub unit: String,
    /// The unit the legend is written in when it differs from `unit`: the one the study's
    /// assessment of this quantity states its bound in. `minimum` and `maximum` stay in `unit`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_unit: Option<String>,
    pub scene: String,
    pub poster: String,
    pub deformation_scale: f64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    /// The field's true extrema over every stored value of the solution, with where they
    /// occur; the legend range above is the same range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extrema: Option<runmat_runtime::analysis::field_scene::extrema::FieldExtrema>,
    /// The largest distance between a saved surface position and the solver's, from the
    /// scene's position quantization; the surface is otherwise the exact solver boundary.
    pub position_error_m: f64,
    /// How the boundary surface is drawn and what that leaves out of the solution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface: Option<runmat_runtime::analysis::field_scene::SurfaceApproximation>,
    /// A view inside the part: what it draws and its stated error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<VolumeView>,
    /// One camera shared by every view, framed on the union of their bounds; serialized as
    /// the state itself (always set before the views are returned).
    pub initial_camera: Option<runmat_runtime::builtins::plotting::PlotSurfaceCameraState>,
    /// The scene's title: what the view's result is of.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    /// CAD files drawn around the result as translucent context geometry. They show shape
    /// only; the result covers only the solved mesh.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<presentation::ContextRecord>,
}

/// A view whose scene is already written: what its poster pass still needs.
#[derive(Serialize)]
pub(super) struct VolumeView {
    pub kind: runmat_runtime::analysis::field_scene::volume::VolumeViewKind,
    /// What is drawn of the field: its magnitude (amplitude of a phasor), its value (the
    /// instantaneous value at phase 0 of a phasor), or its sound pressure level.
    pub drawn: &'static str,
    pub error: runmat_runtime::analysis::field_scene::volume::VolumeViewError,
}

struct Encoded {
    minimum: Option<f64>,
    maximum: Option<f64>,
    display_unit: Option<String>,
    position_error_m: f64,
    bounds: runmat_plot::core::BoundingBox,
    result_bounds: runmat_plot::core::BoundingBox,
    bytes: Vec<u8>,
    deformation_scale: f64,
    scalar_scale: f64,
    surface: Option<runmat_runtime::analysis::field_scene::SurfaceApproximation>,
    volume: Option<runmat_runtime::analysis::field_scene::volume::VolumeViewError>,
}

/// How finely one view is drawn.
#[derive(Clone, Copy)]
struct Detail {
    surface: runmat_runtime::analysis::SurfaceDetail,
    cap_triangle_budget: usize,
}

/// Encoding within a payload budget. A limit miss can retry at lower detail;
/// fatal encoding failures use the outer runtime result.
type EncodingAttempt = std::result::Result<Encoded, ScenePayloadLimit>;

struct ScenePayloadLimit {
    required: usize,
    maximum: usize,
    /// The section cap drawn (triangles, evaluation grid) for a section view.
    cap: Option<(usize, usize)>,
}

/// The next lighter drawing of a view whose scene is `required` bytes against a `maximum`
/// budget, or `None` when the view is already at its lightest. A surface view drops to corner
/// triangles. A section view scales its cap's triangle budget by the share of the payload that
/// must go (with a 10% margin); bytes outside the cap stay, so the loop repeats until the cap is
/// one polygon per cut element.
fn lighter(
    detail: Detail,
    volume: bool,
    required: usize,
    maximum: usize,
    cap: Option<(usize, usize)>,
) -> Option<Detail> {
    use runmat_runtime::analysis::SurfaceDetail;
    if !volume {
        return (detail.surface == SurfaceDetail::Full).then_some(Detail {
            surface: SurfaceDetail::Corners,
            ..detail
        });
    }
    let (triangles, subdivisions) = cap?;
    if subdivisions <= 1 {
        return None;
    }
    let scale = 0.9 * maximum as f64 / required as f64;
    Some(Detail {
        cap_triangle_budget: ((triangles as f64 * scale) as usize).min(triangles.saturating_sub(1)),
        ..detail
    })
}

struct Written {
    view: View,
    /// The whole scene, context geometry included.
    bounds: runmat_plot::core::BoundingBox,
    /// The result alone.
    result_bounds: runmat_plot::core::BoundingBox,
}

/// Mesh coordinates and element corners of one run, kept only while its views are captured.
type MeshEntities = (Vec<[f64; 3]>, Vec<Vec<usize>>);

/// `solver_mesh` gives each captured run's canonical solver mesh artifact; it places each
/// field's extrema.
///
/// Views stream: each scene is built, encoded and written before the next one is built, so
/// at most one view's geometry is in memory. Posters are drawn in a second pass from the
/// written scenes (the persisted representation, the same one the live viewer loads) with one
/// camera framed on the union of every view's bounds (or of their results alone).
pub(super) async fn capture(
    output: &Path,
    profile: CaptureProfile,
    solver_mesh: impl Fn(&str) -> Option<String>,
    presentation: &Presentation,
    terms: &LegendTerms,
) -> Result<Vec<View>> {
    use runmat_runtime::analysis::field_scene::extrema;
    use runmat_runtime::builtins::plotting;
    let mut mesh: Option<(String, Option<MeshEntities>)> = None;
    let mut written = Vec::new();
    let view_count = profile.views.len();
    for (position, spec) in profile.views.into_iter().enumerate() {
        let run_id = spec.run_id.as_str();
        let mesh_path = solver_mesh(run_id);
        // Views are grouped by run in capture order only loosely; reload when the run changes
        // rather than holding every run's mesh at once.
        if mesh.as_ref().is_none_or(|(loaded, _)| loaded != run_id) {
            // Release the previous run's mesh before loading the next.
            drop(mesh.take());
            let entities = mesh_path
                .as_deref()
                .map(extrema::solver_mesh_entities)
                .transpose()
                .map_err(anyhow::Error::msg)?;
            mesh = Some((run_id.to_string(), entities));
        }
        let entities = mesh.as_ref().and_then(|(_, entities)| entities.as_ref());
        let extrema = spec
            .field
            .as_ref()
            .map(|field| {
                extrema::field_extrema(
                    run_id,
                    field,
                    entities.map(|(nodes, _)| nodes.as_slice()),
                    entities.map(|(_, elements)| elements.as_slice()),
                )
            })
            .transpose()
            .map_err(anyhow::Error::msg)?;
        let encode = |detail: Detail| -> Result<EncodingAttempt> {
            let (mut scene, deformation_scale, surface, volume) = match spec.volume {
                Some(volume) => {
                    let solver_mesh = mesh_path.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("view '{}' needs the solver mesh", spec.id)
                    })?;
                    let field = spec
                        .field
                        .clone()
                        .ok_or_else(|| anyhow::anyhow!("view '{}' has no field", spec.id))?;
                    let request =
                        runmat_runtime::analysis::field_scene::volume::VolumeViewRequest {
                            field,
                            kind: volume.kind,
                            reduction: volume.reduction,
                            range: extrema
                                .as_ref()
                                .filter(|_| volume.true_extrema)
                                .map(|extrema| (extrema.minimum.value, extrema.maximum.value)),
                            legend: spec.legend.clone(),
                            unit: spec.unit.clone(),
                            plane: presentation.section,
                            cap_triangle_budget: detail.cap_triangle_budget,
                        };
                    let (scene, error) =
                        runmat_runtime::analysis::field_scene::volume::build_volume_scene(
                            run_id,
                            solver_mesh,
                            &request,
                        )
                        .map_err(anyhow::Error::msg)?;
                    (scene, 0.0, None, Some(error))
                }
                None => {
                    let (scene, deformation_scale, surface) =
                        runmat_runtime::analysis::field_scene::build_scene(
                            run_id,
                            mesh_path.as_deref(),
                            spec.field.as_ref(),
                            spec.deformation.as_ref(),
                            extrema
                                .as_ref()
                                .map(|extrema| (extrema.minimum.value, extrema.maximum.value)),
                            spec.shape,
                            1.0,
                            // Posters show the deformed body alone; an undeformed overlay drawn
                            // over a shape a few micrometres away only fights it for depth.
                            false,
                            detail.surface,
                        )
                        .map_err(anyhow::Error::msg)?;
                    (scene, deformation_scale, Some(surface), None)
                }
            };
            let result_bounds = scene.bounds;
            presentation.dress(&mut scene);
            let mut scalar_scale = 1.0;
            let mut range = None;
            let mut display_unit = None;
            if let Some(legend) = scene
                .overlay
                .as_mut()
                .and_then(|overlay| overlay.scalar_legend.as_mut())
            {
                legend.label = spec.legend.clone();
                legend.unit = spec.unit.clone();
                if spec.relative {
                    anyhow::ensure!(legend.maximum > 0.0, "mode shape has no nonzero amplitude");
                    scalar_scale = 1.0 / legend.maximum;
                    legend.minimum *= scalar_scale;
                    legend.maximum = 1.0;
                }
                range = Some((legend.minimum, legend.maximum));
                if !spec.relative {
                    display_unit = legend::present(
                        legend,
                        &spec.unit,
                        &terms.display_units,
                        terms
                            .assessments
                            .get(run_id)
                            .map(Vec::as_slice)
                            .unwrap_or_default(),
                    )?;
                }
            }
            let bytes = match plotting::encode_geometry_figure_scene_within(
                &runmat_plot::event::FigureScene::from_geometry_scene(&scene),
                presentation.max_scene_bytes,
            ) {
                Ok(bytes) => bytes,
                Err(error) => {
                    if let plotting::GeometrySceneEncodeErrorKind::PayloadTooLarge {
                        required,
                        maximum,
                    } = error.kind()
                    {
                        return Ok(Err(ScenePayloadLimit {
                            required,
                            maximum,
                            cap: volume.as_ref().and_then(|volume| {
                                volume.cap_triangles.zip(volume.cap_subdivisions)
                            }),
                        }));
                    }
                    return Err(anyhow::Error::new(error));
                }
            };
            Ok(Ok(Encoded {
                minimum: range.map(|range| range.0),
                maximum: range.map(|range| range.1),
                display_unit,
                position_error_m: runmat_plot::event::FigureScene::geometry_position_error_m(
                    &scene,
                ),
                bounds: scene.bounds,
                result_bounds,
                bytes,
                deformation_scale,
                scalar_scale,
                surface,
                volume,
            }))
        };
        // A scene larger than the consumer's budget (`--max-scene-bytes`) is drawn lighter until
        // it fits: a dense quadratic surface with one triangle per quadratic face, a section cap
        // on a coarser evaluation grid (down to one polygon per cut element). The view states
        // what that leaves out; values, legend and extrema stay those of the full solution.
        use runmat_runtime::analysis::SurfaceDetail;
        let Encoded {
            minimum,
            maximum,
            display_unit,
            position_error_m,
            bounds,
            result_bounds,
            bytes,
            deformation_scale,
            scalar_scale,
            surface,
            volume,
        } = {
            let mut detail = Detail {
                surface: SurfaceDetail::Full,
                cap_triangle_budget:
                    runmat_runtime::analysis::field_scene::volume::DEFAULT_CAP_TRIANGLE_BUDGET,
            };
            loop {
                let (required, maximum, cap) = match encode(detail)? {
                    Ok(encoded) => break encoded,
                    Err(ScenePayloadLimit {
                        required,
                        maximum,
                        cap,
                    }) => (required, maximum, cap),
                };
                detail = match lighter(detail, spec.volume.is_some(), required, maximum, cap) {
                    Some(lighter) => lighter,
                    None => anyhow::bail!(
                        "snapshot view '{}': its lightest scene is {required} bytes and the \
                         scene budget is {maximum} bytes; raise --max-scene-bytes or select \
                         other views with --view",
                        spec.id
                    ),
                };
                log::info!(
                    "snapshot view '{}': scene is {required} bytes, budget {maximum}; drawing \
                     {:?} surface, section cap of at most {} triangles",
                    spec.id,
                    detail.surface,
                    detail.cap_triangle_budget
                );
            }
        };
        let id = spec.id;
        let scene_path = format!("{id}.scene.json");
        std::fs::write(output.join(&scene_path), bytes)?;
        log::info!(
            "snapshot view {}/{view_count} '{id}' written; peak resident {}",
            position + 1,
            peak_resident_memory()
        );
        let deformation_field = spec
            .deformation
            .filter(|deformation| spec.field.as_ref() != Some(deformation))
            .filter(|_| deformation_scale != 0.0)
            .map(|deformation| deformation.field_id);
        written.push(Written {
            bounds,
            result_bounds,
            view: View {
                poster: format!("{id}.png"),
                id,
                run_id: spec.run_id,
                label: spec.label,
                sample: spec.field.as_ref().and_then(|field| field.sample),
                field: spec.field.map(|field| field.field_id),
                deformation_field,
                scalar_scale,
                unit: spec.unit,
                display_unit,
                scene: scene_path,
                deformation_scale,
                minimum,
                maximum,
                extrema: extrema.filter(|_| {
                    !spec.relative && spec.volume.is_none_or(|volume| volume.true_extrema)
                }),
                position_error_m,
                surface,
                volume: spec.volume.zip(volume).map(|(spec, error)| VolumeView {
                    kind: spec.kind,
                    drawn: match spec.reduction {
                        runmat_plot::field_volume::Reduction::Magnitude => "magnitude",
                        runmat_plot::field_volume::Reduction::InstantaneousAt { .. } => {
                            "value_at_phase_0"
                        }
                        runmat_plot::field_volume::Reduction::SoundPressureLevel { .. } => {
                            "sound_pressure_level"
                        }
                    },
                    error,
                }),
                // Set once every view's bounds are known.
                initial_camera: None,
                caption: presentation.caption.clone(),
                context: presentation.records(),
            },
        });
    }
    drop(mesh);
    let camera = camera::shared_camera(
        written.iter().map(|view| match presentation.frame {
            presentation::Frame::All => view.bounds,
            presentation::Frame::Result => view.result_bounds,
        }),
        written.iter().map(|view| view.bounds),
        presentation.frame_padding,
        presentation
            .view_angle
            .map(|(azimuth, elevation)| camera::view_direction(azimuth, elevation)),
        presentation.aspect_ratio(),
    );
    let mut result = Vec::with_capacity(written.len());
    for Written { mut view, .. } in written {
        // Render the persisted representation: the poster and live view must share
        // the same geometry, field colours, edges, and legend after cold decoding.
        let bytes = std::fs::read(output.join(&view.scene))?;
        let handle = plotting::import_geometry_scene_payload(&bytes)?
            .ok_or_else(|| anyhow::anyhow!("snapshot scene did not contain geometry"))?;
        drop(bytes);
        let poster = plotting::render_geometry_scene_snapshot_with_camera_state_scaled(
            handle,
            presentation.poster_width,
            presentation.poster_height,
            camera.clone(),
            presentation.pixels_per_point(),
            Some(presentation.theme_config()),
        )
        .await;
        plotting::close_geometry_scene(handle);
        std::fs::write(output.join(&view.poster), poster?)?;
        view.initial_camera = Some(camera.clone());
        result.push(view);
    }
    log::info!(
        "snapshot posters drawn; peak resident {}",
        peak_resident_memory()
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{lighter, Detail};
    use runmat_runtime::analysis::SurfaceDetail;

    const FULL: Detail = Detail {
        surface: SurfaceDetail::Full,
        cap_triangle_budget: 600_000,
    };

    #[test]
    fn a_surface_view_drops_to_corner_triangles_once() {
        let corners = lighter(FULL, false, 30, 16, None).unwrap();
        assert_eq!(corners.surface, SurfaceDetail::Corners);
        assert!(lighter(corners, false, 20, 16, None).is_none());
    }

    #[test]
    fn a_section_view_scales_its_cap_to_the_budget_until_one_polygon_per_element() {
        // The six-bolt busbar section: 29.7 MB against the 16 MiB app limit.
        let next = lighter(FULL, true, 29_675_503, 16_777_216, Some((500_000, 2))).unwrap();
        assert!(next.cap_triangle_budget < 500_000 * 16_777_216 / 29_675_503);
        // A cap already one polygon per cut element is the lightest section.
        assert!(lighter(next, true, 20_000_000, 16_777_216, Some((120_000, 1))).is_none());
        // The budget always shrinks, even when the payload is barely over.
        let barely = lighter(FULL, true, 101, 100, Some((1_000, 4))).unwrap();
        assert!(barely.cap_triangle_budget < 1_000);
    }
}
