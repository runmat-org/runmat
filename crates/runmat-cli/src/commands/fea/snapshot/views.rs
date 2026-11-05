use anyhow::Result;
use serde::Serialize;
use std::path::Path;
mod camera;
mod presentation;
mod profiles;
pub(super) use presentation::Presentation;
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
    position_error_m: f64,
    bounds: runmat_plot::core::BoundingBox,
    result_bounds: runmat_plot::core::BoundingBox,
    bytes: Vec<u8>,
    deformation_scale: f64,
    scalar_scale: f64,
    surface: Option<runmat_runtime::analysis::field_scene::SurfaceApproximation>,
    volume: Option<runmat_runtime::analysis::field_scene::volume::VolumeViewError>,
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
        let encode = |detail| -> Result<Encoded> {
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
                            detail,
                        )
                        .map_err(anyhow::Error::msg)?;
                    (scene, deformation_scale, Some(surface), None)
                }
            };
            let result_bounds = scene.bounds;
            presentation.dress(&mut scene);
            let mut scalar_scale = 1.0;
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
            }
            let legend = scene
                .overlay
                .as_ref()
                .and_then(|overlay| overlay.scalar_legend.as_ref());
            Ok(Encoded {
                minimum: legend.map(|legend| legend.minimum),
                maximum: legend.map(|legend| legend.maximum),
                position_error_m: runmat_plot::event::FigureScene::geometry_position_error_m(
                    &scene,
                ),
                bounds: scene.bounds,
                result_bounds,
                bytes: plotting::encode_geometry_figure_scene(
                    &runmat_plot::event::FigureScene::from_geometry_scene(&scene),
                )?,
                deformation_scale,
                scalar_scale,
                surface,
                volume,
            })
        };
        // A dense quadratic surface can exceed the scene payload limit; it is then drawn with one
        // triangle per quadratic face, and the view states what that leaves out. Values, legend
        // and extrema stay those of the full solution.
        use runmat_runtime::analysis::SurfaceDetail;
        let Encoded {
            minimum,
            maximum,
            position_error_m,
            bounds,
            result_bounds,
            bytes,
            deformation_scale,
            scalar_scale,
            surface,
            volume,
        } = match encode(SurfaceDetail::Full) {
            Ok(encoded) => encoded,
            Err(error) if error.to_string().contains("exceeding limit") => {
                log::info!(
                    "snapshot view '{}': {error}; drawing corner triangles",
                    spec.id
                );
                encode(SurfaceDetail::Corners)?
            }
            Err(error) => return Err(error),
        };
        let id = spec.id;
        let scene_path = format!("{id}.scene.json");
        std::fs::write(output.join(&scene_path), bytes)?;
        log::info!(
            "snapshot view {}/{view_count} '{id}' written; peak resident {} MiB",
            position + 1,
            peak_resident_mib()
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
        )
        .await;
        plotting::close_geometry_scene(handle);
        std::fs::write(output.join(&view.poster), poster?)?;
        view.initial_camera = Some(camera.clone());
        result.push(view);
    }
    log::info!(
        "snapshot posters drawn; peak resident {} MiB",
        peak_resident_mib()
    );
    Ok(result)
}

/// This process's peak resident memory so far, for the capture's phase log.
pub(super) fn peak_resident_mib() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes one rusage for this process into the provided buffer.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return 0;
    }
    // SAFETY: getrusage returned success, so it initialized the buffer.
    let maximum = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).unwrap_or(0);
    // macOS reports bytes; Linux reports KiB.
    if cfg!(target_os = "macos") {
        maximum / (1024 * 1024)
    } else {
        maximum / 1024
    }
}
