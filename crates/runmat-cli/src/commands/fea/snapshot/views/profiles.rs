//! Capture plans chosen from the captured run's published field descriptors (quantity,
//! location, components, sample and unit), never from study, file or physics names. Every
//! physics family and every step of a multi-step study goes through this one mechanism.
use anyhow::{anyhow, bail, Result};
use runmat_analysis_core::{
    ComplexRepresentation as R, FieldLocation, FieldOrdinal, ObservationQuantity as Q,
    ObservationValueShape,
};
use runmat_analysis_fea::results::field::RecoveredFieldSample;
use runmat_plot::field_volume::Reduction;
use runmat_runtime::analysis::field_scene::volume::VolumeViewKind;
use runmat_runtime::analysis::{
    analysis_field_page_by_run_id_op, AnalysisFieldDescriptor, AnalysisFieldLocation,
    AnalysisFieldPageValues, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisResultsData,
};
use runmat_runtime::operations::OperationContext;

/// Field views per figure, besides the solver mesh and mode shapes.
const MAXIMUM_FIELD_VIEWS: usize = 5;
/// Mode-shape views per figure.
const MAXIMUM_MODE_VIEWS: usize = 3;

pub(crate) struct CaptureProfile {
    pub(super) views: Vec<ViewSpec>,
}

pub(super) struct ViewSpec {
    /// The run whose fields this view shows.
    pub run_id: String,
    pub id: String,
    /// The view's title, including its sample (mode frequency, harmonic frequency).
    pub label: String,
    /// The color legend's label.
    pub legend: String,
    pub field: Option<AnalysisFieldSelection>,
    /// The nodal vector field that displaces the shape.
    pub deformation: Option<AnalysisFieldSelection>,
    pub unit: String,
    pub shape: &'static str,
    /// Mode shapes: the legend shows amplitude relative to the largest.
    pub relative: bool,
    /// A view inside the part (section, streamlines, arrows) instead of its boundary.
    pub volume: Option<VolumeSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VolumeSpec {
    pub kind: VolumeViewKind,
    pub reduction: Reduction,
    /// The legend is the field's true extrema (a magnitude or a real value); otherwise the
    /// range of the values drawn (a level or an instantaneous phasor value).
    pub true_extrema: bool,
}

/// One displayed physical quantity: the published typed quantities that carry it, in order
/// of preference, the complex representations it draws, and how it is drawn.
struct Display {
    id: &'static str,
    legend: &'static str,
    quantities: &'static [Q],
    /// Drawn representations. A complex (`Complex`) field is drawn as its magnitude.
    representations: &'static [R],
    /// Draw on the deformed shape when the run publishes a displacement.
    deformed: bool,
}

/// Real-valued quantities.
const REAL: &[R] = &[R::Real];
/// Amplitude of a phasor: a published magnitude or the complex field itself.
const AMPLITUDE: &[R] = &[R::Magnitude, R::Complex];

/// The engineering figure's quantities, in priority order. A run shows the first
/// `MAXIMUM_FIELD_VIEWS` that it publishes.
const DISPLAYS: &[Display] = &[
    Display {
        id: "temperature",
        legend: "Temperature",
        quantities: &[Q::Temperature],
        representations: REAL,
        deformed: true,
    },
    Display {
        id: "displacement",
        legend: "Displacement magnitude",
        quantities: &[Q::Displacement],
        representations: AMPLITUDE_OR_REAL,
        deformed: true,
    },
    Display {
        id: "stress",
        legend: "Von Mises stress",
        quantities: &[Q::VonMisesStress],
        representations: REAL,
        deformed: true,
    },
    Display {
        id: "sound-pressure-level",
        legend: "Sound pressure level",
        quantities: &[Q::AcousticPressure],
        representations: &[R::SoundPressureLevel],
        deformed: false,
    },
    Display {
        id: "acoustic-pressure",
        legend: "Acoustic pressure amplitude |p|",
        quantities: &[Q::AcousticPressure],
        representations: AMPLITUDE,
        deformed: false,
    },
    Display {
        id: "velocity",
        legend: "Velocity magnitude",
        quantities: &[Q::FluidVelocity, Q::Velocity],
        representations: REAL,
        deformed: true,
    },
    Display {
        id: "pressure",
        legend: "Pressure",
        quantities: &[Q::FluidPressure],
        representations: REAL,
        deformed: true,
    },
    Display {
        id: "electric-potential",
        legend: "Electric potential",
        quantities: &[Q::ElectricPotential],
        representations: AMPLITUDE_OR_REAL,
        deformed: false,
    },
    Display {
        id: "current-density",
        legend: "Current density |J|",
        quantities: &[Q::ElectricCurrentDensity],
        representations: AMPLITUDE_OR_REAL,
        deformed: false,
    },
    Display {
        id: "magnetic-flux-density",
        legend: "Magnetic flux density |B|",
        quantities: &[Q::MagneticFluxDensity],
        representations: AMPLITUDE_OR_REAL,
        deformed: false,
    },
    Display {
        id: "joule-loss",
        legend: "Joule loss density",
        quantities: &[Q::JoulePowerDensity],
        representations: REAL,
        deformed: false,
    },
    Display {
        id: "electric-field",
        legend: "Electric field |E|",
        quantities: &[Q::ElectricField],
        representations: AMPLITUDE_OR_REAL,
        deformed: false,
    },
    Display {
        id: "heat-flux",
        legend: "Heat-flux magnitude",
        quantities: &[Q::HeatFlux],
        representations: REAL,
        deformed: true,
    },
];

/// Volume views per figure.
const MAXIMUM_VOLUME_VIEWS: usize = 4;

/// A view inside the part: a quantity, how it is drawn, and what is drawn of it.
struct VolumeDisplay {
    id: &'static str,
    legend: &'static str,
    quantities: &'static [Q],
    /// The drawn representation: a real field, or a complex phasor (`Complex`). Published
    /// magnitudes, levels and phases are not interpolated; they are recomputed from the
    /// interpolated phasor.
    representation: R,
    vector: bool,
    kind: VolumeViewKind,
    reduction: Reduction,
    true_extrema: bool,
}

const SIGNED: Reduction = Reduction::InstantaneousAt { phase_radians: 0.0 };

/// Interior views, in priority order: the first `MAXIMUM_VOLUME_VIEWS` a run publishes.
const VOLUME_DISPLAYS: &[VolumeDisplay] = &[
    VolumeDisplay {
        id: "velocity-section",
        legend: "Velocity magnitude",
        quantities: &[Q::FluidVelocity],
        representation: R::Real,
        vector: true,
        kind: VolumeViewKind::Section,
        reduction: Reduction::Magnitude,
        true_extrema: true,
    },
    VolumeDisplay {
        id: "velocity-streamlines",
        legend: "Velocity magnitude",
        quantities: &[Q::FluidVelocity],
        representation: R::Real,
        vector: true,
        kind: VolumeViewKind::Streamlines,
        reduction: Reduction::Magnitude,
        true_extrema: true,
    },
    VolumeDisplay {
        id: "current-density-section",
        legend: "Current density |J|",
        quantities: &[Q::ElectricCurrentDensity],
        representation: R::Real,
        vector: true,
        kind: VolumeViewKind::Section,
        reduction: Reduction::Magnitude,
        true_extrema: true,
    },
    VolumeDisplay {
        id: "current-density-section",
        legend: "Current density amplitude |J|",
        quantities: &[Q::ElectricCurrentDensity],
        representation: R::Complex,
        vector: true,
        kind: VolumeViewKind::Section,
        reduction: Reduction::Magnitude,
        true_extrema: true,
    },
    VolumeDisplay {
        id: "electric-potential-section",
        legend: "Electric potential",
        quantities: &[Q::ElectricPotential],
        representation: R::Real,
        vector: false,
        kind: VolumeViewKind::Section,
        reduction: SIGNED,
        true_extrema: false,
    },
    VolumeDisplay {
        id: "electric-field-arrows",
        legend: "Electric field |E|",
        quantities: &[Q::ElectricField],
        representation: R::Real,
        vector: true,
        kind: VolumeViewKind::Glyphs,
        reduction: Reduction::Magnitude,
        true_extrema: false,
    },
    VolumeDisplay {
        id: "magnetic-flux-density-section",
        legend: "Magnetic flux density |B|",
        quantities: &[Q::MagneticFluxDensity],
        representation: R::Real,
        vector: true,
        kind: VolumeViewKind::Section,
        reduction: Reduction::Magnitude,
        true_extrema: true,
    },
    VolumeDisplay {
        id: "magnetic-flux-density-section",
        legend: "Magnetic flux density amplitude |B|",
        quantities: &[Q::MagneticFluxDensity],
        representation: R::Complex,
        vector: true,
        kind: VolumeViewKind::Section,
        reduction: Reduction::Magnitude,
        true_extrema: true,
    },
    VolumeDisplay {
        id: "sound-pressure-level-section",
        legend: "Sound pressure level",
        quantities: &[Q::AcousticPressure],
        representation: R::Complex,
        vector: false,
        kind: VolumeViewKind::Section,
        reduction: Reduction::SoundPressureLevel {
            reference_pa: runmat_analysis_core::SOUND_PRESSURE_LEVEL_REFERENCE_PA,
        },
        true_extrema: false,
    },
    VolumeDisplay {
        id: "acoustic-pressure-section",
        legend: "Acoustic pressure amplitude |p|",
        quantities: &[Q::AcousticPressure],
        representation: R::Complex,
        vector: false,
        kind: VolumeViewKind::Section,
        reduction: Reduction::Magnitude,
        true_extrema: true,
    },
    VolumeDisplay {
        id: "acoustic-pressure-instantaneous-section",
        legend: "Acoustic pressure at phase 0",
        quantities: &[Q::AcousticPressure],
        representation: R::Complex,
        vector: false,
        kind: VolumeViewKind::Section,
        reduction: SIGNED,
        true_extrema: false,
    },
    VolumeDisplay {
        id: "temperature-section",
        legend: "Temperature",
        quantities: &[Q::Temperature],
        representation: R::Real,
        vector: false,
        kind: VolumeViewKind::Section,
        reduction: SIGNED,
        true_extrema: true,
    },
];

/// A real field, or the amplitude of a harmonic one.
const AMPLITUDE_OR_REAL: &[R] = &[R::Real, R::Magnitude, R::Complex];

/// A solved field on the mesh: values at nodes or elements of the solver mesh. Authored
/// study outputs republish solved fields under authored names with no physics family; the
/// figure shows the solved field itself.
fn drawable(descriptor: &AnalysisFieldDescriptor) -> bool {
    descriptor.family != "unknown"
        && matches!(
            descriptor.location,
            AnalysisFieldLocation::Node | AnalysisFieldLocation::Element
        )
        && descriptor.value_count > 0
}

fn is_mode_shape(descriptor: &AnalysisFieldDescriptor) -> bool {
    matches!(descriptor.layout.ordinal, FieldOrdinal::Mode(_))
}

/// The published 0-based mode index of a mode shape.
fn mode_index(descriptor: &AnalysisFieldDescriptor) -> Option<usize> {
    match descriptor.layout.ordinal {
        FieldOrdinal::Mode(index) => Some(index),
        _ => None,
    }
}

fn selection(descriptor: &AnalysisFieldDescriptor) -> AnalysisFieldSelection {
    AnalysisFieldSelection::new(&descriptor.field_id, descriptor.sample)
}

/// The latest state among descriptors of one quantity: the single or final state, else the
/// latest time, load factor or parameter, else the highest published step.
fn latest<'a>(candidates: &[&'a AnalysisFieldDescriptor]) -> Option<&'a AnalysisFieldDescriptor> {
    let step = |descriptor: &AnalysisFieldDescriptor| match descriptor.layout.ordinal {
        FieldOrdinal::Step(index) => index,
        _ => 0,
    };
    let rank = |descriptor: &AnalysisFieldDescriptor| match descriptor.sample {
        None | Some(RecoveredFieldSample::Final) => (2, 0.0, step(descriptor)),
        Some(RecoveredFieldSample::TimeSeconds(time))
        | Some(RecoveredFieldSample::LoadFactor(time))
        | Some(RecoveredFieldSample::StudyParameterValue(time)) => (1, time, 0),
        _ => (0, 0.0, step(descriptor)),
    };
    candidates.iter().copied().max_by(|left, right| {
        let (left, right) = (rank(left), rank(right));
        left.0
            .cmp(&right.0)
            .then(left.1.total_cmp(&right.1))
            .then(left.2.cmp(&right.2))
    })
}

fn frequency_label(hertz: f64) -> String {
    if hertz != 0.0 && !(0.01..100_000.0).contains(&hertz) {
        format!("{hertz:.3e} Hz")
    } else {
        format!("{hertz:.2} Hz")
    }
}

/// Values of a small published global or per-mode field, read from its first page.
fn small_field_values(run_id: &str, descriptor: &AnalysisFieldDescriptor) -> Result<Vec<f64>> {
    let page = analysis_field_page_by_run_id_op(
        run_id,
        selection(descriptor),
        AnalysisFieldRequestOptions {
            offset: Some(0),
            limit: Some(descriptor.value_count.min(4096)),
        },
        OperationContext::new(None, None),
    )
    .map_err(|error| anyhow!("{}: {}", error.error_code, error.message))?
    .data;
    Ok(match page.values {
        AnalysisFieldPageValues::Real(values) => values,
        AnalysisFieldPageValues::ComplexInterleaved(values) => values
            .chunks_exact(2)
            .map(|pair| pair[0].hypot(pair[1]))
            .collect(),
    })
}

/// The frequency of a published mode, from the run's published frequency list.
fn mode_frequency(
    run_id: &str,
    published: &[AnalysisFieldDescriptor],
    mode: &AnalysisFieldDescriptor,
    position: usize,
) -> Result<Option<f64>> {
    let Some(list) = published.iter().find(|descriptor| {
        descriptor.layout.quantity == Q::ModalFrequency
            && matches!(
                descriptor.location,
                AnalysisFieldLocation::Mode | AnalysisFieldLocation::Global
            )
    }) else {
        return Ok(None);
    };
    let index = mode_index(mode).unwrap_or(position);
    let frequencies = small_field_values(run_id, list)?;
    let frequency = frequencies
        .get(index)
        .copied()
        .ok_or_else(|| anyhow!("mode {} has no published frequency", index + 1))?;
    anyhow::ensure!(
        frequency.is_finite() && frequency >= 0.0,
        "mode {} has an invalid natural frequency",
        index + 1
    );
    Ok(Some(frequency))
}

/// One descriptor of a quantity to show. Frequency-sampled fields show the frequency with the
/// largest peak magnitude (the resonant response of a sweep); other fields their latest state.
fn chosen_sample<'a>(
    run_id: &str,
    candidates: &[&'a AnalysisFieldDescriptor],
) -> Result<Option<(&'a AnalysisFieldDescriptor, Option<f64>)>> {
    let frequencies: Vec<(&AnalysisFieldDescriptor, f64)> = candidates
        .iter()
        .filter_map(|descriptor| match descriptor.sample {
            Some(RecoveredFieldSample::FrequencyHertz(hertz)) => Some((*descriptor, hertz)),
            _ => None,
        })
        .collect();
    match frequencies.as_slice() {
        [] => return Ok(latest(candidates).map(|descriptor| (descriptor, None))),
        [(descriptor, hertz)] => return Ok(Some((descriptor, Some(*hertz)))),
        _ => {}
    }
    let mut best: Option<(&AnalysisFieldDescriptor, f64, f64)> = None;
    for (descriptor, hertz) in frequencies {
        let peak = runmat_runtime::analysis::field_scene::extrema::field_extrema(
            run_id,
            &selection(descriptor),
            None,
            None,
        )
        .map_err(anyhow::Error::msg)?
        .maximum
        .value;
        if best.is_none_or(|(_, _, value)| peak > value) {
            best = Some((descriptor, hertz, peak));
        }
    }
    Ok(best.map(|(descriptor, hertz, _)| (descriptor, Some(hertz))))
}

fn inventory(published: &[AnalysisFieldDescriptor]) -> String {
    published
        .iter()
        .map(|descriptor| {
            format!(
                "{} ({:?} {:?}, {:?}, {} components, {})",
                descriptor.field_id,
                descriptor.layout.quantity,
                descriptor.layout.representation,
                descriptor.location,
                descriptor.component_count.unwrap_or(1),
                descriptor.unit.as_deref().unwrap_or("no unit")
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// One captured run's drawable fields.
struct RunFields<'a> {
    run_id: &'a str,
    /// Every field the run published.
    published: &'a [AnalysisFieldDescriptor],
    descriptors: Vec<&'a AnalysisFieldDescriptor>,
    deformation: Option<AnalysisFieldSelection>,
    /// A modal run: its displacement, strain and stress are those of mass-normalized
    /// eigenvectors, which carry no physical amplitude. It shows only its mode shapes.
    modal: bool,
}

impl<'a> RunFields<'a> {
    fn new(run_id: &'a str, published: &'a [AnalysisFieldDescriptor]) -> Self {
        let mut run = Self {
            run_id,
            published,
            descriptors: published
                .iter()
                .filter(|descriptor| drawable(descriptor))
                .collect(),
            deformation: None,
            modal: false,
        };
        run.modal = !run.modes().is_empty();
        let nodal: Vec<_> = run
            .of_quantity(Q::Displacement, REAL)
            .into_iter()
            .filter(|descriptor| {
                descriptor.layout.location == FieldLocation::Node
                    && descriptor.layout.value_shape == ObservationValueShape::Vector3
            })
            .collect();
        run.deformation = latest(&nodal).map(selection);
        run
    }

    fn of_quantity(&self, quantity: Q, representations: &[R]) -> Vec<&'a AnalysisFieldDescriptor> {
        self.descriptors
            .iter()
            .copied()
            .filter(|descriptor| {
                descriptor.layout.quantity == quantity
                    && representations.contains(&descriptor.layout.representation)
                    && !is_mode_shape(descriptor)
            })
            .collect()
    }

    /// One shape per published mode, in mode order: its displacement eigenvector, or for a
    /// scalar problem its primary variable. Strain, stress and energy recovered from a
    /// mass-normalized eigenvector carry the same mode ordinal but are not its shape.
    fn modes(&self) -> Vec<&'a AnalysisFieldDescriptor> {
        let mut by_mode = std::collections::BTreeMap::new();
        for descriptor in self.descriptors.iter().copied() {
            let Some(index) = mode_index(descriptor) else {
                continue;
            };
            if descriptor.location != AnalysisFieldLocation::Node {
                continue;
            }
            let Some(rank) = MODE_SHAPE_QUANTITIES.iter().position(|(quantity, shape)| {
                descriptor.layout.quantity == *quantity && descriptor.layout.value_shape == *shape
            }) else {
                continue;
            };
            // The run-level copy of mode 1 and its mode-shape field hold the same values; the
            // field id breaks the tie so the choice is stable.
            let key = (rank, descriptor.field_id.as_str());
            by_mode
                .entry(index)
                .and_modify(|chosen: &mut (_, &'a AnalysisFieldDescriptor)| {
                    if key < chosen.0 {
                        *chosen = (key, descriptor);
                    }
                })
                .or_insert((key, descriptor));
        }
        by_mode
            .into_values()
            .map(|(_, descriptor)| descriptor)
            .collect()
    }
}

/// The fields that are a mode's shape, in order of preference.
const MODE_SHAPE_QUANTITIES: &[(Q, ObservationValueShape)] = &[
    (Q::Displacement, ObservationValueShape::Vector3),
    (Q::AcousticPressure, ObservationValueShape::Scalar),
];

impl CaptureProfile {
    /// The views for the captured runs (one per captured case of the study), chosen from their
    /// published field descriptors. Each quantity comes from the first run that publishes it.
    pub(crate) fn from_runs(runs: &[(&str, &AnalysisResultsData)]) -> Result<Self> {
        let published: Vec<(&str, &[AnalysisFieldDescriptor])> = runs
            .iter()
            .map(|(run_id, results)| (*run_id, results.field_descriptors.as_slice()))
            .collect();
        Self::from_published(&published)
    }

    /// Keep only the named views (all of them when none is named), in capture order. An
    /// unknown name is refused with the views the runs offer.
    pub(crate) fn select(mut self, names: &[String]) -> Result<Self> {
        if names.is_empty() {
            return Ok(self);
        }
        let offered = || {
            self.views
                .iter()
                .map(|view| view.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        for name in names {
            if !self.views.iter().any(|view| &view.id == name) {
                bail!("snapshot has no view '{name}'; views: {}", offered());
            }
        }
        self.views.retain(|view| names.contains(&view.id));
        Ok(self)
    }

    /// The views for runs given by their published field descriptors.
    fn from_published(runs: &[(&str, &[AnalysisFieldDescriptor])]) -> Result<Self> {
        let runs: Vec<RunFields> = runs
            .iter()
            .map(|(run_id, published)| RunFields::new(run_id, published))
            .collect();
        let primary = runs.first().ok_or_else(|| anyhow!("snapshot has no run"))?;
        let mut views = vec![ViewSpec {
            run_id: primary.run_id.into(),
            id: "geometry".into(),
            label: "Solver mesh".into(),
            legend: "Solver mesh".into(),
            field: None,
            deformation: None,
            unit: "m".into(),
            shape: "undeformed",
            relative: false,
            volume: None,
        }];
        for display in DISPLAYS {
            if views.len() > MAXIMUM_FIELD_VIEWS {
                break;
            }
            let mut chosen = None;
            'runs: for run in runs.iter().filter(|run| !run.modal) {
                for quantity in display.quantities {
                    let candidates = run.of_quantity(*quantity, display.representations);
                    if let Some(found) = chosen_sample(run.run_id, &candidates)? {
                        chosen = Some((run, found));
                        break 'runs;
                    }
                }
            }
            let Some((run, (descriptor, frequency))) = chosen else {
                continue;
            };
            let unit = descriptor
                .unit
                .clone()
                .filter(|unit| !unit.is_empty())
                .ok_or_else(|| anyhow!("published field '{}' has no unit", descriptor.field_id))?;
            let deformation = run.deformation.clone().filter(|_| display.deformed);
            views.push(ViewSpec {
                run_id: run.run_id.into(),
                id: display.id.into(),
                label: match frequency {
                    Some(hertz) => format!("{} · {}", display.legend, frequency_label(hertz)),
                    None => display.legend.to_string(),
                },
                legend: display.legend.into(),
                shape: if deformation.is_some() {
                    "exaggerated"
                } else {
                    "undeformed"
                },
                deformation,
                field: Some(selection(descriptor)),
                unit,
                relative: false,
                volume: None,
            });
        }
        let mut volume_views = 0;
        for display in VOLUME_DISPLAYS {
            if volume_views == MAXIMUM_VOLUME_VIEWS {
                break;
            }
            if views.iter().any(|view| view.id == display.id) {
                continue;
            }
            let mut chosen = None;
            'volume: for run in runs.iter().filter(|run| !run.modal) {
                for quantity in display.quantities {
                    let candidates: Vec<_> = run
                        .of_quantity(*quantity, &[display.representation])
                        .into_iter()
                        .filter(|descriptor| {
                            descriptor.layout.value_shape
                                == if display.vector {
                                    ObservationValueShape::Vector3
                                } else {
                                    ObservationValueShape::Scalar
                                }
                        })
                        .collect();
                    if let Some(found) = chosen_sample(run.run_id, &candidates)? {
                        chosen = Some((run, found));
                        break 'volume;
                    }
                }
            }
            let Some((run, (descriptor, frequency))) = chosen else {
                continue;
            };
            let unit = match display.reduction {
                Reduction::SoundPressureLevel { .. } => "dB".to_string(),
                _ => descriptor
                    .unit
                    .clone()
                    .filter(|unit| !unit.is_empty())
                    .ok_or_else(|| {
                        anyhow!("published field '{}' has no unit", descriptor.field_id)
                    })?,
            };
            views.push(ViewSpec {
                run_id: run.run_id.into(),
                id: display.id.into(),
                label: match frequency {
                    Some(hertz) => format!("{} · {}", display.legend, frequency_label(hertz)),
                    None => display.legend.to_string(),
                },
                legend: display.legend.into(),
                field: Some(selection(descriptor)),
                deformation: None,
                unit,
                shape: "undeformed",
                relative: false,
                volume: Some(VolumeSpec {
                    kind: display.kind,
                    reduction: display.reduction,
                    true_extrema: display.true_extrema,
                }),
            });
            volume_views += 1;
        }
        let (mode_run, modes) = runs
            .iter()
            .map(|run| (run, run.modes()))
            .find(|(_, modes)| !modes.is_empty())
            .unwrap_or((primary, Vec::new()));
        for (position, mode) in modes.into_iter().take(MAXIMUM_MODE_VIEWS).enumerate() {
            let number = position + 1;
            let legend = format!("Mode {number} relative amplitude");
            let label = match mode_frequency(mode_run.run_id, mode_run.published, mode, position)? {
                Some(hertz) => format!("{legend} · {}", frequency_label(hertz)),
                None => legend.clone(),
            };
            let field = selection(mode);
            // A vector mode shape displaces the body; a scalar mode (acoustic pressure) is
            // drawn on the undeformed body.
            let deforms = mode.layout.quantity == Q::Displacement
                && mode.layout.value_shape == ObservationValueShape::Vector3;
            views.push(ViewSpec {
                run_id: mode_run.run_id.into(),
                id: format!("mode-{number}"),
                label,
                legend,
                deformation: deforms.then(|| field.clone()),
                field: Some(field),
                unit: "relative".into(),
                shape: if deforms { "exaggerated" } else { "undeformed" },
                relative: true,
                volume: None,
            });
        }
        if views.len() == 1 {
            bail!(
                "captured runs publish no displayable field; fields: {}",
                runs.iter()
                    .map(|run| inventory(run.published))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        Ok(Self { views })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::{AnalysisField, AnalysisFieldLayout};

    fn published(
        id: &str,
        layout: AnalysisFieldLayout,
        shape: Vec<usize>,
    ) -> AnalysisFieldDescriptor {
        let count = shape.iter().product();
        let mut descriptor = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
            id,
            layout,
            shape,
            vec![1.0; count],
        ));
        // Units come from the publishing family; the test descriptors carry one.
        descriptor.unit.get_or_insert_with(|| "1".into());
        descriptor
    }

    fn ids(profile: &CaptureProfile) -> Vec<&str> {
        profile.views.iter().map(|view| view.id.as_str()).collect()
    }

    #[test]
    fn element_temperature_is_drawn_without_nodal_averaging() {
        let fields = [
            published(
                "thermo_mechanical.temperature.0",
                AnalysisFieldLayout::scalar(Q::Temperature, FieldLocation::Element).step(0),
                vec![5],
            ),
            published(
                "thermo_mechanical.displacement.0",
                AnalysisFieldLayout::vector3(Q::Displacement, FieldLocation::Node).step(0),
                vec![4, 3],
            ),
        ];
        let profile = CaptureProfile::from_published(&[("run", &fields)]).unwrap();
        assert_eq!(
            ids(&profile),
            [
                "geometry",
                "temperature",
                "displacement",
                "temperature-section"
            ]
        );
        // The section draws the element values themselves.
        let section = &profile.views[3];
        assert_eq!(section.field, profile.views[1].field);
        assert_eq!(section.volume.unwrap().kind, VolumeViewKind::Section);
        let temperature = &profile.views[1];
        assert_eq!(
            temperature.field.as_ref().unwrap().field_id,
            "thermo_mechanical.temperature.0"
        );
        assert_eq!(
            temperature.deformation.as_ref().unwrap().field_id,
            "thermo_mechanical.displacement.0"
        );
    }

    #[test]
    fn integration_point_and_edge_fields_are_not_drawn_as_mesh_colors() {
        let fields = [
            published(
                "thermo_mechanical.temperature.0",
                AnalysisFieldLayout::scalar(Q::Temperature, FieldLocation::IntegrationPoint),
                vec![8],
            ),
            published(
                "electro_thermal.current_density",
                AnalysisFieldLayout::vector3(Q::ElectricCurrentDensity, FieldLocation::Edge),
                vec![6, 3],
            ),
        ];
        assert!(CaptureProfile::from_published(&[("run", &fields)]).is_err());
    }

    #[test]
    fn scalar_mode_shapes_are_relative_undeformed_views_in_mode_order() {
        let fields: Vec<_> = [2, 0, 1]
            .into_iter()
            .map(|mode| {
                published(
                    &format!("acoustic.mode_shape.{mode}"),
                    AnalysisFieldLayout::scalar(Q::AcousticPressure, FieldLocation::Node)
                        .mode(mode),
                    vec![4],
                )
            })
            .collect();
        let profile = CaptureProfile::from_published(&[("run", &fields)]).unwrap();
        assert_eq!(ids(&profile), ["geometry", "mode-1", "mode-2", "mode-3"]);
        for (view, mode) in profile.views[1..].iter().zip(0..) {
            assert_eq!(
                view.field.as_ref().unwrap().field_id,
                format!("acoustic.mode_shape.{mode}")
            );
            assert!(view.relative);
            assert!(view.deformation.is_none());
            assert_eq!(view.shape, "undeformed");
        }
    }

    #[test]
    fn modal_runs_show_geometry_and_one_shape_per_mode_only() {
        let node_vector =
            |mode| AnalysisFieldLayout::vector3(Q::Displacement, FieldLocation::Node).mode(mode);
        let fields = [
            // The run-level copy of mode 1 and the stress recovered from it carry mode 1's
            // ordinal; a stress without the ordinal is still of a mass-normalized shape.
            published("structural.displacement", node_vector(0), vec![4, 3]),
            published(
                "structural.nodal_von_mises",
                AnalysisFieldLayout::scalar(Q::VonMisesStress, FieldLocation::Node).mode(0),
                vec![4],
            ),
            published(
                "structural.von_mises",
                AnalysisFieldLayout::scalar(Q::VonMisesStress, FieldLocation::Element),
                vec![2],
            ),
            published("modal.mode_shape.1", node_vector(0), vec![4, 3]),
            published("modal.mode_shape.2", node_vector(1), vec![4, 3]),
        ];
        let profile = CaptureProfile::from_published(&[("run", &fields)]).unwrap();
        assert_eq!(ids(&profile), ["geometry", "mode-1", "mode-2"]);
        let shapes: Vec<_> = profile.views[1..]
            .iter()
            .map(|view| view.field.as_ref().unwrap().field_id.as_str())
            .collect();
        assert_eq!(shapes, ["modal.mode_shape.1", "modal.mode_shape.2"]);
        for view in &profile.views[1..] {
            assert_eq!(view.shape, "exaggerated");
            assert!(view.relative);
            assert!(view.deformation.is_some());
        }
    }

    #[test]
    fn selected_views_keep_capture_order_and_unknown_names_are_refused() {
        let fields: Vec<_> = (0..3)
            .map(|mode| {
                published(
                    &format!("modal.mode_shape.{}", mode + 1),
                    AnalysisFieldLayout::vector3(Q::Displacement, FieldLocation::Node).mode(mode),
                    vec![4, 3],
                )
            })
            .collect();
        let profile = || CaptureProfile::from_published(&[("run", &fields)]).unwrap();
        let names = |names: &[&str]| {
            names
                .iter()
                .map(|name| name.to_string())
                .collect::<Vec<_>>()
        };
        let selected = profile().select(&names(&["mode-2", "geometry"])).unwrap();
        assert_eq!(ids(&selected), ["geometry", "mode-2"]);
        assert_eq!(ids(&profile().select(&[]).unwrap()).len(), 4);
        let refused = profile()
            .select(&names(&["stress"]))
            .err()
            .unwrap()
            .to_string();
        assert!(
            refused.contains("no view 'stress'") && refused.contains("mode-3"),
            "{refused}"
        );
    }

    #[test]
    fn interior_views_follow_the_published_quantity() {
        let velocity = [published(
            "cfd.velocity",
            AnalysisFieldLayout::vector3(Q::FluidVelocity, FieldLocation::Element),
            vec![4, 3],
        )];
        let profile = CaptureProfile::from_published(&[("run", &velocity)]).unwrap();
        let kinds: Vec<_> = profile
            .views
            .iter()
            .filter_map(|view| view.volume.map(|volume| (view.id.as_str(), volume.kind)))
            .collect();
        assert_eq!(
            kinds,
            [
                ("velocity-section", VolumeViewKind::Section),
                ("velocity-streamlines", VolumeViewKind::Streamlines)
            ]
        );
        // A pressure phasor: its amplitude, level and instantaneous value are recomputed from
        // the interpolated phasor; published magnitudes are not interpolated.
        let pressure = [
            published(
                "acoustic.pressure",
                AnalysisFieldLayout::scalar(Q::AcousticPressure, FieldLocation::Node)
                    .representation(R::Complex),
                vec![4, 2],
            ),
            published(
                "acoustic.pressure_magnitude",
                AnalysisFieldLayout::scalar(Q::AcousticPressure, FieldLocation::Node)
                    .representation(R::Magnitude),
                vec![4],
            ),
        ];
        let profile = CaptureProfile::from_published(&[("run", &pressure)]).unwrap();
        let sections: Vec<_> = profile
            .views
            .iter()
            .filter(|view| view.volume.is_some())
            .map(|view| {
                (
                    view.id.as_str(),
                    view.field.as_ref().unwrap().field_id.as_str(),
                )
            })
            .collect();
        assert_eq!(
            sections,
            [
                ("sound-pressure-level-section", "acoustic.pressure"),
                ("acoustic-pressure-section", "acoustic.pressure"),
                (
                    "acoustic-pressure-instantaneous-section",
                    "acoustic.pressure"
                ),
            ]
        );
        let level = profile
            .views
            .iter()
            .find(|view| view.id == "sound-pressure-level-section")
            .unwrap();
        assert_eq!(level.unit, "dB");
        assert!(!level.volume.unwrap().true_extrema);
    }

    #[test]
    fn stepped_fields_show_their_latest_published_step() {
        let fields: Vec<_> = (0..3)
            .map(|step| {
                published(
                    &format!("thermal.temperature.{step}"),
                    AnalysisFieldLayout::scalar(Q::Temperature, FieldLocation::Node).step(step),
                    vec![4],
                )
            })
            .collect();
        let profile = CaptureProfile::from_published(&[("run", &fields)]).unwrap();
        assert_eq!(
            profile.views[1].field.as_ref().unwrap().field_id,
            "thermal.temperature.2"
        );
    }

    #[test]
    fn phasor_parts_choose_the_amplitude_and_level_never_real_or_phase() {
        let node = |representation| {
            AnalysisFieldLayout::scalar(Q::AcousticPressure, FieldLocation::Node)
                .representation(representation)
        };
        let fields = [
            published("acoustic.pressure_real", node(R::Real), vec![4]),
            published("acoustic.phase", node(R::PhaseRadians), vec![4]),
            published("acoustic.pressure_magnitude", node(R::Magnitude), vec![4]),
            published(
                "acoustic.sound_pressure_level_db",
                node(R::SoundPressureLevel),
                vec![4],
            ),
        ];
        let profile = CaptureProfile::from_published(&[("run", &fields)]).unwrap();
        assert_eq!(
            ids(&profile),
            ["geometry", "sound-pressure-level", "acoustic-pressure"]
        );
        assert_eq!(
            profile.views[2].field.as_ref().unwrap().field_id,
            "acoustic.pressure_magnitude"
        );
    }

    #[test]
    fn each_quantity_comes_from_the_first_run_that_publishes_it() {
        let conduction = [published(
            "electro_thermal.electric_potential",
            AnalysisFieldLayout::scalar(Q::ElectricPotential, FieldLocation::Node),
            vec![4],
        )];
        let heat = [published(
            "electro_thermal.temperature.0",
            AnalysisFieldLayout::scalar(Q::Temperature, FieldLocation::Node).step(0),
            vec![4],
        )];
        let profile =
            CaptureProfile::from_published(&[("conduction", &conduction), ("heat", &heat)])
                .unwrap();
        let by_id = |id: &str| profile.views.iter().find(|view| view.id == id).unwrap();
        assert_eq!(by_id("temperature").run_id, "heat");
        assert_eq!(by_id("electric-potential").run_id, "conduction");
    }
}
