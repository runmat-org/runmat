//! Capture plans chosen from the captured run's published field descriptors (quantity,
//! location, components, sample and unit), never from study, file or physics names. Every
//! physics family and every step of a multi-step study goes through this one mechanism.
use anyhow::{anyhow, bail, Result};
use runmat_analysis_core::{
    ComplexRepresentation as R, FieldLocation, FieldOrdinal, FieldStatistic,
    ObservationQuantity as Q, ObservationValueShape,
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
    /// The statistic the drawn field is (`None`: a state of the solution). A statistic is
    /// drawn on the undeformed body: an RMS or envelope vector is not a displacement state.
    statistic: Option<FieldStatistic>,
    /// Drawn only from time-domain runs: in a run that publishes a phasor of the quantity, a
    /// real field is the phasor's real part, never an instantaneous value.
    time_domain: bool,
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
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "displacement",
        legend: "Displacement magnitude",
        quantities: &[Q::Displacement],
        representations: AMPLITUDE_OR_REAL,
        deformed: true,
        statistic: None,
        time_domain: false,
    },
    // Random response: the root-mean-square of each displacement component; the magnitude of
    // that vector is the RMS of the displacement magnitude.
    Display {
        id: "rms-displacement",
        legend: "RMS displacement magnitude",
        quantities: &[Q::Displacement],
        representations: REAL,
        deformed: false,
        statistic: Some(FieldStatistic::RootMeanSquare),
        time_domain: false,
    },
    // Response spectrum: the combined modal peak of each displacement component.
    Display {
        id: "peak-displacement",
        legend: "Peak displacement magnitude (combined modal envelope)",
        quantities: &[Q::Displacement],
        representations: REAL,
        deformed: false,
        statistic: Some(FieldStatistic::ComponentMagnitudeEnvelope),
        time_domain: false,
    },
    Display {
        id: "stress",
        legend: "Von Mises stress",
        quantities: &[Q::VonMisesStress],
        representations: REAL,
        deformed: true,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "sound-pressure-level",
        legend: "Sound pressure level",
        quantities: &[Q::AcousticPressure],
        representations: &[R::SoundPressureLevel],
        deformed: false,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "acoustic-pressure",
        legend: "Acoustic pressure amplitude |p|",
        quantities: &[Q::AcousticPressure],
        representations: AMPLITUDE,
        deformed: false,
        statistic: None,
        time_domain: false,
    },
    // Acoustic transient: the real pressure of one instant.
    Display {
        id: "sound-pressure",
        legend: "Acoustic pressure",
        quantities: &[Q::AcousticPressure],
        representations: REAL,
        deformed: false,
        statistic: None,
        time_domain: true,
    },
    Display {
        id: "velocity",
        legend: "Velocity magnitude",
        quantities: &[Q::FluidVelocity, Q::Velocity],
        representations: REAL,
        deformed: true,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "pressure",
        legend: "Pressure",
        quantities: &[Q::FluidPressure],
        representations: REAL,
        deformed: true,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "electric-potential",
        legend: "Electric potential",
        quantities: &[Q::ElectricPotential],
        representations: AMPLITUDE_OR_REAL,
        deformed: false,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "current-density",
        legend: "Current density |J|",
        quantities: &[Q::ElectricCurrentDensity],
        representations: AMPLITUDE_OR_REAL,
        deformed: false,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "magnetic-flux-density",
        legend: "Magnetic flux density |B|",
        quantities: &[Q::MagneticFluxDensity],
        representations: AMPLITUDE_OR_REAL,
        deformed: false,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "joule-loss",
        legend: "Joule loss density",
        quantities: &[Q::JoulePowerDensity],
        representations: REAL,
        deformed: false,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "electric-field",
        legend: "Electric field |E|",
        quantities: &[Q::ElectricField],
        representations: AMPLITUDE_OR_REAL,
        deformed: false,
        statistic: None,
        time_domain: false,
    },
    Display {
        id: "heat-flux",
        legend: "Heat-flux magnitude",
        quantities: &[Q::HeatFlux],
        representations: REAL,
        deformed: true,
        statistic: None,
        time_domain: false,
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

/// What numbers a mode: a natural frequency (modal extraction) or a buckling load factor.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ModeEigenvalue {
    FrequencyHertz(f64),
    LoadFactor(f64),
}

/// A published mode's eigenvalue, from the run's published per-mode list: its natural
/// frequency, or for a buckling step its load factor.
fn mode_eigenvalue(
    run_id: &str,
    published: &[AnalysisFieldDescriptor],
    mode: &AnalysisFieldDescriptor,
    position: usize,
    values: &dyn PublishedValues,
) -> Result<Option<ModeEigenvalue>> {
    let per_mode = |quantity: Q| {
        published.iter().find(|descriptor| {
            descriptor.layout.quantity == quantity
                && matches!(
                    descriptor.location,
                    AnalysisFieldLocation::Mode | AnalysisFieldLocation::Global
                )
        })
    };
    let (list, eigenvalue): (_, fn(f64) -> ModeEigenvalue) =
        if let Some(list) = per_mode(Q::ModalFrequency) {
            (list, ModeEigenvalue::FrequencyHertz)
        } else if let Some(list) = per_mode(Q::LoadFactor) {
            (list, ModeEigenvalue::LoadFactor)
        } else {
            return Ok(None);
        };
    let index = mode_index(mode).unwrap_or(position);
    let value = values
        .small(run_id, list)?
        .get(index)
        .copied()
        .ok_or_else(|| anyhow!("mode {} has no published eigenvalue", index + 1))?;
    anyhow::ensure!(
        value.is_finite(),
        "mode {} has an invalid eigenvalue",
        index + 1
    );
    if let ModeEigenvalue::FrequencyHertz(_) = eigenvalue(value) {
        anyhow::ensure!(
            value >= 0.0,
            "mode {} has an invalid natural frequency",
            index + 1
        );
    }
    Ok(Some(eigenvalue(value)))
}

/// The published values a profile reads besides descriptors: a field's peak magnitude and
/// the entries of a small per-mode list.
pub(super) trait PublishedValues {
    fn peak(&self, run_id: &str, descriptor: &AnalysisFieldDescriptor) -> Result<f64>;
    fn small(&self, run_id: &str, descriptor: &AnalysisFieldDescriptor) -> Result<Vec<f64>>;
    /// The time of the retained transient state at step `index`, when the run retains one.
    fn state_time(
        &self,
        run_id: &str,
        descriptor: &AnalysisFieldDescriptor,
        index: usize,
    ) -> Result<Option<f64>>;
}

/// The run's stored original fields, read page by page.
pub(super) struct StoredValues;

impl PublishedValues for StoredValues {
    fn peak(&self, run_id: &str, descriptor: &AnalysisFieldDescriptor) -> Result<f64> {
        Ok(
            runmat_runtime::analysis::field_scene::extrema::field_extrema(
                run_id,
                &selection(descriptor),
                None,
                None,
            )
            .map_err(anyhow::Error::msg)?
            .maximum
            .value,
        )
    }

    fn small(&self, run_id: &str, descriptor: &AnalysisFieldDescriptor) -> Result<Vec<f64>> {
        small_field_values(run_id, descriptor)
    }

    fn state_time(
        &self,
        run_id: &str,
        descriptor: &AnalysisFieldDescriptor,
        index: usize,
    ) -> Result<Option<f64>> {
        let page = analysis_field_page_by_run_id_op(
            run_id,
            AnalysisFieldSelection::transient_index(&descriptor.field_id, index),
            AnalysisFieldRequestOptions {
                offset: Some(0),
                limit: Some(1),
            },
            OperationContext::new(None, None),
        );
        Ok(match page {
            Ok(page) => match page.data.descriptor.sample {
                Some(RecoveredFieldSample::TimeSeconds(time)) => Some(time),
                _ => None,
            },
            // A stepped field of a run that retains no transient schedule has no time.
            Err(_) => None,
        })
    }
}

/// The candidate with the largest peak magnitude.
fn peak_of<'a>(
    run_id: &str,
    candidates: impl IntoIterator<Item = &'a AnalysisFieldDescriptor>,
    values: &dyn PublishedValues,
) -> Result<Option<&'a AnalysisFieldDescriptor>> {
    let mut best: Option<(&AnalysisFieldDescriptor, f64)> = None;
    for descriptor in candidates {
        let peak = values.peak(run_id, descriptor)?;
        if best.is_none_or(|(_, value)| peak > value) {
            best = Some((descriptor, peak));
        }
    }
    Ok(best.map(|(descriptor, _)| descriptor))
}

/// The retained state a stepped field is: its published time, else its step ordinal.
#[derive(Debug, Clone, Copy, PartialEq)]
enum StateKey {
    Time(f64),
    Step(usize),
}

fn state_key(descriptor: &AnalysisFieldDescriptor) -> Option<StateKey> {
    match (descriptor.sample, descriptor.layout.ordinal) {
        (Some(RecoveredFieldSample::TimeSeconds(time)), _) => Some(StateKey::Time(time)),
        (None, FieldOrdinal::Step(index)) => Some(StateKey::Step(index)),
        _ => None,
    }
}

fn time_label(seconds: f64) -> String {
    let trimmed = |value: f64| {
        let text = format!("{value:.4}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    if seconds != 0.0 && seconds.abs() < 1.0 {
        format!("t = {} ms", trimmed(seconds * 1e3))
    } else {
        format!("t = {} s", trimmed(seconds))
    }
}

/// The view title's sample: the frequency, time instant or load factor the field shows.
fn sample_label(descriptor: &AnalysisFieldDescriptor) -> Option<String> {
    match descriptor.sample? {
        RecoveredFieldSample::FrequencyHertz(hertz) => Some(frequency_label(hertz)),
        RecoveredFieldSample::TimeSeconds(seconds) => Some(time_label(seconds)),
        RecoveredFieldSample::LoadFactor(factor) => Some(format!("load factor {factor:.3}")),
        _ => None,
    }
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
    /// A time-sampled run (a transient of any family) is shown at one instant, chosen from its
    /// primary response (see [`PRIMARY_RESPONSES`]). Every time-sampled quantity of the run is
    /// drawn at that instant, on the displacement of that instant when the run publishes one.
    instant: Option<StateKey>,
    /// The instant's title: its time, or its step when the run publishes no time.
    instant_label: Option<String>,
    /// A load path (nonlinear) publishes one load factor per step; its final step is shown,
    /// titled with that step's load factor.
    final_load: Option<(usize, String)>,
    /// A modal run: its displacement, strain and stress are those of mass-normalized
    /// eigenvectors, which carry no physical amplitude. It shows only its mode shapes.
    modal: bool,
}

impl<'a> RunFields<'a> {
    fn new(
        run_id: &'a str,
        published: &'a [AnalysisFieldDescriptor],
        values: &dyn PublishedValues,
    ) -> Result<Self> {
        let mut run = Self {
            run_id,
            published,
            descriptors: published
                .iter()
                .filter(|descriptor| drawable(descriptor))
                .collect(),
            deformation: None,
            instant: None,
            instant_label: None,
            final_load: None,
            modal: false,
        };
        let load_factors: Vec<(usize, &AnalysisFieldDescriptor)> = published
            .iter()
            .filter(|descriptor| {
                descriptor.layout.quantity == Q::LoadFactor
                    && descriptor.location == AnalysisFieldLocation::Global
            })
            .filter_map(|descriptor| match descriptor.layout.ordinal {
                FieldOrdinal::Step(index) => Some((index, descriptor)),
                _ => None,
            })
            .collect();
        if let Some((index, descriptor)) = load_factors.iter().max_by_key(|(index, _)| *index) {
            let factor = values
                .small(run_id, descriptor)?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("load step {index} has no published load factor"))?;
            run.final_load = Some((*index, format!("load factor {factor:.3}")));
        }
        run.modal = !run.modes().is_empty();
        if run.final_load.is_none() {
            if let Some((response, stepped)) = run.primary_response() {
                let at = match response.instant {
                    InstantRule::Peak => peak_of(run_id, stepped, values)?,
                    InstantRule::Final => latest(&stepped),
                };
                if let Some(at) = at {
                    run.instant = state_key(at);
                    run.instant_label = match run.instant {
                        Some(StateKey::Time(time)) => Some(time_label(time)),
                        Some(StateKey::Step(index)) => Some(
                            match values.state_time(run_id, at, index)? {
                                Some(time) => time_label(time),
                                None => format!("step {index}"),
                            },
                        ),
                        None => None,
                    };
                }
            }
        }
        let nodal: Vec<_> = run
            .of_quantity(Q::Displacement, REAL, None)
            .into_iter()
            .filter(|descriptor| {
                descriptor.layout.location == FieldLocation::Node
                    && descriptor.layout.value_shape == ObservationValueShape::Vector3
            })
            .collect();
        let deformation = match run.instant {
            Some(instant) => nodal
                .iter()
                .copied()
                .find(|descriptor| state_key(descriptor) == Some(instant))
                .or_else(|| latest(&nodal)),
            None => latest(&nodal),
        };
        run.deformation = deformation.map(selection);
        Ok(run)
    }

    /// Whether the run publishes `quantity` as a phasor (complex, magnitude, phase or level).
    fn publishes_phasor(&self, quantity: Q) -> bool {
        self.published.iter().any(|descriptor| {
            descriptor.layout.quantity == quantity
                && matches!(
                    descriptor.layout.representation,
                    R::Complex | R::Imaginary | R::Magnitude | R::PhaseRadians | R::SoundPressureLevel
                )
        })
    }

    /// The run's primary response when it is time-stepped (more than one retained state): the
    /// first of [`PRIMARY_RESPONSES`] it publishes as stepped nodal fields.
    fn primary_response(
        &self,
    ) -> Option<(&'static PrimaryResponse, Vec<&'a AnalysisFieldDescriptor>)> {
        PRIMARY_RESPONSES.iter().find_map(|response| {
            let stepped: Vec<_> = self
                .of_quantity(response.quantity, REAL, None)
                .into_iter()
                .filter(|descriptor| {
                    descriptor.layout.location == FieldLocation::Node
                        && descriptor.layout.value_shape == response.shape
                        && state_key(descriptor).is_some()
                })
                .collect();
            (stepped.len() > 1).then_some((response, stepped))
        })
    }

    fn of_quantity(
        &self,
        quantity: Q,
        representations: &[R],
        statistic: Option<FieldStatistic>,
    ) -> Vec<&'a AnalysisFieldDescriptor> {
        self.descriptors
            .iter()
            .copied()
            .filter(|descriptor| {
                descriptor.layout.quantity == quantity
                    && representations.contains(&descriptor.layout.representation)
                    && descriptor.layout.statistic == statistic
                    && !is_mode_shape(descriptor)
            })
            .collect()
    }

    /// The descriptor of a quantity to show. A frequency sweep shows the frequency with the
    /// largest peak magnitude (its resonant response); a time-sampled run its peak instant;
    /// other fields their latest state (the final load factor of a nonlinear path).
    fn chosen(
        &self,
        candidates: &[&'a AnalysisFieldDescriptor],
        values: &dyn PublishedValues,
    ) -> Result<Option<&'a AnalysisFieldDescriptor>> {
        let frequencies: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|descriptor| {
                matches!(
                    descriptor.sample,
                    Some(RecoveredFieldSample::FrequencyHertz(_))
                )
            })
            .collect();
        if frequencies.len() > 1 {
            return peak_of(self.run_id, frequencies, values);
        }
        if let [only] = frequencies.as_slice() {
            return Ok(Some(only));
        }
        if let Some(instant) = self.instant {
            if let Some(at) = candidates
                .iter()
                .copied()
                .find(|descriptor| state_key(descriptor) == Some(instant))
            {
                return Ok(Some(at));
            }
        }
        Ok(latest(candidates))
    }

    /// The view title's sample: the run's instant when the view shows it, else the field's
    /// own frequency, time or load factor.
    fn sample_label(&self, descriptor: &AnalysisFieldDescriptor) -> Option<String> {
        match (self.instant, &self.final_load, descriptor.layout.ordinal) {
            (Some(instant), _, _) if state_key(descriptor) == Some(instant) => {
                self.instant_label.clone()
            }
            (_, Some((last, label)), FieldOrdinal::Step(index)) if index == *last => {
                Some(label.clone())
            }
            _ => sample_label(descriptor),
        }
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

/// How a time-stepped run's instant is chosen from its primary response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstantRule {
    /// The retained state with the largest peak magnitude (a dynamic response's worst instant).
    Peak,
    /// The last retained state (a heating or cooling transient's end state).
    Final,
}

/// A family's primary time response: the nodal quantity that picks the instant a transient is
/// shown at.
struct PrimaryResponse {
    quantity: Q,
    shape: ObservationValueShape,
    instant: InstantRule,
}

/// Primary responses in priority order: structural displacement, acoustic pressure and fluid
/// velocity at their peak; temperature at the end of the transient.
const PRIMARY_RESPONSES: &[PrimaryResponse] = &[
    PrimaryResponse {
        quantity: Q::Displacement,
        shape: ObservationValueShape::Vector3,
        instant: InstantRule::Peak,
    },
    PrimaryResponse {
        quantity: Q::AcousticPressure,
        shape: ObservationValueShape::Scalar,
        instant: InstantRule::Peak,
    },
    PrimaryResponse {
        quantity: Q::FluidVelocity,
        shape: ObservationValueShape::Vector3,
        instant: InstantRule::Peak,
    },
    PrimaryResponse {
        quantity: Q::Temperature,
        shape: ObservationValueShape::Scalar,
        instant: InstantRule::Final,
    },
];

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
        Self::from_published(&published, &StoredValues)
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
    fn from_published(
        runs: &[(&str, &[AnalysisFieldDescriptor])],
        values: &dyn PublishedValues,
    ) -> Result<Self> {
        let runs: Vec<RunFields> = runs
            .iter()
            .map(|(run_id, published)| RunFields::new(run_id, published, values))
            .collect::<Result<_>>()?;
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
                    let mut candidates =
                        run.of_quantity(*quantity, display.representations, display.statistic);
                    if display.time_domain && run.publishes_phasor(*quantity) {
                        candidates.clear();
                    }
                    if let Some(found) = run.chosen(&candidates, values)? {
                        chosen = Some((run, found));
                        break 'runs;
                    }
                }
            }
            let Some((run, descriptor)) = chosen else {
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
                label: match run.sample_label(descriptor) {
                    Some(sample) => format!("{} · {sample}", display.legend),
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
                        .of_quantity(*quantity, &[display.representation], None)
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
                    if let Some(found) = run.chosen(&candidates, values)? {
                        chosen = Some((run, found));
                        break 'volume;
                    }
                }
            }
            let Some((run, descriptor)) = chosen else {
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
                label: match run.sample_label(descriptor) {
                    Some(sample) => format!("{} · {sample}", display.legend),
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
            let eigenvalue =
                mode_eigenvalue(mode_run.run_id, mode_run.published, mode, position, values)?;
            let legend = match eigenvalue {
                Some(ModeEigenvalue::LoadFactor(_)) => {
                    format!("Buckling mode {number} relative amplitude")
                }
                _ => format!("Mode {number} relative amplitude"),
            };
            let label = match eigenvalue {
                Some(ModeEigenvalue::FrequencyHertz(hertz)) => {
                    format!("{legend} · {}", frequency_label(hertz))
                }
                Some(ModeEigenvalue::LoadFactor(factor)) => {
                    format!("{legend} · load factor {factor:.2}")
                }
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

    /// Published values by field id: peak magnitudes and small per-mode lists.
    #[derive(Default)]
    struct Values {
        peaks: std::collections::HashMap<String, f64>,
        lists: std::collections::HashMap<String, Vec<f64>>,
        /// Retained transient times by step.
        times: Vec<f64>,
    }

    impl PublishedValues for Values {
        fn peak(&self, _run: &str, descriptor: &AnalysisFieldDescriptor) -> Result<f64> {
            self.peaks
                .get(&descriptor.field_id)
                .copied()
                .ok_or_else(|| anyhow!("no peak for {}", descriptor.field_id))
        }

        fn small(&self, _run: &str, descriptor: &AnalysisFieldDescriptor) -> Result<Vec<f64>> {
            self.lists
                .get(&descriptor.field_id)
                .cloned()
                .ok_or_else(|| anyhow!("no values for {}", descriptor.field_id))
        }

        fn state_time(
            &self,
            _run: &str,
            _descriptor: &AnalysisFieldDescriptor,
            index: usize,
        ) -> Result<Option<f64>> {
            Ok(self.times.get(index).copied())
        }
    }

    fn sampled(
        id: &str,
        layout: AnalysisFieldLayout,
        shape: Vec<usize>,
        sample: RecoveredFieldSample,
    ) -> AnalysisFieldDescriptor {
        let mut descriptor = published(id, layout, shape);
        descriptor.sample = Some(sample);
        descriptor
    }

    fn view<'a>(profile: &'a CaptureProfile, id: &str) -> &'a ViewSpec {
        profile
            .views
            .iter()
            .find(|view| view.id == id)
            .unwrap_or_else(|| panic!("no view {id}: {:?}", ids(profile)))
    }

    fn field_of(view: &ViewSpec) -> &str {
        view.field.as_ref().unwrap().field_id.as_str()
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
        let profile =
            CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
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
        assert!(CaptureProfile::from_published(&[("run", &fields)], &Values::default()).is_err());
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
        let profile =
            CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
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
        let profile =
            CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
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
        let profile =
            || CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
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
        let profile =
            CaptureProfile::from_published(&[("run", &velocity)], &Values::default()).unwrap();
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
        let profile =
            CaptureProfile::from_published(&[("run", &pressure)], &Values::default()).unwrap();
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
        let profile =
            CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
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
        let profile =
            CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
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
        let profile = CaptureProfile::from_published(
            &[("conduction", &conduction), ("heat", &heat)],
            &Values::default(),
        )
        .unwrap();
        let by_id = |id: &str| profile.views.iter().find(|view| view.id == id).unwrap();
        assert_eq!(by_id("temperature").run_id, "heat");
        assert_eq!(by_id("electric-potential").run_id, "conduction");
    }

    fn node_vector(quantity: Q) -> AnalysisFieldLayout {
        AnalysisFieldLayout::vector3(quantity, FieldLocation::Node)
    }

    /// Transient: every time-sampled quantity is drawn at the instant of peak displacement,
    /// on the displacement of that instant, and the title names the instant.
    #[test]
    fn transient_runs_show_the_peak_displacement_instant() {
        let times = [0.0, 0.002, 0.004, 0.03];
        let mut values = Values::default();
        let mut fields = Vec::new();
        for (step, time) in times.into_iter().enumerate() {
            for (name, quantity) in [("displacement", Q::Displacement), ("velocity", Q::Velocity)] {
                let id = format!("structural.transient.{name}.{step}");
                fields.push(sampled(
                    &id,
                    node_vector(quantity).step(step),
                    vec![4, 3],
                    RecoveredFieldSample::TimeSeconds(time),
                ));
                // Displacement peaks at 4 ms; the final state is smaller.
                values.peaks.insert(id, [0.0, 1.0, 3.0, 2.0][step]);
            }
        }
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        assert_eq!(ids(&profile), ["geometry", "displacement", "velocity"]);
        let displacement = view(&profile, "displacement");
        assert_eq!(
            field_of(displacement),
            "structural.transient.displacement.2"
        );
        assert_eq!(displacement.label, "Displacement magnitude · t = 4 ms");
        assert_eq!(
            displacement.deformation.as_ref().unwrap().field_id,
            "structural.transient.displacement.2"
        );
        let velocity = view(&profile, "velocity");
        assert_eq!(field_of(velocity), "structural.transient.velocity.2");
        assert_eq!(
            velocity.deformation.as_ref().unwrap().field_id,
            "structural.transient.displacement.2"
        );
    }

    /// A structural transient publishes its retained states as stepped fields; the instant of
    /// peak displacement is titled with the time the run retained for that step.
    #[test]
    fn stepped_transient_states_show_the_peak_step_at_its_retained_time() {
        let mut values = Values {
            times: vec![0.0, 0.00025, 0.0005],
            ..Values::default()
        };
        let mut fields = vec![published(
            "structural.displacement",
            node_vector(Q::Displacement),
            vec![4, 3],
        )];
        for step in 0..3 {
            let id = format!("transient.displacement.{step}");
            values.peaks.insert(id.clone(), [0.0, 2.0, 1.0][step]);
            fields.push(published(
                &id,
                node_vector(Q::Displacement).step(step),
                vec![4, 3],
            ));
            fields.push(published(
                &format!("transient.von_mises.{step}"),
                AnalysisFieldLayout::scalar(Q::VonMisesStress, FieldLocation::Element).step(step),
                vec![2],
            ));
        }
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        let displacement = view(&profile, "displacement");
        assert_eq!(field_of(displacement), "transient.displacement.1");
        assert_eq!(displacement.label, "Displacement magnitude · t = 0.25 ms");
        let stress = view(&profile, "stress");
        assert_eq!(field_of(stress), "transient.von_mises.1");
        assert_eq!(stress.label, "Von Mises stress · t = 0.25 ms");
        assert_eq!(
            stress.deformation.as_ref().unwrap().field_id,
            "transient.displacement.1"
        );
    }

    /// Preload -> buckling: the buckling step's mode shapes, titled by their load factors.
    #[test]
    fn buckling_modes_are_titled_by_load_factor() {
        let fields = [
            published(
                "structural.buckling.mode_shape.1",
                node_vector(Q::Displacement).mode(0),
                vec![4, 3],
            ),
            published(
                "structural.buckling.mode_shape.2",
                node_vector(Q::Displacement).mode(1),
                vec![4, 3],
            ),
            published(
                "structural.buckling.load_factor",
                AnalysisFieldLayout::scalar(Q::LoadFactor, FieldLocation::Mode),
                vec![2],
            ),
        ];
        let mut values = Values::default();
        values.lists.insert(
            "structural.buckling.load_factor".into(),
            vec![410.06, 3650.2],
        );
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        assert_eq!(ids(&profile), ["geometry", "mode-1", "mode-2"]);
        let first = view(&profile, "mode-1");
        assert_eq!(field_of(first), "structural.buckling.mode_shape.1");
        assert_eq!(
            first.label,
            "Buckling mode 1 relative amplitude · load factor 410.06"
        );
        assert!(first.relative && first.deformation.is_some());
    }

    /// Modal -> random vibration: the RMS displacement on the undeformed body; the variance is
    /// never drawn as a displacement.
    #[test]
    fn random_response_shows_rms_displacement_never_variance() {
        let fields = [
            published(
                "structural.random_response.variance.displacement",
                node_vector(Q::Displacement).statistic(FieldStatistic::Variance),
                vec![4, 3],
            ),
            published(
                "structural.random_response.rms.displacement",
                node_vector(Q::Displacement).statistic(FieldStatistic::RootMeanSquare),
                vec![4, 3],
            ),
        ];
        let profile =
            CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
        assert_eq!(ids(&profile), ["geometry", "rms-displacement"]);
        let rms = view(&profile, "rms-displacement");
        assert_eq!(field_of(rms), "structural.random_response.rms.displacement");
        assert_eq!(rms.label, "RMS displacement magnitude");
        assert!(rms.deformation.is_none());
        assert_eq!(rms.shape, "undeformed");
    }

    /// Modal -> response spectrum: the combined peak displacement envelope, undeformed.
    #[test]
    fn response_spectrum_shows_the_peak_displacement_envelope() {
        let fields = [published(
            "structural.response_spectrum.nodal.relative_displacement",
            node_vector(Q::Displacement).statistic(FieldStatistic::ComponentMagnitudeEnvelope),
            vec![4, 3],
        )];
        let profile =
            CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
        assert_eq!(ids(&profile), ["geometry", "peak-displacement"]);
        let peak = view(&profile, "peak-displacement");
        assert!(peak.deformation.is_none());
        assert!(peak.label.starts_with("Peak displacement magnitude"));
    }

    /// Nonlinear: displacement and von Mises stress at the final load factor, on the final
    /// deformed shape.
    #[test]
    fn nonlinear_runs_show_the_final_load_factor() {
        let mut fields = Vec::new();
        for (step, factor) in [0.25, 0.5, 1.0].into_iter().enumerate() {
            fields.push(sampled(
                &format!("structural.nonlinear.displacement.{step}"),
                node_vector(Q::Displacement).step(step),
                vec![4, 3],
                RecoveredFieldSample::LoadFactor(factor),
            ));
            fields.push(sampled(
                &format!("structural.nonlinear.von_mises.{step}"),
                AnalysisFieldLayout::scalar(Q::VonMisesStress, FieldLocation::Element).step(step),
                vec![2],
                RecoveredFieldSample::LoadFactor(factor),
            ));
        }
        let profile =
            CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
        assert_eq!(ids(&profile), ["geometry", "displacement", "stress"]);
        let displacement = view(&profile, "displacement");
        assert_eq!(
            field_of(displacement),
            "structural.nonlinear.displacement.2"
        );
        assert_eq!(
            displacement.label,
            "Displacement magnitude · load factor 1.000"
        );
        let stress = view(&profile, "stress");
        assert_eq!(field_of(stress), "structural.nonlinear.von_mises.2");
        assert_eq!(
            stress.deformation.as_ref().unwrap().field_id,
            "structural.nonlinear.displacement.2"
        );
    }

    /// A nonlinear load path publishes stepped states and one load factor per step: the final
    /// step is shown, titled with its load factor, never the step of peak displacement.
    #[test]
    fn stepped_load_paths_show_the_final_step_with_its_load_factor() {
        let mut values = Values::default();
        let mut fields = Vec::new();
        for (step, factor) in [0.5, 1.0].into_iter().enumerate() {
            let displacement = format!("nonlinear.displacement.{step}");
            // A larger earlier displacement must not pull the view off the final load.
            values.peaks.insert(displacement.clone(), [9.0, 1.0][step]);
            fields.push(published(
                &displacement,
                node_vector(Q::Displacement).step(step),
                vec![4, 3],
            ));
            fields.push(published(
                &format!("nonlinear.von_mises.{step}"),
                AnalysisFieldLayout::scalar(Q::VonMisesStress, FieldLocation::Element).step(step),
                vec![2],
            ));
            let load = format!("nonlinear.load_factor.{step}");
            values.lists.insert(load.clone(), vec![factor]);
            fields.push(published(
                &load,
                AnalysisFieldLayout::scalar(Q::LoadFactor, FieldLocation::Global).step(step),
                vec![1],
            ));
        }
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        let displacement = view(&profile, "displacement");
        assert_eq!(field_of(displacement), "nonlinear.displacement.1");
        assert_eq!(
            displacement.label,
            "Displacement magnitude · load factor 1.000"
        );
        let stress = view(&profile, "stress");
        assert_eq!(field_of(stress), "nonlinear.von_mises.1");
        assert_eq!(stress.label, "Von Mises stress · load factor 1.000");
    }

    /// Harmonic: the amplitude at the frequency with the largest peak response.
    #[test]
    fn harmonic_sweeps_show_the_resonant_frequency_amplitude() {
        let mut values = Values::default();
        let fields: Vec<_> = [(20.0, 1.0), (80.0, 9.0), (200.0, 0.5)]
            .into_iter()
            .enumerate()
            .map(|(index, (hertz, peak))| {
                let id = format!("structural.harmonic.displacement.{index}");
                values.peaks.insert(id.clone(), peak);
                sampled(
                    &id,
                    node_vector(Q::Displacement).representation(R::Complex),
                    vec![4, 3, 2],
                    RecoveredFieldSample::FrequencyHertz(hertz),
                )
            })
            .collect();
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        let displacement = view(&profile, "displacement");
        assert_eq!(field_of(displacement), "structural.harmonic.displacement.1");
        assert_eq!(displacement.label, "Displacement magnitude · 80.00 Hz");
    }

    /// Thermal steady: temperature and heat flux on the undeformed body (no displacement).
    #[test]
    fn thermal_steady_shows_temperature_and_heat_flux_undeformed() {
        let fields = [
            published(
                "thermal.temperature",
                AnalysisFieldLayout::scalar(Q::Temperature, FieldLocation::Node),
                vec![4],
            ),
            published(
                "thermal.heat_flux",
                AnalysisFieldLayout::vector3(Q::HeatFlux, FieldLocation::Element),
                vec![2, 3],
            ),
        ];
        let profile = CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
        for (id, field) in [("temperature", "thermal.temperature"), ("heat-flux", "thermal.heat_flux")] {
            let shown = view(&profile, id);
            assert_eq!(field_of(shown), field);
            assert!(shown.deformation.is_none());
            assert_eq!(shown.shape, "undeformed");
        }
        assert_eq!(view(&profile, "temperature").label, "Temperature");
    }

    /// Thermal transient: the end of the transient, titled with its retained time; heat flux
    /// is drawn at the same instant.
    #[test]
    fn thermal_transient_shows_the_final_temperature_at_its_time() {
        let mut fields = Vec::new();
        for (step, time) in [0.0, 30.0, 60.0].into_iter().enumerate() {
            fields.push(sampled(
                &format!("thermal.transient.temperature.{step}"),
                AnalysisFieldLayout::scalar(Q::Temperature, FieldLocation::Node).step(step),
                vec![4],
                RecoveredFieldSample::TimeSeconds(time),
            ));
            fields.push(sampled(
                &format!("thermal.transient.heat_flux.{step}"),
                AnalysisFieldLayout::vector3(Q::HeatFlux, FieldLocation::Element).step(step),
                vec![2, 3],
                RecoveredFieldSample::TimeSeconds(time),
            ));
        }
        // The final-state rule reads no peaks.
        let profile = CaptureProfile::from_published(&[("run", &fields)], &Values::default()).unwrap();
        let temperature = view(&profile, "temperature");
        assert_eq!(field_of(temperature), "thermal.transient.temperature.2");
        assert_eq!(temperature.label, "Temperature · t = 60 s");
        let flux = view(&profile, "heat-flux");
        assert_eq!(field_of(flux), "thermal.transient.heat_flux.2");
        assert_eq!(flux.label, "Heat-flux magnitude · t = 60 s");
    }

    /// Electromagnetic harmonic: each quantity at the frequency of its largest response.
    #[test]
    fn electromagnetic_harmonic_shows_each_quantity_at_its_peak_frequency() {
        let mut values = Values::default();
        let mut fields = Vec::new();
        for (index, hertz) in [50.0, 1000.0, 10000.0].into_iter().enumerate() {
            for (name, quantity, peak) in [
                ("current_density", Q::ElectricCurrentDensity, [1.0, 4.0, 2.0][index]),
                ("magnetic_flux_density", Q::MagneticFluxDensity, [3.0, 2.0, 1.0][index]),
            ] {
                let id = format!("electromagnetic.{name}.{index}");
                values.peaks.insert(id.clone(), peak);
                fields.push(sampled(
                    &id,
                    AnalysisFieldLayout::vector3(quantity, FieldLocation::Element)
                        .representation(R::Complex),
                    vec![2, 3, 2],
                    RecoveredFieldSample::FrequencyHertz(hertz),
                ));
            }
        }
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        let current = view(&profile, "current-density");
        assert_eq!(field_of(current), "electromagnetic.current_density.1");
        assert_eq!(current.label, "Current density |J| · 1000.00 Hz");
        assert!(current.deformation.is_none());
        let flux = view(&profile, "magnetic-flux-density");
        assert_eq!(field_of(flux), "electromagnetic.magnetic_flux_density.0");
        assert_eq!(flux.label, "Magnetic flux density |B| · 50.00 Hz");
    }

    /// Acoustic harmonic: the pressure amplitude at the resonant frequency, undeformed.
    #[test]
    fn acoustic_harmonic_shows_the_pressure_amplitude_at_the_peak_frequency() {
        let mut values = Values::default();
        let fields: Vec<_> = [(100.0, 0.5), (343.0, 7.0), (800.0, 1.0)]
            .into_iter()
            .enumerate()
            .map(|(index, (hertz, peak))| {
                let id = format!("acoustic.pressure.{index}");
                values.peaks.insert(id.clone(), peak);
                sampled(
                    &id,
                    AnalysisFieldLayout::scalar(Q::AcousticPressure, FieldLocation::Node)
                        .representation(R::Complex),
                    vec![4, 2],
                    RecoveredFieldSample::FrequencyHertz(hertz),
                )
            })
            .collect();
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        let pressure = view(&profile, "acoustic-pressure");
        assert_eq!(field_of(pressure), "acoustic.pressure.1");
        assert_eq!(pressure.label, "Acoustic pressure amplitude |p| · 343.00 Hz");
        assert_eq!(pressure.shape, "undeformed");
    }

    /// Acoustic transient: the instant of peak pressure, titled with its time.
    #[test]
    fn acoustic_transient_shows_the_peak_pressure_instant() {
        let mut values = Values::default();
        let fields: Vec<_> = [(0.0, 0.0), (0.001, 5.0), (0.002, 2.0)]
            .into_iter()
            .enumerate()
            .map(|(step, (time, peak))| {
                let id = format!("acoustic.transient.pressure.{step}");
                values.peaks.insert(id.clone(), peak);
                sampled(
                    &id,
                    AnalysisFieldLayout::scalar(Q::AcousticPressure, FieldLocation::Node).step(step),
                    vec![4],
                    RecoveredFieldSample::TimeSeconds(time),
                )
            })
            .collect();
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        let pressure = view(&profile, "sound-pressure");
        assert_eq!(field_of(pressure), "acoustic.transient.pressure.1");
        assert_eq!(pressure.label, "Acoustic pressure · t = 1 ms");
    }

    /// Fluid transient: velocity and pressure at the instant of peak velocity.
    #[test]
    fn fluid_transient_shows_the_peak_velocity_instant() {
        let mut values = Values::default();
        let mut fields = Vec::new();
        for (step, (time, peak)) in [(0.0, 0.1), (0.5, 2.0), (1.0, 1.5)].into_iter().enumerate() {
            let id = format!("fluid.velocity.{step}");
            values.peaks.insert(id.clone(), peak);
            fields.push(sampled(
                &id,
                node_vector(Q::FluidVelocity).step(step),
                vec![4, 3],
                RecoveredFieldSample::TimeSeconds(time),
            ));
            fields.push(sampled(
                &format!("fluid.pressure.{step}"),
                AnalysisFieldLayout::scalar(Q::FluidPressure, FieldLocation::Node).step(step),
                vec![4],
                RecoveredFieldSample::TimeSeconds(time),
            ));
        }
        let profile = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        let velocity = view(&profile, "velocity");
        assert_eq!(field_of(velocity), "fluid.velocity.1");
        assert_eq!(velocity.label, "Velocity magnitude · t = 500 ms");
        assert!(velocity.deformation.is_none());
        let pressure = view(&profile, "pressure");
        assert_eq!(field_of(pressure), "fluid.pressure.1");
        assert_eq!(pressure.label, "Pressure · t = 500 ms");
    }
}
