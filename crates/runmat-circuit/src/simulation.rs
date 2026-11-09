//! Circuit simulation setups: the typed `simulation` sections of a `.circuit` design.
//!
//! ```text
//! simulation front_end {
//!     sheet = channel
//!     source VIN = { p = net.IN, pulse = { from = 0 V, to = 1 V, delay = 1 us, rise = 10 ns, width = 1 ms } }
//!     source V5 = { p = "net.+5V", dc = 5 V }
//!     model OPA171 = { opamp = { a0 = 1000000, gbw = 3 MHz, sr = 1500000 }, pins = ["IN+", "IN-", OUT, "V+", "V-"] }
//!     tolerance resistors = { match = "R*", value = 1 % }
//!     analysis step = tran { stop = 50 us, step = 50 ns }
//!     analysis response = ac { from = 10 Hz, to = 100 MHz, points = 50 }
//!     analysis spread = monte_carlo { of = response, runs = 1000, seed = 1 }
//!     analysis hiss = noise { output = net.AIN, input = VIN, from = 10 Hz, to = 10 MHz, points = 20 }
//!     analysis hot = temperature { of = response, values = [-40 degC, 25 degC, 85 degC] }
//!     probe = [net.IN, net.AMP, net.AIN, RO.2]
//!     measure f3db = { of = response, bandwidth = net.AIN }
//!     measure peak_current = { of = step, max = "RO.2" }
//!     measure en = { of = hiss, output_noise = net.AIN }
//! }
//! ```
//!
//! Quantities are SI internally. A source's `n` defaults to `GND`. Tolerances are `±3σ` of a
//! normal distribution unless the Monte Carlo analysis says `distribution = uniform`.

use runmat_rules::SourceKey;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SimulationDef {
    pub name: String,
    /// Simulate one sheet as the whole circuit (its ports become nets); `None`: the root.
    pub sheet: Option<String>,
    pub sources: BTreeMap<String, SimSource>,
    /// Models by part number (or designator).
    pub models: BTreeMap<String, SimModel>,
    /// Relative tolerances (fractions) by designator pattern, by statement name.
    pub tolerances: BTreeMap<String, SimTolerance>,
    pub analyses: BTreeMap<String, SimAnalysis>,
    /// What is recorded and plotted: net voltages and pin currents.
    pub probes: Vec<SimProbe>,
    pub measures: BTreeMap<String, SimMeasure>,
    /// `.options`-style settings (`reltol`, `method`, `temp`).
    pub options: BTreeMap<String, String>,
    pub source: SourceKey,
}

/// A probe: a net's voltage (`net.OUT`), or the current into a component pin (`RO.2`), which
/// the run measures through a zero-volt sense source in series with that pin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimProbe {
    Net(String),
    Pin { component: String, pin: String },
}

impl SimProbe {
    /// The net (`OUT`) or the pin (`RO.2`).
    pub fn name(&self) -> String {
        match self {
            SimProbe::Net(n) => n.clone(),
            SimProbe::Pin { component, pin } => format!("{component}.{pin}"),
        }
    }
    /// `voltage` or `current`.
    pub fn quantity(&self) -> &'static str {
        match self {
            SimProbe::Net(_) => "voltage",
            SimProbe::Pin { .. } => "current",
        }
    }
    /// `V(OUT)`, `I(RO.2)`.
    pub fn label(&self) -> String {
        format!(
            "{}({})",
            if matches!(self, SimProbe::Net(_)) {
                "V"
            } else {
                "I"
            },
            self.name()
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimTolerance {
    /// Designator glob (`R*`, `C?`).
    pub pattern: String,
    /// Relative tolerance (0.01 for 1 %).
    pub value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Voltage,
    Current,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimSource {
    pub kind: SourceKind,
    pub p: String,
    pub n: String,
    pub dc: f64,
    pub ac: f64,
    pub wave: Option<SimWave>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimWave {
    Pulse {
        from: f64,
        to: f64,
        delay: f64,
        rise: f64,
        fall: f64,
        width: f64,
        period: f64,
    },
    Sin {
        offset: f64,
        amplitude: f64,
        freq: f64,
        delay: f64,
    },
    Pwl(Vec<(f64, f64)>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimModel {
    /// The built-in op-amp macromodel; pins name the part's `in+ in− out v+ v−` pins.
    OpAmp {
        params: BTreeMap<String, f64>,
        pins: Vec<String>,
    },
    /// A vendor subcircuit from a SPICE library, with the part pins in subcircuit order.
    Subckt {
        library: String,
        name: String,
        pins: Vec<String>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimAnalysis {
    Op,
    Dc {
        source: String,
        from: f64,
        to: f64,
        step: f64,
    },
    /// Logarithmic sweep with `points` per decade.
    Ac {
        from: f64,
        to: f64,
        points: u32,
    },
    Tran {
        stop: f64,
        step: f64,
        max_step: Option<f64>,
    },
    MonteCarlo {
        of: String,
        runs: u32,
        seed: u64,
        uniform: bool,
    },
    /// Small-signal noise at net `output` (against `reference`, ground when `None`), referred to
    /// the section source `input`, on a logarithmic sweep with `points` per decade.
    Noise {
        output: String,
        reference: Option<String>,
        input: String,
        from: f64,
        to: f64,
        points: u32,
    },
    /// Reruns analysis `of` at each temperature (°C).
    Temperature {
        of: String,
        values: Vec<f64>,
    },
}

impl SimAnalysis {
    pub fn word(&self) -> &'static str {
        match self {
            SimAnalysis::Op => "op",
            SimAnalysis::Dc { .. } => "dc",
            SimAnalysis::Ac { .. } => "ac",
            SimAnalysis::Tran { .. } => "tran",
            SimAnalysis::MonteCarlo { .. } => "monte_carlo",
            SimAnalysis::Noise { .. } => "noise",
            SimAnalysis::Temperature { .. } => "temperature",
        }
    }

    /// The analysis a Monte Carlo or temperature sweep reruns.
    pub fn reruns(&self) -> Option<&str> {
        match self {
            SimAnalysis::MonteCarlo { of, .. } | SimAnalysis::Temperature { of, .. } => Some(of),
            _ => None,
        }
    }
}

/// What a measurement computes on its signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MeasureKind {
    At,
    Max,
    Min,
    PeakToPeak,
    Average,
    Rms,
    Final,
    RiseTime,
    Settling,
    Overshoot,
    Bandwidth,
    PeakFrequency,
    Q,
    UnityGain,
    PhaseMargin,
    GainAt,
    PhaseAt,
    /// A noise analysis: the output noise integrated over its band (or `from`–`to`), in V rms.
    OutputNoise,
    /// The input-referred noise integrated the same way, in V rms.
    InputNoise,
    /// The output noise density at `at`, in V/√Hz.
    NoiseDensity,
}

impl MeasureKind {
    pub const ALL: &'static [(&'static str, MeasureKind)] = &[
        ("value", MeasureKind::At),
        ("max", MeasureKind::Max),
        ("min", MeasureKind::Min),
        ("peak_to_peak", MeasureKind::PeakToPeak),
        ("average", MeasureKind::Average),
        ("rms", MeasureKind::Rms),
        ("final", MeasureKind::Final),
        ("rise_time", MeasureKind::RiseTime),
        ("settling", MeasureKind::Settling),
        ("overshoot", MeasureKind::Overshoot),
        ("bandwidth", MeasureKind::Bandwidth),
        ("peak_frequency", MeasureKind::PeakFrequency),
        ("q", MeasureKind::Q),
        ("unity_gain", MeasureKind::UnityGain),
        ("phase_margin", MeasureKind::PhaseMargin),
        ("gain", MeasureKind::GainAt),
        ("phase", MeasureKind::PhaseAt),
        ("output_noise", MeasureKind::OutputNoise),
        ("input_noise", MeasureKind::InputNoise),
        ("noise_density", MeasureKind::NoiseDensity),
    ];
    pub fn word(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(_, k)| *k == self)
            .map(|(w, _)| *w)
            .unwrap_or("value")
    }

    /// Whether it reads the signal's level (volts on a net, amperes into a pin).
    pub fn is_level(self) -> bool {
        matches!(
            self,
            MeasureKind::At
                | MeasureKind::Max
                | MeasureKind::Min
                | MeasureKind::PeakToPeak
                | MeasureKind::Average
                | MeasureKind::Rms
                | MeasureKind::Final
        )
    }

    /// Whether it reads a noise analysis.
    pub fn is_noise(self) -> bool {
        matches!(
            self,
            MeasureKind::OutputNoise | MeasureKind::InputNoise | MeasureKind::NoiseDensity
        )
    }

    /// The unit the measured value is in on a net: `V`, `s`, `Hz`, `dB`, `°`, `%` (overshoot,
    /// in percent), `V/√Hz`, or empty (Q). [`SimMeasure::unit`] gives amperes for a pin.
    pub fn unit(self) -> &'static str {
        match self {
            MeasureKind::At
            | MeasureKind::Max
            | MeasureKind::Min
            | MeasureKind::PeakToPeak
            | MeasureKind::Average
            | MeasureKind::Rms
            | MeasureKind::Final => "V",
            MeasureKind::OutputNoise | MeasureKind::InputNoise => "V",
            MeasureKind::NoiseDensity => "V/√Hz",
            MeasureKind::RiseTime | MeasureKind::Settling => "s",
            MeasureKind::Bandwidth | MeasureKind::PeakFrequency | MeasureKind::UnityGain => "Hz",
            MeasureKind::GainAt => "dB",
            MeasureKind::PhaseAt | MeasureKind::PhaseMargin => "°",
            MeasureKind::Overshoot => "%",
            MeasureKind::Q => "",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimMeasure {
    /// The analysis it reads.
    pub of: String,
    pub kind: MeasureKind,
    /// The net (its voltage) or pin (the current into it) it measures.
    pub probe: SimProbe,
    /// `at`, `from`, `to`, `band`, `drop`, in SI (fractions for `band`, dB for `drop`).
    pub args: BTreeMap<String, f64>,
    /// The specification limits (`spec = { min = …, max = … }`), in the measured value's unit
    /// ([`SimMeasure::unit`]): a value passes inside them, and a Monte Carlo's yield is the
    /// fraction of runs that do.
    pub spec_min: Option<f64>,
    pub spec_max: Option<f64>,
}

impl SimMeasure {
    /// The measured value's unit: [`MeasureKind::unit`], in amperes for a level on a pin.
    pub fn unit(&self) -> &'static str {
        match (&self.probe, self.kind.is_level()) {
            (SimProbe::Pin { .. }, true) => "A",
            _ => self.kind.unit(),
        }
    }
}
