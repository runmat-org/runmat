//! What each circuit and board-layout word does: statement words, analyses and measures, for
//! hovers and the language docs. A test keeps every word in the word tables covered.

/// The summary of a `.circuit` or `.circuit.layout` word.
pub fn summary(word: &str) -> Option<&'static str> {
    Some(match word {
        // Circuit statements.
        "circuit" => "A circuit: components joined by nets, on one or more sheets.",
        "sheet" => "A schematic sheet; a sheet used by another becomes a hierarchical block.",
        "root" => "The top sheet of the design.",
        "param" => "A named value the circuit's components can read.",
        "net_class" => "Rules shared by a group of nets: width, clearance, impedance.",
        "interface" => "A named group of ports a sheet exposes.",
        "variant" => "A build variant: components fitted or left out.",
        "port" => "A connection point a sheet exposes, with its direction.",
        "component" => "A placed part: its catalog part, value and parameters.",
        "net" => "A connection between pins.",
        "bus" => "A group of nets carried together.",
        "harness" => "A cable or wiring harness between connectors.",
        "instance" => "A placed copy of a sheet.",
        "connect" => "Joins pins or ports to a net.",
        "no_erc" => "Waives an electrical rule check at a pin or net.",
        "diff_pair" => "Two nets routed as a differential pair.",
        "blanket" => "One directive over several nets.",
        "text" => "A note on a sheet or board.",
        "simulation" => "A circuit simulation: sources, models, analyses and measures.",
        "source" => "A stimulus source in a simulation.",
        "model" => "A device model a simulation uses.",
        "tolerance" => "Component tolerances a Monte Carlo analysis varies.",
        "analysis" => {
            "One simulation analysis: op, dc, ac, tran, noise, monte_carlo or temperature."
        }
        "probe" => "A signal recorded during a simulation.",
        "measure" => "A value computed from a simulation result, with an optional spec.",
        "option" => "A simulator option.",
        "part" => {
            "A catalog part declared in the file: manufacturer part number, package and pins."
        }
        "footprint" => "A footprint: the pads and outline a package places on the board.",
        "symbol" => "A schematic symbol.",
        // Layout statements.
        "layout" => "A board layout for a circuit.",
        "sync" => "The circuit revision the layout was last synchronized with.",
        "version" => "The layout format version.",
        "compare" => "Compares the layout against the circuit.",
        "component_class" => "Rules shared by a group of components.",
        "stackup" => "The board's layer stack: copper, dielectric and their thicknesses.",
        "layer" => "One layer of the stackup.",
        "finish" => "The board's surface finish.",
        "region" => "A board area with its own rules.",
        "outline" => "The board outline.",
        "cutout" => "A hole cut through the board.",
        "limits" => "Fabrication limits the design rules check.",
        "fabrication" => "Fabrication settings: the outputs and their options.",
        "rule" => "A design rule: clearance, width or via size between items.",
        "room" => "An area components of a class stay inside.",
        "fiducial" => "A fiducial mark for assembly machines.",
        "track" => "A copper track on one layer.",
        "arc" => "A curved copper track.",
        "via" => "A plated hole joining layers.",
        "pour" => "A copper pour filling an area with a net.",
        "keepout" => "An area kept clear of copper, components or vias.",
        "graphic" => "A silkscreen or drawing graphic.",
        "dimension" => "A dimension drawn on the board.",
        "package" => "A component package: pads, body and height.",
        "name" => "A name shown on the board.",
        // Analyses.
        "op" => "Operating point: the DC voltages and currents with sources at their set values.",
        "dc" => "DC sweep: the operating point as a source steps through a range.",
        "ac" => "AC small-signal analysis over a frequency range.",
        "tran" => "Transient analysis: voltages and currents over time.",
        "monte_carlo" => {
            "Repeats an analysis with component values varied within their tolerances."
        }
        "noise" => "Noise analysis over a frequency range.",
        "temperature" => "Repeats an analysis at several temperatures.",
        // Measures.
        "value" => "The signal's value at a time or frequency.",
        "max" => "The signal's maximum.",
        "min" => "The signal's minimum.",
        "peak_to_peak" => "Maximum less minimum.",
        "average" => "The signal's mean.",
        "rms" => "The signal's root-mean-square.",
        "final" => "The signal's value at the end of the run.",
        "rise_time" => "The time to rise from 10 % to 90 % of the step.",
        "settling" => "The time until the signal stays within a band of its final value.",
        "overshoot" => "How far the signal passes its final value, as a fraction of the step.",
        "bandwidth" => "The frequency where the gain falls 3 dB below its passband value.",
        "peak_frequency" => "The frequency of the gain's peak.",
        "q" => "The quality factor of a resonance.",
        "unity_gain" => "The frequency where the gain crosses 1 (0 dB).",
        "phase_margin" => "Phase above -180° at unity gain.",
        "gain" => "The gain at a frequency.",
        "phase" => "The phase at a frequency.",
        "output_noise" => "Total output noise over a band.",
        "input_noise" => "Output noise referred to the input.",
        "noise_density" => "Noise spectral density at a frequency.",
        // Measure fields.
        "of" => "The signal measured: a net's voltage or a pin's current.",
        "spec" => "The pass range the measured value must fall in.",
        "at" => "The time or frequency a measure reads at.",
        "from" => "Where a measure's window starts.",
        "to" => "Where a measure's window ends.",
        "band" => "A settling band, as a fraction of the step.",
        "drop" => "How far below the passband a bandwidth is read.",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::summary;

    #[test]
    fn every_word_has_a_summary() {
        let words = super::super::words::ALL
            .iter()
            .chain(super::super::layout::words::ALL)
            .chain(super::super::simulation::ANALYSES)
            .chain(super::super::simulation::MEASURE_FIELDS)
            .copied()
            .chain(crate::simulation::MeasureKind::ALL.iter().map(|(w, _)| *w));
        let missing: Vec<&str> = words.filter(|w| summary(w).is_none()).collect();
        assert!(missing.is_empty(), "no summary: {missing:?}");
    }
}
