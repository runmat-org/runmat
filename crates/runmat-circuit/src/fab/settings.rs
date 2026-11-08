//! A board's fabrication settings: the outputs to make, the layers left out, the variant,
//! the folder and whether to zip. The `.circuit.layout` keeps the last settings used
//! (`fabrication = { … }`), and the Desktop's Fabrication outputs, `design.export` and
//! `runmat-design export --format fabrication` all read them from there, so a re-export makes
//! the same files wherever it runs.
//!
//! ```text
//! fabrication = { preset = fab_house, outputs = [gerber, drill, drill_map, drawing, ipc356], layers = ["F.Cu", "B.Cu", "F.Mask", "B.Mask", "F.Silkscreen", "Edge.Cuts"], folder = "fab/daq-A", zip = true, gerber = { units = inch, decimals = 5 }, pick_place = { sides = separate, units = mm }, bom = { columns = [designators, quantity, value, mpn], group_by_value = false } }
//! ```

use super::assembly::{BomFormat, PickPlaceFormat};
use super::set::FabRequest;
use super::GerberFormat;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// One output of the fabrication set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FabOutput {
    /// Gerber X2 per layer; the job file always comes with it.
    Gerber,
    /// Excellon drill files, plated per span and non-plated.
    Drill,
    DrillMap,
    /// The fabrication drawing with the layer and stackup table.
    Drawing,
    /// IPC-D-356A bare-board test netlist.
    Ipc356,
    /// Pick-and-place (centroid) per variant.
    PickPlace,
    /// Bill of materials per variant.
    Bom,
    /// Assembly drawing: outlines and designators per side.
    AssemblyDrawing,
    Ipc2581,
    Odb,
}

impl FabOutput {
    pub const ALL: [FabOutput; 10] = [
        FabOutput::Gerber,
        FabOutput::Drill,
        FabOutput::DrillMap,
        FabOutput::Drawing,
        FabOutput::Ipc356,
        FabOutput::PickPlace,
        FabOutput::Bom,
        FabOutput::AssemblyDrawing,
        FabOutput::Ipc2581,
        FabOutput::Odb,
    ];
    /// The word the layout and the CLI write (`drill_map`).
    pub fn word(self) -> &'static str {
        match self {
            FabOutput::Gerber => "gerber",
            FabOutput::Drill => "drill",
            FabOutput::DrillMap => "drill_map",
            FabOutput::Drawing => "drawing",
            FabOutput::Ipc356 => "ipc356",
            FabOutput::PickPlace => "pick_place",
            FabOutput::Bom => "bom",
            FabOutput::AssemblyDrawing => "assembly_drawing",
            FabOutput::Ipc2581 => "ipc2581",
            FabOutput::Odb => "odb",
        }
    }
    /// Reads a word; `-` and `_` are the same (`drill-map`).
    pub fn from_word(word: &str) -> Option<FabOutput> {
        let w = word.replace('-', "_");
        FabOutput::ALL.into_iter().find(|o| o.word() == w)
    }
}

/// A named set of outputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FabPreset {
    /// What a board house needs: Gerbers and job file, drill files and map, the fabrication
    /// drawing and the test netlist.
    #[default]
    FabHouse,
    /// What an assembly house needs: pick-and-place, BOM and the assembly drawing.
    Assembly,
    /// Everything, with IPC-2581 and ODB++.
    Full,
    Custom,
}

impl FabPreset {
    pub const ALL: [FabPreset; 4] = [FabPreset::FabHouse, FabPreset::Assembly, FabPreset::Full, FabPreset::Custom];
    pub fn word(self) -> &'static str {
        match self {
            FabPreset::FabHouse => "fab_house",
            FabPreset::Assembly => "assembly",
            FabPreset::Full => "full",
            FabPreset::Custom => "custom",
        }
    }
    pub fn from_word(word: &str) -> Option<FabPreset> {
        let w = word.replace('-', "_");
        FabPreset::ALL.into_iter().find(|p| p.word() == w)
    }
    /// The outputs the preset makes (empty for custom).
    pub fn outputs(self) -> BTreeSet<FabOutput> {
        use FabOutput::*;
        match self {
            FabPreset::FabHouse => [Gerber, Drill, DrillMap, Drawing, Ipc356].into(),
            FabPreset::Assembly => [PickPlace, Bom, AssemblyDrawing].into(),
            FabPreset::Full => FabOutput::ALL.into(),
            FabPreset::Custom => BTreeSet::new(),
        }
    }
    /// The preset whose outputs are exactly `outputs`, else custom.
    pub fn of(outputs: &BTreeSet<FabOutput>) -> FabPreset {
        [FabPreset::FabHouse, FabPreset::Assembly, FabPreset::Full].into_iter().find(|p| &p.outputs() == outputs).unwrap_or(FabPreset::Custom)
    }
}

/// The settings a fabrication export runs with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FabSettings {
    pub preset: FabPreset,
    pub outputs: BTreeSet<FabOutput>,
    /// The Gerber layers to write, by layer name (`F.Cu`, `F.Courtyard`); the default layers
    /// (every one but the courtyards) when `None`.
    pub layers: Option<BTreeSet<String>>,
    /// The variants that get assembly files (`""` is the default variant); every one when empty.
    pub variants: Vec<String>,
    /// The output folder, relative to the board's folder; `fab/<board>` when absent.
    pub folder: Option<String>,
    /// Pack the set into one `.zip` beside the files.
    pub zip: bool,
    /// Pick-and-place lists through-hole parts too.
    pub include_through_hole: bool,
    /// The Gerber files' unit and decimals.
    pub gerber: GerberFormat,
    /// The pick-and-place's sides and unit.
    pub pick_place: PickPlaceFormat,
    /// The bill of materials' columns and grouping.
    pub bom: BomFormat,
    /// The board revision: in the Gerber and job files' project attribute and the default
    /// folder (`fab/<board>-<revision>`).
    pub revision: Option<String>,
    /// The board's part number: an assembly's parts list names the board by it.
    pub part_number: Option<String>,
}

impl Default for FabSettings {
    fn default() -> Self {
        FabSettings {
            preset: FabPreset::FabHouse,
            outputs: FabPreset::FabHouse.outputs(),
            layers: None,
            variants: Vec::new(),
            folder: None,
            zip: true,
            include_through_hole: false,
            gerber: GerberFormat::default(),
            pick_place: PickPlaceFormat::default(),
            bom: BomFormat::default(),
            revision: None,
            part_number: None,
        }
    }
}

impl FabSettings {
    /// The request the settings make.
    pub fn request(&self) -> FabRequest {
        let on = |o| self.outputs.contains(&o);
        FabRequest {
            gerber: on(FabOutput::Gerber),
            drill: on(FabOutput::Drill),
            drill_map: on(FabOutput::DrillMap),
            drawing: on(FabOutput::Drawing),
            pick_place: on(FabOutput::PickPlace),
            bom: on(FabOutput::Bom),
            assembly_drawing: on(FabOutput::AssemblyDrawing),
            ipc2581: on(FabOutput::Ipc2581),
            odb: on(FabOutput::Odb),
            ipc356: on(FabOutput::Ipc356),
            include_through_hole: self.include_through_hole,
            layers: self.layers.clone(),
            date: None,
            gerber_format: self.gerber,
            pick_place_format: self.pick_place,
            bom_format: self.bom.clone(),
            revision: self.revision.clone(),
        }
    }
    /// The folder the set goes to, relative to the board's folder: `fab/<board>-<revision>`
    /// (`fab/<board>` without a revision) unless the settings name one.
    pub fn folder_or_default(&self, board: &str) -> String {
        self.folder.clone().filter(|f| !f.trim().is_empty()).unwrap_or_else(|| match self.revision.as_deref().map(str::trim).filter(|r| !r.is_empty()) {
            Some(rev) => format!("fab/{board}-{rev}"),
            None => format!("fab/{board}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_name_their_outputs_and_a_change_makes_custom() {
        let fab = FabPreset::FabHouse.outputs();
        assert!(fab.contains(&FabOutput::Gerber) && fab.contains(&FabOutput::Ipc356) && !fab.contains(&FabOutput::Bom));
        assert_eq!(FabPreset::of(&fab), FabPreset::FabHouse);
        assert_eq!(FabPreset::of(&FabPreset::Assembly.outputs()), FabPreset::Assembly);
        assert_eq!(FabPreset::of(&FabOutput::ALL.into()), FabPreset::Full);
        let mut custom = fab.clone();
        custom.insert(FabOutput::Odb);
        assert_eq!(FabPreset::of(&custom), FabPreset::Custom);
        let req = FabSettings::default().request();
        assert!(req.gerber && req.drill && !req.bom && !req.odb);
        assert_eq!(FabOutput::from_word("drill-map"), Some(FabOutput::DrillMap));
    }
}
