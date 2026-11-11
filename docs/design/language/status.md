---
description: Design features that the RunMat design docs describe and the current release does not include yet, with what each needs to ship. Generated from the design feature ledger.
keywords: [feature status, roadmap, not yet released, RunMat MCAD, RunMat ECAD]
---
# Feature status

The design docs describe the design language as it is meant to work. Where the docs describe a feature the current release does not include yet, a “Not yet in” badge marks it and links here. Each entry below says what the feature needs before the mark comes off.

This page is generated from `docs/design/ledger.json`; edit the ledger, not this page.

## Design language

### design.language.volatile-lock

**Volatile values recorded in runmat.lock at release.** Releasing a design snapshots every volatile value it used into runmat.lock under [values], and regenerating the released design reads the snapshot.

### design.language.m-quantities

**Values with units across the .m boundary.** A design value function receives quantities for parameters declared quantity.&lt;Kind&gt; in its arguments block and may return a quantity, which the call converts and checks without `as`.

## MCAD

### design.mcad.drawing.standards

**Drawing standards.** A sheet format selects an ISO or ASME standard that sets line weights, text, arrowheads and tolerance display in the Desktop sheet and the DXF, DWG and PDF exports.

### design.mcad.evaluate.surface-analysis

**Surface and shape analysis.** Evaluate runs section, thickness, undercut, curvature, zebra and deviation analyses on a part in the Desktop and through design.inspect, coloring the model by result as draft analysis does.

### design.fea.part-geometry

**Studies that name a design.** A .fea geometry block takes part = &lt;name&gt; or assembly = &lt;name&gt; with inputs, resolved through the folder scope, and the study regenerates the design at those inputs without a STEP export.

## ECAD

### design.ecad.enclosure

**Boards in their enclosures.** An assembly that places a board and its enclosure checks fit and clearance between the board's components and the enclosure's parts.

### design.ecad.board-studies

**Board studies through the pcb source.** Board » Analysis » New study writes a .fea study beside the layout that reads the board's stackup, copper, pads and component power through FEA's pcb geometry source, and FEA checks and runs its thermal, IR drop and warpage studies. Until the source lands, FEA's check answers "pcb geometry source is not yet available" and the Desktop shows the study as planned, with the release flag.

### design.ecad.fab.assembly-drawing

**Polarity marks on assembly drawings.** The assembly drawing of each side marks pin 1 and the polarity of polarized parts (diodes, LEDs, polarized capacitors, ICs) beside their outlines and designators, in the Gerber and PDF.

### design.ecad.import.other-tools

**Imports from other electronics tools.** Clean-room readers for the other common tools' binary, text and JSON schematic and board formats write circuits and boards with part declarations and a report of what did not carry over, tested on an openly licensed corpus.

### design.ecad.import.kicad-roundtrip

**KiCad round-trip gate.** A corpus of KiCad projects reads in and writes back with hierarchy, symbols, footprints, net classes and rules intact, and the not-carried report is empty or explained.

### design.ecad.footprint-wizard

**Footprint wizard and footprint editor.** The wizard builds a footprint from a family and datasheet dimensions on the same generator as package strings, saves a generator string when no value was edited and a footprint declaration when one was, and reopens with its values; the footprint editor edits pads, courtyard, silkscreen and fab with undo.

### design.ecad.library-import

**Footprint, symbol and part library import and export.** A .kicad_mod, a .pretty folder, a .kicad_sym and an XML part library import to library .circuit files with a source note on each declaration, and export writes them back; round-trip tests cover a sample of each.

### design.ecad.field-solved-impedance

**Field-solved trace impedance beside the closed form.** A board study solves each controlled line's cross-section (from the pcb source's lines) for Z0, Zdiff, delay and per-length L and C, and the layer-stack view shows the solved value beside the closed form.

## Tools

### design.tools.runmat-check-design

**runmat check and runmat test on parts, assemblies and drawings.** runmat check and runmat test delegate .part, .assembly and .drawing files to the design checks, as they do .circuit and .circuit.layout files, beside .m and .fea files.

### design.tools.cad-matlab-api

**cad.evaluate, cad.study, cad.check and cad.inputs in MATLAB-language code.** A .m script regenerates a part or an assembly at input values with cad.evaluate and reads its mass properties, bounding box and published values; cad.study runs a .fea study at its inputs; cad.check judges limits and throws cad:limitFailed, which runmat test reports as a failure; cad.inputs lists the inputs; parfor runs one regeneration per worker.
