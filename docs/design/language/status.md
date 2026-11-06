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

**Values with units across the .m boundary.** A design value function receives quantities for parameters declared quantity.<Kind> in its arguments block and may return a quantity, which the call converts and checks without `as`.

## MCAD

### design.mcad.mate.cam

**Cam mates.** An assembly writes kind = { mate = cam, profile = <published profile> }; the solver keeps the follower point on the cam's profile path as the cam turns (a path-tangent residual), with its Jacobian checked.

### design.mcad.drawing.auto

**Drawings laid out automatically.** A command writes a .drawing for a part or an assembly with standard views, dimensions from its PMI and a title block, and the result checks clean.

### design.mcad.drawing.standards

**Drawing standards.** A sheet format selects an ISO or ASME standard that sets line weights, text, arrowheads and tolerance display in the Desktop sheet and the DXF, DWG and PDF exports.

### design.mcad.evaluate.surface-analysis

**Surface and shape analysis.** Evaluate runs section, thickness, draft and undercut, curvature, zebra and deviation analyses on a part in the Desktop and through design.inspect.

### design.fea.part-geometry

**Studies that name a design.** A .fea geometry block takes part = <name> or assembly = <name> with inputs, resolved through the folder scope, and the study regenerates the design at those inputs without a STEP export.

## ECAD

### design.ecad.m-values

**Circuit component values from .m functions.** A .circuit component value or param may call a .m function in the circuit's folder, with the same resolution, units, determinism and cache rules as a .part value.

### design.ecad.enclosure

**Boards in their enclosures.** An assembly that places a board and its enclosure checks fit and clearance between the board's components and the enclosure's parts.

### design.ecad.board-studies

**Board studies through the pcb source.** A .fea study reads a board's stackup, copper and component power through its pcb source and runs thermal and vibration studies of it.

### design.ecad.fab.assembly-drawing

**Assembly drawings in the fabrication set.** The fabrication set writes an assembly drawing per side with outlines, designators and polarity marks, as Gerber and PDF.

### design.ecad.import.other-tools

**Imports from other electronics tools.** Clean-room readers for the other common tools' binary and text schematic and board formats, EAGLE XML and EasyEDA JSON write circuits and boards with part declarations and a report of what did not carry over, tested on an openly licensed corpus.

### design.ecad.import.kicad-roundtrip

**KiCad round-trip gate.** A corpus of KiCad projects reads in and writes back with hierarchy, symbols, footprints, net classes and rules intact, and the not-carried report is empty or explained.

### design.ecad.use-library

**Shared part libraries.** A circuit writes use library <name> to read part declarations from a shared library package, and part declarations propagate through circuits it uses.

## Tools

### design.tools.cli-check-mcad

**runmat-design check on parts, assemblies and drawings.** runmat-design check runs design.check on .part, .assembly and .drawing files, prints each diagnostic and exits 1 on any error.

### design.tools.runmat-check-design

**runmat check and runmat test on design files.** runmat check and runmat test delegate .part, .assembly, .drawing, .circuit and .circuit.layout files to the design checks, beside .m and .fea files.

### design.tools.export-set

**Input values for one export.** design.export takes set (input values for one run), with minimum and maximum range checks, so a study's geometry can come from a part at chosen inputs.
