# The `pcb` geometry source: board → FEA contract (draft)

From: modeling (younger sister). For: FEA (middle sister), as an intent file.
Implementation: `runmat-circuit::fea` (`pcb_source`, `PcbFeaSource`) on `cad/foundations`.

## What the source provides

`pcb_source(board, netlist, pool, op_point)` returns one `PcbFeaSource` per board. It holds:

- **Outline**: the board outline polygon with cutouts, in board nanometres, y up.
- **Thickness**: the total stackup thickness, in metres.
- **Layers**, top to bottom. Each layer is copper, dielectric or mask, and carries:
  - `z_top` and `z_bottom` in metres, measured down from the top surface;
  - its material: Dk, Df, electrical conductivity, in-plane and through-plane thermal
    conductivity, density, heat capacity and CTE;
  - for copper layers, the fraction of the board area that copper covers. This gives an
    effective orthotropic conductivity where copper is not meshed explicitly.
- **Copper regions**: the copper area of each net on each layer, as polygons. It is the
  union of pads, tracks, vias and filled zones, clipped to the outline.
- **Via regions**: the barrels of each net, each with position, drill, depth span and
  plating thickness.
- **Component regions**: one per fitted footprint. Each has:
  - side and outline (the courtyard, otherwise the pad extent);
  - body height;
  - power in watts, and where that number came from;
  - θJC, θJB and Tj,max from the part library.
- **Power map**: region name → watts, for components whose power is known.
- **Lines** (field-solved impedance, PCB-42): one cross-section per controlled-impedance line
  (a net, or a declared pair, whose net class has an `impedance` or `diff_impedance` target)
  per copper layer and width it is routed at. Each carries:
  - its region `pcb/<board>/line/<net>/<layer>` (a pair: `<p>+<n>`) and canonical ID;
  - the nets, the class, the signal layer and its kind (`microstrip`, `stripline`);
  - the conductor's bottom width, a pair's edge-to-edge gap, the copper thickness and the
    class's `etch_factor` (absent: rectangular);
  - the reference planes above and below (copper layer names; none on a microstrip's open
    side);
  - the stack rows from the plane above (or the solder mask over an outer layer) to the plane
    below: kind, `z_top`, thickness, Dk and Df;
  - the closed-form Z0, Zdiff and delay (ps/mm), for the study to report beside its solve;
  - the routed length on that layer at that width.

## Region naming

Every name is a path under `pcb/<board>/`. Each segment is a name authored in the design:

- layer names come from the stackup;
- net names come from the netlist;
- designators come from the components.

Inside a segment, `/`, `%` and whitespace are percent-encoded. Nothing else changes, so
`/power/VOUT` becomes `%2Fpower%2FVOUT`.

| Region | Name |
|---|---|
| whole board | `pcb/<board>/board` |
| all copper of a layer | `pcb/<board>/copper/<layer>`, for example `copper/In1.Cu` |
| one net's copper on one layer | `pcb/<board>/copper/<layer>/net/<net>` |
| copper with no net | `pcb/<board>/copper/<layer>/net/%00unconnected` |
| dielectric | `pcb/<board>/dielectric/<stackup layer name>` |
| solder mask | `pcb/<board>/mask/<stackup layer name>` |
| via barrels of a net | `pcb/<board>/via/<net>` |
| component body | `pcb/<board>/component/<designator>` |
| a pad's copper on each layer it is on | `pcb/<board>/pad/<designator>.<pin>` |
| the board's top or bottom face | `pcb/<board>/surface/top`, `pcb/<board>/surface/bottom` |

**Canonical IDs.** `canonical_id(name)` is a UUID derived from the full region name. The
same authored name always gives the same ID. FEA matches on the ID, so no sidecar or alias
layer is needed. This follows the one-authority rule for region identity.

Renames follow the circuit's lock section. A designator or net rename recorded there as an
ECO rename changes the region name. A study that refers to the old name gets a
"renamed region" diagnostic carrying the new name, and never silently re-binds. (Required behaviour. It is not implemented yet: the circuit side emits the rename from its lock section, and the `.fea` side reports it.)

## Power map sources, in priority order

1. **Authored**: the `power` value on the component in `.circuit`.
2. **Operating point**: SPICE op-point dissipation (I·V over the part's pins), passed as
   `op_point`. `runmat-circuit-spice` will produce it. Metering goes through the
   operation meter (`runmat-simulation-usage`).
3. **Unknown**: the study must supply the value. The source says so (`PowerSource::Unknown`)
   rather than assuming zero.

## Units

- Lengths in the data structure: metres.
- Polygons: board nanometres. Multiply by 1e-9 for metres.
- Power: watts.
- θ: K/W.
- k: W/(m·K).
- Density: kg/m³.
- cp: J/(kg·K).

## What FEA is asked to add

- Ledger rows for the `pcb` source and the families it serves:
  - board thermal, steady and transient;
  - DC IR drop;
  - warpage.
- "Not yet in current release" flags until those families ship.
- A `.fea` form: `source pcb "board.circuit"`, with regions referenced by the names above.
- A 2D electrostatic study on a line's cross-section (C, and C0 with every dielectric at
  vacuum) giving Z0, Zdiff and Zcommon, delay and per-length L and C. Design shows the solved
  value beside the closed form in the layer-stack view once the study returns it.

## Open points for FEA

- Should copper be smeared per layer into effective k from `copper_fraction`, or meshed
  explicitly per net region? The contract carries both, so the study can choose.
- Via arrays: should we pre-compute an effective k_z per area, or leave that to FEA from the
  barrels?
- Two-resistor compact models (θJC/θJB): does FEA want the junction node as its own region
  (`component/<ref>/junction`)? Today the component region carries the values.
