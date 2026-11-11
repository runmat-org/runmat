---
description: RunMat ECAD footprints. Name a standard package with a generator string and get its IPC-7351B land pattern, or declare a custom footprint in a library .circuit file and use it by name.
keywords: [footprints, land patterns, IPC-7351, footprint generator, QFN footprint, SOIC footprint, custom footprint, footprint library, symbols, library .circuit]
---
# Footprints

A part's `package` names its footprint. For a standard package, write a generator string such as `"QFN24_4mm_P0.5_EP2.6"`: RunMat generates the land pattern from the IPC-7351B equations, with pads, courtyard, silkscreen with a pin-1 mark, fabrication outline and a 3D body. For a package no generator family covers, such as a module or an unusual connector, declare a `footprint` in a library `.circuit` file and name it: `package = gps_lcc12`.

Use a generator string whenever a family below fits. A custom footprint is for the rest.

## Generator strings

Each string names the family and every dimension the generator needs. Letters are not case-sensitive, dimensions are in mm, and the `mm` after a body size is optional. When a circuit is printed, each string is written in the canonical form shown here.

| Family | String | Example | Pads |
|---|---|---|---|
| Chip resistor, capacitor, inductor, diode, fuse, LED (imperial size code 01005 to 2512) | `R`, `C`, `L`, `D`, `F` or `LED`, then the size | `R0402`, `C0603`, `LED0805` | 1, 2 |
| SOIC (3.9 mm body; `W` for the 7.5 mm body) | `SOIC<pins>`, `SOIC<pins>W` | `SOIC8`, `SOIC16W` | 1 to pins |
| TSSOP (4.4 mm body, 0.65 mm pitch) | `TSSOP<pins>` | `TSSOP20` | 1 to pins |
| MSOP (3.0 mm body, 0.5 mm pitch) | `MSOP<pins>` | `MSOP10` | 1 to pins |
| QFP (square body) | `QFP<pins>_<body>mm_P<pitch>` | `QFP48_7mm_P0.5` | 1 to pins |
| QFN (square body, optional exposed pad) | `QFN<pins>_<body>mm_P<pitch>[_EP<pad>]` | `QFN24_4mm_P0.5_EP2.6` | 1 to pins, then the exposed pad |
| DFN (two rows, optional exposed pad) | `DFN<pins>_<w>x<l>mm_P<pitch>[_EP<w>x<l>]` | `DFN8_3x3mm_P0.65_EP1.6x2.4` | 1 to pins, then the exposed pad |
| BGA (full array) | `BGA<balls>_<rows>x<cols>_P<pitch>[_BALL<diameter>]` | `BGA256_16x16_P1` | `A1`, `A2`, … by row and column |
| SOT-23 (3, 5 or 6 pins) | `SOT23-<pins>` | `SOT23-6` | 1 to pins |
| SOT-223 | `SOT223` | `SOT223` | 1 to 3, tab 4 |
| Pin header, through-hole | `PinHeader<cols>x<rows>[_P<pitch>]` (pitch 2.54 when left out) | `PinHeader1x04`, `PinHeader2x05_P2` | 1 to pins, pin 1 square |
| SMD connector, one row | `SMDConn<pins>_P<pitch>[_Pad<w>x<l>][_MP<w>x<l>]` | `SMDConn4_P1_MP1.2x1.8` | 1 to pins, then `MP1`, `MP2` |
| Plated mounting hole | `MountingHole_<drill>mm[_Pad<diameter>]` | `MountingHole_3.2mm_Pad6` | 1 |

Pads number in the package's standard order: pin 1 at the top left, then counter-clockwise for QFN and QFP. A part maps its pins to these pad numbers, so a QFN24 with an exposed pad has pads 1 to 25 and its ground pin usually lists `"25"` among its pads.

The density suffix picks the IPC-7351B land size: `@A` is most material (hand soldering, rework), `@C` is least material (dense boards), and the default is nominal (`@B`). `C0402@A` and `QFN32_5mm_P0.5_EP3.3@C` are examples. Pin headers, SMD connectors and mounting holes have one size and take no suffix.

The BGA ball diameter defaults to half the pitch, and the SMD connector's pads default to 0.6 of the pitch by 1.5 mm. A mounting hole's pad defaults to 1.875 times the drill, rounded to 0.01 mm.

## Custom footprints

A `footprint` declaration is the full land pattern: pads, courtyard, silkscreen, fabrication drawing and 3D body. Put each one in its own library file under `electrical/library/footprints/`, so every board in the project can use it:

```circuit file=electrical/library/footprints/gps_lcc12.circuit
runmat 0.7
std 1

footprint gps_lcc12 = {
    description = "12-pad LCC GPS module, 10.1 x 9.7 mm",
    pads = [
        { number = "1", shape = rect, size = [1.5, 0.8], at = [-4.3, 3.175], layers = smd },
        { number = "2", shape = rect, size = [1.5, 0.8], at = [-4.3, 1.905], layers = smd },
        { number = "3", shape = rect, size = [1.5, 0.8], at = [-4.3, 0.635], layers = smd },
        { number = "4", shape = rect, size = [1.5, 0.8], at = [-4.3, -0.635], layers = smd },
        { number = "5", shape = rect, size = [1.5, 0.8], at = [-4.3, -1.905], layers = smd },
        { number = "6", shape = rect, size = [1.5, 0.8], at = [-4.3, -3.175], layers = smd },
        { number = "7", shape = rect, size = [1.5, 0.8], at = [4.3, -3.175], layers = smd },
        { number = "8", shape = rect, size = [1.5, 0.8], at = [4.3, -1.905], layers = smd },
        { number = "9", shape = rect, size = [1.5, 0.8], at = [4.3, -0.635], layers = smd },
        { number = "10", shape = rect, size = [1.5, 0.8], at = [4.3, 0.635], layers = smd },
        { number = "11", shape = rect, size = [1.5, 0.8], at = [4.3, 1.905], layers = smd },
        { number = "12", shape = rect, size = [1.5, 0.8], at = [4.3, 3.175], layers = smd }
    ],
    courtyard = rect([-5.3, -5.1], [5.3, 5.1]),
    silk = [line([-5.05, 4.85], [5.05, 4.85]), line([-5.05, -4.85], [5.05, -4.85]), circle([-5.6, 4.1], 0.15, fill = true)],
    fab = [rect([-5.05, -4.85], [5.05, 4.85]), text([0, 0], "${REFERENCE}")],
    body = { height = 2.4, model = "gps_module.step" }
}
```

Lengths are in mm, with x to the right and y up, and the footprint's origin is its placement point.

| Field | Meaning |
|---|---|
| `description` | one line on what the footprint is |
| `pads` | one record per pad (below) |
| `courtyard` | the keep-out outline: `rect([x, y], [x, y])`, `polygon([[x, y], …])`, `circle([x, y], r)`, or `line` and `arc` drawings that join end to end into a closed outline |
| `silk` | silkscreen drawings, 0.12 mm wide unless a drawing gives `width` |
| `fab` | fabrication-layer drawings, 0.10 mm wide unless a drawing gives `width` |
| `body` | the 3D body: `height`, optional `size = [w, l]` (the fab outline's extent when left out), and `model`, a STEP file or an MCAD `.part` |

A pad record takes:

| Field | Meaning |
|---|---|
| `number` | the pad number a part's pins land on, quoted (`"1"`, `"A1"`, `"MP1"`); a mechanical hole may leave it empty |
| `shape` | `rect`, `roundrect` (with `ratio`, the corner radius as a fraction of the shorter side, 0.25 when left out), `circle`, `oval`, `trapezoid` (with `delta`), `chamfered` (with `chamfer` and `corners`) or `custom` (with `polygon` and optional `holes`) |
| `size` | `[width, height]`, before rotation |
| `at` | the pad centre |
| `rotation` | degrees, counter-clockwise |
| `layers` | `smd` (top copper, paste and mask), `through` (every copper layer and mask, plated), `hole` (an unplated hole) or `edge` (edge-connector fingers: copper and mask, no paste) |
| `drill` | the hole diameter for `through` and `hole` pads, or `{ diameter, slot }` for a slot |
| `side` | `back` for an SMD pad on the bottom |
| `paste`, `mask` | `false` to leave out the paste or mask opening |
| `mask_expand`, `paste_expand` | the opening's margin beyond the pad, in mm; negative shrinks it |
| `thermal` | `true` for an exposed thermal pad, which connects to its zone solidly |

Drawings are `line([x, y], [x, y], …)`, `rect([x, y], [x, y])`, `circle([x, y], r)`, `arc([start], [mid], [end])`, `polygon([[x, y], …])` and `text([x, y], "words", size = 1)`. Each takes `width`, and the closed shapes take `fill = true`.

A generated package prints as the same declaration, so you can start from the nearest standard package and edit it. The footprint wizard writes a generator string when you keep its values, and a full declaration when you edit one by hand [not yet in the current release](pending:design.ecad.footprint-wizard).

## Footprint checks

A library file compiles on its own, and its checks name the footprint, part or pad they are about:

- pad numbers are unique within a footprint;
- the courtyard is closed and contains every pad;
- each drill is smaller than its pad;
- every pin of a part lands on a pad its footprint has;
- every pad has a pin, or the part lists the pad in `nc`.

For example, a part that maps 10 of the module's 12 pads and leaves the other two unlisted is reported on the part:

```text
part `GPS-M12`: its footprint has 12 pads and its pins land on 10; pads `10`, `11` have no pin (map each to a pin or list them in `nc`)
```

## Libraries

A `.circuit` file with no `circuit` block is a library. It holds `footprint`, `symbol` and `part` declarations in any mix. A file uses a library with `use "<path>.circuit"`, with the path relative to the file, and a library can use other libraries:

```circuit file=electrical/library/parts.circuit
runmat 0.7
std 1

use "footprints/gps_lcc12.circuit"

part "GPS-M12" = {
    manufacturer = "Sample Catalog",
    description = "GPS receiver module, 12-pad LCC",
    category = module,
    package = gps_lcc12,
    pins = [
        { name = VCC, pad = "1", type = power_in },
        { name = GND, pads = ["2", "6", "7", "12"], type = power_in },
        { name = TXD, pad = "3", type = output },
        { name = RXD, pad = "4", type = input },
        { name = PPS, pad = "5", type = output },
        { name = RESET, pad = "8", type = input },
        { name = ANT, pad = "9", type = passive }
    ],
    nc = ["10", "11"]
}

part "ADC-12B-QFN24" = {
    manufacturer = "Sample Catalog",
    description = "12-bit 4-channel ADC, I2C",
    category = adc,
    package = "QFN24_4mm_P0.5_EP2.6",
    pins = [
        { name = VDD, pad = "1", type = power_in },
        { name = GND, pads = ["2", "25"], type = power_in },
        { name = AIN0, pad = "3", type = input },
        { name = SCL, pad = "4", type = input },
        { name = SDA, pad = "5", type = bidirectional }
    ],
    nc = ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"]
}
```

A board names the parts by MPN as it names built-in ones:

```circuit file=electrical/sensor/sensor.circuit
runmat 0.7
std 1

use "../library/parts.circuit"

circuit sensor {
    sheet main {
        component U1 = { part = "GPS-M12" }
        component U2 = { part = "ADC-12B-QFN24" }
        component P1 = { part = "PAD-WIRE-20AWG" }
        net GND = { global = true, power_flag = true }
        net "+3V3" = { global = true, power_flag = true }
        connect = [net.GND, U1.GND, U2.GND]
        connect = ["net.+3V3", U1.VCC, U2.VDD, U1.RESET]
        connect = [net.SCL, U2.SCL, U1.TXD]
        connect = [net.SDA, U2.SDA, U1.RXD]
        connect = [net.SENSE, U2.AIN0, "P1.1"]
        no_erc = [U1.PPS, U1.ANT]
    }
}
```

Names resolve the same way every time:

- Parts are named by MPN, footprints and symbols by identifier.
- `use` lines are followed in order, depth first, and each file is read once, so two libraries that both use a third see the same declarations.
- A name declared in two files is an error that names both files. That includes a file's own declaration and a used one: rename one or remove one.
- Libraries that use each other in a cycle are an error that lists the chain.
- A `use` line whose file can't be read is an error that names the path. In the browser, a library that hasn't loaded yet says so on its `use` line, and the file is checked again when it arrives.
- A declared part replaces a built-in part of the same MPN.

The standard project layout keeps libraries in `electrical/library/`: one file per custom footprint in `footprints/`, and parts in `parts/` or one `parts.circuit`.

## Symbols

A part's schematic symbol is generated from its pins unless the part names a `symbol` declaration. A symbol lists its pins by name, and the part's pins map to them by name, so one symbol can serve every part with the same pin names:

```circuit fragment
symbol buffer = {
    body = [polygon([[-2.54, -2.54], [-2.54, 2.54], [2.54, 0]])],
    pins = [
        { name = IN, at = [-5.08, 0], dir = right },
        { name = OUT, at = [5.08, 0], dir = left },
        { name = VCC, at = [0, 5.08], dir = down }
    ]
}
```

A pin's `at` is the end that wires connect to, and `dir` is the way the pin runs from there toward the body: `right`, `up`, `left` or `down`. `length` is 2.54 mm unless given. A part names the symbol with `symbol = buffer`. Every part pin needs a symbol pin of the same name, and every symbol pin needs a part pin.

## 3D bodies

A footprint's `body` gives its 3D body. A part can give its own with `body = "<file>"`, a STEP file or an MCAD `.part`, which replaces the footprint's model for that part only.

## Foreign footprint files

A part's `package` can also name a `.kicad_mod` file, which is read in place. Use this for a one-off file you received; for a footprint the project keeps, convert it to a `footprint` declaration. See [Import](/docs/ecad/import).
