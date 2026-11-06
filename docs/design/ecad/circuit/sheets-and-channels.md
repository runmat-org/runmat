---
description: Reference for sheets in a RunMat circuit. The root sheet, sheet ports and their directions, sheets placed as blocks with instance, channels repeated with repeat and bound to buses, and how designators and nets are named in each copy.
keywords: [schematic sheets, hierarchical schematic, sheet ports, sheet symbol, multi-channel design, repeat, channels, bus binding, hierarchy]
---
# Sheets, hierarchy and channels

A circuit is drawn on one or more sheets. `root = <sheet>` names the top sheet; every other sheet is placed in it, directly or through other sheets, as a block. A sheet placed several times is a set of channels, each with its own components and nets.

## Ports

A sheet's `port` statements are the pins of its block:

| Form | Declares |
|---|---|
| `port IN : in` | an input |
| `port OUT : out` | an output |
| `port SDA : inout` | a bidirectional signal |
| `port SENSE : passive` | a passive connection, such as a sense resistor's node |
| `port VBAT : power` | a supply the parent provides |
| `port D : in = bus [D0, D1]` | a bus port |

Inside the sheet, `port.IN` is an endpoint of `connect`. A net marked `global = true`, such as ground and the supply rails, reaches every sheet without a port.

## Blocks

`instance <name> = { sheet = <sheet> }` places a sheet as a block, and the parent connects to its ports as `instance.<name>.<port>`:

```circuit fragment
sheet sensor {
    port SDA : inout
    port SCL : in
    component U2 = { part = "BARO-LGA8" }
    connect = [port.SDA, U2.SDA]
    connect = [port.SCL, U2.SCL]
}

sheet main {
    instance baro = { sheet = sensor }
    connect = [net.SDA, instance.baro.SDA]
    connect = [net.SCL, instance.baro.SCL]
}
```

## Channels

`repeat = N` on an instance places its sheet N times as channels: each copy gets its own designators and its own nets (`/ch[0]/FBK`, `/ch[1]/FBK`), and `bind` maps the sheet's ports onto the parent's nets or onto consecutive members of a bus.

```circuit fragment
sheet channel {
    port IN : in
    port OUT : out
    component U = { part = OPA4171 }
    component RF = { part = "R0402-10K" }
    connect = [port.IN, net.IN, "U.A.IN+"]
    connect = [net.FBK, "U.A.IN-", "RF.1"]
    connect = [port.OUT, net.OUT, U.A.OUT, "RF.2"]
}

sheet main {
    bus AIN = [AIN0, AIN1, AIN2]
    bus ADC = [ADC0, ADC1, ADC2]
    instance ch = { sheet = channel, repeat = 3, bind = { IN = { bus = AIN, offset = 0 }, OUT = { bus = ADC, offset = 0 } } }
}
```

Channel `k` binds `IN` to `AIN<k>` and `OUT` to `ADC<k>`. The board can place each channel in its own room, so a layout made for one channel can be copied to the rest.

## Names in a hierarchy

A component or net inside a block is named by its path: `/baro/U2`, `/ch[1]/RF`, `/ch[1]/FBK`. Board footprints link to components by these paths in the board's `sync` section, and a variant names a component in a channel the same way (`"/ch[0]/CD2"`). See [Names and selectors](/docs/design-language/names-and-selectors#pins-nets-and-designators).

A circuit can also use a sheet, an interface or a net class that a circuit in another folder declares, named by symbol with the folder it comes from. See [Folders, modules and use](/docs/design-language/modules).
