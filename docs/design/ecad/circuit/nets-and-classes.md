---
description: Reference for nets in a RunMat circuit. Connect lists and endpoints, net settings (global, power flag, class), buses, interfaces and harnesses, differential pairs, and net classes whose track width and clearance the board's design rules use.
keywords: [nets, connect, net class, power flag, global net, bus, harness, interface, differential pair, track width, clearance]
---
# Nets and net classes

## Connections

`connect = [<endpoints>]` joins every endpoint it lists into one net. A `net.<name>` endpoint in the list names the net.

| Endpoint | Names |
|---|---|
| `U1.VIN` | a pin by name |
| `"C1.1"` | a pin by number |
| `net.GND`, `"net.+3V3"` | a net; quote a name with characters outside letters, digits and `_` |
| `port.IN` | a port of the sheet |
| `bus.D.D0` | a member of a bus |
| `harness.USB.DP` | a member of a harness |
| `instance.pwr.VIN` | a port of a placed block |

## Net settings

A net joins pins. `global = true` joins nets of the same name across every sheet, as power rails and ground do; `power_flag = true` marks a net as driven by a source outside the schematic, such as a connector. A net class carries the rules its nets share:

```circuit fragment
net_class power = { track_width = "0.40 mm", clearance = "0.20 mm" }
net_class analog = { clearance = "0.20 mm" }
interface usb2 = { members = [DP, DN], diff_pairs = [[DP, DN]] }
```

The board reads each class's track width and clearance, and DRC checks them.

A `blanket` gives several nets on a sheet one class and set of parameters:

```circuit fragment
blanket rails = { nets = [VIN, GND, "+3V3"], class = power, max_length = 50 mm }
```

Each listed net takes the class unless its own `net` statement names one; ERC notes each net that keeps a class of its own (`blanket-class-kept`) and warns about a listed net with no pins (`blanket-net-empty`). A name that isn't a net on the sheet is an error. The schematic draws a dashed box round the listed nets' wires, labels and pins, captioned with the blanket's class and parameters.

## Buses, interfaces and harnesses

`bus D = [D0, D1, D2, D3]` groups nets under one name; `bus.D.D2` is one member. An `interface` is a bundle type with named members and differential pairs, and `harness USB = usb2` declares a harness of that type, whose members connect as `harness.USB.DP`.

## Differential pairs

`diff_pair USB = { p = net.DP, n = net.DN }` pairs two nets. The board routes the pair together and checks its gap, its uncoupled length and its skew. See [Board layout](/docs/ecad/layout#design-rules).
