---
description: Reference for the checks on a RunMat circuit and board. The electrical rules and their pin-type matrix, the design-review suites for decoupling, I2C, crystals, straps, USB-C and switching regulators, the board checks, and running them all from the command line and in CI.
keywords: [ERC, electrical rules check, design review, decoupling check, I2C pull-up check, crystal load capacitance, USB-C, DRC, circuit check, CI]
---
# Rules and checks

A circuit check elaborates the design and runs the electrical rules:

| Code | Severity | Finds |
|---|---|---|
| `pin-conflict` | from the pin-type matrix | two pin types on one net that may not meet |
| `multiple-drivers` | error | more than one push-pull driver on a net |
| `power-not-driven` | error | power inputs on a net with no power output and no power flag |
| `input-not-driven` | warning | inputs on a net nothing drives |
| `pin-unconnected` | warning | a pin on no net, not no-connect and not in `no_erc` |
| `single-pin-net` | warning | a net with one pin |
| `no-connect-connected` | error | a no-connect pin on a net with other pins |

Design-review suites check circuits the way a reviewer would, each finding with a typed fix:

| Suite | Checks |
|---|---|
| `decoupling` | every IC power pin has a capacitor from its rail to ground |
| `i2c` | each SCL and SDA net has a pull-up of 1 kΩ to 10 kΩ |
| `crystal` | each crystal pin has a load capacitor, and the load matches the crystal within 10 % |
| `straps` | reset and boot pins are pulled to a defined level |
| `usb-c` | a USB-C sink has 5.1 kΩ pull-downs on CC1 and CC2, and D+ and D− are paired |
| `switcher` | a switching regulator has its inductor, capacitors, and a feedback divider within 3 % of the rail |

A board check adds DRC (see [Board layout](/docs/ecad/layout#design-rules)), unrouted connections and schematic changes the board has not taken. 

## Leaving pins alone

`no_erc = [U1.NC, "J1.5"]` on a sheet tells the rules those pins are meant as they are, such as a no-connect pin left open.

## Running the checks

```sh
runmat-design check board/sensor.circuit.layout
runmat check board/sensor.circuit
```

`runmat-design check` and `runmat check` run every rule, suite and simulation measure on each `.circuit` or `.circuit.layout` named, print each finding and each measure with its verdict, and exit with status 1 on an error, a DRC violation, an unrouted connection, a schematic change the board hasn't taken, or a failed measure. `--no-simulate` skips the simulations. `runmat test` runs them as tests beside `.m` tests. See [Checks and diagnostics](/docs/design-language/checks-and-diagnostics#checking-from-the-command-line).
