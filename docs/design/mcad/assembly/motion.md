---
description: Reference for motion studies in a RunMat assembly. Drive a revolute, slider, cylindrical, distance or angle mate through a range over time, simulate the mechanism under gravity, springs and motors, send joint loads to a simulation, play the solved motion, plot speed and acceleration, check the motion for collisions, and save named mate positions.
keywords: [motion study, animation, mechanism, drive a mate, kinematics, dynamics, gravity, spring, motor, joint reaction, motion loads, collision detection, speed plot, acceleration, cycloidal motion]
---
# Motion studies

A motion study moves a mechanism by driving one of its mates through a range of values over time. RunMat solves the mechanism's position at every frame with the same mate solve that places the assembly, so each frame the Desktop plays is a real pose of the mechanism. It then reports the drive value and the speed and acceleration of a component you choose at each frame.

This hinge turns an arm a quarter turn on its base over 2 seconds:

```part file=block.part
runmat 0.7
std 1

part block {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])
        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])
        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])
        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 15 mm)
    output block = { body = feature.body.body }
}
```

```assembly file=hinge.assembly
runmat 0.7
std 1

assembly hinge {
    use part block as base
    use part block as arm {
        fixed = false
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 15 mm] } }
    }
    connector pivot = { occurrence = [{ name = base }], anchor = { anchor = explicit, offset = { translation = [0 mm, 0 mm, 15 mm] } } }
    connector root = { occurrence = [{ name = arm }], anchor = { anchor = origin } }
    mate hinge = { kind = { mate = revolute }, a = pivot, b = root }

    motion open_close = { duration = 2 s, drives = [{ mate = hinge, from = 0 deg, to = 90 deg }], law = cycloidal, track = arm }
}
```

## Define a study

`motion <name> = { … }` takes:

| Field | Meaning |
| --- | --- |
| `duration` | How long the motion takes, as a time (`2 s`). |
| `drives` | The driven mates. Each one names a `mate` and the values it moves `from` and `to`. |
| `law` | `linear` (the default) moves at constant speed. `cycloidal` starts and stops at rest, with zero speed and acceleration at both ends. |
| `track` | The component whose speed and acceleration are plotted. Without it, the first moving component is plotted. |
| `rate` | Solved frames per second, 12 by default. A study solves at most 241 frames. |

A drive's values set the mate's own value:

- an angle drives a revolute mate, an angle mate or a cylindrical mate's turn;
- a length drives a slider mate, a distance mate or a cylindrical mate's travel.

Fasten, planar, ball and the other mate kinds have no value to drive, and a study that names one reports a problem.

## How the motion is solved

At each frame time, RunMat sets each driven mate to its value for that time and solves the mates for every component's position. Each solve starts from the previous frame's pose, so a linkage stays on the branch it started on. Speed and acceleration are the derivatives of the solved positions over time, computed with central differences. The plotted point is the center of the tracked component's bounding box.

When a frame doesn't solve, the study ends at the previous frame and its row reports the time and the reason. A drive that pulls a linkage past the position it can reach, or that conflicts with another mate, ends the study this way.

## Play a study

In the Desktop, **Animation study** on the Motion tab adds a study: pick the mate, choose Angle or Distance, and enter the range, duration and motion. The Outline lists each study under **Motion studies**. Selecting a study shows it in the view, and the pane under the canvas plays it:

- **Play** steps through the solved frames in real time, and the slider moves to any frame.
- The pane shows the time, the drive value and the tracked component's speed at the frame.
- The plot draws the drive value, speed and acceleration over the whole study, and the view draws the tracked component's path.

The tabs at the bottom of the view switch between **As placed** and each study. To change a study, start Animation study from its row. Delete removes it.

## Check for collisions

**Collision detection** checks a study for components that meet as they move. It places the moving components at every solved frame and checks their exact solids against the rest of the assembly:

- **Stop** reports the first frame where components that started apart meet, and ends the check there.
- **Highlight** checks every frame and lists each pair that touches, the times it starts and stops touching, and its largest overlap volume.

Pick components under **Check** to check only their pairs. Pairs that already overlap at the start of the study are listed as warnings.

## Drag through the remaining freedom

**Drag through DOF** on the Motion tab moves a component as far as its mates let it. Pick a floating component, then drag the handle at its center. The mates hold the whole time: the component goes as near the handle as they allow, and the components mated to it follow. For example, an arm on a revolute mate dragged sideways swings about the hinge, and a pull along the hinge axis doesn't move it. The view shows the solved pose while you drag, and Apply keeps it.

The X, Y and Z fields hold how far the handle was pulled, in millimeters, so a drag can also be typed. With **Stop at collisions** on, a drag that would push the component into another one it started clear of stops at the last pose before they meet, and the message names the pair.

**Move component** in **Free drag** mode drags the same way. Each component the drag moves keeps its new pose as its placement, and the mate solve starts from that pose next time.

## Save mate positions

A mate position saves values for a set of mates under a name, so the mechanism can move between named states such as open and closed:

```text
position closed = { values = [{ mate = hinge, value = 0 deg }] }
```

**Mate controller** on the Motion tab works with positions:

- **Save values** keeps the picked mates' current values under the position's name. Each mate needs a value to save, such as a revolute mate's angle. Saving under an existing name replaces that position, and saving from a position's row with no mates picked saves its mates' values again.
- **Move to position** sets each of the position's mates to its saved value. The mate solve then moves the mechanism there, and Undo moves it back.

The Outline lists positions under **Mate positions**. Start Mate controller from a position's row to move to it, and use Delete to remove it.

## Simulate the mechanism under its loads

A dynamic study lets the mechanism move under gravity, springs and motors instead of driving a mate through set values. RunMat integrates the mechanism's equations of motion over the study's duration: the mates are the joints, and each moving component carries the mass and inertia of its parts, computed from their bodies and materials. Every part that moves needs a material.

This study releases the arm and lets it swing under gravity along −Y for 2 seconds, with a torsion spring and damper on the hinge:

```text
motion fall = { kind = dynamic, duration = 2 s, gravity = [0 m/s^2, -9.80665 m/s^2, 0 m/s^2],
                springs = [{ mate = hinge, stiffness = 0.05 N*m/rad, damping = 0.0005 N*m*s/rad, free = 0 deg }] }
```

| Field | Meaning |
| --- | --- |
| `kind = dynamic` | Integrates the motion under the loads. A dynamic study has no drives. |
| `gravity` | Three accelerations, `[x, y, z]`. The default is `[0, 0, -9.80665] m/s^2`. |
| `springs` | Springs and dampers on revolute, slider or cylindrical mates. Each names a `mate`, a `stiffness`, a `damping` and the `free` value where the spring is unloaded (0 by default). On a revolute mate they act as a torque about its axis; on a slider, as a force along it. |
| `motors` | A constant `torque` about a revolute mate's axis, or a constant `force` along a slider's. |
| `rate` | Solved frames per second, 30 by default. |

The integration uses a fixed step of at most 2 ms and holds the joints with constraint stabilization and projection, so an undamped mechanism keeps its energy: over ten periods of a pendulum check, the energy drifts by less than one millionth of m·g·d, where m is the mass, g gravity and d the distance from the hinge to the center of mass. Joint limits are not enforced during a dynamic study, and components pass through each other: contact is not modeled. A study on a mechanism with joint limits lists that as a problem.

In the Desktop, **Basic motion** on the Motion tab adds a dynamic study: enter the name and duration, pick the gravity direction, and optionally pick a mate for a spring, damper and motor. The study plays like any other, and its plot adds the first mate's reaction force on the component it moves.

## Send loads to a simulation

Each frame of a dynamic study records every mate's reaction on the components it joins: a force and a moment about the mate's connector. **Results to .fea** hands those loads on one part, at one instant, to a new structural simulation:

1. Start Results to .fea from a dynamic study's row (or name the study).
2. Enter the time. The loads come from the solved frame nearest that time.
3. Pick the part to simulate, or leave it empty for the first part a mate loads.

RunMat turns the loads into the part's own axes at that frame and starts the Simulation setup guide with them, as **New study** does. The guide asks for the supports, material and acceptance limit, and writes the `.fea` file. The request lists, for each mate, the connector's position in millimeters, the force in newtons and the moment in newton-meters, and gravity in the part's axes.

The joint loads hold the part in balance with its weight and its inertia at that instant. A static study of the part under the joint loads alone is out of balance unless it also carries the part's weight and the acceleration loads at that time.

## Next

- [Occurrences and mates](/docs/mcad/assembly/mates) lists every mate kind and its values.
- [Evaluate](/docs/mcad/evaluate) checks interference and clearance in the assembly's current pose.
