# Third-party KiCad boards

These are real KiCad 7 and 8 projects (board formats `20221018` and `20240108`) from other
authors. They are copied unchanged, and each one's license allows that.

- `tests/kicad_third_party.rs` reads each board, writes it back, and checks that the written
  file has the same tokens. It then exports the board from RunMat's own model and checks that
  every pad comes back where it was, on the same net.
- `tests/kicad_third_party_sch.rs` reads each schematic hierarchy into RunMat's model, checks
  that every part on the board is in it, exports it back to KiCad and checks that the export
  reads back with the same parts and the same nets.

| File | Board | Source | Commit | License |
|---|---|---|---|---|
| `atmega328p-datalogger-4l.kicad_pcb` | ATmega328P datalogger with 512 KB storage, 4 layers | [JMVI/KiCAD-PCB-Design-Examples](https://github.com/JMVI/KiCAD-PCB-Design-Examples), `ATMega328P-512K-Datalogger-4L/ATMega328P-512K-Datalogger-4L.kicad_pcb` | `d29c9ea257d068571b0bb1f7b1b92bac1368b5f1` | MIT, Copyright (c) 2025 Jaime Manuel Villegas Issa |
| `unipa-bioamp-pro.kicad_pcb` | UniPa BioAmp Pro, a biopotential (ECG/EMG/EEG) analog front end | [Mustafa-Yagci/unipa-bioamp-pro](https://github.com/Mustafa-Yagci/unipa-bioamp-pro), `hardware/UniPa_BioAmp_Pro.kicad_pcb` | `367965f53d1c62c64423019f24b6b5f661149be1` | MIT, Copyright (c) 2026 Mustafa Yağcı |
| `esp32-flight-controller/` (board and schematic, KiCad 8) | ESP32 drone flight controller, rev 2, 4 layers | [shazmeenShz/ESP32-Flight-Controller](https://github.com/shazmeenShz/ESP32-Flight-Controller), `Rev2/KiCad/flight_controller_rev2.kicad_pcb` and `.kicad_sch` | `d6dfbbb90a5bd5004c119a9a1ffc37b7ca904f28` | MIT, Copyright (c) 2026 Shazmeen Siddiqui |
| `ottercast-audio-v2/` (board and five schematic sheets, KiCad 7) | OtterCast Audio V2, a network audio receiver, 4 layers | [Ottercast/OtterCastAudioV2](https://github.com/Ottercast/OtterCastAudioV2), `OtterCastAudioV2.kicad_pcb`, `OtterCastAudioV2.kicad_sch`, `Audio.kicad_sch`, `MIPI.kicad_sch`, `Power.kicad_sch` and `WifiBT.kicad_sch` | `199395d7249aaf12432e42bff828dd665546d6a4` | MIT, Copyright (c) 2024 Ottercast, Jana Marie Hemsing |

Each MIT license asks that the copyright notice and the permission notice go with copies.
Here is the notice, which is the same in all four repositories apart from the copyright line:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
> BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Before adding a board here, check its license file at the commit you copy from. Use only
permissive licenses: MIT, BSD, Apache-2.0, CC0 or CERN-OHL-P. Share-alike licenses such as
CC BY-SA, CERN-OHL-S and the GPL are not allowed. Then add a row to the table above.
