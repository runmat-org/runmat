# Third-party KiCad boards

These are real KiCad 8 boards (file format `20240108`) from other authors. They are copied
unchanged, and each one's license allows that. `tests/kicad_third_party.rs` reads each board,
writes it back, and checks that the written file has the same tokens. It then exports the
board from RunMat's own model and checks that every pad comes back where it was, on the same net.

| File | Board | Source | Commit | License |
|---|---|---|---|---|
| `atmega328p-datalogger-4l.kicad_pcb` | ATmega328P datalogger with 512 KB storage, 4 layers | [JMVI/KiCAD-PCB-Design-Examples](https://github.com/JMVI/KiCAD-PCB-Design-Examples), `ATMega328P-512K-Datalogger-4L/ATMega328P-512K-Datalogger-4L.kicad_pcb` | `d29c9ea257d068571b0bb1f7b1b92bac1368b5f1` | MIT, Copyright (c) 2025 Jaime Manuel Villegas Issa |
| `unipa-bioamp-pro.kicad_pcb` | UniPa BioAmp Pro, a biopotential (ECG/EMG/EEG) analog front end | [Mustafa-Yagci/unipa-bioamp-pro](https://github.com/Mustafa-Yagci/unipa-bioamp-pro), `hardware/UniPa_BioAmp_Pro.kicad_pcb` | `367965f53d1c62c64423019f24b6b5f661149be1` | MIT, Copyright (c) 2026 Mustafa Yağcı |

Both MIT licenses ask that the copyright notice and the permission notice go with copies.
Here is the notice, which is the same in both repositories apart from the copyright line:

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
