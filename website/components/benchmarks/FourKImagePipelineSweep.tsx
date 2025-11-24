type SweepRow = {
  batchSize: number;
  runMatMs: number;
  vsPyTorch: string;
  vsNumPy: string;
};

const SWEEP_ROWS: SweepRow[] = [
  { batchSize: 4, runMatMs: 143, vsPyTorch: "5.6× faster", vsNumPy: "3.5× faster" },
  { batchSize: 8, runMatMs: 213, vsPyTorch: "3.8× faster", vsNumPy: "4.4× faster" },
  { batchSize: 16, runMatMs: 242, vsPyTorch: "3.8× faster", vsNumPy: "7.4× faster" },
  { batchSize: 32, runMatMs: 389, vsPyTorch: "2.9× faster", vsNumPy: "9.3× faster" },
  { batchSize: 64, runMatMs: 684, vsPyTorch: "1.8× faster", vsNumPy: "10.2× faster" },
];

export default function FourKImagePipelineSweep() {
  return (
    <div className="mx-auto w-full max-w-[40rem] rounded-2xl border border-border/60 bg-background/60 overflow-hidden shadow-lg">
      {/* Header */}
      <div className="bg-[#0E1421] px-4 sm:px-6 py-4 border-b border-border/60">
        <div className="text-sm sm:text-base font-semibold uppercase tracking-wide text-gray-200">
          4K Image Pipeline
        </div>
        <div className="mt-1 text-xs sm:text-sm text-gray-400">
          Batch size sweep: 4 → 64 images
        </div>
      </div>

      {/* Table */}
      <div className="bg-[#0E1421]">
        <div className="overflow-x-auto">
          <table className="w-full text-sm sm:text-base">
            <thead className="border-b border-border/60 bg-muted/40 text-center">
              <tr>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center text-gray-400 font-medium">
                  Batch size (images)
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-gray-400">
                  RunMat (ms)
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-gray-400">
                  RunMat vs PyTorch
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-gray-400">
                  RunMat vs NumPy
                </th>
              </tr>
            </thead>
            <tbody className="bg-[#0E1421] text-gray-200">
              {SWEEP_ROWS.map((row) => (
                <tr
                  key={row.batchSize}
                  className="border-b border-border/40 last:border-b-0"
                >
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="font-mono text-sm sm:text-base">
                      {row.batchSize}
                    </span>
                  </td>
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center font-mono text-sm sm:text-base">
                    {row.runMatMs}
                  </td>
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="inline-flex items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 px-3 py-1 text-sm sm:text-base text-purple-100">
                      {row.vsPyTorch}
                    </span>
                  </td>
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="inline-flex items-center justify-center rounded-full border border-blue-500/40 bg-blue-500/10 px-3 py-1 text-sm sm:text-base text-blue-100">
                      {row.vsNumPy}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="bg-gradient-to-r from-blue-500 via-purple-500 to-purple-600 text-white text-sm sm:text-base">
                <td className="px-3 sm:px-6 py-2 sm:py-3 font-medium" colSpan={2}>
                  RunMat speedup range
                </td>
                <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                  1.8× – 5.6× vs PyTorch
                </td>
                <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                  3.5× – 10.2× vs NumPy
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
}


