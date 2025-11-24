type ElementwiseRow = {
  points: string;
  runMatMs: number;
  vsPyTorch: string;
  vsNumPy: string;
};

const ELEMENTWISE_ROWS: ElementwiseRow[] = [
  { points: "10M", runMatMs: 143, vsPyTorch: "9.6× faster", vsNumPy: "1.1× faster" },
  { points: "100M", runMatMs: 145, vsPyTorch: "113.3× faster", vsNumPy: "7.4× faster" },
  { points: "200M", runMatMs: 157, vsPyTorch: "105.5× faster", vsNumPy: "13.5× faster" },
  { points: "500M", runMatMs: 138, vsPyTorch: "130× faster", vsNumPy: "36.5× faster" },
  { points: "1B", runMatMs: 144, vsPyTorch: "144.3× faster", vsNumPy: "82.6× faster" },
];

export default function ElementwiseMathSweep() {
  return (
    <div className="mx-auto w-full max-w-[40rem] rounded-2xl border border-border/60 bg-background/60 overflow-hidden shadow-lg">
      <div className="bg-[#0E1421] px-4 sm:px-6 py-4 border-b border-border/60">
        <div className="text-sm sm:text-base font-semibold uppercase tracking-wide text-gray-200">
          Elementwise Math
        </div>
        <div className="mt-1 text-xs sm:text-sm text-gray-400">
          Problem size sweep: 10M → 1B points
        </div>
      </div>
      <div className="bg-[#0E1421]">
        <div className="overflow-x-auto">
          <table className="w-full text-sm sm:text-base">
            <thead className="border-b border-border/60 bg-muted/40 text-center">
              <tr>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center text-gray-400 font-medium">
                  Points (elements)
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
              {ELEMENTWISE_ROWS.map((row) => (
                <tr key={row.points} className="border-b border-border/40 last:border-b-0">
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="font-mono text-sm sm:text-base">{row.points}</span>
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
                  9.6× – 144.3× vs PyTorch
                </td>
                <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                  1.1× – 82.6× vs NumPy
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
}

