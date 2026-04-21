import { type FC } from "react";

const LIGHT_FINE = "rgba(0,0,0,0.06)";
const LIGHT_MAJOR = "rgba(0,0,0,0.10)";
const LIGHT_CROSS = "rgba(0,0,0,0.18)";

const DARK_FINE = "rgba(255,255,255,0.12)";
const DARK_MAJOR = "rgba(255,255,255,0.22)";
const DARK_CROSS = "rgba(255,255,255,0.35)";

function Svg({
  uid,
  bg,
  fine,
  major,
  cross,
  children,
}: {
  uid: string;
  bg: string;
  fine: string;
  major: string;
  cross: string;
  children: React.ReactNode;
}) {
  return (
    <svg viewBox="0 0 400 220" preserveAspectRatio="xMidYMid slice" className="absolute inset-0 w-full h-full">
      <defs>
        <pattern id={`f-${uid}`} width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M10 0V10H0" fill="none" stroke={fine} strokeWidth="0.4" />
        </pattern>
        <pattern id={`m-${uid}`} width="50" height="50" patternUnits="userSpaceOnUse">
          <rect width="50" height="50" fill={`url(#f-${uid})`} />
          <path d="M50 0V50H0" fill="none" stroke={major} strokeWidth="0.6" />
        </pattern>
      </defs>
      <rect width="400" height="220" fill={bg} />
      <rect width="400" height="220" fill={`url(#m-${uid})`} />
      {children}
      <g stroke={cross} strokeWidth="0.5">
        <line x1={47} y1={45} x2={53} y2={45} /><line x1={50} y1={42} x2={50} y2={48} />
        <line x1={347} y1={195} x2={353} y2={195} /><line x1={350} y1={192} x2={350} y2={198} />
      </g>
    </svg>
  );
}

function Wrap({
  id,
  lightBg,
  darkBg,
  children,
}: {
  id: string;
  lightBg: string;
  darkBg: string;
  children: React.ReactNode;
}) {
  return (
    <>
      <div className="absolute inset-0 w-full h-full block dark:hidden">
        <Svg uid={`${id}-l`} bg={lightBg} fine={LIGHT_FINE} major={LIGHT_MAJOR} cross={LIGHT_CROSS}>
          {children}
        </Svg>
      </div>
      <div className="absolute inset-0 w-full h-full hidden dark:block" style={{ filter: "brightness(1.6)" }}>
        <Svg uid={`${id}-d`} bg={darkBg} fine={DARK_FINE} major={DARK_MAJOR} cross={DARK_CROSS}>
          {children}
        </Svg>
      </div>
    </>
  );
}

const BlueArcs: FC = () => (
  <Wrap id="blue" lightBg="#e8f0fb" darkBg="#1a2333">
    <path d="M80 200 A150 150 0 0 1 230 50" fill="none" stroke="#4F8CFF" strokeWidth="1.5" opacity="0.25" />
    <path d="M120 210 A130 130 0 0 1 250 80" fill="none" stroke="#4F8CFF" strokeWidth="1" opacity="0.18" />
    <path d="M320 10 A160 160 0 0 1 160 170" fill="none" stroke="#4F8CFF" strokeWidth="1.5" opacity="0.22" />
    <path d="M350 30 A140 140 0 0 1 210 170" fill="none" stroke="#4F8CFF" strokeWidth="1" opacity="0.15" />
    <line x1="30" y1="180" x2="280" y2="30" stroke="#4F8CFF" strokeWidth="0.6" opacity="0.12" />
    <line x1="370" y1="15" x2="120" y2="200" stroke="#4F8CFF" strokeWidth="0.6" opacity="0.12" />
    <circle cx="195" cy="108" r="3" fill="#4F8CFF" opacity="0.25" />
    <circle cx="172" cy="130" r="2" fill="#4F8CFF" opacity="0.2" />
    <circle cx="220" cy="85" r="2.5" fill="#4F8CFF" opacity="0.22" />
  </Wrap>
);

const GreenRings: FC = () => (
  <Wrap id="green" lightBg="#e8f5eb" darkBg="#1a2b1e">
    <circle cx="140" cy="95" r="75" fill="none" stroke="#3a9e52" strokeWidth="0.8" opacity="0.14" />
    <circle cx="140" cy="95" r="55" fill="none" stroke="#3a9e52" strokeWidth="0.8" opacity="0.18" />
    <circle cx="140" cy="95" r="35" fill="none" stroke="#3a9e52" strokeWidth="1" opacity="0.22" />
    <circle cx="140" cy="95" r="15" fill="none" stroke="#3a9e52" strokeWidth="1" opacity="0.26" />
    <circle cx="300" cy="140" r="60" fill="none" stroke="#3a9e52" strokeWidth="0.8" opacity="0.12" />
    <circle cx="300" cy="140" r="40" fill="none" stroke="#3a9e52" strokeWidth="0.8" opacity="0.16" />
    <circle cx="300" cy="140" r="20" fill="none" stroke="#3a9e52" strokeWidth="0.8" opacity="0.2" />
    <line x1="60" y1="20" x2="220" y2="170" stroke="#3a9e52" strokeWidth="0.5" opacity="0.1" />
    <line x1="240" y1="80" x2="360" y2="200" stroke="#3a9e52" strokeWidth="0.5" opacity="0.1" />
  </Wrap>
);

const VioletCubes: FC = () => (
  <Wrap id="violet" lightBg="#f0ecf6" darkBg="#231e2e">
    <g stroke="#7a5ab8" opacity="0.22" fill="none" strokeWidth="0.8">
      <path d="M100 130 L130 115 L160 130 L130 145 Z" />
      <path d="M100 130 L100 100 L130 85 L130 115" />
      <path d="M130 85 L160 100 L160 130" />
      <path d="M160 130 L190 115 L220 130 L190 145 Z" />
      <path d="M160 130 L160 100 L190 85 L190 115" />
      <path d="M190 85 L220 100 L220 130" />
      <path d="M130 85 L130 55 L160 40 L160 70" />
      <path d="M130 55 L100 70 L100 100" />
      <path d="M160 40 L190 55 L190 85" />
    </g>
    <path d="M280 60 L310 110 L250 110 Z" fill="none" stroke="#7a5ab8" strokeWidth="1" opacity="0.15" />
    <path d="M295 75 L315 110 L275 110 Z" fill="none" stroke="#7a5ab8" strokeWidth="0.8" opacity="0.18" />
    <g stroke="#7a5ab8" strokeWidth="0.6" opacity="0.15">
      <line x1="310" y1="150" x2="340" y2="150" />
      <line x1="310" y1="150" x2="310" y2="175" />
      <path d="M310 158 L318 158 L318 150" fill="none" />
    </g>
  </Wrap>
);

const GoldenSpiral: FC = () => (
  <Wrap id="spiral" lightBg="#eaf0fa" darkBg="#1b2230">
    <path d="M200 110 A30 30 0 0 1 230 80" fill="none" stroke="#4F8CFF" strokeWidth="1.2" opacity="0.25" />
    <path d="M230 80 A50 50 0 0 1 280 130" fill="none" stroke="#4F8CFF" strokeWidth="1.2" opacity="0.22" />
    <path d="M280 130 A80 80 0 0 1 200 210" fill="none" stroke="#4F8CFF" strokeWidth="1.2" opacity="0.18" />
    <path d="M200 210 A130 130 0 0 1 70 80" fill="none" stroke="#4F8CFF" strokeWidth="1" opacity="0.15" />
    <rect x="170" y="80" width="60" height="60" fill="none" stroke="#4F8CFF" strokeWidth="0.6" opacity="0.15" />
    <rect x="230" y="80" width="100" height="100" fill="none" stroke="#4F8CFF" strokeWidth="0.6" opacity="0.12" />
    <rect x="70" y="80" width="100" height="160" fill="none" stroke="#4F8CFF" strokeWidth="0.5" opacity="0.1" />
    <line x1="170" y1="140" x2="230" y2="80" stroke="#4F8CFF" strokeWidth="0.4" opacity="0.12" strokeDasharray="3 3" />
    <line x1="230" y1="180" x2="330" y2="80" stroke="#4F8CFF" strokeWidth="0.4" opacity="0.1" strokeDasharray="3 3" />
  </Wrap>
);

const HexTessellation: FC = () => (
  <Wrap id="hex" lightBg="#f2eef6" darkBg="#25202e">
    <g fill="none" stroke="#6a3d9a" strokeWidth="0.8" opacity="0.16">
      <polygon points="120,50 145,38 170,50 170,75 145,87 120,75" />
      <polygon points="170,50 195,38 220,50 220,75 195,87 170,75" />
      <polygon points="220,50 245,38 270,50 270,75 245,87 220,75" />
      <polygon points="145,87 170,75 195,87 195,112 170,124 145,112" />
      <polygon points="195,87 220,75 245,87 245,112 220,124 195,112" />
      <polygon points="245,87 270,75 295,87 295,112 270,124 245,112" />
      <polygon points="120,125 145,112 170,124 170,150 145,162 120,150" />
      <polygon points="170,124 195,112 220,124 220,150 195,162 170,150" />
      <polygon points="220,124 245,112 270,124 270,150 245,162 220,150" />
    </g>
    <polygon points="170,50 195,38 220,50 220,75 195,87 170,75" fill="#6a3d9a" opacity="0.06" />
    <polygon points="195,87 220,75 245,87 245,112 220,124 195,112" fill="#6a3d9a" opacity="0.1" />
    <polygon points="145,87 170,75 195,87 195,112 170,124 145,112" fill="#6a3d9a" opacity="0.04" />
    <polygon points="220,124 245,112 270,124 270,150 245,162 220,150" fill="#6a3d9a" opacity="0.07" />
    <circle cx="195" cy="87" r="2" fill="#6a3d9a" opacity="0.18" />
    <circle cx="220" cy="75" r="2" fill="#6a3d9a" opacity="0.18" />
    <circle cx="245" cy="87" r="2" fill="#6a3d9a" opacity="0.18" />
    <circle cx="170" cy="75" r="2" fill="#6a3d9a" opacity="0.15" />
    <circle cx="220" cy="124" r="2" fill="#6a3d9a" opacity="0.15" />
  </Wrap>
);

const BezierCurves: FC = () => (
  <Wrap id="bezier" lightBg="#f2f2f0" darkBg="#242422">
    <path d="M30 180 C100 30 200 30 370 180" fill="none" stroke="#555" strokeWidth="1.2" opacity="0.18" />
    <path d="M30 180 C130 60 270 60 370 180" fill="none" stroke="#555" strokeWidth="1" opacity="0.15" />
    <path d="M30 180 C80 100 150 50 250 50 C320 50 360 100 370 180" fill="none" stroke="#555" strokeWidth="0.8" opacity="0.12" />
    <path d="M30 40 C100 190 200 190 370 40" fill="none" stroke="#555" strokeWidth="1.2" opacity="0.18" />
    <path d="M30 40 C130 160 270 160 370 40" fill="none" stroke="#555" strokeWidth="1" opacity="0.15" />
    <ellipse cx="200" cy="110" rx="40" ry="15" fill="#555" opacity="0.03" />
    <circle cx="115" cy="110" r="2" fill="#555" opacity="0.18" />
    <circle cx="285" cy="110" r="2" fill="#555" opacity="0.18" />
    <circle cx="200" cy="85" r="1.5" fill="#555" opacity="0.15" />
    <circle cx="200" cy="135" r="1.5" fill="#555" opacity="0.15" />
  </Wrap>
);

const VoronoiCells: FC = () => (
  <Wrap id="voronoi" lightBg="#f6f2ec" darkBg="#28241c">
    <g fill="none" stroke="#b8860a" strokeWidth="0.7" opacity="0.18">
      <path d="M0 80 L60 50 L110 70 L90 130 L30 120 Z" />
      <path d="M60 50 L140 20 L180 60 L110 70 Z" />
      <path d="M110 70 L180 60 L200 120 L150 140 L90 130 Z" />
      <path d="M140 20 L240 10 L250 70 L180 60 Z" />
      <path d="M180 60 L250 70 L270 130 L200 120 Z" />
      <path d="M250 70 L340 40 L360 100 L270 130 Z" />
      <path d="M90 130 L150 140 L140 200 L50 190 L30 120 Z" />
      <path d="M150 140 L200 120 L230 180 L180 210 L140 200 Z" />
      <path d="M200 120 L270 130 L290 190 L230 180 Z" />
      <path d="M270 130 L360 100 L380 170 L290 190 Z" />
    </g>
    <g fill="#b8860a" opacity="0.15">
      <circle cx="60" cy="90" r="2" />
      <circle cx="140" cy="55" r="2" />
      <circle cx="155" cy="105" r="2" />
      <circle cx="225" cy="55" r="2" />
      <circle cx="230" cy="110" r="2" />
      <circle cx="310" cy="85" r="2" />
      <circle cx="110" cy="165" r="2" />
      <circle cx="190" cy="165" r="2" />
      <circle cx="270" cy="160" r="2" />
      <circle cx="340" cy="145" r="2" />
    </g>
  </Wrap>
);

const Lissajous: FC = () => (
  <Wrap id="lissajous" lightBg="#e8f0f5" darkBg="#1a2228">
    <path d="M200 30 C260 30 320 65 340 110 C360 155 340 190 280 190 C220 190 160 155 140 110 C120 65 140 30 200 30" fill="none" stroke="#c44060" strokeWidth="1.5" opacity="0.22" />
    <path d="M200 110 C220 60 280 40 310 70 C340 100 310 140 280 150 C250 160 220 140 200 110 C180 80 150 60 120 70 C90 80 60 120 90 150 C120 180 180 160 200 110" fill="none" stroke="#c44060" strokeWidth="1" opacity="0.18" />
    <ellipse cx="200" cy="110" rx="25" ry="40" fill="none" stroke="#c44060" strokeWidth="0.8" opacity="0.15" transform="rotate(30 200 110)" />
    <ellipse cx="200" cy="110" rx="25" ry="40" fill="none" stroke="#c44060" strokeWidth="0.8" opacity="0.15" transform="rotate(-30 200 110)" />
    <circle cx="200" cy="110" r="2.5" fill="#c44060" opacity="0.18" />
    <circle cx="310" cy="70" r="1.5" fill="#c44060" opacity="0.12" />
    <circle cx="90" cy="150" r="1.5" fill="#c44060" opacity="0.12" />
  </Wrap>
);

const PenroseTriangle: FC = () => (
  <Wrap id="penrose" lightBg="#eef3ec" darkBg="#1e261c">
    <g fill="none" stroke="#4a6080" strokeWidth="1" opacity="0.18">
      <path d="M200 30 L330 180 L70 180 Z" />
      <path d="M200 155 L135 80 L265 80 Z" />
    </g>
    <g fill="none" stroke="#4a6080" strokeWidth="0.6" opacity="0.12">
      <line x1="200" y1="30" x2="135" y2="80" />
      <line x1="330" y1="180" x2="265" y2="80" />
      <line x1="70" y1="180" x2="200" y2="155" />
      <line x1="200" y1="30" x2="265" y2="80" />
      <line x1="330" y1="180" x2="200" y2="155" />
      <line x1="70" y1="180" x2="135" y2="80" />
    </g>
    <g fill="#4a6080" opacity="0.12">
      <circle cx="200" cy="30" r="2.5" />
      <circle cx="330" cy="180" r="2.5" />
      <circle cx="70" cy="180" r="2.5" />
      <circle cx="135" cy="80" r="2" />
      <circle cx="265" cy="80" r="2" />
      <circle cx="200" cy="155" r="2" />
    </g>
  </Wrap>
);

const DelaunayMesh: FC = () => (
  <Wrap id="delaunay" lightBg="#eceff6" darkBg="#1e2128">
    <g fill="none" stroke="#3a9e52" strokeWidth="0.6" opacity="0.16">
      <polygon points="50,40 130,25 100,90" />
      <polygon points="130,25 220,35 180,80" />
      <polygon points="100,90 180,80 130,25" />
      <polygon points="220,35 310,20 280,75" />
      <polygon points="180,80 280,75 220,35" />
      <polygon points="310,20 380,50 340,95" />
      <polygon points="280,75 340,95 310,20" />
      <polygon points="50,40 100,90 40,130" />
      <polygon points="100,90 180,80 160,145" />
      <polygon points="180,80 280,75 250,140" />
      <polygon points="280,75 340,95 320,155" />
      <polygon points="40,130 100,90 80,170" />
      <polygon points="100,90 160,145 80,170" />
      <polygon points="160,145 250,140 200,195" />
      <polygon points="250,140 320,155 280,200" />
      <polygon points="80,170 160,145 120,210" />
      <polygon points="160,145 200,195 120,210" />
      <polygon points="250,140 280,200 200,195" />
      <polygon points="320,155 370,180 280,200" />
    </g>
    <polygon points="180,80 280,75 250,140" fill="#3a9e52" opacity="0.04" />
    <polygon points="100,90 180,80 160,145" fill="#3a9e52" opacity="0.06" />
    <polygon points="160,145 250,140 200,195" fill="#3a9e52" opacity="0.05" />
    <g fill="#3a9e52" opacity="0.15">
      <circle cx="130" cy="25" r="2" />
      <circle cx="220" cy="35" r="2" />
      <circle cx="310" cy="20" r="2" />
      <circle cx="100" cy="90" r="2" />
      <circle cx="180" cy="80" r="2" />
      <circle cx="280" cy="75" r="2" />
      <circle cx="340" cy="95" r="2" />
      <circle cx="160" cy="145" r="2" />
      <circle cx="250" cy="140" r="2" />
      <circle cx="200" cy="195" r="2" />
    </g>
  </Wrap>
);

const MoireRings: FC = () => (
  <Wrap id="moire" lightBg="#edf0f5" darkBg="#1e2128">
    <circle cx="160" cy="110" r="90" fill="none" stroke="#2a8a7a" strokeWidth="0.5" opacity="0.12" />
    <circle cx="160" cy="110" r="70" fill="none" stroke="#2a8a7a" strokeWidth="0.5" opacity="0.15" />
    <circle cx="160" cy="110" r="50" fill="none" stroke="#2a8a7a" strokeWidth="0.5" opacity="0.18" />
    <circle cx="160" cy="110" r="30" fill="none" stroke="#2a8a7a" strokeWidth="0.6" opacity="0.2" />
    <circle cx="240" cy="110" r="90" fill="none" stroke="#2a8a7a" strokeWidth="0.5" opacity="0.12" />
    <circle cx="240" cy="110" r="70" fill="none" stroke="#2a8a7a" strokeWidth="0.5" opacity="0.15" />
    <circle cx="240" cy="110" r="50" fill="none" stroke="#2a8a7a" strokeWidth="0.5" opacity="0.18" />
    <circle cx="240" cy="110" r="30" fill="none" stroke="#2a8a7a" strokeWidth="0.6" opacity="0.2" />
    <circle cx="160" cy="110" r="2" fill="#2a8a7a" opacity="0.18" />
    <circle cx="240" cy="110" r="2" fill="#2a8a7a" opacity="0.18" />
  </Wrap>
);

const CrossHatch: FC = () => (
  <Wrap id="hatch" lightBg="#f5f0ea" darkBg="#28231a">
    <g stroke="#4040a0" strokeWidth="0.6" opacity="0.12">
      <line x1="40" y1="20" x2="160" y2="200" />
      <line x1="70" y1="20" x2="190" y2="200" />
      <line x1="100" y1="20" x2="220" y2="200" />
      <line x1="130" y1="20" x2="250" y2="200" />
      <line x1="160" y1="20" x2="280" y2="200" />
      <line x1="190" y1="20" x2="310" y2="200" />
      <line x1="220" y1="20" x2="340" y2="200" />
      <line x1="250" y1="20" x2="370" y2="200" />
    </g>
    <g stroke="#4040a0" strokeWidth="0.6" opacity="0.1">
      <line x1="40" y1="200" x2="160" y2="20" />
      <line x1="70" y1="200" x2="190" y2="20" />
      <line x1="100" y1="200" x2="220" y2="20" />
      <line x1="130" y1="200" x2="250" y2="20" />
      <line x1="160" y1="200" x2="280" y2="20" />
      <line x1="190" y1="200" x2="310" y2="20" />
      <line x1="220" y1="200" x2="340" y2="20" />
      <line x1="250" y1="200" x2="370" y2="20" />
    </g>
    <g fill="#4040a0" opacity="0.08">
      <rect x="147" y="107" width="6" height="6" transform="rotate(45 150 110)" />
      <rect x="207" y="107" width="6" height="6" transform="rotate(45 210 110)" />
      <rect x="267" y="107" width="6" height="6" transform="rotate(45 270 110)" />
    </g>
  </Wrap>
);

const StackedRects: FC = () => (
  <Wrap id="rects" lightBg="#f0eef5" darkBg="#222028">
    <rect x="80" y="50" width="120" height="80" rx="8" fill="none" stroke="#c06040" strokeWidth="0.8" opacity="0.15" transform="rotate(-8 140 90)" />
    <rect x="100" y="60" width="120" height="80" rx="8" fill="none" stroke="#c06040" strokeWidth="0.8" opacity="0.18" transform="rotate(5 160 100)" />
    <rect x="90" y="55" width="120" height="80" rx="8" fill="none" stroke="#c06040" strokeWidth="1" opacity="0.12" transform="rotate(-2 150 95)" />
    <rect x="230" y="90" width="100" height="70" rx="6" fill="none" stroke="#c06040" strokeWidth="0.8" opacity="0.12" transform="rotate(12 280 125)" />
    <rect x="240" y="100" width="100" height="70" rx="6" fill="none" stroke="#c06040" strokeWidth="0.8" opacity="0.15" transform="rotate(-6 290 135)" />
    <circle cx="80" cy="50" r="1.5" fill="#c06040" opacity="0.12" />
    <circle cx="200" cy="50" r="1.5" fill="#c06040" opacity="0.12" />
    <circle cx="200" cy="130" r="1.5" fill="#c06040" opacity="0.12" />
    <circle cx="80" cy="130" r="1.5" fill="#c06040" opacity="0.12" />
  </Wrap>
);

const WaveStack: FC = () => (
  <Wrap id="waves" lightBg="#e8eef5" darkBg="#1a2028">
    <path d="M0 60 Q50 30 100 60 T200 60 T300 60 T400 60" fill="none" stroke="#2a3a6a" strokeWidth="1" opacity="0.15" />
    <path d="M0 90 Q25 70 50 90 T100 90 T150 90 T200 90 T250 90 T300 90 T350 90 T400 90" fill="none" stroke="#2a3a6a" strokeWidth="0.8" opacity="0.12" />
    <path d="M0 120 Q15 108 30 120 T60 120 T90 120 T120 120 T150 120 T180 120 T210 120 T240 120 T270 120 T300 120 T330 120 T360 120 T390 120" fill="none" stroke="#2a3a6a" strokeWidth="0.6" opacity="0.1" />
    <path d="M0 40 C100 10 200 10 300 25 C350 35 380 50 400 55" fill="none" stroke="#2a3a6a" strokeWidth="0.8" opacity="0.1" strokeDasharray="4 4" />
    <path d="M0 80 C100 110 200 110 300 95 C350 85 380 70 400 65" fill="none" stroke="#2a3a6a" strokeWidth="0.8" opacity="0.1" strokeDasharray="4 4" />
    <line x1="0" y1="150" x2="400" y2="150" stroke="#2a3a6a" strokeWidth="0.4" opacity="0.08" />
    <line x1="0" y1="170" x2="400" y2="170" stroke="#2a3a6a" strokeWidth="0.4" opacity="0.08" />
    <line x1="0" y1="190" x2="400" y2="190" stroke="#2a3a6a" strokeWidth="0.4" opacity="0.08" />
  </Wrap>
);

const TangentCircles: FC = () => (
  <Wrap id="tangent" lightBg="#ecf5f0" darkBg="#1c2822">
    <circle cx="130" cy="90" r="40" fill="none" stroke="#4a7a40" strokeWidth="0.8" opacity="0.15" />
    <circle cx="200" cy="60" r="35" fill="none" stroke="#4a7a40" strokeWidth="0.8" opacity="0.18" />
    <circle cx="260" cy="100" r="45" fill="none" stroke="#4a7a40" strokeWidth="0.8" opacity="0.12" />
    <circle cx="180" cy="145" r="38" fill="none" stroke="#4a7a40" strokeWidth="0.8" opacity="0.15" />
    <circle cx="300" cy="160" r="30" fill="none" stroke="#4a7a40" strokeWidth="0.8" opacity="0.12" />
    <circle cx="166" cy="72" r="1.5" fill="#4a7a40" opacity="0.18" />
    <circle cx="228" cy="82" r="1.5" fill="#4a7a40" opacity="0.18" />
    <circle cx="155" cy="120" r="1.5" fill="#4a7a40" opacity="0.18" />
    <circle cx="210" cy="114" r="1.5" fill="#4a7a40" opacity="0.15" />
    <circle cx="276" cy="142" r="1.5" fill="#4a7a40" opacity="0.15" />
    <line x1="166" y1="72" x2="228" y2="82" stroke="#4a7a40" strokeWidth="0.4" opacity="0.1" strokeDasharray="2 3" />
    <line x1="155" y1="120" x2="210" y2="114" stroke="#4a7a40" strokeWidth="0.4" opacity="0.1" strokeDasharray="2 3" />
  </Wrap>
);

const RadialBurst: FC = () => (
  <Wrap id="burst" lightBg="#f5f0ec" darkBg="#282220">
    <g stroke="#804070" strokeWidth="0.5" opacity="0.12">
      <line x1="280" y1="80" x2="20" y2="30" />
      <line x1="280" y1="80" x2="10" y2="80" />
      <line x1="280" y1="80" x2="20" y2="140" />
      <line x1="280" y1="80" x2="50" y2="200" />
      <line x1="280" y1="80" x2="120" y2="210" />
      <line x1="280" y1="80" x2="200" y2="210" />
      <line x1="280" y1="80" x2="300" y2="210" />
      <line x1="280" y1="80" x2="380" y2="200" />
      <line x1="280" y1="80" x2="400" y2="140" />
      <line x1="280" y1="80" x2="400" y2="60" />
      <line x1="280" y1="80" x2="380" y2="10" />
      <line x1="280" y1="80" x2="300" y2="0" />
      <line x1="280" y1="80" x2="200" y2="0" />
      <line x1="280" y1="80" x2="120" y2="0" />
    </g>
    <path d="M230 30 A60 60 0 0 1 330 130" fill="none" stroke="#804070" strokeWidth="0.6" opacity="0.12" />
    <path d="M210 10 A80 80 0 0 1 360 150" fill="none" stroke="#804070" strokeWidth="0.6" opacity="0.1" />
    <circle cx="280" cy="80" r="3" fill="#804070" opacity="0.15" />
    <circle cx="280" cy="80" r="8" fill="none" stroke="#804070" strokeWidth="0.6" opacity="0.12" />
  </Wrap>
);

export const CARD_PATTERNS: FC[] = [
  BlueArcs,
  GreenRings,
  VioletCubes,
  GoldenSpiral,
  HexTessellation,
  BezierCurves,
  VoronoiCells,
  Lissajous,
  PenroseTriangle,
  DelaunayMesh,
  MoireRings,
  CrossHatch,
  StackedRects,
  WaveStack,
  TangentCircles,
  RadialBurst,
];
