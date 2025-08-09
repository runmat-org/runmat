import * as React from "react";

type LogoProps = {
  height?: number | string;
  className?: string;
};

export default function Logo({ height = 20, className }: LogoProps) {
  const id = React.useId();
  const g0 = `${id}-g0`;
  const g1 = `${id}-g1`;
  const g2 = `${id}-g2`;
  const g3 = `${id}-g3`;
  const g4 = `${id}-g4`;
  const g5 = `${id}-g5`;

  return (
    <svg
      height={height}
      viewBox="0 0 264 134"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="RunMat logo"
    >
      <path d="M30 33H0V101H30V33Z" fill={`url(#${g0})`} />
      <path d="M264 33H230V101H264V33Z" fill={`url(#${g1})`} />
      <path d="M30 33V0H96V33H30Z" fill={`url(#${g2})`} />
      <path d="M165 134V101H230V134H165Z" fill={`url(#${g3})`} />
      <path d="M96 65V33H132V65H96Z" fill={`url(#${g4})`} />
      <path d="M132 101V66H165V101H132Z" fill={`url(#${g5})`} />
      <defs>
        <linearGradient id={g0} x1="15" y1="33" x2="15" y2="101" gradientUnits="userSpaceOnUse">
          <stop stopColor="#CA26D6" />
          <stop offset="1" stopColor="#D500C6" />
        </linearGradient>
        <linearGradient id={g1} x1="247" y1="33" x2="247" y2="101" gradientUnits="userSpaceOnUse">
          <stop stopColor="#CA26D6" />
          <stop offset="1" stopColor="#D500C6" />
        </linearGradient>
        <linearGradient id={g2} x1="63" y1="0" x2="63" y2="33" gradientUnits="userSpaceOnUse">
          <stop stopColor="#CA26D6" />
          <stop offset="1" stopColor="#D500C6" />
        </linearGradient>
        <linearGradient id={g3} x1="197.5" y1="101" x2="197.5" y2="134" gradientUnits="userSpaceOnUse">
          <stop stopColor="#CA26D6" />
          <stop offset="1" stopColor="#D500C6" />
        </linearGradient>
        <linearGradient id={g4} x1="114" y1="33" x2="114" y2="65" gradientUnits="userSpaceOnUse">
          <stop stopColor="#CA26D6" />
          <stop offset="1" stopColor="#D500C6" />
        </linearGradient>
        <linearGradient id={g5} x1="148.5" y1="66" x2="148.5" y2="101" gradientUnits="userSpaceOnUse">
          <stop stopColor="#CA26D6" />
          <stop offset="1" stopColor="#D500C6" />
        </linearGradient>
      </defs>
    </svg>
  );
}


