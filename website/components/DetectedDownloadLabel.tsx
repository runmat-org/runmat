"use client";

import { useEffect, useState } from "react";

type OS = "macOS" | "Linux" | "Windows";

function detectOS(): OS | null {
  const userAgent = window.navigator.userAgent.toLowerCase();

  if (userAgent.includes("android") || userAgent.includes("iphone") || userAgent.includes("ipad")) {
    return null;
  }
  if (userAgent.includes("win")) return "Windows";
  if (userAgent.includes("linux")) return "Linux";
  if (userAgent.includes("mac")) return "macOS";

  return null;
}

export function DetectedDownloadLabel() {
  const [os, setOs] = useState<OS | null>(null);

  useEffect(() => {
    setOs(detectOS());
  }, []);

  return <>Download{os ? ` for ${os}` : " RunMat"}</>;
}
