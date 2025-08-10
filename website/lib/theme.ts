// Theme tokens extracted from globals.css and converted to hex values
// These are built-time constants to avoid Node.js dependencies in edge runtime

// Helper function to convert HSL tokens to hex colors
function hslToHex(h: number, s: number, l: number): string {
  const hue = h / 360;
  const saturation = s / 100;
  const lightness = l / 100;
  
  const c = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const x = c * (1 - Math.abs((hue * 6) % 2 - 1));
  const m = lightness - c / 2;
  
  let r = 0, g = 0, b = 0;
  
  if (0 <= hue && hue < 1/6) {
    r = c; g = x; b = 0;
  } else if (1/6 <= hue && hue < 2/6) {
    r = x; g = c; b = 0;
  } else if (2/6 <= hue && hue < 3/6) {
    r = 0; g = c; b = x;
  } else if (3/6 <= hue && hue < 4/6) {
    r = 0; g = x; b = c;
  } else if (4/6 <= hue && hue < 5/6) {
    r = x; g = 0; b = c;
  } else if (5/6 <= hue && hue < 1) {
    r = c; g = 0; b = x;
  }
  
  const red = Math.round((r + m) * 255);
  const green = Math.round((g + m) * 255);
  const blue = Math.round((b + m) * 255);
  
  return `#${red.toString(16).padStart(2, '0')}${green.toString(16).padStart(2, '0')}${blue.toString(16).padStart(2, '0')}`;
}

// Theme tokens converted to hex colors for programmatic use (e.g., OpenGraph)
export const theme = {
  dark: {
    // From .dark CSS tokens in globals.css
    background: hslToHex(20, 14.3, 4.1),    // --background: 20 14.3% 4.1%
    card: hslToHex(20, 14.3, 4.1),          // --card: 20 14.3% 4.1%
    border: hslToHex(12, 6.5, 15.1),        // --border: 12 6.5% 15.1%
    foreground: hslToHex(60, 9.1, 97.8),    // --foreground: 60 9.1% 97.8%
    muted: hslToHex(24, 5.4, 63.9),         // --muted-foreground: 24 5.4% 63.9%
    primary: hslToHex(221.2, 83.2, 53.3),   // --primary: 221.2 83.2% 53.3%
    // Brand gradient from globals.css
    brandGradient: 'linear-gradient(135deg, #3ea7fd 0%, #bb51ff 100%)',
  },
  light: {
    // From :root CSS tokens in globals.css
    background: hslToHex(0, 0, 100),         // --background: 0 0% 100%
    card: hslToHex(0, 0, 100),               // --card: 0 0% 100%
    border: hslToHex(20, 5.9, 90),           // --border: 20 5.9% 90%
    foreground: hslToHex(20, 14.3, 4.1),    // --foreground: 20 14.3% 4.1%
    muted: hslToHex(25, 5.3, 44.7),         // --muted-foreground: 25 5.3% 44.7%
    primary: hslToHex(221.2, 83.2, 53.3),   // --primary: 221.2 83.2% 53.3%
    // Brand gradient from globals.css
    brandGradient: 'linear-gradient(135deg, #3ea7fd 0%, #bb51ff 100%)',
  },
} as const;

export type Theme = typeof theme;