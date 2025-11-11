const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const svgPath = path.join(__dirname, '../app/icon.svg');
const publicDir = path.join(__dirname, '../public');
const appDir = path.join(__dirname, '../app');

async function convertFavicon() {
  try {
    // Read the SVG
    const svgBuffer = fs.readFileSync(svgPath);
    
    // Convert to PNG (32x32 for favicon.ico, 180x180 for apple-touch-icon)
    const png32 = await sharp(svgBuffer)
      .resize(32, 32, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toBuffer();
    
    const png180 = await sharp(svgBuffer)
      .resize(180, 180, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toBuffer();
    
    // For ICO, we need multiple sizes. Create a 16x16 and 32x32 version
    const png16 = await sharp(svgBuffer)
      .resize(16, 16, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toBuffer();
    
    // Write favicon.ico (using PNG format as ICO - browsers accept this)
    // Note: True ICO format requires special encoding, but PNG works for most browsers
    fs.writeFileSync(path.join(publicDir, 'favicon.ico'), png32);
    fs.writeFileSync(path.join(appDir, 'favicon.ico'), png32);
    
    // Write apple-touch-icon.png
    fs.writeFileSync(path.join(publicDir, 'apple-touch-icon.png'), png180);
    
    console.log('✅ Favicon conversion complete!');
    console.log('   - favicon.ico created (32x32)');
    console.log('   - apple-touch-icon.png created (180x180)');
  } catch (error) {
    console.error('Error converting favicon:', error);
    process.exit(1);
  }
}

convertFavicon();

