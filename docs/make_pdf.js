// Erzeugt Kurzanleitung_Freezing_Tool.pdf aus kurzanleitung.html.
// Logo: liegt docs/optima_pharma_logo.svg oder .png vor, wird es oben rechts eingesetzt.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  let html = fs.readFileSync(path.join(__dirname, 'kurzanleitung.html'), 'utf8');
  const logos = [['optima_pharma_logo.svg', 'image/svg+xml'], ['optima_pharma_logo.png', 'image/png']];
  const logo = logos.find(([f]) => fs.existsSync(path.join(__dirname, f)));
  if (logo) {
    const data = fs.readFileSync(path.join(__dirname, logo[0])).toString('base64');
    html = html.replace('<img id="logo" alt="">', `<img id="logo" alt="OPTIMA pharma" src="data:${logo[1]};base64,${data}">`);
  } else {
    html = html.replace('<img id="logo" alt="">', '');
  }
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: 'load' });
  await page.pdf({ path: path.join(__dirname, 'Kurzanleitung_Freezing_Tool.pdf'), format: 'A4', printBackground: true, preferCSSPageSize: true });
  await browser.close();
  console.log(logo ? `PDF erstellt (Logo: ${logo[0]})` : 'PDF erstellt (ohne Logo)');
})();
