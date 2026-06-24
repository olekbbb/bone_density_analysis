const DEFAULT_CSV = "../wyniki_biomechaniczne.csv";

const columns = {
  slice: "slice",
  area: "area_cortex_mm2",
  ratio: "cortical_ratio",
  young: "E_mean_GPa",
  ieSag: "IE_sagittal_GPa_mm4",
  ieFront: "IE_frontal_GPa_mm4",
  polar: "JE_polar_GPa_mm4",
};

let currentRows = [];

const statusEl = document.querySelector("#status");
const summaryEl = document.querySelector("#summary");
const smoothRange = document.querySelector("#smooth-range");
const smoothValue = document.querySelector("#smooth-value");
const sliceMinInput = document.querySelector("#slice-min");
const sliceMaxInput = document.querySelector("#slice-max");

document.querySelector("#reload-button").addEventListener("click", loadDefaultCsv);
document.querySelector("#csv-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => loadCsvText(reader.result, file.name);
  reader.readAsText(file);
});

smoothRange.addEventListener("input", () => {
  smoothValue.textContent = `${smoothRange.value} przekrojów`;
  renderDashboard();
});

sliceMinInput.addEventListener("input", renderDashboard);
sliceMaxInput.addEventListener("input", renderDashboard);

loadDefaultCsv();

async function loadDefaultCsv() {
  try {
    const response = await fetch(`${DEFAULT_CSV}?t=${Date.now()}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    loadCsvText(text, "wyniki_biomechaniczne.csv");
  } catch (error) {
    statusEl.textContent = "Nie udało się automatycznie wczytać CSV. Użyj przycisku Wczytaj CSV albo uruchom lokalny serwer HTTP.";
  }
}

function loadCsvText(text, label) {
  currentRows = parseCsv(text)
    .map(normalizeRow)
    .filter((row) => Number.isFinite(row.slice));

  if (!currentRows.length) {
    statusEl.textContent = "CSV nie zawiera poprawnych danych.";
    return;
  }

  statusEl.textContent = `Wczytano ${currentRows.length} przekrojów z ${label}.`;
  renderDashboard();
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = splitCsvLine(lines.shift());
  return lines.map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function splitCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;

  for (const char of line) {
    if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += char;
    }
  }

  values.push(value);
  return values;
}

function normalizeRow(row) {
  return {
    slice: toNumber(row[columns.slice]),
    area: toNumber(row[columns.area]),
    ratio: toNumber(row[columns.ratio]),
    young: toNumber(row[columns.young]),
    ieSag: toNumber(row[columns.ieSag]),
    ieFront: toNumber(row[columns.ieFront]),
    polar: toNumber(row[columns.polar]),
  };
}

function toNumber(value) {
  const number = Number.parseFloat(value);
  return Number.isFinite(number) ? number : 0;
}

function renderDashboard() {
  if (!currentRows.length) return;

  const sliceMin = Number.parseFloat(sliceMinInput.value);
  const sliceMax = Number.parseFloat(sliceMaxInput.value);
  const filteredRows = currentRows.filter((row) => {
    if (Number.isFinite(sliceMin) && row.slice < sliceMin) return false;
    if (Number.isFinite(sliceMax) && row.slice > sliceMax) return false;
    return true;
  });

  if (!filteredRows.length) {
    statusEl.textContent = "Brak przekrojów w wybranym zakresie.";
    return;
  }

  const smoothWindow = Number.parseInt(smoothRange.value, 10);
  const rows = smoothRows(filteredRows, smoothWindow);

  renderSummary(filteredRows);
  drawLineChart("chart-young", rows, [
    { key: "young", label: "E mean [GPa]", color: "#0f766e" },
  ]);
  drawLineChart("chart-ratio", rows, [
    { key: "ratio", label: "Cortical ratio", color: "#b45309" },
  ]);
  drawLineChart("chart-area", rows, [
    { key: "area", label: "Pole kory [mm2]", color: "#2563eb" },
  ]);
  drawLineChart("chart-ie", rows, [
    { key: "ieSag", label: "IE sagittal", color: "#be123c" },
    { key: "ieFront", label: "IE frontal", color: "#7c3aed" },
  ]);
  drawLineChart("chart-polar", rows, [
    { key: "polar", label: "JE polar", color: "#0f766e" },
  ]);
}

function renderSummary(rows) {
  const metrics = [
    ["Przekroje", rows.length, `od ${min(rows, "slice")} do ${max(rows, "slice")}`],
    ["Śr. moduł Younga", `${format(mean(rows, "young"))} GPa`, `max ${format(max(rows, "young"))}`],
    ["Śr. pole kory", `${format(mean(rows, "area"))} mm²`, `max ${format(max(rows, "area"))}`],
    ["Śr. sztywność polarna", format(mean(rows, "polar")), `max ${format(max(rows, "polar"))}`],
  ];

  summaryEl.innerHTML = metrics
    .map(([label, value, detail]) => `
      <article class="metric">
        <span>${label}</span>
        <strong>${value}</strong>
        <small>${detail}</small>
      </article>
    `)
    .join("");
}

function smoothRows(rows, windowSize) {
  const half = Math.floor(windowSize / 2);
  return rows.map((row, index) => {
    const start = Math.max(0, index - half);
    const end = Math.min(rows.length, index + half + 1);
    const subset = rows.slice(start, end);
    return {
      ...row,
      area: mean(subset, "area"),
      ratio: mean(subset, "ratio"),
      young: mean(subset, "young"),
      ieSag: mean(subset, "ieSag"),
      ieFront: mean(subset, "ieFront"),
      polar: mean(subset, "polar"),
    };
  });
}

function drawLineChart(canvasId, rows, series) {
  const canvas = document.querySelector(`#${canvasId}`);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { top: 24, right: 24, bottom: 44, left: 62 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const xMin = min(rows, "slice");
  const xMax = max(rows, "slice");
  const yValues = series.flatMap((item) => rows.map((row) => row[item.key]));
  let yMin = Math.min(...yValues);
  let yMax = Math.max(...yValues);
  if (yMin === yMax) yMax = yMin + 1;
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  drawGrid(ctx, pad, plotW, plotH, xMin, xMax, yMin, yMax);

  series.forEach((item) => {
    ctx.beginPath();
    rows.forEach((row, index) => {
      const x = pad.left + ((row.slice - xMin) / (xMax - xMin || 1)) * plotW;
      const y = pad.top + (1 - (row[item.key] - yMin) / (yMax - yMin)) * plotH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 3;
    ctx.stroke();
  });

  drawLegend(ctx, series, pad.left, 18);
}

function drawGrid(ctx, pad, plotW, plotH, xMin, xMax, yMin, yMax) {
  ctx.strokeStyle = "#d8dee6";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#667085";
  ctx.font = "18px Segoe UI, Arial, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (plotH / 4) * i;
    const value = yMax - ((yMax - yMin) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillText(format(value), pad.left - 10, y);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i += 1) {
    const x = pad.left + (plotW / 4) * i;
    const value = xMin + ((xMax - xMin) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.fillText(Math.round(value), x, pad.top + plotH + 14);
  }
}

function drawLegend(ctx, series, x, y) {
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.font = "18px Segoe UI, Arial, sans-serif";

  let offset = 0;
  series.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(x + offset, y - 5, 22, 10);
    ctx.fillStyle = "#17202a";
    ctx.fillText(item.label, x + offset + 30, y);
    offset += ctx.measureText(item.label).width + 74;
  });
}

function mean(rows, key) {
  return rows.reduce((sum, row) => sum + row[key], 0) / rows.length;
}

function min(rows, key) {
  return Math.min(...rows.map((row) => row[key]));
}

function max(rows, key) {
  return Math.max(...rows.map((row) => row[key]));
}

function format(value) {
  if (!Number.isFinite(value)) return "0";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}
