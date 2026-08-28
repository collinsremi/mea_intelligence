const DEFAULT_PAYLOAD = {
  Flue_Gas_Flow_kmolhr: 1200,
  "Inlet_CO2_mol%": 12,
  Absorber_Temp_C: 45,
  "MEA_Concentration_wt%": 30,
  LG_Ratio: 4.2,
};

const featureOrder = [
  'Flue_Gas_Flow_kmolhr',
  'Inlet_CO2_mol%',
  'Absorber_Temp_C',
  'MEA_Concentration_wt%',
  'LG_Ratio',
];

function readPayload() {
  return featureOrder.reduce((acc, key) => {
    const id = key;
    const value = document.getElementById(id)?.value;
    acc[key] = Number(value || 0);
    return acc;
  }, {});
}

function setMetricValues(data) {
  document.getElementById('captureValue').textContent = Number(data.capture_efficiency_pct).toFixed(2);
  document.getElementById('dutyValue').textContent = Number(data.reboiler_duty_mj_kg_co2).toFixed(4);
  document.getElementById('removedValue').textContent = Number(data.estimated_co2_removed_kg_hr).toFixed(2);
  document.getElementById('removedKmol').textContent = Number(data.estimated_co2_removed_kmol_hr).toFixed(4);
  document.getElementById('thermalLoad').textContent = Number(data.reboiler_total_mj_hr).toFixed(2);
}

window.runPrediction = async function runPrediction() {
  const payload = readPayload();
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  setMetricValues(data);
  drawCharts(data);
}

function drawCharts(data) {
  const capture = data.capture_efficiency_pct;
  const duty = data.reboiler_duty_mj_kg_co2;

  const scatter = {
    x: [capture - 10, capture, capture + 10],
    y: [duty + 0.8, duty, duty + 0.8],
    mode: 'markers+lines',
    type: 'scatter',
    line: { color: '#4da3ff', width: 3 },
    marker: { size: 10, color: '#50e0ff' },
    name: 'Predicted',
  };

  const layout1 = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 40, r: 20, t: 24, b: 32 },
    font: { color: '#edf6ff' },
    xaxis: { title: 'Capture efficiency (%)', color: '#cfeaff' },
    yaxis: { title: 'Reboiler duty (MJ/kg)', color: '#cfeaff' },
    modebar: { color: '#dfeaff' }
  };

  Plotly.newPlot('captureDutyChart', [scatter], layout1, { responsive: true, displayModeBar: false });

  const sensitivityData = [
    {
      type: 'bar',
      x: ['Flue gas', 'CO₂ inlet', 'Absorber T', 'MEA', 'L/G'],
      y: [0.78, 0.92, 0.43, 0.86, 0.97],
      marker: {
        color: ['#4da3ff', '#50e0ff', '#ffb84d', '#38d39f', '#8db9ff'],
      },
    }
  ];

  const layout2 = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 40, r: 20, t: 24, b: 40 },
    font: { color: '#edf6ff' },
    xaxis: { color: '#cfeaff' },
    yaxis: { title: 'Relative influence', color: '#cfeaff' },
    modebar: { color: '#dfeaff' }
  };

  Plotly.newPlot('sensitivityChart', sensitivityData, layout2, { responsive: true, displayModeBar: false });
}

function apply3DMovement() {
  const tiltElements = document.querySelectorAll('.tilt-panel');
  tiltElements.forEach((el) => {
    el.addEventListener('pointermove', (event) => {
      const rect = el.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      const rotateY = (x - 0.5) * 12;
      const rotateX = (0.5 - y) * 12;
      el.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px)`;
    });

    el.addEventListener('pointerleave', () => {
      el.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg)';
    });
  });
}

document.getElementById('runPrediction').addEventListener('click', runPrediction);

window.addEventListener('load', async () => {
  apply3DMovement();
  await runPrediction();
});

window.addEventListener('resize', () => {
  Plotly.Plots.resize(document.getElementById('captureDutyChart'));
  Plotly.Plots.resize(document.getElementById('sensitivityChart'));
});
