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

function getFlagLabel(value, type) {
  if (type === 'capture') {
    if (value >= 75) return { label: 'Good', className: 'good' };
    if (value >= 45) return { label: 'Moderate', className: 'moderate' };
    return { label: 'Bad', className: 'bad' };
  }

  if (type === 'duty') {
    if (value <= 3.5) return { label: 'Good', className: 'good' };
    if (value <= 5.0) return { label: 'Moderate', className: 'moderate' };
    return { label: 'Bad', className: 'bad' };
  }

  if (value >= 3500) return { label: 'Good', className: 'good' };
  if (value >= 2500) return { label: 'Moderate', className: 'moderate' };
  return { label: 'Bad', className: 'bad' };
}

function updateFlag(elementId, value, type) {
  const { label, className } = getFlagLabel(value, type);
  const element = document.getElementById(elementId);
  if (!element) return;
  element.textContent = label;
  element.className = `status-badge ${className}`;
}

function setMetricValues(data) {
  const capture = Number(data.capture_efficiency_pct);
  const duty = Number(data.reboiler_duty_mj_kg_co2);
  const co2Removed = Number(data.estimated_co2_removed_kg_hr);

  document.getElementById('captureValue').textContent = capture.toFixed(2);
  document.getElementById('dutyValue').textContent = duty.toFixed(4);
  document.getElementById('removedValue').textContent = co2Removed.toFixed(2);
  document.getElementById('removedKmol').textContent = Number(data.estimated_co2_removed_kmol_hr).toFixed(4);
  document.getElementById('thermalLoad').textContent = Number(data.reboiler_total_mj_hr).toFixed(2);

  updateFlag('captureFlag', capture, 'capture');
  updateFlag('dutyFlag', duty, 'duty');
  updateFlag('removedFlag', co2Removed, 'co2');
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
      x: ['Flue gas', 'CO₂ inlet', 'Absorber T', 'L/G', 'MEA', 'Lean load'],
      y: [1.15, 2.13, 0.60, 4.31, 2.45, 6.40],
      marker: {
        color: ['#4da3ff', '#50e0ff', '#ffb84d', '#7f8cff', '#38d39f', '#ff7a7a'],
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

function setupGalleryModal() {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  const modalTitle = document.getElementById('modalTitle');
  const closeButton = document.querySelector('.image-modal-close');

  document.querySelectorAll('.gallery-card').forEach((card) => {
    card.addEventListener('click', () => {
      modalImg.src = card.dataset.src;
      modalTitle.textContent = card.dataset.title;
      modal.classList.add('is-visible');
      modal.setAttribute('aria-hidden', 'false');
    });
  });

  const closeModal = () => {
    modal.classList.remove('is-visible');
    modal.setAttribute('aria-hidden', 'true');
  };

  closeButton.addEventListener('click', closeModal);
  modal.addEventListener('click', (event) => {
    if (event.target.dataset.close === 'true' || event.target === modal) {
      closeModal();
    }
  });
}

window.addEventListener('load', async () => {
  apply3DMovement();
  setupGalleryModal();
  await runPrediction();
});

window.addEventListener('resize', () => {
  Plotly.Plots.resize(document.getElementById('captureDutyChart'));
  Plotly.Plots.resize(document.getElementById('sensitivityChart'));
});
