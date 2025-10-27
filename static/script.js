

// Gaussian analysis functionality
let symmetryFile = null;
let assessmentFile = null;



async function performGaussianAnalysis() {
  if (!symmetryFile || !assessmentFile) {
    alert('Please upload both symmetry and assessment CSV files');
    return;
  }
  
  const formData = new FormData();
  formData.append('symmetry_file', symmetryFile);
  formData.append('assessment_file', assessmentFile);
  formData.append('category', document.getElementById('category').value);
  formData.append('sex', document.getElementById('sex').value);
  formData.append('color', document.getElementById('gaussianColor').value);
  
  const statsDiv = document.getElementById('gaussianStats');
  statsDiv.style.display = 'none';
  
  try {
    const response = await fetch('/plot-gaussian', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.error) {
      alert(`Error: ${data.error}`);
    } else {
      document.getElementById('plot').src = "data:image/png;base64," + data.plot;
      
      // Display statistics
      statsDiv.innerHTML = `
        <h4>Gaussian Fit Statistics</h4>
        <p><strong>Mean (μ):</strong> ${data.stats.mean.toFixed(2)}%</p>
        <p><strong>Standard Deviation (σ):</strong> ${data.stats.std_dev.toFixed(2)}%</p>
        <p><strong>Amplitude:</strong> ${data.stats.amplitude.toFixed(2)}</p>
        <p><strong>Sample Size:</strong> ${data.stats.sample_size}</p>
      `;
      statsDiv.style.display = 'block';
    }
  } catch (error) {
    alert(`Analysis failed: ${error.message}`);
  }
}

function setupGaussianDropZones() {
  // Symmetry file drop zone
  const symmetryDropZone = document.getElementById('symmetryDropZone');
  const symmetryInput = document.getElementById('symmetryInput');
  const symmetryInfo = document.getElementById('symmetryInfo');
  
  symmetryDropZone.addEventListener('click', () => symmetryInput.click());
  
  symmetryDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    symmetryDropZone.classList.add('dragover');
  });
  
  symmetryDropZone.addEventListener('dragleave', () => {
    symmetryDropZone.classList.remove('dragover');
  });
  
  symmetryDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    symmetryDropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleSymmetryFile(files[0]);
    }
  });
  
  symmetryInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleSymmetryFile(e.target.files[0]);
    }
  });
  
  function handleSymmetryFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      symmetryInfo.textContent = 'Please select a CSV file';
      symmetryInfo.style.color = 'red';
      return;
    }
    
    symmetryFile = file;
    symmetryInfo.textContent = `Symmetry file loaded: ${file.name}`;
    symmetryInfo.style.color = 'green';
  }
  
  // Assessment file drop zone
  const assessmentDropZone = document.getElementById('assessmentDropZone');
  const assessmentInput = document.getElementById('assessmentInput');
  const assessmentInfo = document.getElementById('assessmentInfo');
  
  assessmentDropZone.addEventListener('click', () => assessmentInput.click());
  
  assessmentDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    assessmentDropZone.classList.add('dragover');
  });
  
  assessmentDropZone.addEventListener('dragleave', () => {
    assessmentDropZone.classList.remove('dragover');
  });
  
  assessmentDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    assessmentDropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleAssessmentFile(files[0]);
    }
  });
  
  assessmentInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleAssessmentFile(e.target.files[0]);
    }
  });
  
  function handleAssessmentFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      assessmentInfo.textContent = 'Please select a CSV file';
      assessmentInfo.style.color = 'red';
      return;
    }
    
    assessmentFile = file;
    assessmentInfo.textContent = `Assessment file loaded: ${file.name}`;
    assessmentInfo.style.color = 'green';
  }
}



// Setup event listeners
function setupEventListeners() {
  document.getElementById('analyzeBtn').addEventListener('click', performGaussianAnalysis);
}

// Initialize on page load
window.onload = () => {
  setupGaussianDropZones();
  setupEventListeners();
};

