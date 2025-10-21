// Update plot with mathematical function
async function updatePlot() {
  const func = document.getElementById('function').value;
  const color = document.getElementById('color').value;
  
  const response = await fetch('/plot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ function: func, color: color })
  });
  
  const data = await response.json();
  document.getElementById('plot').src = "data:image/png;base64," + data.plot;
}

// Upload and plot file
async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('color', document.getElementById('csvColor').value);
  
  const fileInfo = document.getElementById('fileInfo');
  fileInfo.textContent = `Uploading ${file.name}...`;
  fileInfo.style.color = '#666';
  
  try {
    const response = await fetch('/plot-csv', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.error) {
      fileInfo.textContent = `Error: ${data.error}`;
      fileInfo.style.color = 'red';
    } else {
      document.getElementById('plot').src = "data:image/png;base64," + data.plot;
      fileInfo.textContent = `Successfully plotted data from ${file.name}`;
      fileInfo.style.color = 'green';
    }
  } catch (error) {
    fileInfo.textContent = `Error: ${error.message}`;
    fileInfo.style.color = 'red';
  }
}

// Setup drag and drop functionality
function setupDropZone() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  
  // Click to browse
  dropZone.addEventListener('click', () => fileInput.click());
  
  // Drag over
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  
  // Drag leave
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });
  
  // Drop
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      uploadFile(files[0]);
    }
  });
  
  // File input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      uploadFile(e.target.files[0]);
    }
  });
}

// Setup event listeners
function setupEventListeners() {
  document.getElementById('plotBtn').addEventListener('click', updatePlot);
  document.getElementById('function').addEventListener('change', updatePlot);
  document.getElementById('color').addEventListener('change', updatePlot);
}

// Initialize on page load
window.onload = () => {
  setupDropZone();
  setupEventListeners();
};

