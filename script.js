const API_URL = "https://portal-eska.onrender.com/"

async function getMessage() {
  const response = await fetch(API_URL);
  const data = await response.json();
  document.getElementById('output').innerText = data.message;
}

