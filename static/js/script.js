const form = document.getElementById('predict-form');
const resultBox = document.getElementById('resultBox');
const svmLabel = document.getElementById('svmLabel');
const svmConfidence = document.getElementById('svmConfidence');
const lstmLabel = document.getElementById('lstmLabel');
const lstmConfidence = document.getElementById('lstmConfidence');
const clearBtn = document.getElementById('clearBtn');

if (form) {
  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const review = document.getElementById('reviewInput').value.trim();
    if (!review) {
      alert('Please enter a review before analyzing.');
      return;
    }

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review }),
      });

      const data = await response.json();

      if (!response.ok) {
        alert(data.error || 'Prediction failed.');
        return;
      }

      svmLabel.textContent = data.svm.label;
      svmConfidence.textContent = data.svm.confidence !== null
        ? `Confidence: ${data.svm.confidence}`
        : 'Confidence unavailable';

      lstmLabel.textContent = data.lstm.label;
      lstmConfidence.textContent = data.lstm.confidence !== null
        ? `Confidence: ${data.lstm.confidence}`
        : 'Confidence unavailable';

      resultBox.classList.remove('hidden');
    } catch (error) {
      alert('There was an error connecting to the prediction server.');
      console.error(error);
    }
  });
}

if (clearBtn) {
  clearBtn.addEventListener('click', () => {
    document.getElementById('reviewInput').value = '';
    resultBox.classList.add('hidden');
  });
}
