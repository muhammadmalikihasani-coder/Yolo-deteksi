// Global variables
let model;
let video;
let canvas;
let ctx;
let isCameraActive = false;

// Initialize the application
async function init() {
    console.log('Initializing YOLO Object Detection...');
    
    // Get DOM elements
    video = document.getElementById('video');
    canvas = document.getElementById('canvasOutput');
    ctx = canvas.getContext('2d');
    
    // Load COCO-SSD model (YOLO-based)
    try {
        showLoading();
        model = await cocoSsd.load();
        hideLoading();
        console.log('Model loaded successfully');
        
        // Enable file input
        setupFileInput();
        
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Error loading AI model. Please refresh the page.');
    }
}

// Setup file input handler
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            if (file.size > 5 * 1024 * 1024) {
                alert('File terlalu besar! Maksimal 5MB.');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                canvas.style.display = 'none';
                video.style.display = 'none';
                isCameraActive = false;
                
                // Process the image
                processImage(imagePreview);
            };
            reader.readAsDataURL(file);
        }
    });
}

// Open camera
async function openCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        video.srcObject = stream;
        video.style.display = 'block';
        imagePreview.style.display = 'none';
        canvas.style.display = 'none';
        isCameraActive = true;
        
        // Start real-time detection
        detectRealTime();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Tidak dapat mengakses kamera. Pastikan izin kamera sudah diberikan.');
    }
}

// Take picture from camera
function takePicture() {
    if (!isCameraActive) {
        alert('Buka kamera terlebih dahulu!');
        return;
    }
    
    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Hide video and show canvas
    video.style.display = 'none';
    canvas.style.display = 'block';
    isCameraActive = false;
    
    // Process the captured image
    processImage(canvas);
}

// Real-time detection from camera
async function detectRealTime() {
    if (!isCameraActive || !model) return;
    
    try {
        // Set canvas size to video size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw video frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Perform detection
        const startTime = performance.now();
        const predictions = await model.detect(canvas);
        const endTime = performance.now();
        
        // Draw bounding boxes
        drawPredictions(predictions);
        
        // Update results
        updateResults(predictions, endTime - startTime);
        
        // Continue detection
        requestAnimationFrame(detectRealTime);
        
    } catch (error) {
        console.error('Detection error:', error);
        setTimeout(detectRealTime, 100);
    }
}

// Process uploaded image
async function processImage(imgElement) {
    if (!model) {
        alert('Model belum siap. Tunggu sebentar...');
        return;
    }
    
    showLoading();
    
    try {
        const startTime = performance.now();
        const predictions = await model.detect(imgElement);
        const endTime = performance.now();
        
        // Draw on canvas
        canvas.width = imgElement.naturalWidth || imgElement.width;
        canvas.height = imgElement.naturalHeight || imgElement.height;
        
        // Draw original image to canvas
        ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);
        
        // Draw predictions
        drawPredictions(predictions);
        
        // Show canvas and hide image preview
        canvas.style.display = 'block';
        imgElement.style.display = 'none';
        
        // Update results
        updateResults(predictions, endTime - startTime);
        
    } catch (error) {
        console.error('Detection error:', error);
        alert('Error processing image: ' + error.message);
    }
    
    hideLoading();
}

// Draw bounding boxes and labels
function drawPredictions(predictions) {
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(isCameraActive ? video : document.getElementById('imagePreview'), 0, 0, canvas.width, canvas.height);
    
    predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const label = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
        
        // Draw bounding box
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        
        // Draw label background
        ctx.fillStyle = '#FF6B6B';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y - 20, textWidth + 10, 20);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(label, x + 5, y - 5);
    });
}

// Update results display
function updateResults(predictions, processingTime) {
    const resultList = document.getElementById('resultList');
    const objectCount = document.getElementById('objectCount');
    const confidenceAvg = document.getElementById('confidenceAvg');
    const processingTimeElem = document.getElementById('processingTime');
    
    // Clear previous results
    resultList.innerHTML = '';
    
    if (predictions.length === 0) {
        resultList.innerHTML = '<div class="result-item">Tidak ada objek terdeteksi</div>';
        objectCount.textContent = '0';
        confidenceAvg.textContent = '0%';
        processingTimeElem.textContent = '0ms';
        return;
    }
    
    // Calculate statistics
    const totalConfidence = predictions.reduce((sum, pred) => sum + pred.score, 0);
    const averageConfidence = (totalConfidence / predictions.length) * 100;
    
    // Update stats
    objectCount.textContent = predictions.length;
    confidenceAvg.textContent = Math.round(averageConfidence) + '%';
    processingTimeElem.textContent = Math.round(processingTime) + 'ms';
    
    // Display individual results
    predictions.forEach((prediction, index) => {
        const confidencePercent = Math.round(prediction.score * 100);
        
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        resultItem.innerHTML = `
            <strong>${index + 1}. ${prediction.class}</strong>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                <span>Akurasi: ${confidencePercent}%</span>
                <span style="font-size: 0.8rem; color: #666;">${Math.round(prediction.bbox[2])}×${Math.round(prediction.bbox[3])}px</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
            </div>
        `;
        
        resultList.appendChild(resultItem);
    });
}

// Utility functions
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

// Drag and drop functionality
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.querySelector('.upload-area');
    
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.background = '#eef2ff';
        uploadArea.style.borderColor = '#764ba2';
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.style.background = '#f8f9ff';
        uploadArea.style.borderColor = '#667eea';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.background = '#f8f9ff';
        uploadArea.style.borderColor = '#667eea';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            document.getElementById('fileInput').files = files;
            const event = new Event('change');
            document.getElementById('fileInput').dispatchEvent(event);
        }
    });
});

// Initialize when page loads
window.addEventListener('load', init);
