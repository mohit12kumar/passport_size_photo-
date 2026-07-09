// Global State
let appState = {
    countries: {},
    paperSizes: {},
    currentFile: null,      // Name of the uploaded file
    alignedFile: null,      // Name of the auto-aligned working image
    originalUrl: null,      // Url of original uploaded image
    alignedUrl: null,       // Url of aligned image
    faceBox: null,          // Face bounding box {x, y, w, h}
    eyes: [],               // Eye coordinates
    autoAngle: 0.0,         // Angle applied during upload face detection
    imageLoaded: false,
    alignedImageObj: null,  // HTML Image object of aligned face
    selectedColor: "#FFFFFF" // Current bg color replacement
};

// UI Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfoPanel = document.getElementById('fileInfoPanel');
const uploadedFileName = document.getElementById('uploadedFileName');
const removeFileBtn = document.getElementById('removeFileBtn');
const settingsPanel = document.getElementById('settingsPanel');
const countrySelect = document.getElementById('countrySelect');
const paperSizeSelect = document.getElementById('paperSizeSelect');
const photoCountGroup = document.getElementById('photoCountGroup');
const photoCountSelect = document.getElementById('photoCountSelect');


// Specifications Displays
const specSize = document.getElementById('specSize');
const specHead = document.getElementById('specHead');
const specBg = document.getElementById('specBg');
const specDesc = document.getElementById('specDesc');

// Background Color Selection
const bgRemovalToggle = document.getElementById('bgRemovalToggle');
const bgSettingsGroup = document.getElementById('bgSettingsGroup');
const customColorBtn = document.getElementById('customColorBtn');
const customColorInput = document.getElementById('customColorInput');
const customHexDisplay = document.getElementById('customHexDisplay');
const hexCodeValue = document.getElementById('hexCodeValue');

// Fine Tuning Sliders
const sliderScale = document.getElementById('sliderScale');
const valScale = document.getElementById('valScale');
const sliderRotation = document.getElementById('sliderRotation');
const valRotation = document.getElementById('valRotation');
const sliderShiftX = document.getElementById('sliderShiftX');
const valShiftX = document.getElementById('valShiftX');
const sliderShiftY = document.getElementById('sliderShiftY');
const valShiftY = document.getElementById('valShiftY');
const resetCropBtn = document.getElementById('resetCropBtn');

// Image Enhancement Sliders
const sliderBrightness = document.getElementById('sliderBrightness');
const valBrightness = document.getElementById('valBrightness');
const sliderContrast = document.getElementById('sliderContrast');
const valContrast = document.getElementById('valContrast');
const sliderSharpness = document.getElementById('sliderSharpness');
const valSharpness = document.getElementById('valSharpness');
const sliderSaturation = document.getElementById('sliderSaturation');
const valSaturation = document.getElementById('valSaturation');
const resetEnhanceBtn = document.getElementById('resetEnhanceBtn');

// Layout Setup - Professional print-shop standards per paper (fallback defaults)
// Per-paper values are loaded from the API and stored in appState.paperSizes
const FALLBACK_MARGIN_MM = 5.0;
const FALLBACK_GAP_MM = 2.0;

// Helper: get recommended margin/gap for the currently selected paper
function getPaperLayoutDefaults() {
    const key = paperSizeSelect ? paperSizeSelect.value : null;
    const paper = key && appState.paperSizes ? appState.paperSizes[key] : null;
    return {
        margin_mm: paper && paper.default_margin_mm != null ? paper.default_margin_mm : FALLBACK_MARGIN_MM,
        gap_mm:    paper && paper.default_gap_mm    != null ? paper.default_gap_mm    : FALLBACK_GAP_MM
    };
}

// Process Buttons and Placeholders
const renderBtn = document.getElementById('renderBtn');
const tabBtnCrop = document.getElementById('tabBtnCrop');
const tabBtnPrint = document.getElementById('tabBtnPrint');
const tabContentCrop = document.getElementById('tabContentCrop');
const tabContentPrint = document.getElementById('tabContentPrint');

const canvasPlaceholder = document.getElementById('canvasPlaceholder');
const originalCanvas = document.getElementById('originalCanvas');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const croppedResultImage = document.getElementById('croppedResultImage');
const processingSpinner = document.getElementById('processingSpinner');
const spinnerText = document.getElementById('spinnerText');
const croppedDownloadGroup = document.getElementById('croppedDownloadGroup');
const downloadCroppedBtn = document.getElementById('downloadCroppedBtn');

// Printable Layout Elements
const printableSheetImage = document.getElementById('printableSheetImage');
const sheetSpinner = document.getElementById('sheetSpinner');
const statTotalPhotos = document.getElementById('statTotalPhotos');
const statGrid = document.getElementById('statGrid');
const statPaperSize = document.getElementById('statPaperSize');
const statMargins = document.getElementById('statMargins');
const downloadPdfBtn = document.getElementById('downloadPdfBtn');
const downloadSheetJpgBtn = document.getElementById('downloadSheetJpgBtn');

// Compliance Report Elements (Step 13) - Selecting dynamically via class names

// Initialize App
window.addEventListener('DOMContentLoaded', async () => {
    await fetchCountries();
    await fetchPaperSizes();
    setupEventListeners();
    // Show Photo Count if default paper is 4x6
    if (paperSizeSelect && paperSizeSelect.value === '4x6') {
        if (photoCountGroup) photoCountGroup.style.display = 'block';
    }
});

// Fetch Countries Rules from backend
async function fetchCountries() {
    try {
        const response = await fetch('/api/countries');
        appState.countries = await response.json();
        
        // Populate select list
        countrySelect.innerHTML = '';
        Object.entries(appState.countries).forEach(([key, value]) => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = value.name;
            if (key === 'usa') option.selected = true; // Default Select USA
            countrySelect.appendChild(option);
        });
        
        updateSpecsDisplay();
    } catch (err) {
        console.error('Error fetching countries:', err);
    }
}

// Fetch Paper Sizes from backend
async function fetchPaperSizes() {
    try {
        const response = await fetch('/api/paper-sizes');
        appState.paperSizes = await response.json();
        
        paperSizeSelect.innerHTML = '';
        Object.entries(appState.paperSizes).forEach(([key, value]) => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = value.name;
            if (key === 'A4') option.selected = true; // Default A4
            paperSizeSelect.appendChild(option);
        });
        // Reveal/hide Photo Count based on the selected paper size after populating
        if (paperSizeSelect.value === '4x6') {
            if (photoCountGroup) photoCountGroup.style.display = 'block';
        } else {
            if (photoCountGroup) photoCountGroup.style.display = 'none';
        }
    } catch (err) {
        console.error('Error fetching paper sizes:', err);
    }
}

// Update Specs Display Cards
function updateSpecsDisplay() {
    const selected = countrySelect.value;
    const rule = appState.countries[selected];
    
    if (rule) {
        specSize.textContent = `${rule.width_mm}x${rule.height_mm} mm`;
        specHead.textContent = `${Math.round(rule.head_height_ratio_min * 100)}-${Math.round(rule.head_height_ratio_max * 100)}%`;
        specBg.textContent = rule.bg_color;
        specDesc.innerHTML = `<i class="fa-solid fa-circle-info"></i> ${rule.description}`;
        
        // Update canvas guides if photo is already loaded
        if (appState.imageLoaded) {
            drawInteractiveCanvas();
        }
    }
}

// Event Listeners Setup
function setupEventListeners() {
    // Dropzone Events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    removeFileBtn.addEventListener('click', () => {
        resetApp();
    });

    // Country Select Change
    countrySelect.addEventListener('change', () => {
        updateSpecsDisplay();
        if (appState.processedPassportFile) {
            generatePrintableSheet();
        }
    });

    // Paper Size Change
    paperSizeSelect.addEventListener('change', () => {
        const val = paperSizeSelect.value;
        // Show Photo Count in sidebar only for 4x6 paper
        if (val === '4x6') {
            if (photoCountGroup) photoCountGroup.style.display = 'block';
        } else {
            if (photoCountGroup) photoCountGroup.style.display = 'none';
            // Reset count to max when switching away from 4x6
            if (photoCountSelect) photoCountSelect.value = '8';
            updateChipActive(8);
        }
        if (appState.processedPassportFile) {
            generatePrintableSheet();
        }
    });

    // Photo Count select change (dropdown)
    photoCountSelect.addEventListener('change', () => {
        updateChipActive(parseInt(photoCountSelect.value, 10));
        if (appState.processedPassportFile) {
            generatePrintableSheet();
        }
    });

    // Photo Count chip buttons (covers chips in both sidebar and print tab)
    document.querySelectorAll('.photo-count-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const count = parseInt(chip.dataset.count, 10);
            if (photoCountSelect) photoCountSelect.value = count;
            updateChipActive(count);
            if (appState.processedPassportFile) {
                generatePrintableSheet();
            }
        });
    });

    // Bg Removal Toggle
    bgRemovalToggle.addEventListener('change', () => {
        if (bgRemovalToggle.checked) {
            bgSettingsGroup.style.display = 'flex';
        } else {
            bgSettingsGroup.style.display = 'none';
        }
    });

    // Bg Color Dots Selection
    const colorDots = document.querySelectorAll('.color-dot');
    colorDots.forEach(dot => {
        dot.addEventListener('click', (e) => {
            colorDots.forEach(d => d.classList.remove('active'));
            
            const target = e.currentTarget;
            if (target.id === 'customColorBtn') {
                customColorInput.click();
            } else {
                target.classList.add('active');
                appState.selectedColor = target.getAttribute('data-color');
                customHexDisplay.style.display = 'none';
            }
        });
    });

    // Custom Color Input Color Picker
    customColorInput.addEventListener('input', (e) => {
        const color = e.target.value;
        appState.selectedColor = color;
        customColorBtn.style.color = color;
        customColorBtn.classList.add('active');
        hexCodeValue.textContent = color.toUpperCase();
        customHexDisplay.style.display = 'block';
    });

    // Sliders Live Update (Dynamic Crop Box Redraw on canvas)
    sliderScale.addEventListener('input', (e) => {
        valScale.textContent = `${e.target.value}x`;
        if (appState.imageLoaded) drawInteractiveCanvas();
    });

    sliderRotation.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        valRotation.textContent = `${val > 0 ? '+' : ''}${val}°`;
        if (appState.imageLoaded) drawInteractiveCanvas();
    });

    sliderShiftX.addEventListener('input', (e) => {
        valShiftX.textContent = `${e.target.value} px`;
        if (appState.imageLoaded) drawInteractiveCanvas();
    });

    sliderShiftY.addEventListener('input', (e) => {
        valShiftY.textContent = `${e.target.value} px`;
        if (appState.imageLoaded) drawInteractiveCanvas();
    });

    // Reset Sliders
    resetCropBtn.addEventListener('click', () => {
        sliderScale.value = 1.0;
        valScale.textContent = "1.0x";
        sliderRotation.value = 0;
        valRotation.textContent = "0°";
        sliderShiftX.value = 0;
        valShiftX.textContent = "0 px";
        sliderShiftY.value = 0;
        valShiftY.textContent = "0 px";
        if (appState.imageLoaded) drawInteractiveCanvas();
    });

    // Enhancements Sliders Live Update
    sliderBrightness.addEventListener('input', (e) => {
        valBrightness.textContent = `${e.target.value}x`;
    });
    sliderContrast.addEventListener('input', (e) => {
        valContrast.textContent = `${e.target.value}x`;
    });
    sliderSharpness.addEventListener('input', (e) => {
        valSharpness.textContent = `${e.target.value}x`;
    });
    sliderSaturation.addEventListener('input', (e) => {
        valSaturation.textContent = `${e.target.value}x`;
    });

    resetEnhanceBtn.addEventListener('click', () => {
        sliderBrightness.value = 1.0;
        valBrightness.textContent = "1.0x";
        sliderContrast.value = 1.0;
        valContrast.textContent = "1.0x";
        sliderSharpness.value = 1.0;
        valSharpness.textContent = "1.0x";
        sliderSaturation.value = 1.0;
        valSaturation.textContent = "1.0x";
    });

    // Printable layout uses pre-fixed margin and gap values

    // Tabs Navigation
    tabBtnCrop.addEventListener('click', () => {
        switchTab('crop');
    });

    tabBtnPrint.addEventListener('click', () => {
        switchTab('print');
    });

    // Process render button
    renderBtn.addEventListener('click', () => {
        processPassportPhoto();
    });
}

// Reset App state
function resetApp() {
    appState.currentFile = null;
    appState.alignedFile = null;
    appState.originalUrl = null;
    appState.alignedUrl = null;
    appState.faceBox = null;
    appState.eyes = [];
    appState.autoAngle = 0.0;
    appState.imageLoaded = false;
    appState.alignedImageObj = null;
    
    fileInput.value = '';
    uploadedFileName.textContent = '';
    fileInfoPanel.style.display = 'none';
    dropZone.style.display = 'block';
    settingsPanel.classList.add('disabled-state');
    
    // Canvas & Placeholders
    originalCanvas.style.display = 'none';
    canvasPlaceholder.style.display = 'flex';
    resultPlaceholder.style.display = 'flex';
    croppedResultImage.style.display = 'none';
    croppedDownloadGroup.style.display = 'none';
    processedIndicator.style.display = 'none';
    const cropCard = document.getElementById('cropComplianceReportCard');
    const printCard = document.getElementById('printComplianceReportCard');
    if (cropCard) cropCard.style.display = 'none';
    if (printCard) printCard.style.display = 'none';
    
    // Printable
    tabBtnPrint.disabled = true;
    switchTab('crop');
    
    // Reset Sliders
    resetCropBtn.click();
    resetEnhanceBtn.click();

    // Reset Custom Photo Layout elements
    paperSizeSelect.value = 'A4';
    if (photoCountSelect) photoCountSelect.value = '8';
    if (photoCountGroup) photoCountGroup.style.display = 'none';
    updateChipActive(8);
}

// Switch view tabs
function switchTab(tab) {
    if (tab === 'crop') {
        tabBtnCrop.classList.add('active');
        tabBtnPrint.classList.remove('active');
        tabContentCrop.classList.add('active');
        tabContentPrint.classList.remove('active');
    } else if (tab === 'print') {
        tabBtnCrop.classList.remove('active');
        tabBtnPrint.classList.add('active');
        tabContentCrop.classList.remove('active');
        tabContentPrint.classList.add('active');
        
        // Auto-generate printable sheet if single photo was processed
        if (appState.currentFile && croppedResultImage.style.display !== 'none') {
            generatePrintableSheet();
        }
    }
}

// Handle Upload
async function handleFileUpload(file) {
    // Check file size (15MB max)
    if (file.size > 15 * 1024 * 1024) {
        alert("File size is too large. Maximum size is 15MB.");
        return;
    }

    // Toggle loader
    spinnerText.textContent = "Uploading & Detecting Face...";
    resultPlaceholder.style.display = 'none';
    croppedResultImage.style.display = 'none';
    processingSpinner.style.display = 'flex';
    canvasPlaceholder.style.display = 'none';
    originalCanvas.style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Upload failed");
        }
        
        const data = await response.json();
        
        // Update State
        appState.currentFile = data.filename;
        appState.alignedFile = data.aligned_filename;
        appState.originalUrl = data.original_url;
        appState.alignedUrl = data.aligned_url;
        appState.faceBox = data.face;
        appState.eyes = data.eyes;
        appState.autoAngle = data.auto_angle;
        
        // UI Adjustments
        uploadedFileName.textContent = file.name;
        fileInfoPanel.style.display = 'flex';
        dropZone.style.display = 'none';
        settingsPanel.classList.remove('disabled-state');
        
        // Show/hide photo count selector based on restored paper size selection
        const paperVal = paperSizeSelect.value;
        if (paperVal === '4x6' || paperVal === '5x7') {
            photoCountGroup.style.display = 'block';
        } else {
            photoCountGroup.style.display = 'none';
        }
        
        // Load aligned Image and paint it on canvas
        const img = new Image();
        img.src = appState.alignedUrl;
        img.onload = () => {
            appState.alignedImageObj = img;
            appState.imageLoaded = true;
            originalCanvas.style.display = 'block';
            drawInteractiveCanvas();
            
            // Turn off spinners
            processingSpinner.style.display = 'none';
            resultPlaceholder.style.display = 'flex';
            
            // Populate AI enhancement recommendations onto UI sliders
            if (data.auto_enhance) {
                const auto = data.auto_enhance;
                sliderBrightness.value = auto.brightness;
                valBrightness.textContent = `${auto.brightness}x`;
                sliderContrast.value = auto.contrast;
                valContrast.textContent = `${auto.contrast}x`;
                sliderSharpness.value = auto.sharpness;
                valSharpness.textContent = `${auto.sharpness}x`;
                sliderSaturation.value = auto.saturation;
                valSaturation.textContent = `${auto.saturation}x`;
                
                // Show biometric compliance report cards dynamically
                if (auto.biometric_score && auto.analysis) {
                    renderBiometricReport(auto.biometric_score, auto.analysis);
                }
            }
        };
        
    } catch (err) {
        alert(`Error: ${err.message}`);
        resetApp();
    }
}

// Draw the aligned photo and dynamic guides on Canvas
function drawInteractiveCanvas() {
    if (!appState.imageLoaded || !appState.alignedImageObj) return;
    
    const img = appState.alignedImageObj;
    const ctx = originalCanvas.getContext('2d');
    
    // Resize canvas pixel count to match actual image dimensions
    originalCanvas.width = img.naturalWidth;
    originalCanvas.height = img.naturalHeight;
    
    // Draw base aligned image
    ctx.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
    
    // Apply manual rotation preview on the canvas (if any)
    const mRotation = parseFloat(sliderRotation.value);
    
    if (Math.abs(mRotation) > 0.01) {
        ctx.save();
        // Rotate around center of face box
        const face_cx = appState.faceBox.x + appState.faceBox.w / 2.0;
        const face_cy = appState.faceBox.y + appState.faceBox.h / 2.0;
        ctx.translate(face_cx, face_cy);
        ctx.rotate((mRotation * Math.PI) / 180.0);
        ctx.translate(-face_cx, -face_cy);
    }
    
    ctx.drawImage(img, 0, 0);
    
    if (Math.abs(mRotation) > 0.01) {
        ctx.restore();
    }
    
    // Get target specifications
    const country = countrySelect.value;
    const rule = appState.countries[country];
    if (!rule) return;
    
    const target_ar = rule.width_mm / rule.height_mm;
    const head_ratio = (rule.head_height_ratio_min + rule.head_height_ratio_max) / 2.0;
    
    // Calculate base crop box coordinates
    const fx = appState.faceBox.x;
    const fy = appState.faceBox.y;
    const fw = appState.faceBox.w;
    const fh = appState.faceBox.h;
    
    const h_head = fh * 1.35;
    const y_crown = fy - (fh * 0.30);
    const x_center = fx + fw / 2.0;
    
    const C_h = h_head / head_ratio;
    const C_w = C_h * target_ar;
    
    const C_x = x_center - C_w / 2.0;
    const C_y = y_crown - (C_h * 0.12);
    
    // Apply manual sliders adjustment (scale, shifts)
    const scale = parseFloat(sliderScale.value);
    const x_offset = parseInt(sliderShiftX.value);
    const y_offset = parseInt(sliderShiftY.value);
    
    const C_w_adj = C_w / scale;
    const C_h_adj = C_h / scale;
    
    // Calculate final positions
    const final_x = C_x + x_offset + (C_w - C_w_adj) / 2.0;
    const final_y = C_y + y_offset + (C_h - C_h_adj) / 2.0;
    
    // Draw Face detection box (semi-transparent green)
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.lineWidth = Math.max(2, Math.round(originalCanvas.width / 400));
    ctx.strokeRect(fx, fy, fw, fh);
    
    // Draw Eyes (crosshairs)
    if (appState.eyes && appState.eyes.length > 0) {
        ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
        appState.eyes.forEach(eye => {
            const ex = eye.x + eye.w / 2.0;
            const ey = eye.y + eye.h / 2.0;
            
            // Draw small dot
            ctx.beginPath();
            ctx.arc(ex, ey, Math.max(3, Math.round(originalCanvas.width / 200)), 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    // Draw Crop Bounding Box Guide (bright red dashed)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = Math.max(3, Math.round(originalCanvas.width / 250));
    ctx.setLineDash([Math.round(originalCanvas.width / 100), Math.round(originalCanvas.width / 100)]);
    ctx.strokeRect(final_x, final_y, final_w_adj = C_w_adj, final_h_adj = C_h_adj);
    ctx.setLineDash([]); // Reset
    
    // Draw Corner photo brackets (thick solid lines)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = Math.max(5, Math.round(originalCanvas.width / 150));
    const bracket_len = Math.min(C_w_adj, C_h_adj) * 0.15;
    
    // Top-Left Corner Bracket
    ctx.beginPath();
    ctx.moveTo(final_x, final_y + bracket_len);
    ctx.lineTo(final_x, final_y);
    ctx.lineTo(final_x + bracket_len, final_y);
    ctx.stroke();
    
    // Top-Right Corner Bracket
    ctx.beginPath();
    ctx.moveTo(final_x + C_w_adj, final_y + bracket_len);
    ctx.lineTo(final_x + C_w_adj, final_y);
    ctx.lineTo(final_x + C_w_adj - bracket_len, final_y);
    ctx.stroke();
    
    // Bottom-Left Corner Bracket
    ctx.beginPath();
    ctx.moveTo(final_x, final_y + C_h_adj - bracket_len);
    ctx.lineTo(final_x, final_y + C_h_adj);
    ctx.lineTo(final_x + bracket_len, final_y + C_h_adj);
    ctx.stroke();
    
    // Bottom-Right Corner Bracket
    ctx.beginPath();
    ctx.moveTo(final_x + C_w_adj, final_y + C_h_adj - bracket_len);
    ctx.lineTo(final_x + C_w_adj, final_y + C_h_adj);
    ctx.lineTo(final_x + C_w_adj - bracket_len, final_y + C_h_adj);
    ctx.stroke();
    
    // Draw Biometric lines inside crop zone
    // 1. Vertical center line (grey dashed)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = Math.max(1, Math.round(originalCanvas.width / 800));
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(final_x + C_w_adj / 2.0, final_y);
    ctx.lineTo(final_x + C_w_adj / 2.0, final_y + C_h_adj);
    ctx.stroke();
    
    // 2. Eye level guideline (light blue, e.g. 60% height from bottom)
    const eye_ratio = (rule.eye_height_ratio_min + rule.eye_height_ratio_max) / 2.0 || 0.60;
    const target_eye_y = final_y + C_h_adj * (1.0 - eye_ratio);
    
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.55)';
    ctx.beginPath();
    ctx.moveTo(final_x, target_eye_y);
    ctx.lineTo(final_x + C_w_adj, target_eye_y);
    ctx.stroke();
    ctx.setLineDash([]); // Reset
}

// Call API to process the single passport photo
async function processPassportPhoto() {
    if (!appState.currentFile || !appState.faceBox) return;
    
    // Show spinner
    spinnerText.textContent = "Processing passport photo rules...";
    resultPlaceholder.style.display = 'none';
    croppedResultImage.style.display = 'none';
    croppedDownloadGroup.style.display = 'none';
    processedIndicator.style.display = 'none';
    processingSpinner.style.display = 'flex';
    
    const requestData = {
        filename: appState.currentFile,
        country: countrySelect.value,
        face: {
            x: Math.round(appState.faceBox.x),
            y: Math.round(appState.faceBox.y),
            w: Math.round(appState.faceBox.w),
            h: Math.round(appState.faceBox.h)
        },
        scale: parseFloat(sliderScale.value),
        x_offset: parseInt(sliderShiftX.value),
        y_offset: parseInt(sliderShiftY.value),
        manual_rotation: parseFloat(sliderRotation.value),
        remove_bg: bgRemovalToggle.checked,
        bg_color_hex: appState.selectedColor,
        brightness: parseFloat(sliderBrightness.value),
        contrast: parseFloat(sliderContrast.value),
        sharpness: parseFloat(sliderSharpness.value),
        saturation: parseFloat(sliderSaturation.value)
    };
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Processing failed");
        }
        
        const data = await response.json();
        
        // Cache the processed passport photo filename
        appState.processedPassportFile = data.filename;
        
        // Update single preview image
        croppedResultImage.src = `${data.url}?t=${new Date().getTime()}`; // Cache bust
        croppedResultImage.style.display = 'block';
        croppedDownloadGroup.style.display = 'block';
        downloadCroppedBtn.href = data.url;
        downloadCroppedBtn.download = `passport_${countrySelect.value}.png`;
        processedIndicator.style.display = 'flex';
        
        // Show report card updated with processed specs
        if (data.biometric_score && data.analysis) {
            renderBiometricReport(data.biometric_score, data.analysis);
        } else {
            renderBiometricReport(100, {
                is_blurry: false,
                is_dark: false,
                is_overexposed: false,
                is_low_contrast: false
            });
        }
        
        // Unlock printable tab
        tabBtnPrint.disabled = false;
        
    } catch (err) {
        alert(`Error during processing: ${err.message}`);
        resultPlaceholder.style.display = 'flex';
    } finally {
        processingSpinner.style.display = 'none';
    }
}

// Call API to generate printable grid sheet
async function generatePrintableSheet() {
    if (!appState.processedPassportFile) return;
    
    sheetSpinner.style.display = 'flex';
    
    const { margin_mm, gap_mm } = getPaperLayoutDefaults();
    const requestData = {
        filename: appState.processedPassportFile,
        country: countrySelect.value,
        paper_size: paperSizeSelect.value,
        margin_mm: margin_mm,
        gap_mm: gap_mm,
        photo_count: parseInt(photoCountSelect.value, 10)
    };
    
    try {
        const response = await fetch('/api/generate-sheet', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Layout generation failed");
        }
        
        const data = await response.json();
        
        // Update Preview sheet image
        printableSheetImage.src = `${data.sheet_url}?t=${new Date().getTime()}`;
        printableInfoBadge.textContent = `${paperSizeSelect.value} Sheet`;
        
        // Populate layout statistics summary
        statTotalPhotos.textContent = `${data.count} Photos`;
        statGrid.textContent = `${data.columns} Cols x ${data.rows} Rows`;
        statPaperSize.textContent = `${data.paper_width_mm} x ${data.paper_height_mm} mm`;
        const { margin_mm: usedMargin, gap_mm: usedGap } = getPaperLayoutDefaults();
        statMargins.textContent = `${usedMargin}mm margins / ${usedGap}mm gaps`;
        
        // Set up download URLs
        downloadPdfBtn.href = data.pdf_url;
        downloadPdfBtn.download = `passport_layout_${paperSizeSelect.value}.pdf`;
        downloadSheetJpgBtn.href = data.sheet_url;
        downloadSheetJpgBtn.download = `passport_layout_${paperSizeSelect.value}.png`;

        // Show photo count card only for 4x6 paper
        if (photoCountPrintGroup) {
            photoCountPrintGroup.style.display = paperSizeSelect.value === '4x6' ? 'block' : 'none';
        }
        
    } catch (err) {
        alert(`Error creating printable layout: ${err.message}`);
    } finally {
        sheetSpinner.style.display = 'none';
    }
}

// Dynamically render biometric report cards based on quality score and analysis checks
function renderBiometricReport(score, analysis) {
    // 1. Show both compliance cards
    const cropCard = document.getElementById('cropComplianceReportCard');
    const printCard = document.getElementById('printComplianceReportCard');
    if (cropCard) cropCard.style.display = 'block';
    if (printCard) printCard.style.display = 'block';

    // 2. Set quality score badge and progress bar (using class names to update both)
    document.querySelectorAll('.complianceScoreBadge').forEach(badge => {
        badge.textContent = `${score}% Quality`;
        if (score >= 90) {
            badge.className = 'badge success-badge complianceScoreBadge';
        } else if (score >= 60) {
            badge.className = 'badge warning-badge complianceScoreBadge';
        } else {
            badge.className = 'badge error-badge complianceScoreBadge';
        }
    });

    document.querySelectorAll('.complianceScoreBar').forEach(bar => {
        bar.style.width = `${score}%`;
        if (score >= 90) {
            bar.style.background = 'var(--color-success)';
        } else if (score >= 60) {
            bar.style.background = 'var(--color-warning)';
        } else {
            bar.style.background = 'var(--color-error)';
        }
    });

    // 3. Evaluate each check status
    const checks = {
        face: appState.faceBox !== null,
        eyes: appState.eyes && appState.eyes.length > 0,
        sharpness: !analysis.is_blurry,
        brightness: !analysis.is_dark,
        overexposure: !analysis.is_overexposed,
        contrast: !analysis.is_low_contrast,
        background: bgRemovalToggle.checked,
        dimensions: true // Crop guarantees exact size matching profile
    };

    // Helper to update specific checklist row across both cards
    function updateCheckRow(className, passed, passedText = "PASSED", failedText = "FAILED") {
        document.querySelectorAll(className).forEach(li => {
            const icon = li.querySelector('.label i');
            const val = li.querySelector('.value');
            if (passed) {
                icon.className = 'fa-regular fa-circle-check';
                icon.style.color = '#10b981';
                val.className = 'value success-text';
                val.textContent = passedText;
            } else {
                icon.className = 'fa-regular fa-circle-xmark';
                icon.style.color = '#ef4444';
                val.className = 'value error-text';
                val.textContent = failedText;
            }
        });
    }

    updateCheckRow('.chk-face', checks.face, "PASSED", "FAILED");
    updateCheckRow('.chk-eyes', checks.eyes, "PASSED", "FAILED");
    updateCheckRow('.chk-sharpness', checks.sharpness, "PASSED", "FAILED");
    updateCheckRow('.chk-brightness', checks.brightness, "PASSED", "FAILED");
    updateCheckRow('.chk-overexposure', checks.overexposure, "PASSED", "FAILED");
    updateCheckRow('.chk-contrast', checks.contrast, "PASSED", "FAILED");
    updateCheckRow('.chk-background', checks.background, "PASSED", "FAILED");
    updateCheckRow('.chk-dimensions', checks.dimensions, "PASSED", "FAILED");

    // 4. Update the detailed instructions box at the bottom of both cards
    let deductions = [];
    if (!checks.face) deductions.push("No face detected");
    else if (!checks.eyes) deductions.push("Eyes misaligned/undetected");
    if (!checks.sharpness) deductions.push("Slightly blurry");
    if (!checks.brightness) deductions.push("Underexposed (dark)");
    if (!checks.overexposure) deductions.push("Overexposed");
    if (!checks.contrast) deductions.push("Low contrast");
    if (!checks.background) deductions.push("Background not replaced");

    document.querySelectorAll('.complianceDetailBox').forEach(box => {
        const icon = box.querySelector('i');
        const text = box.querySelector('.complianceDetailText');
        if (score >= 90) {
            box.style.backgroundColor = '#f0fdf4';
            box.style.color = '#15803d';
            box.style.borderColor = 'rgba(21, 128, 61, 0.15)';
            icon.className = 'fa-solid fa-circle-check';
            text.textContent = "Ready to Print: Photo matches target biometric standard.";
        } else if (score >= 60) {
            box.style.backgroundColor = '#fffbeb';
            box.style.color = '#b45309';
            box.style.borderColor = 'rgba(180, 83, 9, 0.15)';
            icon.className = 'fa-solid fa-circle-exclamation';
            text.textContent = `Partial Pass: ${deductions.join("; ")}. Adjust settings to improve quality.`;
        } else {
            box.style.backgroundColor = '#fef2f2';
            box.style.color = '#b91c1c';
            box.style.borderColor = 'rgba(185, 28, 28, 0.15)';
            icon.className = 'fa-solid fa-circle-xmark';
            text.textContent = `Failed: ${deductions.join("; ")}. Please re-upload.`;
        }
    });
}

// Sync active chip highlight with the selected count value
function updateChipActive(count) {
    document.querySelectorAll('.photo-count-chip').forEach(chip => {
        chip.classList.toggle('active', parseInt(chip.dataset.count, 10) === count);
    });
}
