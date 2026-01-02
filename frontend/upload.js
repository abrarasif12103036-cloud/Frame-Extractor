// Improved upload script with drag/drop, progress bar, and client-side zip extraction using JSZip
const fileInput = document.getElementById('file')
const browseBtn = document.getElementById('browse')
const uploadBtn = document.getElementById('upload')
const statusEl = document.getElementById('status')
const errorBar = document.getElementById('error-bar')
const progressWrap = document.getElementById('progress-wrap')
const progressBar = document.getElementById('progress')
const dropArea = document.getElementById('drop-area')
const fileInfo = document.getElementById('file-info')
const frameCount = document.getElementById('frame-count')
const downloadLink = document.getElementById('download-link')
const showThumbsBtn = document.getElementById('show-thumbs')
const clearBtn = document.getElementById('clear')
const thumbGrid = document.getElementById('thumb-grid')

let currentFile = null
let lastZipBlob = null
let thumbnails = []

function setStatus(msg, type='') {
  statusEl.textContent = msg
  statusEl.className = type
  // Clear error bar when status updates
  if (!msg.includes('failed') && !msg.includes('error') && !msg.includes('Error')) {
    errorBar.style.display = 'none'
  }
}

function setError(msg) {
  errorBar.textContent = '❌ ' + msg
  errorBar.style.display = 'block'
}

function clearError() {
  errorBar.style.display = 'none'
}

function updateProgress(pct) {
  progressWrap.style.display = pct >= 0 ? 'block' : 'none'
  progressBar.style.width = pct + '%'
  progressBar.textContent = pct + '%'
}

browseBtn.addEventListener('click', () => fileInput.click())

fileInput.addEventListener('change', (e) => {
  const f = e.target.files[0]
  onFileSelected(f)
})

dropArea.addEventListener('dragover', (e) => { e.preventDefault(); dropArea.classList.add('dragover') })
dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'))
dropArea.addEventListener('drop', (e) => {
  e.preventDefault(); dropArea.classList.remove('dragover')
  const f = e.dataTransfer.files[0]
  onFileSelected(f)
})

function onFileSelected(f) {
  if (!f) return
  currentFile = f
  const intervalVal = parseFloat(document.getElementById('interval').value || '0.1')
  fileInfo.textContent = `${f.name} — ${Math.round(f.size/1024)} KB — interval ${intervalVal}s`
  uploadBtn.disabled = false
  setStatus('Extracting first frame for color sampling...')
  
  // Extract and display first frame
  const form = new FormData()
  form.append('video', f)
  
  fetch('http://127.0.0.1:5000/extract_first_frame', {
    method: 'POST',
    body: form
  })
  .then(r => r.json())
  .then(data => {
    if (data.success && data.frame) {
      // Display first frame in eyedropper canvas
      const img = new Image()
      
      img.onload = () => {
        eyedropperCanvas.width = img.width
        eyedropperCanvas.height = img.height
        const ctx = eyedropperCanvas.getContext('2d')
        ctx.drawImage(img, 0, 0)
        
        eyedropperCanvasContainer.style.display = 'block'
        eyedropperActive = true
        eyedropperBtn.textContent = 'Click to pick color from video'
        eyedropperBtn.classList.add('active')
        
        setStatus('Ready! Click on the video preview to pick a color, then upload')
      }
      img.src = data.frame
    } else {
      setError('Failed to extract first frame')
      setStatus('Ready to upload')
    }
  })
  .catch(e => {
    console.error('Error extracting first frame:', e)
    setError('Could not extract first frame: ' + e.message)
    setStatus('Ready to upload (without color preview)')
  })
}

clearBtn.addEventListener('click', () => {
  fileInput.value = ''
  fileInfo.textContent = ''
  uploadBtn.disabled = true
  setStatus('Cleared')
  downloadLink.classList.add('d-none')
  showThumbsBtn.classList.add('d-none')
  frameCount.textContent = 'No frames yet'
  thumbGrid.innerHTML = ''
  thumbnails = []
})

showThumbsBtn.addEventListener('click', () => {
  renderThumbnails()
})

document.getElementById('demo').addEventListener('click', () => {
  setStatus('Requesting demo from server...')
  updateProgress(0)

  // Get tracking enabled status
  const trackingEnabled = document.getElementById('enable-tracking').checked
  
  // Get color parameters only if tracking is enabled
  let colorHex = ''
  if (trackingEnabled) {
    colorHex = document.getElementById('color-hex').value.replace('#', '')
    if (!colorHex) {
      setError('Please select a color for demo.')
      return
    }
  }
  
  const tolerance = 20  // Stricter for better accuracy
  const accuracy = document.getElementById('accuracy').value  // User-selected accuracy

  const form = new FormData()
  if (trackingEnabled && colorHex) {
    form.append('track_color', colorHex)
    form.append('color_tolerance', String(tolerance))
    form.append('min_pixels', String(accuracy))
  }

  const xhr = new XMLHttpRequest()
  xhr.open('POST', 'http://127.0.0.1:5000/demo')
  xhr.responseType = 'blob'
  xhr.onload = async () => {
    updateProgress(100)
    if (xhr.status >= 200 && xhr.status < 300) {
      const detector = xhr.getResponseHeader('X-Used-Detector') || 'none'
      const markers = xhr.getResponseHeader('X-Markers-Count') || '0'
      const frames = xhr.getResponseHeader('X-Frames-Count') || '0'
      const detectionMode = trackingEnabled ? `detector: ${detector}, markers: ${markers}` : 'frame extraction only'
      setStatus(`Demo completed — ${detectionMode}`)
      lastZipBlob = xhr.response
      const url = window.URL.createObjectURL(lastZipBlob)
      downloadLink.href = url
      downloadLink.classList.remove('d-none')

      // Try to extract preview thumbnails if possible
      try {
        const jszip = new JSZip()
        const zip = await jszip.loadAsync(lastZipBlob)
        thumbnails = []
        for (const fname of Object.keys(zip.files)) {
          const lower = fname.toLowerCase()
          if (lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.png') || lower.endsWith('.webp')) {
            thumbnails.push(fname)
          }
        }
        const serverFrames = parseInt(frames, 10) || 0
        frameCount.textContent = `${serverFrames} frames (preview ${thumbnails.length})`
        if (thumbnails.length > 0) {
          showThumbsBtn.classList.remove('d-none')
          // preload first few
          thumbGrid.innerHTML = ''
          const toShow = thumbnails.slice(0, 8)
          for (const name of toShow) {
            const entry = zip.files[name]
            const blob = await entry.async('blob')
            const imgUrl = URL.createObjectURL(blob)
            const col = document.createElement('div')
            col.className = 'col-6 col-md-4'
            const img = document.createElement('img')
            img.src = imgUrl
            img.alt = name
            col.appendChild(img)
            thumbGrid.appendChild(col)
          }
          setStatus(`Frames ready — detector: ${detector}, markers: ${markers} — click "Show thumbnails" for full grid`)
        } else if (serverFrames > 0) {
          setStatus(`Frames ZIP received: ${serverFrames} frames, but no image previews found (you can still download frames.zip)`)
        } else {
          setStatus('Demo completed but no frames in ZIP')
        }
      } catch (err) {
        setStatus('Demo completed but error reading ZIP: ' + err.message)
      }

    } else {
      setError('Demo failed: ' + xhr.statusText)
    }
  }
  xhr.onerror = () => setError('Network error during demo request')
  xhr.send(form)
})

uploadBtn.addEventListener('click', () => {
  if (!currentFile) { setError('Choose a video first.'); return }

  // Validate interval client-side
  let intervalVal = parseFloat(document.getElementById('interval').value || '0.1')
  if (isNaN(intervalVal) || intervalVal < 0.01) intervalVal = 0.1
  if (intervalVal > 5.0) intervalVal = 5.0

  // Get tracking enabled status
  const trackingEnabled = document.getElementById('enable-tracking').checked
  
  // Get color parameters only if tracking is enabled
  let colorHex = ''
  if (trackingEnabled) {
    colorHex = document.getElementById('color-hex').value.trim()
    if (!colorHex) {
      // Fallback to color-picker if hex is empty
      const pickerVal = document.getElementById('color-picker').value
      colorHex = pickerVal.replace('#', '')
    }
    colorHex = colorHex.replace('#', '')
    
    if (!colorHex) { setError('Please select a color.'); return }
  }
  
  const tolerance = 20  // High-accuracy strict tolerance
  const accuracy = document.getElementById('accuracy').value  // User-selected accuracy

  const form = new FormData()
  form.append('video', currentFile)
  form.append('interval', String(intervalVal))
  if (trackingEnabled && colorHex) {
    form.append('track_color', colorHex)
    form.append('color_tolerance', String(tolerance))
    form.append('min_pixels', String(accuracy))
  }

  setStatus('Uploading...')
  clearError()
  updateProgress(0)

  const xhr = new XMLHttpRequest()
  xhr.open('POST', 'http://127.0.0.1:5000/upload')
  xhr.responseType = 'blob'

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100)
      updateProgress(pct)
      setStatus('Uploading: ' + pct + '%')
    }
  }

  xhr.onload = async () => {
    updateProgress(100)
    const detector = xhr.getResponseHeader('X-Used-Detector') || 'none'
    const markers = xhr.getResponseHeader('X-Markers-Count') || '0'
    const frames = xhr.getResponseHeader('X-Frames-Count') || '0'
    const trackingEnabled = document.getElementById('enable-tracking').checked

    if (xhr.status >= 200 && xhr.status < 300) {
      const detectionMode = trackingEnabled ? `detector: ${detector}, markers found: ${markers}` : 'frame extraction only'
      setStatus(`Processing completed — ${detectionMode}`)
      lastZipBlob = xhr.response
      const url = window.URL.createObjectURL(lastZipBlob)
      downloadLink.href = url
      downloadLink.classList.remove('d-none')

      // Use JSZip to read and display thumbnails
      try {
        setStatus('Reading ZIP and extracting thumbnails...')
        const jszip = new JSZip()
        const zip = await jszip.loadAsync(lastZipBlob)
        thumbnails = []
        for (const fname of Object.keys(zip.files)) {
          const lower = fname.toLowerCase()
          if (lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.png') || lower.endsWith('.webp')) {
            thumbnails.push(fname)
          }
        }
        const serverFrames = parseInt(frames, 10) || 0
        frameCount.textContent = `${serverFrames} frames (preview ${thumbnails.length})`
        if (thumbnails.length > 0) {
          showThumbsBtn.classList.remove('d-none')
          // Preload up to first 8 thumbnails to show immediately
          thumbGrid.innerHTML = ''
          const toShow = thumbnails.slice(0, 8)
          for (const name of toShow) {
            const entry = zip.files[name]
            const blob = await entry.async('blob')
            const imgUrl = URL.createObjectURL(blob)
            const col = document.createElement('div')
            col.className = 'col-6 col-md-4'
            const img = document.createElement('img')
            img.src = imgUrl
            img.alt = name
            col.appendChild(img)
            thumbGrid.appendChild(col)
          }
          setStatus(`Frames ready — detector: ${detector}, markers: ${markers} — click "Show thumbnails" for full grid`)
        } else if (serverFrames > 0) {
          setStatus(`Frames ZIP received: ${serverFrames} frames, but no image previews found (you can still download frames.zip)`)
          showThumbsBtn.classList.add('d-none')
        } else {
          setStatus('No frames found in ZIP')
        }

        // If server included detections.json, show a small summary
        if (zip.files['detections.json']) {
          try {
            const txt = await zip.files['detections.json'].async('string')
            const det = JSON.parse(txt)
            const keys = Object.keys(det).slice(0, 6)
            const lines = keys.map(k => `${k}: ${JSON.stringify(det[k])}`)
            setStatus((serverFrames || thumbnails.length) + ' frames — detector: ' + detector + ', markers: ' + markers + '\n' + lines.join('\n'))
          } catch (e) {
            // ignore
          }
        }

        // Extract and display frame_data.csv as a table
        if (zip.files['frame_data.csv']) {
          try {
            const csvText = await zip.files['frame_data.csv'].async('string')
            displayFrameDataTable(csvText)
            
            // Also provide CSV download
            const csvBlob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' })
            const csvUrl = URL.createObjectURL(csvBlob)
            const downloadCsv = document.getElementById('download-csv')
            downloadCsv.href = csvUrl
            downloadCsv.classList.remove('d-none')
          } catch (e) {
            console.error('Error reading CSV:', e)
          }
        }
      } catch (err) {
        setStatus('Error reading ZIP: ' + err.message)
      }
    } else {
      try { const json = JSON.parse(await blobText(xhr.response)); setError('Upload failed: ' + (json.error || xhr.statusText)) } catch (e) { setError('Upload failed: ' + xhr.statusText) }
    }
  }

  xhr.onerror = () => { setError('Network error during upload') }
  xhr.send(form)
})

async function blobText(b) { try { return await (new Response(b)).text() } catch (e) { return '' } }

function renderThumbnails() {
  if (!lastZipBlob) return
  setStatus('Loading all thumbnails...')
  const jszip = new JSZip()
  jszip.loadAsync(lastZipBlob).then(async (zip) => {
    thumbGrid.innerHTML = ''
    for (const name of thumbnails) {
      const entry = zip.files[name]
      const blob = await entry.async('blob')
      const imgUrl = URL.createObjectURL(blob)
      const col = document.createElement('div')
      col.className = 'col-6 col-md-3'
      const img = document.createElement('img')
      img.src = imgUrl
      img.alt = name
      col.appendChild(img)
      thumbGrid.appendChild(col)
    }
    setStatus('All thumbnails loaded')
  }).catch(err => setStatus('Error loading thumbnails: ' + err.message))
}
// Color picker event listeners
document.getElementById('color-picker').addEventListener('change', (e) => {
  const hex = e.target.value.substring(1)
  document.getElementById('color-hex').value = hex
  document.getElementById('color-preview').style.backgroundColor = e.target.value
})

document.getElementById('color-hex').addEventListener('change', (e) => {
  let hex = e.target.value.replace('#', '').toUpperCase()
  if (hex.length === 6 && /^[0-9A-F]{6}$/.test(hex)) {
    document.getElementById('color-picker').value = '#' + hex
    document.getElementById('color-preview').style.backgroundColor = '#' + hex
  }
})

// Eyedropper functionality
const eyedropperBtn = document.getElementById('eyedropper-btn')
const eyedropperFileInput = document.getElementById('eyedropper-file-input')
const eyedropperCanvas = document.getElementById('eyedropper-canvas')
const eyedropperCanvasContainer = document.getElementById('eyedropper-canvas-container')

let eyedropperActive = false

if (eyedropperBtn && eyedropperFileInput) {
  eyedropperBtn.addEventListener('click', (e) => {
    e.preventDefault()
    // If canvas already has video frame, don't open file dialog
    if (eyedropperCanvasContainer.style.display !== 'none' && eyedropperCanvas.width > 0) {
      return
    }
    // Otherwise open file dialog for manual image selection
    eyedropperFileInput.click()
  })
}

if (eyedropperFileInput) {
  eyedropperFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    const reader = new FileReader()
    reader.onload = (evt) => {
      const img = new Image()
      img.onload = () => {
        eyedropperCanvas.width = img.width
        eyedropperCanvas.height = img.height
        const ctx = eyedropperCanvas.getContext('2d')
        ctx.drawImage(img, 0, 0)
        
        eyedropperCanvasContainer.style.display = 'block'
        eyedropperActive = true
        eyedropperBtn.textContent = 'Click image to pick'
        eyedropperBtn.classList.add('active')
      }
      img.src = evt.target.result
    }
    reader.readAsDataURL(file)
  })
}

eyedropperCanvas.addEventListener('click', (e) => {
  if (!eyedropperActive) return
  
  const rect = eyedropperCanvas.getBoundingClientRect()
  const x = Math.floor((e.clientX - rect.left) * (eyedropperCanvas.width / rect.width))
  const y = Math.floor((e.clientY - rect.top) * (eyedropperCanvas.height / rect.height))
  
  const ctx = eyedropperCanvas.getContext('2d')
  const imageData = ctx.getImageData(x, y, 1, 1)
  const data = imageData.data
  
  const hex = ('0' + data[0].toString(16)).slice(-2) +
              ('0' + data[1].toString(16)).slice(-2) +
              ('0' + data[2].toString(16)).slice(-2)
  
  document.getElementById('color-picker').value = '#' + hex.toUpperCase()
  document.getElementById('color-hex').value = hex.toUpperCase()
  document.getElementById('color-preview').style.backgroundColor = '#' + hex.toUpperCase()
  
  resetEyedropper()
  setStatus('Color picked: #' + hex.toUpperCase())
})

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && eyedropperActive) {
    resetEyedropper()
  }
})

function resetEyedropper() {
  eyedropperActive = false
  eyedropperCanvasContainer.style.display = 'none'
  eyedropperBtn.textContent = 'Eyedropper'
  eyedropperBtn.classList.remove('active')
  eyedropperFileInput.value = ''
}

// Display frame data as a table
function displayFrameDataTable(csvText) {
  const dataSheetContainer = document.getElementById('data-sheet-container')
  const dataSheetBody = document.getElementById('data-sheet-body')
  
  // Parse CSV text and store frame data globally
  const lines = csvText.trim().split('\n')
  if (lines.length < 2) {
    dataSheetContainer.style.display = 'none'
    return
  }
  
  // Store frame data for plotting
  window.frameData = []
  
  // Clear table body
  dataSheetBody.innerHTML = ''
  
  // Process each line (skip header)
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',')
    if (values.length < 4) continue
    
    const frame = values[0].trim()
    const x = values[1].trim()
    const y = values[2].trim()
    const time = values[3].trim()
    
    // Store data point
    window.frameData.push({
      frame: parseInt(frame),
      x: x && x !== 'None' ? parseFloat(x) : null,
      y: y && y !== 'None' ? parseFloat(y) : null,
      time: parseFloat(time)
    })
    
    const row = document.createElement('tr')
    row.innerHTML = `
      <td><strong>${frame}</strong></td>
      <td>${x && x !== 'None' ? x : '—'}</td>
      <td>${y && y !== 'None' ? y : '—'}</td>
      <td>${time}</td>
    `
    dataSheetBody.appendChild(row)
  }
  
  // Show the container and plot buttons
  dataSheetContainer.style.display = 'block'
  document.getElementById('download-csv').classList.remove('d-none')
  document.getElementById('plot-x-btn').classList.remove('d-none')
  document.getElementById('plot-y-btn').classList.remove('d-none')
  
  // Show conversion section
  document.getElementById('conversion-section').style.display = 'block'
  
  // Setup plot button listeners
  document.getElementById('plot-x-btn').addEventListener('click', () => showPlot('x'))
  document.getElementById('plot-y-btn').addEventListener('click', () => showPlot('y'))
}

// Global variable to store frame data
window.frameData = []

// Function to show plots
function showPlot(type) {
  if (!window.frameData || window.frameData.length === 0) return
  
  const modal = document.getElementById('plot-modal')
  const plotContainer = document.getElementById('plot-container')
  const plotTitle = document.getElementById('plot-title')
  const saveBtn = document.getElementById('plot-save-btn')
  
  // Clear previous plot
  plotContainer.innerHTML = ''
  
  // Prepare data
  const validData = window.frameData.filter(d => {
    const val = type === 'x' ? d.x : d.y
    return val !== null && val !== undefined
  })
  
  const validTimes = validData.map(d => d.time)
  const validValues = validData.map(d => type === 'x' ? d.x : d.y)
  
  if (validValues.length === 0) {
    alert(`No ${type.toUpperCase()} values to plot`)
    return
  }
  
  // Set title
  const titleText = type === 'x' ? 'X Coordinate vs Time' : 'Y Coordinate vs Time'
  plotTitle.textContent = titleText
  
  // Create trace
  const trace = {
    x: validTimes,
    y: validValues,
    mode: 'lines+markers',
    type: 'scatter',
    name: type.toUpperCase(),
    line: { color: type === 'x' ? '#0066cc' : '#cc0000', width: 2 },
    marker: { size: 6 }
  }
  
  // Create layout
  const layout = {
    title: { text: titleText },
    xaxis: { title: 'Time (seconds)' },
    yaxis: { title: type === 'x' ? 'X Coordinate (px)' : 'Y Coordinate (px)' },
    hovermode: 'closest',
    plot_bgcolor: '#f8f9fa',
    paper_bgcolor: 'white',
    margin: { l: 80, r: 60, t: 40, b: 60 },
    width: 1000,
    height: 500
  }
  
  const config = { responsive: false, displayModeBar: true }
  
  // Plot the data using setTimeout to ensure DOM is ready
  setTimeout(() => {
    try {
      Plotly.newPlot(plotContainer, [trace], layout, config)
    } catch(e) {
      console.error('Plot error:', e)
      plotContainer.innerHTML = '<p style="padding: 20px; color: red;">Error creating plot</p>'
    }
  }, 100)
  
  // Save button handler
  saveBtn.onclick = () => {
    Plotly.downloadImage(plotContainer, {
      format: 'png',
      width: 1200,
      height: 600,
      filename: `${type.toUpperCase()}_vs_Time`
    })
  }
  
  // Show modal with proper display
  modal.style.display = 'block'
  modal.offsetHeight
  modal.style.visibility = 'visible'
  modal.style.opacity = '1'
}

// Modal close handlers
document.getElementById('plot-close-btn').addEventListener('click', () => {
  const modal = document.getElementById('plot-modal')
  modal.style.opacity = '0'
  modal.style.visibility = 'hidden'
  setTimeout(() => { modal.style.display = 'none' }, 300)
})
document.getElementById('plot-close-btn2').addEventListener('click', () => {
  const modal = document.getElementById('plot-modal')
  modal.style.opacity = '0'
  modal.style.visibility = 'hidden'
  setTimeout(() => { modal.style.display = 'none' }, 300)
})

// Pixel to Millimeter Conversion System
window.conversionFactors = {
  x: 1,  // px/mm
  y: 1   // px/mm
}

// Apply X conversion
document.getElementById('apply-x-conversion').addEventListener('click', () => {
  const pxPerMm = parseFloat(document.getElementById('px-mm-x-input').value)
  if (isNaN(pxPerMm) || pxPerMm <= 0) {
    alert('Please enter a valid positive number for X conversion factor')
    return
  }
  window.conversionFactors.x = pxPerMm
  displayConvertedDataSheet('x')
})

// Apply Y conversion
document.getElementById('apply-y-conversion').addEventListener('click', () => {
  const pxPerMm = parseFloat(document.getElementById('px-mm-y-input').value)
  if (isNaN(pxPerMm) || pxPerMm <= 0) {
    alert('Please enter a valid positive number for Y conversion factor')
    return
  }
  window.conversionFactors.y = pxPerMm
  displayConvertedDataSheet('y')
})

// Reset conversions
document.getElementById('reset-conversions').addEventListener('click', () => {
  window.conversionFactors.x = 1
  window.conversionFactors.y = 1
  document.getElementById('px-mm-x-input').value = '1'
  document.getElementById('px-mm-y-input').value = '1'
  document.getElementById('mm-x-data-sheet-container').style.display = 'none'
  document.getElementById('mm-y-data-sheet-container').style.display = 'none'
})

// Display converted data sheet (pixels to millimeters)
function displayConvertedDataSheet(axis) {
  if (!window.frameData || window.frameData.length === 0) return
  
  const containerId = axis === 'x' ? 'mm-x-data-sheet-container' : 'mm-y-data-sheet-container'
  const bodyId = axis === 'x' ? 'mm-x-data-sheet-body' : 'mm-y-data-sheet-body'
  const container = document.getElementById(containerId)
  const tbody = document.getElementById(bodyId)
  
  const pxPerMm = axis === 'x' ? window.conversionFactors.x : window.conversionFactors.y
  
  // Clear table body
  tbody.innerHTML = ''
  
  // Process frame data and convert to mm
  for (const data of window.frameData) {
    const pixelValue = axis === 'x' ? data.x : data.y
    
    if (pixelValue === null || pixelValue === undefined) {
      const row = document.createElement('tr')
      row.innerHTML = `
        <td><strong>${data.frame}</strong></td>
        <td>—</td>
        <td>${data.time}</td>
      `
      tbody.appendChild(row)
    } else {
      const mmValue = (pixelValue / pxPerMm).toFixed(2)
      const row = document.createElement('tr')
      row.innerHTML = `
        <td><strong>${data.frame}</strong></td>
        <td>${mmValue}</td>
        <td>${data.time}</td>
      `
      tbody.appendChild(row)
    }
  }
  
  // Show the container and plot button
  container.style.display = 'block'
  document.getElementById('conversion-section').style.display = 'block'
  document.getElementById('conversion-data-section').style.display = 'block'
}

// Show mm plots
function showMmPlot(axis) {
  if (!window.frameData || window.frameData.length === 0) return
  
  const modal = document.getElementById('plot-modal')
  const plotContainer = document.getElementById('plot-container')
  const plotTitle = document.getElementById('plot-title')
  const saveBtn = document.getElementById('plot-save-btn')
  
  // Clear previous plot
  plotContainer.innerHTML = ''
  
  const pxPerMm = axis === 'x' ? window.conversionFactors.x : window.conversionFactors.y
  
  // Filter and convert data
  const validData = window.frameData.filter(d => {
    const val = axis === 'x' ? d.x : d.y
    return val !== null && val !== undefined
  })
  
  if (validData.length === 0) {
    alert(`No ${axis.toUpperCase()} values to plot`)
    return
  }
  
  const validTimes = validData.map(d => d.time)
  const validMmValues = validData.map(d => {
    const pixelValue = axis === 'x' ? d.x : d.y
    return pixelValue / pxPerMm
  })
  
  // Set title
  const titleText = axis === 'x' ? `X Distance vs Time (${window.conversionFactors.x} px/mm)` : `Y Distance vs Time (${window.conversionFactors.y} px/mm)`
  plotTitle.textContent = titleText
  
  // Create trace
  const trace = {
    x: validTimes,
    y: validMmValues,
    mode: 'lines+markers',
    type: 'scatter',
    name: 'Distance (mm)',
    line: { color: '#0d6efd', width: 2 },
    marker: { size: 6 }
  }
  
  // Create layout
  const layout = {
    title: { text: titleText },
    xaxis: { title: 'Time (seconds)' },
    yaxis: { title: 'Distance (mm)' },
    hovermode: 'closest',
    plot_bgcolor: '#f8f9fa',
    paper_bgcolor: 'white',
    margin: { l: 80, r: 60, t: 40, b: 60 },
    width: 1000,
    height: 500
  }
  
  const config = { responsive: false, displayModeBar: true }
  
  // Plot the data using setTimeout
  setTimeout(() => {
    try {
      Plotly.newPlot(plotContainer, [trace], layout, config)
    } catch(e) {
      console.error('Plot error:', e)
      plotContainer.innerHTML = '<p style="padding: 20px; color: red;">Error creating plot</p>'
    }
  }, 100)
  
  // Save button handler
  saveBtn.onclick = () => {
    Plotly.downloadImage(plotContainer, {
      format: 'png',
      width: 1200,
      height: 600,
      filename: `${axis.toUpperCase()}_Distance_vs_Time`
    })
  }
  
  // Show modal with proper display
  modal.style.display = 'block'
  modal.offsetHeight
  modal.style.visibility = 'visible'
  modal.style.opacity = '1'
}

// Attach mm plot button listeners
document.getElementById('plot-mm-x-btn').addEventListener('click', () => showMmPlot('x'))
document.getElementById('plot-mm-y-btn').addEventListener('click', () => showMmPlot('y'))

// Range slider value display
document.getElementById('accuracy').addEventListener('input', (e) => {
  document.getElementById('accuracy-value').textContent = e.target.value + ' pixels'
})