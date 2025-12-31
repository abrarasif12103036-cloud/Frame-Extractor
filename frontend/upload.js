// Improved upload script with drag/drop, progress bar, and client-side zip extraction using JSZip
const fileInput = document.getElementById('file')
const browseBtn = document.getElementById('browse')
const uploadBtn = document.getElementById('upload')
const statusEl = document.getElementById('status')
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
  setStatus('Ready to upload')
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
  const xhr = new XMLHttpRequest()
  xhr.open('POST', 'http://127.0.0.1:5000/demo')
  xhr.responseType = 'blob'
  xhr.onload = async () => {
    updateProgress(100)
    if (xhr.status >= 200 && xhr.status < 300) {
      const detector = xhr.getResponseHeader('X-Used-Detector') || 'unknown'
      const markers = xhr.getResponseHeader('X-Markers-Count') || '0'
      const frames = xhr.getResponseHeader('X-Frames-Count') || '0'
      setStatus(`Demo completed — detector: ${detector}, markers: ${markers}`)
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
      setStatus('Demo failed: ' + xhr.statusText)
    }
  }
  xhr.onerror = () => setStatus('Network error during demo request')
  xhr.send()
})

uploadBtn.addEventListener('click', () => {
  if (!currentFile) { setStatus('Choose a video first.'); return }

  // Validate interval client-side
  let intervalVal = parseFloat(document.getElementById('interval').value || '0.1')
  if (isNaN(intervalVal) || intervalVal < 0.01) intervalVal = 0.1
  if (intervalVal > 5.0) intervalVal = 5.0

  const form = new FormData()
  form.append('video', currentFile)
  form.append('interval', String(intervalVal))
  const track = document.getElementById('track-red').checked ? '1' : '0'
  form.append('track_red', track)
  // Pass detection tuning parameters
  const minPixels = document.getElementById('min_pixels') ? document.getElementById('min_pixels').value : '80'
  const redMin = document.getElementById('red_min') ? document.getElementById('red_min').value : '150'
  form.append('min_pixels', String(minPixels))
  form.append('red_min', String(redMin))

  setStatus('Uploading...')
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
    const detector = xhr.getResponseHeader('X-Used-Detector') || 'unknown'
    const markers = xhr.getResponseHeader('X-Markers-Count') || '0'
    const frames = xhr.getResponseHeader('X-Frames-Count') || '0'

    if (xhr.status >= 200 && xhr.status < 300) {
      setStatus(`Processing completed — detector: ${detector}, markers found: ${markers}`)
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
      } catch (err) {
        setStatus('Error reading ZIP: ' + err.message)
      }
    } else {
      try { const json = JSON.parse(await blobText(xhr.response)); setStatus('Upload failed: ' + (json.error || xhr.statusText)) } catch (e) { setStatus('Upload failed: ' + xhr.statusText) }
    }
  }

  xhr.onerror = () => { setStatus('Network error during upload') }
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
