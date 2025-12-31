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

uploadBtn.addEventListener('click', () => {
  if (!currentFile) { setStatus('Choose a video first.'); return }

  // Validate interval client-side
  let intervalVal = parseFloat(document.getElementById('interval').value || '0.1')
  if (isNaN(intervalVal) || intervalVal < 0.01) intervalVal = 0.1
  if (intervalVal > 5.0) intervalVal = 5.0

  const form = new FormData()
  form.append('video', currentFile)
  form.append('interval', String(intervalVal))

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
    if (xhr.status >= 200 && xhr.status < 300) {
      setStatus('Processing completed — received frames.zip')
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
          if (fname.toLowerCase().endsWith('.jpg') || fname.toLowerCase().endsWith('.png')) {
            thumbnails.push(fname)
          }
        }
        frameCount.textContent = `${thumbnails.length} frames`
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
          setStatus('Frames ready — click "Show thumbnails" for full grid')
        } else {
          setStatus('No frames found in ZIP')
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
