import { ImagePlus, RotateCcw, ScanLine } from 'lucide-react'
import { useRef, useState } from 'react'

function ImageUploader({ file, previewUrl, loading, onFileSelect, onAnalyze, onReset }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const chooseFile = (selectedFile) => {
    if (selectedFile) {
      onFileSelect(selectedFile)
    }
  }

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  return (
    <section className="panel">
      <div className="section-head">
        <div>
          <h2>Upload and analyze</h2>
          <p>A clear close-up leaf photo works best. Supported formats: JPG, PNG, JPEG.</p>
        </div>
        <span className="pill">EfficientNet-B3</span>
      </div>

      <label
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          onChange={(event) => chooseFile(event.target.files?.[0])}
        />
        {previewUrl ? (
          <div className="preview">
            <img src={previewUrl} alt="Uploaded leaf preview" />
          </div>
        ) : (
          <div>
            <div className="upload-icon">
              <ImagePlus size={26} />
            </div>
            <p className="upload-title">Drop a leaf image here</p>
            <p className="upload-copy">Browse from your device or drag a file into this panel.</p>
          </div>
        )}
      </label>

      {file && (
        <div className="file-meta">
          <div className="detail">
            <span>Filename</span>
            <strong>{file.name}</strong>
          </div>
          <div className="detail">
            <span>File size</span>
            <strong>{(file.size / 1024).toFixed(1)} KB</strong>
          </div>
        </div>
      )}

      <div className="actions">
        <button className="primary-btn" type="button" onClick={onAnalyze} disabled={!file || loading}>
          <ScanLine size={18} />
          {loading ? 'Analyzing...' : 'Analyze leaf'}
        </button>
        {file && (
          <button className="ghost-btn" type="button" onClick={onReset} disabled={loading}>
            <RotateCcw size={18} />
            Reset
          </button>
        )}
      </div>
    </section>
  )
}

export default ImageUploader
